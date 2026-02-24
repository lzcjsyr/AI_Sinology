from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.llm_client import OpenAICompatClient
from core.utils import append_jsonl, parse_json_from_text, read_json, read_jsonl, write_json


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _heuristic_matches(
    fragment_text: str,
    target_themes: list[dict[str, str]],
) -> list[dict[str, Any]]:
    text = fragment_text.lower()
    matches: list[dict[str, Any]] = []
    for theme_item in target_themes:
        theme = theme_item["theme"]
        desc = theme_item.get("description", "")
        tokens = [theme.lower()] + [t for t in desc.lower().split() if len(t) > 2]
        found = any(token and token in text for token in tokens[:5])
        if found:
            matches.append(
                {
                    "theme": theme,
                    "is_relevant": True,
                    "reason": "启发式命中主题关键词",
                    "relevance_level": "LOW",
                }
            )
        else:
            matches.append(
                {
                    "theme": theme,
                    "is_relevant": False,
                    "reason": None,
                    "relevance_level": None,
                }
            )
    return matches


def _normalize_matches(
    payload: dict[str, Any],
    target_themes: list[dict[str, str]],
) -> list[dict[str, Any]]:
    raw = payload.get("matches")
    by_theme: dict[str, dict[str, Any]] = {}
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            theme = str(item.get("theme", "")).strip()
            if theme:
                by_theme[theme] = item

    result: list[dict[str, Any]] = []
    for theme_item in target_themes:
        theme = theme_item["theme"]
        src = by_theme.get(theme)
        if src is None:
            result.append(
                {
                    "theme": theme,
                    "is_relevant": False,
                    "reason": None,
                    "relevance_level": None,
                }
            )
            continue

        is_relevant = bool(src.get("is_relevant", False))
        reason = src.get("reason")
        relevance_level = src.get("relevance_level")
        if not is_relevant:
            reason = None
            relevance_level = None
        elif relevance_level is not None:
            relevance_level = str(relevance_level).upper()
            if relevance_level not in {"HIGH", "MEDIUM", "LOW"}:
                relevance_level = "LOW"

        result.append(
            {
                "theme": theme,
                "is_relevant": is_relevant,
                "reason": reason,
                "relevance_level": relevance_level,
            }
        )

    return result


def _classify_fragment(
    *,
    llm_client: OpenAICompatClient,
    model: str,
    fragment: dict[str, str],
    target_themes: list[dict[str, str]],
    logger,
) -> list[dict[str, Any]]:
    theme_lines = [f"- {t['theme']}: {t.get('description', '')}" for t in target_themes]
    prompt = (
        "你是古籍史料筛选助手。请严格返回 JSON 对象，格式："
        '{"matches":[{"theme":"...","is_relevant":true/false,"reason":"...或null","relevance_level":"HIGH|MEDIUM|LOW|null"}]}'
        "。\n要求：\n"
        "1) 每个主题都必须返回一条 matches 记录。\n"
        "2) 若 is_relevant=false，则 reason 和 relevance_level 必须为 null。\n"
        "3) 禁止返回任何 JSON 之外的文字。\n\n"
        f"目标主题：\n{chr(10).join(theme_lines)}\n\n"
        f"史料片段 piece_id={fragment['piece_id']}\n"
        f"source_file={fragment['source_file']}\n"
        f"original_text=\n{fragment['original_text']}"
    )

    messages = [
        {"role": "system", "content": "你是严谨的结构化信息抽取器。"},
        {"role": "user", "content": prompt},
    ]

    try:
        response = llm_client.chat(messages, model=model, temperature=0.0)
        payload = parse_json_from_text(response.content)
        return _normalize_matches(payload, target_themes)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "片段筛选失败，回退启发式。piece_id=%s model=%s error=%s",
            fragment.get("piece_id"),
            model,
            e,
        )
        return _heuristic_matches(fragment.get("original_text", ""), target_themes)


def _flatten_records(
    fragment: dict[str, str],
    matches: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for match in matches:
        is_relevant = bool(match.get("is_relevant", False))
        record = {
            "piece_id": fragment["piece_id"],
            "source_file": fragment["source_file"],
            "original_text": fragment["original_text"],
            "matched_theme": match.get("theme"),
            "is_relevant": is_relevant,
            "reason": match.get("reason") if is_relevant else None,
            "relevance_level": match.get("relevance_level") if is_relevant else None,
        }
        records.append(record)
    return records


async def _run_single_model(
    *,
    tag: str,
    model: str,
    llm_client: OpenAICompatClient,
    target_themes: list[dict[str, str]],
    fragments: list[dict[str, str]],
    raw_output_path: Path,
    cursor_path: Path,
    logger,
) -> None:
    next_index = 0
    if cursor_path.exists():
        try:
            cursor = read_json(cursor_path)
            next_index = int(cursor.get("next_index", 0))
        except Exception:  # noqa: BLE001
            next_index = 0

    total = len(fragments)
    if next_index >= total:
        logger.info("%s 无需继续，已完成。", tag)
        return

    logger.info("%s 从 index=%s 开始处理，总量=%s", tag, next_index, total)

    for idx in range(next_index, total):
        fragment = fragments[idx]
        matches = await asyncio.to_thread(
            _classify_fragment,
            llm_client=llm_client,
            model=model,
            fragment=fragment,
            target_themes=target_themes,
            logger=logger,
        )
        records = _flatten_records(fragment, matches)
        for record in records:
            append_jsonl(raw_output_path, record)

        write_json(
            cursor_path,
            {
                "tag": tag,
                "model": model,
                "next_index": idx + 1,
                "last_piece_id": fragment.get("piece_id"),
                "updated_at": _now_iso(),
            },
        )


async def run_archival_screening(
    *,
    project_dir: Path,
    fragments_path: Path,
    target_themes: list[dict[str, str]],
    llm_client: OpenAICompatClient,
    model_llm1: str,
    model_llm2: str,
    logger,
) -> tuple[Path, Path]:
    fragments = read_jsonl(fragments_path)
    if not fragments:
        raise RuntimeError(f"阶段2.2无法继续：未找到可筛选碎片 {fragments_path}")

    llm1_raw_path = project_dir / "2_llm1_raw.jsonl"
    llm2_raw_path = project_dir / "2_llm2_raw.jsonl"
    cursor1_path = project_dir / ".cursor_llm1.json"
    cursor2_path = project_dir / ".cursor_llm2.json"

    await asyncio.gather(
        _run_single_model(
            tag="llm1",
            model=model_llm1,
            llm_client=llm_client,
            target_themes=target_themes,
            fragments=fragments,
            raw_output_path=llm1_raw_path,
            cursor_path=cursor1_path,
            logger=logger,
        ),
        _run_single_model(
            tag="llm2",
            model=model_llm2,
            llm_client=llm_client,
            target_themes=target_themes,
            fragments=fragments,
            raw_output_path=llm2_raw_path,
            cursor_path=cursor2_path,
            logger=logger,
        ),
    )

    logger.info("阶段2.2完成: %s, %s", llm1_raw_path, llm2_raw_path)
    return llm1_raw_path, llm2_raw_path
