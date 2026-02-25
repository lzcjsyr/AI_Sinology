from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.llm_client import OpenAICompatClient
from core.utils import append_jsonl, parse_json_from_text, read_json, read_jsonl, write_json


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ScreeningStats:
    model_tag: str
    model_name: str
    start_index: int
    end_index: int
    processed_fragments: int
    raw_records_written: int
    piece_records_written: int
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_tag": self.model_tag,
            "model_name": self.model_name,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "processed_fragments": self.processed_fragments,
            "raw_records_written": self.raw_records_written,
            "piece_records_written": self.piece_records_written,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


def _theme_names(target_themes: list[dict[str, str]]) -> list[str]:
    names = [str(item.get("theme") or "").strip() for item in target_themes]
    names = [name for name in names if name]
    if len(set(names)) != len(names):
        raise RuntimeError("阶段2.2失败：target_themes 存在重复主题名称")
    if not names:
        raise RuntimeError("阶段2.2失败：target_themes 为空")
    return names


def _normalize_matches_strict(
    payload: dict[str, Any],
    target_themes: list[dict[str, str]],
) -> list[dict[str, Any]]:
    raw = payload.get("matches")
    if not isinstance(raw, list):
        raise ValueError("LLM 返回缺少 matches 数组")

    theme_names = [str(item.get("theme") or "").strip() for item in target_themes]
    id_to_theme = {f"T{i+1}": theme for i, theme in enumerate(theme_names)}
    normalized_name_to_id = {theme: theme_id for theme_id, theme in id_to_theme.items()}

    by_theme_id: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        theme_id = str(item.get("theme_id") or "").strip().upper()
        theme = str(item.get("theme") or "").strip()

        resolved_theme_id = ""
        if theme_id in id_to_theme:
            resolved_theme_id = theme_id
        elif theme in normalized_name_to_id:
            resolved_theme_id = normalized_name_to_id[theme]

        if resolved_theme_id:
            by_theme_id[resolved_theme_id] = item

    result: list[dict[str, Any]] = []
    missing: list[str] = []
    for idx, theme_item in enumerate(target_themes, start=1):
        theme = str(theme_item.get("theme") or "").strip()
        theme_id = f"T{idx}"
        src = by_theme_id.get(theme_id)
        if src is None:
            missing.append(theme_id)
            continue

        if not isinstance(src.get("is_relevant"), bool):
            raise ValueError(f"主题 `{theme_id}` 的 is_relevant 不是布尔值")
        is_relevant = bool(src.get("is_relevant"))

        reason = src.get("reason")
        relevance_level = src.get("relevance_level")

        if not is_relevant:
            reason = None
            relevance_level = None
        else:
            if relevance_level is None:
                raise ValueError(f"主题 `{theme_id}` 判定为相关但缺少 relevance_level")
            relevance_level = str(relevance_level).upper().strip()
            if relevance_level not in {"HIGH", "MEDIUM", "LOW"}:
                raise ValueError(f"主题 `{theme_id}` 的 relevance_level 非法: {relevance_level}")
            reason = str(reason or "").strip()
            if not reason:
                raise ValueError(f"主题 `{theme_id}` 判定为相关但缺少 reason")

        result.append(
            {
                "theme": theme,
                "theme_id": theme_id,
                "is_relevant": is_relevant,
                "reason": reason,
                "relevance_level": relevance_level,
            }
        )

    if missing:
        raise ValueError(f"LLM 返回缺少主题判定: {missing}")
    return result


async def _classify_fragment_strict(
    *,
    llm_client: OpenAICompatClient,
    model: str,
    fragment: dict[str, str],
    target_themes: list[dict[str, str]],
    logger,
    max_attempts: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    theme_lines = [
        f"- T{i+1} | theme: {t['theme']} | description: {t.get('description', '')}"
        for i, t in enumerate(target_themes)
    ]
    prompt = (
        "你是古籍史料筛选助手。请严格返回 JSON 对象，格式："
        '{"matches":[{"theme_id":"T1","theme":"...","is_relevant":true/false,"reason":"...或null","relevance_level":"HIGH|MEDIUM|LOW|null"}]}'
        "。\n要求：\n"
        "1) 每个主题都必须返回一条 matches 记录。\n"
        "2) theme_id 必须严格使用给定的 T1/T2/...，不要自造。\n"
        "3) 若 is_relevant=false，则 reason 和 relevance_level 必须为 null。\n"
        "4) 禁止返回任何 JSON 之外的文字。\n\n"
        f"目标主题：\n{chr(10).join(theme_lines)}\n\n"
        f"史料片段 piece_id={fragment['piece_id']}\n"
        f"source_file={fragment['source_file']}\n"
        f"original_text=\n{fragment['original_text']}"
    )

    messages = [
        {"role": "system", "content": "你是严谨的结构化信息抽取器。"},
        {"role": "user", "content": prompt},
    ]

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = await llm_client.achat(messages, model=model, temperature=0.0)
            payload = parse_json_from_text(response.content)
            matches = _normalize_matches_strict(payload, target_themes)
            return matches, response.usage
        except Exception as e:  # noqa: BLE001
            last_error = e
            logger.warning(
                "片段筛选失败，准备重试。piece_id=%s model=%s attempt=%s error=%s",
                fragment.get("piece_id"),
                model,
                attempt,
                e,
            )

    raise RuntimeError(
        f"片段筛选失败：piece_id={fragment.get('piece_id')} model={model} last_error={last_error}"
    )


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


def _piece_level_record(
    *,
    model_tag: str,
    model_name: str,
    fragment: dict[str, str],
    matches: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "piece_id": fragment["piece_id"],
        "source_file": fragment["source_file"],
        "original_text": fragment["original_text"],
        "model_tag": model_tag,
        "model": model_name,
        "matches": matches,
    }


def _usage_tokens(usage: dict[str, Any] | None) -> tuple[int, int, int]:
    if not usage:
        return 0, 0, 0
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    total = int(usage.get("total_tokens") or (prompt + completion))
    return prompt, completion, total


async def _run_single_model(
    *,
    tag: str,
    model: str,
    llm_client: OpenAICompatClient,
    target_themes: list[dict[str, str]],
    fragments: list[dict[str, str]],
    raw_output_path: Path,
    piece_output_path: Path,
    cursor_path: Path,
    logger,
    concurrency: int,
    fragment_max_attempts: int,
) -> ScreeningStats:
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
        return ScreeningStats(
            model_tag=tag,
            model_name=model,
            start_index=next_index,
            end_index=next_index,
            processed_fragments=0,
            raw_records_written=0,
            piece_records_written=0,
            total_tokens=0,
            prompt_tokens=0,
            completion_tokens=0,
        )

    # Idempotent resume support in case cursor lags behind already written rows.
    existing_flat_rows = read_jsonl(raw_output_path)
    seen_flat = {
        (str(row.get("piece_id") or ""), str(row.get("matched_theme") or ""))
        for row in existing_flat_rows
        if row.get("piece_id") and row.get("matched_theme")
    }
    existing_piece_rows = read_jsonl(piece_output_path)
    seen_piece_ids = {
        str(row.get("piece_id") or "") for row in existing_piece_rows if row.get("piece_id")
    }

    logger.info(
        "%s 从 index=%s 开始处理，总量=%s，并发=%s",
        tag,
        next_index,
        total,
        concurrency,
    )

    async def run_one(idx: int) -> tuple[int, list[dict[str, Any]], dict[str, Any] | None]:
        fragment = fragments[idx]
        matches, usage = await _classify_fragment_strict(
            llm_client=llm_client,
            model=model,
            fragment=fragment,
            target_themes=target_themes,
            logger=logger,
            max_attempts=fragment_max_attempts,
        )
        return idx, matches, usage

    in_flight: dict[asyncio.Task, int] = {}
    buffered_results: dict[int, tuple[list[dict[str, Any]], dict[str, Any] | None]] = {}

    write_index = next_index
    submit_index = next_index
    safe_concurrency = max(1, int(concurrency))

    stats = ScreeningStats(
        model_tag=tag,
        model_name=model,
        start_index=next_index,
        end_index=next_index,
        processed_fragments=0,
        raw_records_written=0,
        piece_records_written=0,
        total_tokens=0,
        prompt_tokens=0,
        completion_tokens=0,
    )

    def schedule_more() -> None:
        nonlocal submit_index
        while submit_index < total and len(in_flight) < safe_concurrency:
            task = asyncio.create_task(run_one(submit_index))
            in_flight[task] = submit_index
            submit_index += 1

    schedule_more()

    try:
        while in_flight:
            done, _ = await asyncio.wait(in_flight.keys(), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                idx = in_flight.pop(task)
                try:
                    result_idx, matches, usage = task.result()
                except Exception:
                    for pending_task in in_flight:
                        pending_task.cancel()
                    raise
                if result_idx != idx:
                    raise RuntimeError(f"并发调度错误：task idx={idx} result idx={result_idx}")
                buffered_results[idx] = (matches, usage)

            while write_index in buffered_results:
                matches, usage = buffered_results.pop(write_index)
                fragment = fragments[write_index]

                piece_id = str(fragment.get("piece_id") or "")
                if piece_id and piece_id not in seen_piece_ids:
                    append_jsonl(
                        piece_output_path,
                        _piece_level_record(
                            model_tag=tag,
                            model_name=model,
                            fragment=fragment,
                            matches=matches,
                        ),
                    )
                    seen_piece_ids.add(piece_id)
                    stats.piece_records_written += 1

                records = _flatten_records(fragment, matches)
                for record in records:
                    key = (str(record.get("piece_id") or ""), str(record.get("matched_theme") or ""))
                    if key in seen_flat:
                        continue
                    append_jsonl(raw_output_path, record)
                    seen_flat.add(key)
                    stats.raw_records_written += 1

                prompt, completion, total_token = _usage_tokens(usage)
                stats.prompt_tokens += prompt
                stats.completion_tokens += completion
                stats.total_tokens += total_token
                stats.processed_fragments += 1
                write_index += 1
                stats.end_index = write_index

                write_json(
                    cursor_path,
                    {
                        "tag": tag,
                        "model": model,
                        "next_index": write_index,
                        "last_piece_id": fragment.get("piece_id"),
                        "updated_at": _now_iso(),
                    },
                )

            schedule_more()

    finally:
        for task in in_flight:
            if not task.done():
                task.cancel()

    return stats


async def run_archival_screening(
    *,
    project_dir: Path,
    fragments_path: Path,
    target_themes: list[dict[str, str]],
    llm_client: OpenAICompatClient,
    model_llm1: str,
    model_llm2: str,
    logger,
    concurrency_per_model: int = 4,
    fragment_max_attempts: int = 3,
) -> tuple[Path, Path, dict[str, Any]]:
    _theme_names(target_themes)

    fragments = read_jsonl(fragments_path)
    if not fragments:
        raise RuntimeError(f"阶段2.2无法继续：未找到可筛选碎片 {fragments_path}")

    llm1_raw_path = project_dir / "2_llm1_raw.jsonl"
    llm2_raw_path = project_dir / "2_llm2_raw.jsonl"
    llm1_piece_raw_path = project_dir / "2_llm1_piece_raw.jsonl"
    llm2_piece_raw_path = project_dir / "2_llm2_piece_raw.jsonl"
    cursor1_path = project_dir / ".cursor_llm1.json"
    cursor2_path = project_dir / ".cursor_llm2.json"

    stats1, stats2 = await asyncio.gather(
        _run_single_model(
            tag="llm1",
            model=model_llm1,
            llm_client=llm_client,
            target_themes=target_themes,
            fragments=fragments,
            raw_output_path=llm1_raw_path,
            piece_output_path=llm1_piece_raw_path,
            cursor_path=cursor1_path,
            logger=logger,
            concurrency=concurrency_per_model,
            fragment_max_attempts=fragment_max_attempts,
        ),
        _run_single_model(
            tag="llm2",
            model=model_llm2,
            llm_client=llm_client,
            target_themes=target_themes,
            fragments=fragments,
            raw_output_path=llm2_raw_path,
            piece_output_path=llm2_piece_raw_path,
            cursor_path=cursor2_path,
            logger=logger,
            concurrency=concurrency_per_model,
            fragment_max_attempts=fragment_max_attempts,
        ),
    )

    stats_payload = {
        "total_fragments": len(fragments),
        "target_theme_count": len(target_themes),
        "llm1": stats1.to_dict(),
        "llm2": stats2.to_dict(),
    }
    write_json(project_dir / "2_screening_audit.json", stats_payload)

    logger.info("阶段2.2完成: %s, %s", llm1_raw_path, llm2_raw_path)
    return llm1_raw_path, llm2_raw_path, stats_payload
