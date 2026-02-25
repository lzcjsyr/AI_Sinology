from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from core.llm_client import OpenAICompatClient
from core.utils import clamp_text, parse_json_from_text, read_jsonl, write_json, write_yaml


def _record_key(record: dict[str, Any]) -> tuple[str, str]:
    return str(record.get("piece_id", "")), str(record.get("matched_theme", ""))


def _normalize_level(level: Any) -> str | None:
    if level is None:
        return None
    value = str(level).upper().strip()
    if value in {"HIGH", "MEDIUM", "LOW"}:
        return value
    return "LOW"


def _consensus_record(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    reason_a = str(a.get("reason") or "")
    reason_b = str(b.get("reason") or "")
    chosen_reason = reason_a if len(reason_a) >= len(reason_b) else reason_b

    level_a = _normalize_level(a.get("relevance_level"))
    level_b = _normalize_level(b.get("relevance_level"))
    level = level_a or level_b or "LOW"

    return {
        "piece_id": a["piece_id"],
        "source_file": a.get("source_file") or b.get("source_file"),
        "original_text": a.get("original_text") or b.get("original_text"),
        "matched_theme": a["matched_theme"],
        "is_relevant": True,
        "reason": chosen_reason or "双模型一致判定相关",
        "relevance_level": level,
    }


def _build_maps(
    llm1_records: list[dict[str, Any]],
    llm2_records: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    map1: dict[tuple[str, str], dict[str, Any]] = {}
    map2: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in llm1_records:
        map1[_record_key(rec)] = rec
    for rec in llm2_records:
        map2[_record_key(rec)] = rec
    return map1, map2


def _consensus_and_disputes(
    llm1_records: list[dict[str, Any]],
    llm2_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    map1, map2 = _build_maps(llm1_records, llm2_records)
    keys = sorted(set(map1.keys()) | set(map2.keys()))

    consensus: list[dict[str, Any]] = []
    disputes: list[dict[str, Any]] = []

    for key in keys:
        a = map1.get(key)
        b = map2.get(key)
        if a is None and b is None:
            continue

        if a is None:
            a = {
                "piece_id": b["piece_id"],
                "source_file": b.get("source_file"),
                "original_text": b.get("original_text"),
                "matched_theme": b["matched_theme"],
                "is_relevant": False,
                "reason": None,
                "relevance_level": None,
            }
        if b is None:
            b = {
                "piece_id": a["piece_id"],
                "source_file": a.get("source_file"),
                "original_text": a.get("original_text"),
                "matched_theme": a["matched_theme"],
                "is_relevant": False,
                "reason": None,
                "relevance_level": None,
            }

        a_rel = bool(a.get("is_relevant"))
        b_rel = bool(b.get("is_relevant"))

        if a_rel and b_rel:
            consensus.append(_consensus_record(a, b))
            continue
        if (not a_rel) and (not b_rel):
            continue

        disputes.append(
            {
                "piece_id": a["piece_id"],
                "source_file": a.get("source_file") or b.get("source_file"),
                "original_text": a.get("original_text") or b.get("original_text"),
                "matched_theme": a.get("matched_theme") or b.get("matched_theme"),
                "llm1_result": {
                    "is_relevant": a_rel,
                    "relevance_level": _normalize_level(a.get("relevance_level")),
                    "reason": a.get("reason") if a_rel else None,
                },
                "llm2_result": {
                    "is_relevant": b_rel,
                    "relevance_level": _normalize_level(b.get("relevance_level")),
                    "reason": b.get("reason") if b_rel else None,
                },
            }
        )

    return consensus, disputes


def _arbitrate_single_dispute(
    *,
    llm_client: OpenAICompatClient,
    model: str,
    dispute: dict[str, Any],
    logger,
    max_attempts: int = 3,
) -> dict[str, Any]:
    prompt = (
        "你是第三方学术仲裁模型。请在给定主题下判断该史料是否相关。"
        "仅返回 JSON："
        '{"is_relevant":true/false,"reason":"...或null","relevance_level":"HIGH|MEDIUM|LOW|null"}'
        "。如果不相关，reason 和 relevance_level 必须为 null。\n\n"
        f"主题: {dispute['matched_theme']}\n"
        f"史料坐标: {dispute['piece_id']}\n"
        f"原文片段:\n{clamp_text(str(dispute['original_text']), 2800)}\n\n"
        f"LLM1判定: {dispute['llm1_result']}\n"
        f"LLM2判定: {dispute['llm2_result']}\n"
    )

    messages = [
        {"role": "system", "content": "你是严谨的学术仲裁助手。"},
        {"role": "user", "content": prompt},
    ]

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = llm_client.chat(messages, model=model, temperature=0.0)
            data = parse_json_from_text(response.content)
            if not isinstance(data.get("is_relevant"), bool):
                raise ValueError("is_relevant 不是布尔值")
            is_relevant = bool(data.get("is_relevant"))
            if not is_relevant:
                return {"is_relevant": False, "reason": None, "relevance_level": None}

            reason = str(data.get("reason") or "").strip()
            level = _normalize_level(data.get("relevance_level"))
            if not reason:
                raise ValueError("相关仲裁缺少 reason")
            if level is None:
                raise ValueError("相关仲裁缺少 relevance_level")
            return {"is_relevant": True, "reason": reason, "relevance_level": level}
        except Exception as e:  # noqa: BLE001
            last_error = e
            logger.warning(
                "仲裁失败，准备重试。piece_id=%s attempt=%s error=%s",
                dispute.get("piece_id"),
                attempt,
                e,
            )

    raise RuntimeError(
        f"仲裁失败：piece_id={dispute.get('piece_id')} theme={dispute.get('matched_theme')} last_error={last_error}"
    )


async def run_archival_arbitration(
    *,
    project_dir: Path,
    llm1_raw_path: Path,
    llm2_raw_path: Path,
    llm_client: OpenAICompatClient,
    model_llm3: str,
    logger,
) -> list[dict[str, Any]]:
    llm1_records = read_jsonl(llm1_raw_path)
    llm2_records = read_jsonl(llm2_raw_path)
    if not llm1_records or not llm2_records:
        raise RuntimeError("阶段2.3无法继续：双模型原始结果为空")

    consensus, disputes = _consensus_and_disputes(llm1_records, llm2_records)

    consensus_yaml_path = project_dir / "2_consensus_data.yaml"
    disputed_yaml_path = project_dir / "2_disputed_data.yaml"
    write_yaml(consensus_yaml_path, consensus)
    write_yaml(disputed_yaml_path, disputes)

    write_json(project_dir / "2_consensus_data.json", consensus)
    write_json(project_dir / "2_disputed_data.json", disputes)

    verified: list[dict[str, Any]] = []
    for dispute in disputes:
        result = await asyncio.to_thread(
            _arbitrate_single_dispute,
            llm_client=llm_client,
            model=model_llm3,
            dispute=dispute,
            logger=logger,
        )
        if not result["is_relevant"]:
            continue

        verified.append(
            {
                "piece_id": dispute["piece_id"],
                "source_file": dispute.get("source_file"),
                "original_text": dispute.get("original_text"),
                "matched_theme": dispute["matched_theme"],
                "is_relevant": True,
                "reason": result.get("reason") or "仲裁判定相关",
                "relevance_level": result.get("relevance_level") or "LOW",
            }
        )

    llm3_yaml_path = project_dir / "2_llm3_verified.yaml"
    write_yaml(llm3_yaml_path, verified)
    write_json(project_dir / "2_llm3_verified.json", verified)

    final_corpus = consensus + verified

    final_yaml_path = project_dir / "2_final_corpus.yaml"
    write_yaml(final_yaml_path, final_corpus)
    write_json(project_dir / "2_final_corpus.json", final_corpus)

    logger.info(
        "阶段2.3-2.4完成: consensus=%s disputed=%s verified=%s final=%s",
        len(consensus),
        len(disputes),
        len(verified),
        len(final_corpus),
    )
    return final_corpus
