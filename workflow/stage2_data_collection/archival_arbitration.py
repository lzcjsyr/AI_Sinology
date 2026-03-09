from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Any

from core.config import LLMEndpointConfig
from core.llm_client import OpenAICompatClient
from core.project_paths import stage2_internal_dir, stage2_internal_json_path
from core.prompt_loader import PromptSpec, build_messages, load_prompt
from core.utils import clamp_text, parse_json_from_text, read_jsonl, write_json, write_yaml
from workflow.stage2_data_collection.rate_control import DualRateLimiter, RateLimits


def _record_key(record: dict[str, Any]) -> tuple[str, str]:
    return str(record.get("piece_id", "")), str(record.get("matched_theme", ""))


def _pick_shared_field(a: dict[str, Any], b: dict[str, Any], key: str) -> Any:
    return a.get(key) or b.get(key)


def _consensus_record(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    reason_a = str(a.get("reason") or "")
    reason_b = str(b.get("reason") or "")
    chosen_reason = reason_a if len(reason_a) >= len(reason_b) else reason_b

    return {
        "piece_id": a["piece_id"],
        "source_file": _pick_shared_field(a, b, "source_file"),
        "original_text": _pick_shared_field(a, b, "original_text"),
        "matched_theme": a["matched_theme"],
        "is_relevant": True,
        "reason": chosen_reason or "双模型一致判定相关",
        "screening_batch_id": _pick_shared_field(a, b, "screening_batch_id"),
        "localization_method": _pick_shared_field(a, b, "localization_method"),
        "target_span": _pick_shared_field(a, b, "target_span"),
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


def _dispute_side(record: dict[str, Any] | None) -> dict[str, Any]:
    if record is None:
        return {
            "is_relevant": False,
            "reason": None,
            "target_span": None,
            "localization_method": None,
            "screening_batch_id": None,
        }
    return {
        "is_relevant": bool(record.get("is_relevant")),
        "reason": record.get("reason") if record.get("is_relevant") else None,
        "target_span": record.get("target_span") if record.get("is_relevant") else None,
        "localization_method": record.get("localization_method"),
        "screening_batch_id": record.get("screening_batch_id"),
    }


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

        a_rel = bool(a and a.get("is_relevant"))
        b_rel = bool(b and b.get("is_relevant"))
        if a_rel and b_rel:
            consensus.append(_consensus_record(a, b))
            continue
        if (not a_rel) and (not b_rel):
            continue

        primary = a or b or {}
        counterpart = b or a or {}
        disputes.append(
            {
                "piece_id": str(primary.get("piece_id") or counterpart.get("piece_id") or ""),
                "source_file": _pick_shared_field(primary, counterpart, "source_file"),
                "original_text": _pick_shared_field(primary, counterpart, "original_text"),
                "matched_theme": str(primary.get("matched_theme") or counterpart.get("matched_theme") or ""),
                "llm1_result": _dispute_side(a),
                "llm2_result": _dispute_side(b),
            }
        )

    return consensus, disputes


def _extract_total_tokens(usage: dict[str, Any] | None) -> int | None:
    if not usage:
        return None
    total = usage.get("total_tokens")
    if total is not None:
        return int(total)
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    combined = prompt + completion
    return combined or None


def _estimate_total_tokens(messages: list[dict[str, str]]) -> int:
    text = "\n".join(str(message.get("content") or "") for message in messages)
    return max(192, int(len(text) * 0.72) + 128)


def _estimate_arbitration_request_tokens(
    *,
    prompt_spec: PromptSpec,
    disputes: list[dict[str, Any]],
    sample_size: int = 8,
) -> int:
    if not disputes:
        return 512
    upper = max(1, min(int(sample_size), len(disputes)))
    estimated: list[int] = []
    for dispute in disputes[:upper]:
        messages = build_messages(
            prompt_spec,
            matched_theme=dispute["matched_theme"],
            piece_id=dispute["piece_id"],
            original_text=clamp_text(str(dispute["original_text"]), 2800),
            llm1_result_json=json.dumps(dispute["llm1_result"], ensure_ascii=False),
            llm2_result_json=json.dumps(dispute["llm2_result"], ensure_ascii=False),
        )
        estimated.append(_estimate_total_tokens(messages))
    return max(1, int(sum(estimated) / len(estimated)))


def _derive_auto_concurrency(
    *,
    limits: RateLimits,
    estimated_tokens_per_request: int,
    avg_latency_seconds: float = 1.2,
    utilization: float = 0.9,
    hard_cap: int = 128,
) -> int:
    safe_limits = limits.normalized()
    safe_tokens = max(1, int(estimated_tokens_per_request))
    safe_latency = max(0.2, float(avg_latency_seconds))
    safe_utilization = min(1.0, max(0.1, float(utilization)))
    req_bound = int((safe_limits.rpm * safe_latency / 60.0) * safe_utilization)
    tok_bound = int((safe_limits.tpm * safe_latency / (60.0 * safe_tokens)) * safe_utilization)
    return max(1, min(int(hard_cap), max(1, req_bound), max(1, tok_bound)))


async def _arbitrate_single_dispute(
    *,
    llm_client: OpenAICompatClient,
    llm_endpoint: LLMEndpointConfig,
    prompt_spec: PromptSpec,
    dispute: dict[str, Any],
    logger,
    rate_limiter: DualRateLimiter | None,
    retry_backoff_seconds: float,
    max_attempts: int = 3,
) -> dict[str, Any]:
    model = llm_endpoint.model
    messages = build_messages(
        prompt_spec,
        matched_theme=dispute["matched_theme"],
        piece_id=dispute["piece_id"],
        original_text=clamp_text(str(dispute["original_text"]), 2800),
        llm1_result_json=json.dumps(dispute["llm1_result"], ensure_ascii=False),
        llm2_result_json=json.dumps(dispute["llm2_result"], ensure_ascii=False),
    )
    estimated_tokens = _estimate_total_tokens(messages)

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            reservation = await rate_limiter.acquire(estimated_tokens) if rate_limiter else None
            response = await llm_client.achat(
                messages,
                model=model,
                api_key=llm_endpoint.api_key,
                api_keys=llm_endpoint.api_keys,
                api_base=llm_endpoint.base_url,
                temperature=0.0,
            )
            if rate_limiter is not None and reservation is not None:
                await rate_limiter.commit(reservation, _extract_total_tokens(response.usage))
            data = parse_json_from_text(response.content)
            if not isinstance(data.get("is_relevant"), bool):
                raise ValueError("is_relevant 不是布尔值")
            is_relevant = bool(data.get("is_relevant"))
            if not is_relevant:
                return {"is_relevant": False, "reason": None}

            reason = str(data.get("reason") or "").strip()
            if not reason:
                raise ValueError("相关仲裁缺少 reason")
            return {"is_relevant": True, "reason": reason}
        except Exception as e:  # noqa: BLE001
            last_error = e
            logger.warning(
                "仲裁失败，准备重试。piece_id=%s attempt=%s error=%s",
                dispute.get("piece_id"),
                attempt,
                e,
            )
            if attempt < max_attempts:
                base = max(0.05, float(retry_backoff_seconds))
                backoff = base * (2 ** (attempt - 1))
                jitter = random.uniform(0.0, backoff * 0.25)
                await asyncio.sleep(backoff + jitter)

    raise RuntimeError(
        f"仲裁失败：piece_id={dispute.get('piece_id')} theme={dispute.get('matched_theme')} last_error={last_error}"
    )


async def run_archival_arbitration(
    *,
    project_dir: Path,
    llm1_raw_path: Path,
    llm2_raw_path: Path,
    llm_client: OpenAICompatClient,
    llm3_endpoint: LLMEndpointConfig,
    logger,
    concurrency: int | None = None,
    retry_backoff_seconds: float = 2.0,
) -> list[dict[str, Any]]:
    prompt_spec = load_prompt("stage2_arbitration")
    llm1_records = read_jsonl(llm1_raw_path)
    llm2_records = read_jsonl(llm2_raw_path)
    if not llm1_records or not llm2_records:
        raise RuntimeError("阶段2.3无法继续：双模型原始结果为空")

    internal_dir = stage2_internal_dir(project_dir)
    internal_dir.mkdir(parents=True, exist_ok=True)

    consensus, disputes = _consensus_and_disputes(llm1_records, llm2_records)

    consensus_yaml_path = project_dir / "2_consensus_data.yaml"
    disputed_yaml_path = project_dir / "2_disputed_data.yaml"
    write_yaml(consensus_yaml_path, consensus)
    write_yaml(disputed_yaml_path, disputes)

    write_json(stage2_internal_json_path(project_dir, "2_consensus_data.json"), consensus)
    write_json(stage2_internal_json_path(project_dir, "2_disputed_data.json"), disputes)

    provider_limits = RateLimits(
        rpm=llm3_endpoint.effective_rpm,
        tpm=llm3_endpoint.effective_tpm,
    ).normalized()
    limiter = DualRateLimiter(
        name=f"model:llm3:{llm3_endpoint.provider}/{llm3_endpoint.model}",
        limits=provider_limits,
    )
    if concurrency is not None and concurrency < 1:
        raise RuntimeError("阶段2.3参数 concurrency 必须 >= 1")
    estimated_tokens_per_request = _estimate_arbitration_request_tokens(
        prompt_spec=prompt_spec,
        disputes=disputes,
    )
    safe_concurrency = concurrency
    if safe_concurrency is None:
        safe_concurrency = _derive_auto_concurrency(
            limits=provider_limits,
            estimated_tokens_per_request=estimated_tokens_per_request,
        )
        logger.info(
            "阶段2.3自动并发 arbitration=%s (rpm=%s tpm=%s est_tokens=%s)",
            safe_concurrency,
            provider_limits.rpm,
            provider_limits.tpm,
            estimated_tokens_per_request,
        )
    safe_concurrency = max(1, int(safe_concurrency))
    semaphore = asyncio.Semaphore(safe_concurrency)
    verified_results: list[dict[str, Any] | None] = [None] * len(disputes)

    async def _worker(i: int, dispute: dict[str, Any]) -> None:
        async with semaphore:
            result = await _arbitrate_single_dispute(
                llm_client=llm_client,
                llm_endpoint=llm3_endpoint,
                prompt_spec=prompt_spec,
                dispute=dispute,
                logger=logger,
                rate_limiter=limiter,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            if not result["is_relevant"]:
                return
            verified_results[i] = {
                "piece_id": dispute["piece_id"],
                "source_file": dispute.get("source_file"),
                "original_text": dispute.get("original_text"),
                "matched_theme": dispute["matched_theme"],
                "is_relevant": True,
                "reason": result.get("reason") or "仲裁判定相关",
            }

    await asyncio.gather(*[_worker(i, dispute) for i, dispute in enumerate(disputes)])

    verified: list[dict[str, Any]] = []
    for row in verified_results:
        if row is not None:
            verified.append(row)

    llm3_yaml_path = project_dir / "2_llm3_verified.yaml"
    write_yaml(llm3_yaml_path, verified)
    write_json(stage2_internal_json_path(project_dir, "2_llm3_verified.json"), verified)

    final_corpus = consensus + verified

    final_yaml_path = project_dir / "2_final_corpus.yaml"
    write_yaml(final_yaml_path, final_corpus)
    write_json(stage2_internal_json_path(project_dir, "2_final_corpus.json"), final_corpus)

    logger.info(
        "阶段2.3-2.4完成: consensus=%s disputed=%s verified=%s final=%s",
        len(consensus),
        len(disputes),
        len(verified),
        len(final_corpus),
    )
    return final_corpus
