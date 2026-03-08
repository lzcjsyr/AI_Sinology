from __future__ import annotations

import asyncio
import json
import random
import shutil
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Callable

from core.config import LLMEndpointConfig
from core.llm_client import OpenAICompatClient
from core.project_paths import (
    resolve_stage2_internal_path,
    stage2_internal_dir,
    stage2_internal_path,
)
from core.prompt_loader import PromptSpec, build_messages, load_prompt
from core.utils import append_jsonl, parse_json_from_text, read_json, read_jsonl, write_json, write_jsonl
from workflow.stage2_data_collection.rate_control import (
    AcquireReservation,
    DualRateLimiter,
    RateLimits,
    build_lowest_shared_limits,
)


_JSON_MODE_RESPONSE_FORMAT: dict[str, str] = {"type": "json_object"}
_JSON_MODE_SUPPORT_CACHE: dict[tuple[str, str], bool] = {}
_JSON_MODE_WARNED_CACHE: set[tuple[str, str]] = set()
_SCREENING_BATCHES_FILE = "kanripo_screening_batches.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ScreeningStats:
    model_tag: str
    model_name: str
    provider: str
    start_index: int
    end_index: int
    processed_batches: int
    processed_fragments: int
    raw_records_written: int
    refined_batches: int
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    provider_wait_seconds: float
    sync_wait_seconds: float
    provider_throttled_count: int
    sync_throttled_count: int
    retry_count: int
    failed_batches: int = 0
    failed_fragments: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_tag": self.model_tag,
            "model_name": self.model_name,
            "provider": self.provider,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "processed_batches": self.processed_batches,
            "processed_fragments": self.processed_fragments,
            "raw_records_written": self.raw_records_written,
            "refined_batches": self.refined_batches,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "provider_wait_seconds": round(self.provider_wait_seconds, 3),
            "sync_wait_seconds": round(self.sync_wait_seconds, 3),
            "provider_throttled_count": self.provider_throttled_count,
            "sync_throttled_count": self.sync_throttled_count,
            "retry_count": self.retry_count,
            "failed_batches": self.failed_batches,
            "failed_fragments": self.failed_fragments,
        }


@dataclass
class RequestControlStats:
    provider_wait_seconds: float = 0.0
    sync_wait_seconds: float = 0.0
    provider_throttled_count: int = 0
    sync_throttled_count: int = 0
    retry_count: int = 0

    def absorb(self, other: "RequestControlStats") -> None:
        self.provider_wait_seconds += other.provider_wait_seconds
        self.sync_wait_seconds += other.sync_wait_seconds
        self.provider_throttled_count += other.provider_throttled_count
        self.sync_throttled_count += other.sync_throttled_count
        self.retry_count += other.retry_count


@dataclass
class BatchScreeningResult:
    records: list[dict[str, Any]]
    usage: dict[str, Any] | None
    control_stats: RequestControlStats
    used_refine: bool = False
    failed: bool = False


def _theme_names(target_themes: list[dict[str, str]]) -> list[str]:
    names = [str(item.get("theme") or "").strip() for item in target_themes]
    names = [name for name in names if name]
    if len(set(names)) != len(names):
        raise RuntimeError("阶段2.2失败：target_themes 存在重复主题名称")
    if not names:
        raise RuntimeError("阶段2.2失败：target_themes 为空")
    return names


def _screening_batches_path(project_dir: Path) -> Path:
    return project_dir / "_processed_data" / _SCREENING_BATCHES_FILE


def _fragment_char_count(text: str) -> int:
    return len(str(text).replace("\n", ""))


def _build_screening_batches(
    fragments: list[dict[str, str]],
    *,
    batch_max_chars: int,
) -> list[dict[str, Any]]:
    safe_limit = max(1, int(batch_max_chars))
    batches: list[dict[str, Any]] = []
    current: list[dict[str, str]] = []
    current_source = ""
    current_char_count = 0

    def flush() -> None:
        nonlocal current, current_source, current_char_count
        if not current:
            return

        batch_index = len(batches) + 1
        parts: list[str] = []
        piece_offsets: list[dict[str, Any]] = []
        cursor = 0
        for idx, fragment in enumerate(current):
            text = str(fragment.get("original_text") or "")
            if idx > 0:
                parts.append("\n")
                cursor += 1
            start = cursor
            parts.append(text)
            cursor += len(text)
            piece_offsets.append(
                {
                    "piece_id": str(fragment.get("piece_id") or ""),
                    "start": start,
                    "end": cursor,
                }
            )

        batches.append(
            {
                "batch_id": f"batch_{batch_index:08d}",
                "source_file": current_source,
                "piece_ids": [str(fragment.get("piece_id") or "") for fragment in current],
                "batch_text": "".join(parts),
                "char_count": current_char_count,
                "piece_offsets": piece_offsets,
            }
        )
        current = []
        current_source = ""
        current_char_count = 0

    for fragment in fragments:
        piece_id = str(fragment.get("piece_id") or "").strip()
        text = str(fragment.get("original_text") or "")
        source_file = str(fragment.get("source_file") or "").strip()
        if not piece_id or not text:
            continue

        fragment_chars = _fragment_char_count(text)
        if not current:
            current = [fragment]
            current_source = source_file
            current_char_count = fragment_chars
            continue

        should_flush = (
            source_file != current_source
            or (current_char_count + fragment_chars) > safe_limit
        )
        if should_flush:
            flush()
            current = [fragment]
            current_source = source_file
            current_char_count = fragment_chars
            continue

        current.append(fragment)
        current_char_count += fragment_chars

    flush()
    return batches


def _load_or_build_screening_batches(
    *,
    project_dir: Path,
    fragments: list[dict[str, str]],
    batch_max_chars: int,
    logger,
) -> list[dict[str, Any]]:
    output_path = _screening_batches_path(project_dir)
    existing = read_jsonl(output_path)
    if existing:
        logger.info("阶段2.2复用 screening batches: %s (records=%s)", output_path, len(existing))
        return existing

    batches = _build_screening_batches(fragments, batch_max_chars=batch_max_chars)
    if not batches:
        raise RuntimeError("阶段2.2无法继续：未生成任何 screening batch")
    write_jsonl(output_path, batches)
    logger.info(
        "阶段2.2生成 screening batches: %s (batches=%s, max_chars=%s)",
        output_path,
        len(batches),
        batch_max_chars,
    )
    return batches


def _normalize_matches_strict(
    payload: dict[str, Any],
    target_themes: list[dict[str, str]],
) -> list[dict[str, Any]]:
    raw = payload.get("matches")
    if not isinstance(raw, list):
        for key in ("results", "items", "themes", "judgments"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                raw = candidate
                break
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

        is_relevant_value = src.get("is_relevant")
        if isinstance(is_relevant_value, bool):
            is_relevant = is_relevant_value
        elif isinstance(is_relevant_value, (int, float)):
            if int(is_relevant_value) in {0, 1}:
                is_relevant = bool(int(is_relevant_value))
            else:
                raise ValueError(f"主题 `{theme_id}` 的 is_relevant 不是布尔值")
        elif isinstance(is_relevant_value, str):
            normalized = is_relevant_value.strip().lower()
            if normalized in {"true", "1", "yes"}:
                is_relevant = True
            elif normalized in {"false", "0", "no"}:
                is_relevant = False
            else:
                raise ValueError(f"主题 `{theme_id}` 的 is_relevant 不是布尔值")
        else:
            raise ValueError(f"主题 `{theme_id}` 的 is_relevant 不是布尔值")

        reason = src.get("reason")
        target_span = src.get("target_span")
        related_spans_raw = src.get("related_spans")
        related_spans: list[str] = []
        if isinstance(related_spans_raw, list):
            for item in related_spans_raw:
                text = str(item or "").strip()
                if text:
                    related_spans.append(text)

        if not is_relevant:
            reason = None
            target_span = None
            related_spans = []
        else:
            reason = str(reason or "").strip()
            if not reason:
                raise ValueError(f"主题 `{theme_id}` 判定为相关但缺少 reason")
            target_span = str(target_span or "").strip()
            if not target_span:
                raise ValueError(f"主题 `{theme_id}` 判定为相关但缺少 target_span")

        result.append(
            {
                "theme": theme,
                "theme_id": theme_id,
                "is_relevant": is_relevant,
                "reason": reason,
                "target_span": target_span,
                "related_spans": related_spans,
            }
        )

    if missing:
        raise ValueError(f"LLM 返回缺少主题判定: {missing}")
    return result


def _normalize_refined_matches(
    payload: dict[str, Any],
    unresolved_matches: list[dict[str, Any]],
    batch_piece_ids: list[str],
) -> list[dict[str, Any]]:
    raw = payload.get("matches")
    if not isinstance(raw, list):
        raise ValueError("细筛返回缺少 matches 数组")

    piece_order = {piece_id: idx for idx, piece_id in enumerate(batch_piece_ids)}
    by_theme_id: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        theme_id = str(item.get("theme_id") or "").strip().upper()
        if theme_id:
            by_theme_id[theme_id] = item

    result: list[dict[str, Any]] = []
    missing: list[str] = []
    for coarse_match in unresolved_matches:
        theme_id = str(coarse_match.get("theme_id") or "").strip().upper()
        src = by_theme_id.get(theme_id)
        if src is None:
            missing.append(theme_id)
            continue

        raw_piece_ids = src.get("relevant_piece_ids")
        if not isinstance(raw_piece_ids, list):
            raise ValueError(f"主题 `{theme_id}` 缺少 relevant_piece_ids 数组")

        deduped: list[str] = []
        seen: set[str] = set()
        for piece_id in raw_piece_ids:
            value = str(piece_id or "").strip()
            if not value or value in seen:
                continue
            if value not in piece_order:
                raise ValueError(f"主题 `{theme_id}` 返回了批次外的 piece_id: {value}")
            seen.add(value)
            deduped.append(value)
        if not deduped:
            raise ValueError(f"主题 `{theme_id}` 的 relevant_piece_ids 为空")

        ordered = sorted(deduped, key=lambda item: piece_order[item])
        positions = [piece_order[item] for item in ordered]
        if positions[-1] - positions[0] + 1 != len(positions):
            raise ValueError(f"主题 `{theme_id}` 的 relevant_piece_ids 必须连续")

        reason = str(src.get("reason") or "").strip()
        if not reason:
            raise ValueError(f"主题 `{theme_id}` 缺少定位 reason")

        result.append(
            {
                "theme": coarse_match["theme"],
                "theme_id": theme_id,
                "relevant_piece_ids": ordered,
                "reason": reason,
            }
        )

    if missing:
        raise ValueError(f"细筛返回缺少主题判定: {missing}")
    return result


def _is_response_format_unsupported(error: Exception) -> bool:
    message = str(error).lower()
    if "response_format" not in message and "json_object" not in message:
        return False
    signals = (
        "unsupported",
        "not support",
        "not supported",
        "unknown",
        "invalid",
        "not allow",
        "not permitted",
    )
    return any(signal in message for signal in signals)


def _model_cache_key(llm_endpoint: LLMEndpointConfig) -> tuple[str, str]:
    return (llm_endpoint.provider, llm_endpoint.model)


def _known_json_mode_support(llm_endpoint: LLMEndpointConfig) -> bool | None:
    provider = str(llm_endpoint.provider or "").strip().lower()
    model = str(llm_endpoint.model or "").strip().lower()
    if provider == "volcengine" and model.startswith("doubao-seed-2-0"):
        return False
    return None


async def _probe_json_mode_support(
    *,
    llm_client: OpenAICompatClient,
    llm_endpoint: LLMEndpointConfig,
    logger,
) -> bool:
    cache_key = _model_cache_key(llm_endpoint)
    cached = _JSON_MODE_SUPPORT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    known = _known_json_mode_support(llm_endpoint)
    if known is not None:
        _JSON_MODE_SUPPORT_CACHE[cache_key] = known
        if not known:
            logger.info(
                "模型 `%s` 已知不支持 JSON mode，阶段2.2将直接关闭 response_format。",
                llm_endpoint.model,
            )
        return known

    try:
        response = await llm_client.achat(
            [
                {"role": "system", "content": "Return only a JSON object."},
                {"role": "user", "content": "{\"ok\": true}"},
            ],
            model=llm_endpoint.model,
            api_key=llm_endpoint.api_key,
            api_keys=llm_endpoint.api_keys,
            api_base=llm_endpoint.base_url,
            temperature=0.0,
            max_tokens=32,
            response_format=_JSON_MODE_RESPONSE_FORMAT,
        )
        parse_json_from_text(response.content)
        _JSON_MODE_SUPPORT_CACHE[cache_key] = True
        return True
    except Exception as e:  # noqa: BLE001
        if _is_response_format_unsupported(e):
            logger.warning(
                "模型 `%s` 不支持 JSON mode，本轮将关闭 response_format。provider=%s error=%s",
                llm_endpoint.model,
                llm_endpoint.provider,
                e,
            )
            _JSON_MODE_SUPPORT_CACHE[cache_key] = False
            return False
        logger.warning(
            "JSON mode 预检异常，先继续启用 JSON mode。provider=%s model=%s error=%s",
            llm_endpoint.provider,
            llm_endpoint.model,
            e,
        )
        _JSON_MODE_SUPPORT_CACHE[cache_key] = True
        return True


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


def _usage_tokens(usage: dict[str, Any] | None) -> tuple[int, int, int]:
    if not usage:
        return 0, 0, 0
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    total = int(usage.get("total_tokens") or (prompt + completion))
    return prompt, completion, total


def _merge_usage(*usages: dict[str, Any] | None) -> dict[str, int] | None:
    prompt_total = 0
    completion_total = 0
    total_total = 0
    any_usage = False
    for usage in usages:
        if not usage:
            continue
        any_usage = True
        prompt, completion, total = _usage_tokens(usage)
        prompt_total += prompt
        completion_total += completion
        total_total += total
    if not any_usage:
        return None
    return {
        "prompt_tokens": prompt_total,
        "completion_tokens": completion_total,
        "total_tokens": total_total,
    }


def _estimate_total_tokens(messages: list[dict[str, str]]) -> int:
    text = "\n".join(str(message.get("content") or "") for message in messages)
    return max(256, int(len(text) * 0.72) + 160)


def _estimate_screening_request_tokens(
    *,
    prompt_spec: PromptSpec,
    target_themes: list[dict[str, str]],
    batches: list[dict[str, Any]],
    sample_size: int = 8,
) -> int:
    if not batches:
        return 512
    themes_block = "\n".join(
        f"- T{i+1} | theme: {t['theme']} | description: {t.get('description', '')}"
        for i, t in enumerate(target_themes)
    )
    upper = max(1, min(int(sample_size), len(batches)))
    estimated: list[int] = []
    for batch in batches[:upper]:
        messages = build_messages(
            prompt_spec,
            themes_block=themes_block,
            batch_id=batch["batch_id"],
            source_file=batch["source_file"],
            batch_text=batch["batch_text"],
        )
        estimated.append(_estimate_total_tokens(messages))
    return max(1, int(sum(estimated) / len(estimated)))


def _derive_auto_concurrency(
    *,
    limits: RateLimits,
    estimated_tokens_per_request: int,
    avg_latency_seconds: float = 1.5,
    utilization: float = 0.9,
    hard_cap: int = 256,
) -> int:
    safe_limits = limits.normalized()
    safe_tokens = max(1, int(estimated_tokens_per_request))
    safe_latency = max(0.2, float(avg_latency_seconds))
    safe_utilization = min(1.0, max(0.1, float(utilization)))
    req_bound = int((safe_limits.rpm * safe_latency / 60.0) * safe_utilization)
    tok_bound = int((safe_limits.tpm * safe_latency / (60.0 * safe_tokens)) * safe_utilization)
    return max(1, min(int(hard_cap), max(1, req_bound), max(1, tok_bound)))


def _counterpart_tag(tag: str) -> str:
    return "llm2" if tag == "llm1" else "llm1"


class SyncProgressGate:
    def __init__(self, *, max_ahead: int) -> None:
        self.max_ahead = max(0, int(max_ahead))
        self._condition = asyncio.Condition()
        self._processed: dict[str, int] = {"llm1": 0, "llm2": 0}

    async def set_processed(self, tag: str, value: int) -> None:
        async with self._condition:
            self._processed[tag] = max(0, int(value))
            self._condition.notify_all()

    async def wait_until_allowed(self, tag: str) -> None:
        other = _counterpart_tag(tag)
        async with self._condition:
            while (self._processed.get(tag, 0) - self._processed.get(other, 0)) > self.max_ahead:
                await self._condition.wait()


def _build_progress_bar(completed: int, total: int, *, width: int = 28) -> str:
    safe_total = max(1, int(total))
    safe_width = max(8, int(width))
    ratio = min(1.0, max(0.0, float(completed) / float(safe_total)))
    filled = min(safe_width, int(ratio * safe_width))
    return f"[{'#' * filled}{'-' * (safe_width - filled)}]"


def _format_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--:--"
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class _InlineScreeningProgress:
    def __init__(self, *, total: int) -> None:
        self.total = max(1, int(total))
        self._stream = sys.stdout
        self.enabled = bool(hasattr(self._stream, "isatty") and self._stream.isatty())
        self._last_line_len = 0
        self._last_draw_at = 0.0
        self._draw_interval_seconds = 0.2
        self._state: dict[str, tuple[int, float | None]] = {}

    def update(self, *, tag: str, completed: int, eta_seconds: float | None, force: bool = False) -> None:
        self._state[tag] = (completed, eta_seconds)
        if not self.enabled:
            return

        if "llm1" not in self._state or "llm2" not in self._state:
            return

        now = monotonic()
        if not force and (now - self._last_draw_at) < self._draw_interval_seconds and completed < self.total:
            return

        llm1_completed, _llm1_eta = self._state["llm1"]
        llm2_completed, _llm2_eta = self._state["llm2"]
        line = (
            f"阶段2.2筛选进度 llm1 {llm1_completed}/{self.total} | "
            f"llm2 {llm2_completed}/{self.total}"
        )

        width = shutil.get_terminal_size((140, 30)).columns
        if width > 24:
            max_len = width - 1
            if len(line) > max_len:
                line = line[:max_len]

        pad = max(0, self._last_line_len - len(line))
        self._stream.write("\r\033[2K" + line + (" " * pad))
        self._stream.flush()
        self._last_line_len = len(line)
        self._last_draw_at = now

    def finalize(self) -> None:
        if not self.enabled:
            return
        llm1_done = self._state.get("llm1", (0, None))[0] >= self.total
        llm2_done = self._state.get("llm2", (0, None))[0] >= self.total
        if not (llm1_done and llm2_done) and "llm1" in self._state:
            self.update(
                tag="llm1",
                completed=self._state.get("llm1", (0, None))[0],
                eta_seconds=None,
                force=True,
            )
        self._stream.write("\r\033[2K\n")
        self._stream.flush()


def _matchable_char(ch: str) -> bool:
    category = unicodedata.category(ch)
    return category.startswith("L") or category.startswith("N")


def _normalize_text_with_map(text: str) -> tuple[str, list[int]]:
    normalized_chars: list[str] = []
    index_map: list[int] = []
    for idx, ch in enumerate(str(text)):
        if not _matchable_char(ch):
            continue
        normalized_chars.append(ch.lower())
        index_map.append(idx)
    return "".join(normalized_chars), index_map


def _normalize_match_query(text: str) -> str:
    return "".join(ch.lower() for ch in str(text) if _matchable_char(ch))


def _find_candidate_piece_ids(
    *,
    batch_text: str,
    piece_offsets: list[dict[str, Any]],
    candidate_text: str,
) -> list[str]:
    normalized_candidate = _normalize_match_query(candidate_text)
    if len(normalized_candidate) < 2:
        return []

    normalized_batch, index_map = _normalize_text_with_map(batch_text)
    if not normalized_batch:
        return []

    resolved_sets: list[tuple[str, ...]] = []
    search_start = 0
    while True:
        found = normalized_batch.find(normalized_candidate, search_start)
        if found < 0:
            break
        start = index_map[found]
        end = index_map[found + len(normalized_candidate) - 1] + 1
        piece_ids: list[str] = []
        for item in piece_offsets:
            piece_id = str(item.get("piece_id") or "")
            piece_start = int(item.get("start") or 0)
            piece_end = int(item.get("end") or 0)
            if max(start, piece_start) < min(end, piece_end):
                piece_ids.append(piece_id)
        if piece_ids:
            resolved_sets.append(tuple(piece_ids))
        search_start = found + 1

    unique_sets: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for item in resolved_sets:
        if item in seen:
            continue
        seen.add(item)
        unique_sets.append(item)
    if len(unique_sets) != 1:
        return []
    return list(unique_sets[0])


def _build_themes_block(target_themes: list[dict[str, str]]) -> str:
    return "\n".join(
        f"- T{i+1} | theme: {t['theme']} | description: {t.get('description', '')}"
        for i, t in enumerate(target_themes)
    )


def _build_unresolved_themes_block(unresolved_matches: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            " | ".join(
                [
                    f"{match['theme_id']}",
                    f"theme={match['theme']}",
                    f"reason={match['reason']}",
                    f"target_span={match['target_span']}",
                    f"related_spans={json.dumps(match['related_spans'], ensure_ascii=False)}",
                ]
            )
            for match in unresolved_matches
        ]
    )


async def _classify_fragment_strict(
    *,
    llm_client: OpenAICompatClient,
    model: str,
    llm_endpoint: LLMEndpointConfig,
    prompt_spec: PromptSpec,
    fragment: dict[str, Any],
    target_themes: list[dict[str, str]],
    logger,
    max_attempts: int,
    retry_backoff_seconds: float,
    provider_limiter: DualRateLimiter | None = None,
    sync_limiter: DualRateLimiter | None = None,
    prefer_json_mode: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, RequestControlStats]:
    themes_block = _build_themes_block(target_themes)
    batch_id = str(fragment.get("batch_id") or fragment.get("piece_id") or "").strip()
    batch_text = str(fragment.get("batch_text") or fragment.get("original_text") or "")

    messages = build_messages(
        prompt_spec,
        themes_block=themes_block,
        batch_id=batch_id,
        source_file=fragment["source_file"],
        batch_text=batch_text,
    )
    estimated_tokens = _estimate_total_tokens(messages)
    cache_key = _model_cache_key(llm_endpoint)
    use_json_mode = bool(prefer_json_mode and _JSON_MODE_SUPPORT_CACHE.get(cache_key, True))

    last_error: Exception | None = None
    control_stats = RequestControlStats()
    for attempt in range(1, max_attempts + 1):
        provider_reservation: AcquireReservation | None = None
        sync_reservation: AcquireReservation | None = None
        try:
            if provider_limiter is not None:
                provider_reservation = await provider_limiter.acquire(estimated_tokens)
                control_stats.provider_wait_seconds += provider_reservation.wait_seconds
                if provider_reservation.throttled:
                    control_stats.provider_throttled_count += 1
            if sync_limiter is not None:
                sync_reservation = await sync_limiter.acquire(estimated_tokens)
                control_stats.sync_wait_seconds += sync_reservation.wait_seconds
                if sync_reservation.throttled:
                    control_stats.sync_throttled_count += 1

            response = await llm_client.achat(
                messages,
                model=model,
                api_key=llm_endpoint.api_key,
                api_keys=llm_endpoint.api_keys,
                api_base=llm_endpoint.base_url,
                temperature=0.0,
                response_format=_JSON_MODE_RESPONSE_FORMAT if use_json_mode else None,
            )
            payload = parse_json_from_text(response.content)
            matches = _normalize_matches_strict(payload, target_themes)
            if use_json_mode:
                _JSON_MODE_SUPPORT_CACHE[cache_key] = True
            actual_total_tokens = _extract_total_tokens(response.usage)
            if provider_limiter is not None and provider_reservation is not None:
                await provider_limiter.commit(provider_reservation, actual_total_tokens)
            if sync_limiter is not None and sync_reservation is not None:
                await sync_limiter.commit(sync_reservation, actual_total_tokens)
            return matches, response.usage, control_stats
        except Exception as e:  # noqa: BLE001
            if use_json_mode and _is_response_format_unsupported(e):
                _JSON_MODE_SUPPORT_CACHE[cache_key] = False
                use_json_mode = False
                if cache_key not in _JSON_MODE_WARNED_CACHE:
                    logger.warning(
                        "模型不支持 JSON mode，自动降级文本解析。batch_id=%s model=%s error=%s",
                        batch_id,
                        model,
                        e,
                    )
                    _JSON_MODE_WARNED_CACHE.add(cache_key)
                if attempt < max_attempts:
                    continue
            last_error = e
            control_stats.retry_count += 1
            logger.warning(
                "批次粗筛失败，准备重试。batch_id=%s model=%s attempt=%s error=%s",
                batch_id,
                model,
                attempt,
                e,
            )
            if attempt < max_attempts:
                base = max(0.05, float(retry_backoff_seconds))
                backoff = base * (2 ** (attempt - 1))
                jitter = random.uniform(0.0, backoff * 0.25)
                await asyncio.sleep(backoff + jitter)

    raise RuntimeError(
        f"批次粗筛失败：batch_id={batch_id} model={model} last_error={last_error}"
    )


async def _refine_batch_localization_strict(
    *,
    llm_client: OpenAICompatClient,
    model: str,
    llm_endpoint: LLMEndpointConfig,
    prompt_spec: PromptSpec,
    batch: dict[str, Any],
    unresolved_matches: list[dict[str, Any]],
    fragment_map: dict[str, dict[str, str]],
    logger,
    max_attempts: int,
    retry_backoff_seconds: float,
    provider_limiter: DualRateLimiter | None = None,
    sync_limiter: DualRateLimiter | None = None,
    prefer_json_mode: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, RequestControlStats]:
    piece_catalog = [
        {
            "piece_id": piece_id,
            "text": str(fragment_map.get(piece_id, {}).get("original_text") or ""),
        }
        for piece_id in batch["piece_ids"]
    ]
    messages = build_messages(
        prompt_spec,
        unresolved_themes_block=_build_unresolved_themes_block(unresolved_matches),
        batch_id=batch["batch_id"],
        source_file=batch["source_file"],
        batch_text=batch["batch_text"],
        piece_catalog_json=json.dumps(piece_catalog, ensure_ascii=False),
    )
    estimated_tokens = _estimate_total_tokens(messages)
    cache_key = _model_cache_key(llm_endpoint)
    use_json_mode = bool(prefer_json_mode and _JSON_MODE_SUPPORT_CACHE.get(cache_key, True))

    last_error: Exception | None = None
    control_stats = RequestControlStats()
    for attempt in range(1, max_attempts + 1):
        provider_reservation: AcquireReservation | None = None
        sync_reservation: AcquireReservation | None = None
        try:
            if provider_limiter is not None:
                provider_reservation = await provider_limiter.acquire(estimated_tokens)
                control_stats.provider_wait_seconds += provider_reservation.wait_seconds
                if provider_reservation.throttled:
                    control_stats.provider_throttled_count += 1
            if sync_limiter is not None:
                sync_reservation = await sync_limiter.acquire(estimated_tokens)
                control_stats.sync_wait_seconds += sync_reservation.wait_seconds
                if sync_reservation.throttled:
                    control_stats.sync_throttled_count += 1

            response = await llm_client.achat(
                messages,
                model=model,
                api_key=llm_endpoint.api_key,
                api_keys=llm_endpoint.api_keys,
                api_base=llm_endpoint.base_url,
                temperature=0.0,
                response_format=_JSON_MODE_RESPONSE_FORMAT if use_json_mode else None,
            )
            payload = parse_json_from_text(response.content)
            matches = _normalize_refined_matches(payload, unresolved_matches, batch["piece_ids"])
            if use_json_mode:
                _JSON_MODE_SUPPORT_CACHE[cache_key] = True
            actual_total_tokens = _extract_total_tokens(response.usage)
            if provider_limiter is not None and provider_reservation is not None:
                await provider_limiter.commit(provider_reservation, actual_total_tokens)
            if sync_limiter is not None and sync_reservation is not None:
                await sync_limiter.commit(sync_reservation, actual_total_tokens)
            return matches, response.usage, control_stats
        except Exception as e:  # noqa: BLE001
            if use_json_mode and _is_response_format_unsupported(e):
                _JSON_MODE_SUPPORT_CACHE[cache_key] = False
                use_json_mode = False
                if cache_key not in _JSON_MODE_WARNED_CACHE:
                    logger.warning(
                        "模型不支持 JSON mode，细筛自动降级文本解析。batch_id=%s model=%s error=%s",
                        batch.get("batch_id"),
                        model,
                        e,
                    )
                    _JSON_MODE_WARNED_CACHE.add(cache_key)
                if attempt < max_attempts:
                    continue
            last_error = e
            control_stats.retry_count += 1
            logger.warning(
                "批内细筛失败，准备重试。batch_id=%s model=%s attempt=%s error=%s",
                batch.get("batch_id"),
                model,
                attempt,
                e,
            )
            if attempt < max_attempts:
                base = max(0.05, float(retry_backoff_seconds))
                backoff = base * (2 ** (attempt - 1))
                jitter = random.uniform(0.0, backoff * 0.25)
                await asyncio.sleep(backoff + jitter)

    raise RuntimeError(
        f"批内细筛失败：batch_id={batch.get('batch_id')} model={model} last_error={last_error}"
    )


def _build_positive_record(
    *,
    batch: dict[str, Any],
    fragment: dict[str, str],
    match: dict[str, Any],
    reason: str,
    localization_method: str,
) -> dict[str, Any]:
    return {
        "piece_id": fragment["piece_id"],
        "source_file": fragment["source_file"],
        "original_text": fragment["original_text"],
        "matched_theme": match["theme"],
        "is_relevant": True,
        "reason": reason,
        "screening_batch_id": batch["batch_id"],
        "localization_method": localization_method,
        "target_span": match.get("target_span"),
        "related_spans": list(match.get("related_spans") or []),
    }


def _build_failed_records(
    *,
    batch: dict[str, Any],
    fragment_map: dict[str, dict[str, str]],
    target_themes: list[dict[str, str]],
    error: Exception,
) -> list[dict[str, Any]]:
    error_text = str(error).strip()
    records: list[dict[str, Any]] = []
    for piece_id in batch["piece_ids"]:
        fragment = fragment_map.get(piece_id)
        if not fragment:
            continue
        for theme_item in target_themes:
            theme = str(theme_item.get("theme") or "").strip()
            records.append(
                {
                    "piece_id": fragment["piece_id"],
                    "source_file": fragment["source_file"],
                    "original_text": fragment["original_text"],
                    "matched_theme": theme,
                    "is_relevant": False,
                    "reason": None,
                    "screening_batch_id": batch["batch_id"],
                    "localization_method": "screening_error",
                    "target_span": None,
                    "related_spans": [],
                    "screening_error": error_text,
                }
            )
    return records


async def _screen_batch_strict(
    *,
    llm_client: OpenAICompatClient,
    model: str,
    llm_endpoint: LLMEndpointConfig,
    prompt_spec: PromptSpec,
    refine_prompt_spec: PromptSpec,
    batch: dict[str, Any],
    target_themes: list[dict[str, str]],
    fragment_map: dict[str, dict[str, str]],
    logger,
    max_attempts: int,
    retry_backoff_seconds: float,
    provider_limiter: DualRateLimiter | None = None,
    sync_limiter: DualRateLimiter | None = None,
    prefer_json_mode: bool = True,
) -> BatchScreeningResult:
    matches, coarse_usage, control_stats = await _classify_fragment_strict(
        llm_client=llm_client,
        model=model,
        llm_endpoint=llm_endpoint,
        prompt_spec=prompt_spec,
        fragment=batch,
        target_themes=target_themes,
        logger=logger,
        max_attempts=max_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
        provider_limiter=provider_limiter,
        sync_limiter=sync_limiter,
        prefer_json_mode=prefer_json_mode,
    )

    relevant_matches = [match for match in matches if bool(match.get("is_relevant"))]
    if not relevant_matches:
        return BatchScreeningResult(
            records=[],
            usage=coarse_usage,
            control_stats=control_stats,
            used_refine=False,
        )

    records: list[dict[str, Any]] = []
    if len(batch["piece_ids"]) == 1:
        piece_id = batch["piece_ids"][0]
        fragment = fragment_map[piece_id]
        for match in relevant_matches:
            records.append(
                _build_positive_record(
                    batch=batch,
                    fragment=fragment,
                    match=match,
                    reason=str(match.get("reason") or ""),
                    localization_method="single_piece",
                )
            )
        return BatchScreeningResult(
            records=records,
            usage=coarse_usage,
            control_stats=control_stats,
            used_refine=False,
        )

    unresolved: list[dict[str, Any]] = []
    for match in relevant_matches:
        localized_piece_ids = _find_candidate_piece_ids(
            batch_text=batch["batch_text"],
            piece_offsets=batch["piece_offsets"],
            candidate_text=str(match.get("target_span") or ""),
        )
        if not localized_piece_ids:
            for span in list(match.get("related_spans") or []):
                localized_piece_ids = _find_candidate_piece_ids(
                    batch_text=batch["batch_text"],
                    piece_offsets=batch["piece_offsets"],
                    candidate_text=span,
                )
                if localized_piece_ids:
                    break

        if not localized_piece_ids:
            unresolved.append(match)
            continue

        for piece_id in localized_piece_ids:
            fragment = fragment_map.get(piece_id)
            if not fragment:
                continue
            records.append(
                _build_positive_record(
                    batch=batch,
                    fragment=fragment,
                    match=match,
                    reason=str(match.get("reason") or ""),
                    localization_method="regex",
                )
            )

    if not unresolved:
        return BatchScreeningResult(
            records=records,
            usage=coarse_usage,
            control_stats=control_stats,
            used_refine=False,
        )

    refined_matches, refine_usage, refine_stats = await _refine_batch_localization_strict(
        llm_client=llm_client,
        model=model,
        llm_endpoint=llm_endpoint,
        prompt_spec=refine_prompt_spec,
        batch=batch,
        unresolved_matches=unresolved,
        fragment_map=fragment_map,
        logger=logger,
        max_attempts=max_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
        provider_limiter=provider_limiter,
        sync_limiter=sync_limiter,
        prefer_json_mode=prefer_json_mode,
    )
    control_stats.absorb(refine_stats)

    coarse_by_theme_id = {
        str(match.get("theme_id") or "").strip().upper(): match for match in unresolved
    }
    for refined_match in refined_matches:
        coarse_match = coarse_by_theme_id[str(refined_match["theme_id"]).upper()]
        for piece_id in refined_match["relevant_piece_ids"]:
            fragment = fragment_map.get(piece_id)
            if not fragment:
                continue
            records.append(
                _build_positive_record(
                    batch=batch,
                    fragment=fragment,
                    match=coarse_match,
                    reason=str(refined_match.get("reason") or ""),
                    localization_method="llm_refine",
                )
            )

    return BatchScreeningResult(
        records=records,
        usage=_merge_usage(coarse_usage, refine_usage),
        control_stats=control_stats,
        used_refine=True,
    )


async def _run_single_model(
    *,
    tag: str,
    llm_endpoint: LLMEndpointConfig,
    llm_client: OpenAICompatClient,
    prompt_spec: PromptSpec,
    refine_prompt_spec: PromptSpec,
    target_themes: list[dict[str, str]],
    batches: list[dict[str, Any]],
    fragment_map: dict[str, dict[str, str]],
    raw_output_path: Path,
    cursor_path: Path,
    logger,
    concurrency: int,
    fragment_max_attempts: int,
    retry_backoff_seconds: float = 2.0,
    provider_limiter: DualRateLimiter | None = None,
    sync_limiter: DualRateLimiter | None = None,
    sync_gate: SyncProgressGate | None = None,
    progress_callback: Callable[[str, int, float | None, bool], None] | None = None,
    json_mode_enabled: bool = True,
    fragment_failure_fallback: bool = True,
) -> ScreeningStats:
    model = llm_endpoint.model

    next_index = 0
    if cursor_path.exists():
        try:
            cursor = read_json(cursor_path)
            next_index = int(cursor.get("next_batch_index", 0))
        except Exception:  # noqa: BLE001
            next_index = 0

    total = len(batches)
    if next_index >= total:
        if sync_gate is not None:
            await sync_gate.set_processed(tag, total)
        logger.info("%s 无需继续，已完成。", tag)
        return ScreeningStats(
            model_tag=tag,
            model_name=model,
            provider=llm_endpoint.provider,
            start_index=next_index,
            end_index=next_index,
            processed_batches=0,
            processed_fragments=0,
            raw_records_written=0,
            refined_batches=0,
            total_tokens=0,
            prompt_tokens=0,
            completion_tokens=0,
            provider_wait_seconds=0.0,
            sync_wait_seconds=0.0,
            provider_throttled_count=0,
            sync_throttled_count=0,
            retry_count=0,
            failed_batches=0,
            failed_fragments=0,
        )

    existing_flat_rows = read_jsonl(raw_output_path)
    seen_flat = {
        (str(row.get("piece_id") or ""), str(row.get("matched_theme") or ""))
        for row in existing_flat_rows
        if row.get("piece_id") and row.get("matched_theme")
    }

    logger.info(
        "%s 从 batch_index=%s 开始处理，总批次=%s，并发=%s，provider=%s，model=%s",
        tag,
        next_index,
        total,
        concurrency,
        llm_endpoint.provider,
        llm_endpoint.model,
    )

    async def run_one(idx: int) -> tuple[int, BatchScreeningResult]:
        if sync_gate is not None:
            await sync_gate.wait_until_allowed(tag)
        batch = batches[idx]
        result = await _screen_batch_strict(
            llm_client=llm_client,
            model=model,
            llm_endpoint=llm_endpoint,
            prompt_spec=prompt_spec,
            refine_prompt_spec=refine_prompt_spec,
            batch=batch,
            target_themes=target_themes,
            fragment_map=fragment_map,
            logger=logger,
            max_attempts=fragment_max_attempts,
            retry_backoff_seconds=retry_backoff_seconds,
            provider_limiter=provider_limiter,
            sync_limiter=sync_limiter,
            prefer_json_mode=json_mode_enabled,
        )
        return idx, result

    in_flight: dict[asyncio.Task, int] = {}
    buffered_results: dict[int, BatchScreeningResult] = {}

    write_index = next_index
    submit_index = next_index
    safe_concurrency = max(1, int(concurrency))

    stats = ScreeningStats(
        model_tag=tag,
        model_name=model,
        provider=llm_endpoint.provider,
        start_index=next_index,
        end_index=next_index,
        processed_batches=0,
        processed_fragments=0,
        raw_records_written=0,
        refined_batches=0,
        total_tokens=0,
        prompt_tokens=0,
        completion_tokens=0,
        provider_wait_seconds=0.0,
        sync_wait_seconds=0.0,
        provider_throttled_count=0,
        sync_throttled_count=0,
        retry_count=0,
        failed_batches=0,
        failed_fragments=0,
    )

    progress_step = max(1, total // 200)
    next_progress_checkpoint = min(total, (((next_index // progress_step) + 1) * progress_step))
    started_at = monotonic()
    last_progress_log_at = started_at
    last_logged_completed = next_index
    progress_heartbeat_seconds = 5.0

    def log_progress(completed: int, *, force: bool = False) -> None:
        nonlocal next_progress_checkpoint, last_progress_log_at, last_logged_completed
        now = monotonic()
        hit_checkpoint = completed >= next_progress_checkpoint
        hit_done = completed >= total
        hit_heartbeat = (
            completed > last_logged_completed and (now - last_progress_log_at) >= progress_heartbeat_seconds
        )
        if not force and not hit_checkpoint and not hit_done and not hit_heartbeat:
            return

        processed_in_run = max(0, completed - next_index)
        eta_seconds: float | None = None
        if processed_in_run > 0 and completed < total:
            elapsed = max(now - started_at, 1e-6)
            throughput = processed_in_run / elapsed
            if throughput > 0:
                eta_seconds = (total - completed) / throughput

        percentage = 100.0 if total <= 0 else (completed / total) * 100.0
        if progress_callback is not None:
            progress_callback(tag, completed, eta_seconds, force)
        else:
            logger.info(
                "%s 筛选进度 %s 已完成条目 %s/%s (%.2f%%) ETA=%s",
                tag,
                _build_progress_bar(completed, total),
                completed,
                total,
                percentage,
                _format_eta(eta_seconds),
            )
        last_progress_log_at = now
        last_logged_completed = completed

        if completed >= total:
            next_progress_checkpoint = total
            return

        while next_progress_checkpoint <= completed and next_progress_checkpoint < total:
            next_progress_checkpoint = min(total, next_progress_checkpoint + progress_step)

    def schedule_more() -> None:
        nonlocal submit_index
        while submit_index < total and len(in_flight) < safe_concurrency:
            task = asyncio.create_task(run_one(submit_index))
            in_flight[task] = submit_index
            submit_index += 1

    if sync_gate is not None:
        await sync_gate.set_processed(tag, next_index)
    schedule_more()
    log_progress(next_index, force=True)

    try:
        while in_flight:
            done, _ = await asyncio.wait(in_flight.keys(), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                idx = in_flight.pop(task)
                try:
                    result_idx, batch_result = task.result()
                except Exception as e:  # noqa: BLE001
                    batch = batches[idx]
                    if not fragment_failure_fallback:
                        for pending_task in in_flight:
                            pending_task.cancel()
                        raise
                    logger.error(
                        "批次筛选最终失败，按不相关兜底继续。tag=%s model=%s batch_id=%s error=%s",
                        tag,
                        model,
                        batch.get("batch_id"),
                        e,
                    )
                    batch_result = BatchScreeningResult(
                        records=_build_failed_records(
                            batch=batch,
                            fragment_map=fragment_map,
                            target_themes=target_themes,
                            error=e,
                        ),
                        usage=None,
                        control_stats=RequestControlStats(),
                        used_refine=False,
                        failed=True,
                    )
                    result_idx = idx
                if result_idx != idx:
                    raise RuntimeError(f"并发调度错误：task idx={idx} result idx={result_idx}")
                buffered_results[idx] = batch_result

            while write_index in buffered_results:
                batch_result = buffered_results.pop(write_index)
                batch = batches[write_index]

                for record in batch_result.records:
                    key = (str(record.get("piece_id") or ""), str(record.get("matched_theme") or ""))
                    if key in seen_flat:
                        continue
                    append_jsonl(raw_output_path, record)
                    seen_flat.add(key)
                    stats.raw_records_written += 1

                prompt, completion, total_token = _usage_tokens(batch_result.usage)
                stats.prompt_tokens += prompt
                stats.completion_tokens += completion
                stats.total_tokens += total_token
                stats.provider_wait_seconds += batch_result.control_stats.provider_wait_seconds
                stats.sync_wait_seconds += batch_result.control_stats.sync_wait_seconds
                stats.provider_throttled_count += batch_result.control_stats.provider_throttled_count
                stats.sync_throttled_count += batch_result.control_stats.sync_throttled_count
                stats.retry_count += batch_result.control_stats.retry_count
                stats.processed_batches += 1
                stats.processed_fragments += len(batch["piece_ids"])
                if batch_result.used_refine:
                    stats.refined_batches += 1
                if batch_result.failed:
                    stats.failed_batches += 1
                    stats.failed_fragments += len(batch["piece_ids"])

                write_index += 1
                stats.end_index = write_index
                log_progress(write_index)
                if sync_gate is not None:
                    await sync_gate.set_processed(tag, write_index)

                write_json(
                    cursor_path,
                    {
                        "tag": tag,
                        "model": model,
                        "provider": llm_endpoint.provider,
                        "schema_version": 2,
                        "next_batch_index": write_index,
                        "last_batch_id": batch.get("batch_id"),
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
    llm1_endpoint: LLMEndpointConfig,
    llm2_endpoint: LLMEndpointConfig,
    logger,
    llm1_concurrency: int | None = None,
    llm2_concurrency: int | None = None,
    sync_headroom: float = 0.85,
    sync_max_ahead: int = 128,
    sync_mode: str = "lowest_shared",
    fragment_max_attempts: int = 3,
    retry_backoff_seconds: float = 2.0,
    screening_batch_max_chars: int = 300,
) -> tuple[Path, Path, dict[str, Any]]:
    _theme_names(target_themes)
    prompt_spec = load_prompt("stage2_screening")
    refine_prompt_spec = load_prompt("stage2_localization")
    internal_dir = stage2_internal_dir(project_dir)
    internal_dir.mkdir(parents=True, exist_ok=True)

    fragments = read_jsonl(fragments_path)
    if not fragments:
        raise RuntimeError(f"阶段2.2无法继续：未找到可筛选碎片 {fragments_path}")

    if llm1_concurrency is not None and llm1_concurrency < 1:
        raise RuntimeError("阶段2.2参数 llm1_concurrency 必须 >= 1")
    if llm2_concurrency is not None and llm2_concurrency < 1:
        raise RuntimeError("阶段2.2参数 llm2_concurrency 必须 >= 1")
    if screening_batch_max_chars < 1:
        raise RuntimeError("阶段2.2参数 screening_batch_max_chars 必须 >= 1")

    fragment_map = {
        str(fragment.get("piece_id") or ""): {
            "piece_id": str(fragment.get("piece_id") or ""),
            "source_file": str(fragment.get("source_file") or ""),
            "original_text": str(fragment.get("original_text") or ""),
        }
        for fragment in fragments
        if fragment.get("piece_id")
    }
    batches = _load_or_build_screening_batches(
        project_dir=project_dir,
        fragments=fragments,
        batch_max_chars=screening_batch_max_chars,
        logger=logger,
    )

    llm1_raw_path = project_dir / "2_llm1_raw.jsonl"
    llm2_raw_path = project_dir / "2_llm2_raw.jsonl"
    cursor1_path = resolve_stage2_internal_path(project_dir, ".cursor_llm1.json")
    cursor2_path = resolve_stage2_internal_path(project_dir, ".cursor_llm2.json")
    progress = _InlineScreeningProgress(total=len(batches))

    llm1_provider_limits = RateLimits(
        rpm=llm1_endpoint.effective_rpm,
        tpm=llm1_endpoint.effective_tpm,
    ).normalized()
    llm2_provider_limits = RateLimits(
        rpm=llm2_endpoint.effective_rpm,
        tpm=llm2_endpoint.effective_tpm,
    ).normalized()
    llm1_provider_limiter = DualRateLimiter(
        name=f"model:llm1:{llm1_endpoint.provider}/{llm1_endpoint.model}",
        limits=llm1_provider_limits,
    )
    llm2_provider_limiter = DualRateLimiter(
        name=f"model:llm2:{llm2_endpoint.provider}/{llm2_endpoint.model}",
        limits=llm2_provider_limits,
    )
    estimated_tokens_per_request = _estimate_screening_request_tokens(
        prompt_spec=prompt_spec,
        target_themes=target_themes,
        batches=batches,
    )
    if llm1_concurrency is None:
        llm1_concurrency = _derive_auto_concurrency(
            limits=llm1_provider_limits,
            estimated_tokens_per_request=estimated_tokens_per_request,
        )
        logger.info(
            "阶段2.2自动并发 llm1=%s (rpm=%s tpm=%s est_tokens=%s)",
            llm1_concurrency,
            llm1_provider_limits.rpm,
            llm1_provider_limits.tpm,
            estimated_tokens_per_request,
        )
    if llm2_concurrency is None:
        llm2_concurrency = _derive_auto_concurrency(
            limits=llm2_provider_limits,
            estimated_tokens_per_request=estimated_tokens_per_request,
        )
        logger.info(
            "阶段2.2自动并发 llm2=%s (rpm=%s tpm=%s est_tokens=%s)",
            llm2_concurrency,
            llm2_provider_limits.rpm,
            llm2_provider_limits.tpm,
            estimated_tokens_per_request,
        )

    if sync_mode != "lowest_shared":
        raise RuntimeError(f"阶段2.2不支持 sync_mode={sync_mode}")
    sync_limits = build_lowest_shared_limits(
        llm1_limits=llm1_provider_limits,
        llm2_limits=llm2_provider_limits,
        headroom=sync_headroom,
    )
    sync_limiter = DualRateLimiter(name="sync:llm1-llm2", limits=sync_limits)
    sync_gate = SyncProgressGate(max_ahead=sync_max_ahead)

    def on_progress(tag: str, completed: int, eta_seconds: float | None, force: bool) -> None:
        progress.update(tag=tag, completed=completed, eta_seconds=eta_seconds, force=force)

    progress_callback = on_progress if progress.enabled else None
    llm1_json_mode_enabled, llm2_json_mode_enabled = await asyncio.gather(
        _probe_json_mode_support(
            llm_client=llm_client,
            llm_endpoint=llm1_endpoint,
            logger=logger,
        ),
        _probe_json_mode_support(
            llm_client=llm_client,
            llm_endpoint=llm2_endpoint,
            logger=logger,
        ),
    )

    try:
        stats1, stats2 = await asyncio.gather(
            _run_single_model(
                tag="llm1",
                llm_endpoint=llm1_endpoint,
                llm_client=llm_client,
                prompt_spec=prompt_spec,
                refine_prompt_spec=refine_prompt_spec,
                target_themes=target_themes,
                batches=batches,
                fragment_map=fragment_map,
                raw_output_path=llm1_raw_path,
                cursor_path=cursor1_path,
                logger=logger,
                concurrency=llm1_concurrency,
                fragment_max_attempts=fragment_max_attempts,
                retry_backoff_seconds=retry_backoff_seconds,
                provider_limiter=llm1_provider_limiter,
                sync_limiter=sync_limiter,
                sync_gate=sync_gate,
                progress_callback=progress_callback,
                json_mode_enabled=llm1_json_mode_enabled,
                fragment_failure_fallback=True,
            ),
            _run_single_model(
                tag="llm2",
                llm_endpoint=llm2_endpoint,
                llm_client=llm_client,
                prompt_spec=prompt_spec,
                refine_prompt_spec=refine_prompt_spec,
                target_themes=target_themes,
                batches=batches,
                fragment_map=fragment_map,
                raw_output_path=llm2_raw_path,
                cursor_path=cursor2_path,
                logger=logger,
                concurrency=llm2_concurrency,
                fragment_max_attempts=fragment_max_attempts,
                retry_backoff_seconds=retry_backoff_seconds,
                provider_limiter=llm2_provider_limiter,
                sync_limiter=sync_limiter,
                sync_gate=sync_gate,
                progress_callback=progress_callback,
                json_mode_enabled=llm2_json_mode_enabled,
                fragment_failure_fallback=True,
            ),
        )
    finally:
        progress.finalize()

    stats_payload = {
        "total_fragments": len(fragments),
        "total_batches": len(batches),
        "target_theme_count": len(target_themes),
        "screening_batch_max_chars": int(screening_batch_max_chars),
        "llm1": stats1.to_dict(),
        "llm2": stats2.to_dict(),
        "sync": {
            "mode": sync_mode,
            "headroom": sync_headroom,
            "max_ahead": sync_max_ahead,
            "rpm": sync_limits.rpm,
            "tpm": sync_limits.tpm,
        },
        "effective_concurrency": {
            "llm1": llm1_concurrency,
            "llm2": llm2_concurrency,
            "estimated_tokens_per_request": estimated_tokens_per_request,
        },
    }

    logger.info(
        "阶段2.2完成: %s, %s (fragments=%s, batches=%s)",
        llm1_raw_path,
        llm2_raw_path,
        len(fragments),
        len(batches),
    )
    return llm1_raw_path, llm2_raw_path, stats_payload
