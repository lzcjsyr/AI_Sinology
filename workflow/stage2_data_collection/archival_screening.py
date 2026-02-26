from __future__ import annotations

import asyncio
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Callable

from core.config import LLMEndpointConfig
from core.llm_client import OpenAICompatClient
from core.prompt_loader import PromptSpec, build_messages, load_prompt
from workflow.stage2_data_collection.rate_control import (
    AcquireReservation,
    DualRateLimiter,
    RateLimits,
    build_lowest_shared_limits,
)
from core.utils import append_jsonl, parse_json_from_text, read_json, read_jsonl, write_json


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ScreeningStats:
    model_tag: str
    model_name: str
    provider: str
    start_index: int
    end_index: int
    processed_fragments: int
    raw_records_written: int
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    provider_wait_seconds: float
    sync_wait_seconds: float
    provider_throttled_count: int
    sync_throttled_count: int
    retry_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_tag": self.model_tag,
            "model_name": self.model_name,
            "provider": self.provider,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "processed_fragments": self.processed_fragments,
            "raw_records_written": self.raw_records_written,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "provider_wait_seconds": round(self.provider_wait_seconds, 3),
            "sync_wait_seconds": round(self.sync_wait_seconds, 3),
            "provider_throttled_count": self.provider_throttled_count,
            "sync_throttled_count": self.sync_throttled_count,
            "retry_count": self.retry_count,
        }


@dataclass
class RequestControlStats:
    provider_wait_seconds: float = 0.0
    sync_wait_seconds: float = 0.0
    provider_throttled_count: int = 0
    sync_throttled_count: int = 0
    retry_count: int = 0


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
    llm_endpoint: LLMEndpointConfig,
    prompt_spec: PromptSpec,
    fragment: dict[str, str],
    target_themes: list[dict[str, str]],
    logger,
    max_attempts: int,
    retry_backoff_seconds: float,
    provider_limiter: DualRateLimiter | None = None,
    sync_limiter: DualRateLimiter | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, RequestControlStats]:
    themes_block = "\n".join(
        f"- T{i+1} | theme: {t['theme']} | description: {t.get('description', '')}"
        for i, t in enumerate(target_themes)
    )

    messages = build_messages(
        prompt_spec,
        themes_block=themes_block,
        piece_id=fragment["piece_id"],
        source_file=fragment["source_file"],
        original_text=fragment["original_text"],
    )
    estimated_tokens = _estimate_total_tokens(messages)

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
                api_base=llm_endpoint.base_url,
                temperature=0.0,
            )
            payload = parse_json_from_text(response.content)
            matches = _normalize_matches_strict(payload, target_themes)
            actual_total_tokens = _extract_total_tokens(response.usage)
            if provider_limiter is not None and provider_reservation is not None:
                await provider_limiter.commit(provider_reservation, actual_total_tokens)
            if sync_limiter is not None and sync_reservation is not None:
                await sync_limiter.commit(sync_reservation, actual_total_tokens)
            return matches, response.usage, control_stats
        except Exception as e:  # noqa: BLE001
            last_error = e
            control_stats.retry_count += 1
            logger.warning(
                "片段筛选失败，准备重试。piece_id=%s model=%s attempt=%s error=%s",
                fragment.get("piece_id"),
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


def _usage_tokens(usage: dict[str, Any] | None) -> tuple[int, int, int]:
    if not usage:
        return 0, 0, 0
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    total = int(usage.get("total_tokens") or (prompt + completion))
    return prompt, completion, total


def _extract_total_tokens(usage: dict[str, Any] | None) -> int | None:
    if not usage:
        return None
    total = usage.get("total_tokens")
    if total is None:
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        combined = prompt + completion
        return combined or None
    return int(total)


def _estimate_total_tokens(messages: list[dict[str, str]]) -> int:
    text = "\n".join(str(message.get("content") or "") for message in messages)
    # Roughly estimate OpenAI-compatible token usage without tokenizer dependency.
    return max(256, int(len(text) * 0.72) + 160)


def _estimate_screening_request_tokens(
    *,
    prompt_spec: PromptSpec,
    target_themes: list[dict[str, str]],
    fragments: list[dict[str, str]],
    sample_size: int = 8,
) -> int:
    if not fragments:
        return 512
    themes_block = "\n".join(
        f"- T{i+1} | theme: {t['theme']} | description: {t.get('description', '')}"
        for i, t in enumerate(target_themes)
    )
    upper = max(1, min(int(sample_size), len(fragments)))
    estimated: list[int] = []
    for fragment in fragments[:upper]:
        messages = build_messages(
            prompt_spec,
            themes_block=themes_block,
            piece_id=fragment["piece_id"],
            source_file=fragment["source_file"],
            original_text=fragment["original_text"],
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

        # Wait until both model streams are available, to avoid colliding with
        # startup logs and drawing a partial single-model line.
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


async def _run_single_model(
    *,
    tag: str,
    llm_endpoint: LLMEndpointConfig,
    llm_client: OpenAICompatClient,
    prompt_spec: PromptSpec,
    target_themes: list[dict[str, str]],
    fragments: list[dict[str, str]],
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
) -> ScreeningStats:
    model = llm_endpoint.model

    next_index = 0
    if cursor_path.exists():
        try:
            cursor = read_json(cursor_path)
            next_index = int(cursor.get("next_index", 0))
        except Exception:  # noqa: BLE001
            next_index = 0

    total = len(fragments)
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
            processed_fragments=0,
            raw_records_written=0,
            total_tokens=0,
            prompt_tokens=0,
            completion_tokens=0,
            provider_wait_seconds=0.0,
            sync_wait_seconds=0.0,
            provider_throttled_count=0,
            sync_throttled_count=0,
            retry_count=0,
        )

    # Idempotent resume support in case cursor lags behind already written rows.
    existing_flat_rows = read_jsonl(raw_output_path)
    seen_flat = {
        (str(row.get("piece_id") or ""), str(row.get("matched_theme") or ""))
        for row in existing_flat_rows
        if row.get("piece_id") and row.get("matched_theme")
    }

    logger.info(
        "%s 从 index=%s 开始处理，总量=%s，并发=%s，provider=%s，model=%s",
        tag,
        next_index,
        total,
        concurrency,
        llm_endpoint.provider,
        llm_endpoint.model,
    )

    async def run_one(
        idx: int,
    ) -> tuple[int, list[dict[str, Any]], dict[str, Any] | None, RequestControlStats]:
        if sync_gate is not None:
            await sync_gate.wait_until_allowed(tag)
        fragment = fragments[idx]
        matches, usage, control_stats = await _classify_fragment_strict(
            llm_client=llm_client,
            model=model,
            llm_endpoint=llm_endpoint,
            prompt_spec=prompt_spec,
            fragment=fragment,
            target_themes=target_themes,
            logger=logger,
            max_attempts=fragment_max_attempts,
            retry_backoff_seconds=retry_backoff_seconds,
            provider_limiter=provider_limiter,
            sync_limiter=sync_limiter,
        )
        return idx, matches, usage, control_stats

    in_flight: dict[asyncio.Task, int] = {}
    buffered_results: dict[
        int, tuple[list[dict[str, Any]], dict[str, Any] | None, RequestControlStats]
    ] = {}

    write_index = next_index
    submit_index = next_index
    safe_concurrency = max(1, int(concurrency))

    stats = ScreeningStats(
        model_tag=tag,
        model_name=model,
        provider=llm_endpoint.provider,
        start_index=next_index,
        end_index=next_index,
        processed_fragments=0,
        raw_records_written=0,
        total_tokens=0,
        prompt_tokens=0,
        completion_tokens=0,
        provider_wait_seconds=0.0,
        sync_wait_seconds=0.0,
        provider_throttled_count=0,
        sync_throttled_count=0,
        retry_count=0,
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
                    result_idx, matches, usage, control_stats = task.result()
                except Exception:
                    for pending_task in in_flight:
                        pending_task.cancel()
                    raise
                if result_idx != idx:
                    raise RuntimeError(f"并发调度错误：task idx={idx} result idx={result_idx}")
                buffered_results[idx] = (matches, usage, control_stats)

            while write_index in buffered_results:
                matches, usage, control_stats = buffered_results.pop(write_index)
                fragment = fragments[write_index]

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
                stats.provider_wait_seconds += control_stats.provider_wait_seconds
                stats.sync_wait_seconds += control_stats.sync_wait_seconds
                stats.provider_throttled_count += control_stats.provider_throttled_count
                stats.sync_throttled_count += control_stats.sync_throttled_count
                stats.retry_count += control_stats.retry_count
                stats.processed_fragments += 1
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
) -> tuple[Path, Path, dict[str, Any]]:
    _theme_names(target_themes)
    prompt_spec = load_prompt("stage2_screening")

    fragments = read_jsonl(fragments_path)
    if not fragments:
        raise RuntimeError(f"阶段2.2无法继续：未找到可筛选碎片 {fragments_path}")

    if llm1_concurrency is not None and llm1_concurrency < 1:
        raise RuntimeError("阶段2.2参数 llm1_concurrency 必须 >= 1")
    if llm2_concurrency is not None and llm2_concurrency < 1:
        raise RuntimeError("阶段2.2参数 llm2_concurrency 必须 >= 1")

    llm1_raw_path = project_dir / "2_llm1_raw.jsonl"
    llm2_raw_path = project_dir / "2_llm2_raw.jsonl"
    cursor1_path = project_dir / ".cursor_llm1.json"
    cursor2_path = project_dir / ".cursor_llm2.json"
    progress = _InlineScreeningProgress(total=len(fragments))

    llm1_provider_limits = RateLimits(rpm=llm1_endpoint.rpm, tpm=llm1_endpoint.tpm).normalized()
    llm2_provider_limits = RateLimits(rpm=llm2_endpoint.rpm, tpm=llm2_endpoint.tpm).normalized()
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
        fragments=fragments,
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

    try:
        stats1, stats2 = await asyncio.gather(
            _run_single_model(
                tag="llm1",
                llm_endpoint=llm1_endpoint,
                llm_client=llm_client,
                prompt_spec=prompt_spec,
                target_themes=target_themes,
                fragments=fragments,
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
            ),
            _run_single_model(
                tag="llm2",
                llm_endpoint=llm2_endpoint,
                llm_client=llm_client,
                prompt_spec=prompt_spec,
                target_themes=target_themes,
                fragments=fragments,
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
            ),
        )
    finally:
        progress.finalize()

    stats_payload = {
        "total_fragments": len(fragments),
        "target_theme_count": len(target_themes),
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

    logger.info("阶段2.2完成: %s, %s", llm1_raw_path, llm2_raw_path)
    return llm1_raw_path, llm2_raw_path, stats_payload
