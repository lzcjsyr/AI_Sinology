from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from core.config import LLMEndpointConfig
from core.llm_client import OpenAICompatClient
from core.project_paths import (
    resolve_stage2_internal_path,
    resolve_stage2_manifest_path,
    resolve_stage2_json_path,
    stage2_internal_dir,
)
from core.utils import ensure_dir, read_json, read_jsonl, write_json, write_text
from workflow.stage2_data_collection.archival_arbitration import run_archival_arbitration
from workflow.stage2_data_collection.archival_screening import run_archival_screening
from workflow.stage2_data_collection.data_ingestion.parse_kanripo import (
    ScopeOption,
    list_available_scope_dirs,
    list_available_scope_options,
    list_available_scopes,
    parse_kanripo_to_fragments,
)

def _signature(
    selected_scopes: list[str],
    target_themes: list[dict[str, str]],
    llm1_endpoint: LLMEndpointConfig,
    llm2_endpoint: LLMEndpointConfig,
    llm3_endpoint: LLMEndpointConfig,
    screening_batch_max_chars: int,
) -> dict[str, Any]:
    return {
        "stage2_schema_version": 2,
        "scopes": sorted(selected_scopes),
        "target_themes": [
            {
                "theme": str(item.get("theme") or "").strip(),
                "description": str(item.get("description") or "").strip(),
            }
            for item in target_themes
        ],
        "screening_batch_max_chars": int(screening_batch_max_chars),
        "stage2_llms": [
            {
                "stage": llm1_endpoint.stage,
                "provider": llm1_endpoint.provider,
                "model": llm1_endpoint.model,
            },
            {
                "stage": llm2_endpoint.stage,
                "provider": llm2_endpoint.provider,
                "model": llm2_endpoint.model,
            },
            {
                "stage": llm3_endpoint.stage,
                "provider": llm3_endpoint.provider,
                "model": llm3_endpoint.model,
            },
        ],
    }


def _read_manifest(project_dir: Path) -> dict[str, Any] | None:
    path = resolve_stage2_manifest_path(project_dir)
    if not path.exists():
        return None
    try:
        data = read_json(path)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(data, dict):
        return None
    return data


def _write_manifest(project_dir: Path, payload: dict[str, Any]) -> None:
    write_json(stage2_internal_dir(project_dir) / "2_stage_manifest.json", payload)


def read_cached_scopes(project_dir: Path, available_scopes: list[str]) -> list[str]:
    manifest = _read_manifest(project_dir)
    if not manifest:
        return []
    signature = manifest.get("signature")
    if not isinstance(signature, dict):
        return []
    cached = signature.get("scopes")
    if not isinstance(cached, list):
        return []
    available = set(available_scopes)
    return [str(scope) for scope in cached if str(scope) in available]


def _reset_stage2_artifacts(project_dir: Path) -> None:
    cleanup_files = [
        project_dir / "_processed_data" / "kanripo_fragments.jsonl",
        project_dir / "_processed_data" / "kanripo_screening_batches.jsonl",
        project_dir / "2_stage_manifest.json",
        resolve_stage2_manifest_path(project_dir),
        project_dir / "2_llm1_raw.jsonl",
        project_dir / "2_llm2_raw.jsonl",
        project_dir / ".cursor_llm1.json",
        resolve_stage2_internal_path(project_dir, ".cursor_llm1.json"),
        project_dir / ".cursor_llm2.json",
        resolve_stage2_internal_path(project_dir, ".cursor_llm2.json"),
        project_dir / "2_consensus_data.yaml",
        project_dir / "2_consensus_data.json",
        resolve_stage2_json_path(project_dir, "2_consensus_data.json"),
        project_dir / "2_disputed_data.yaml",
        project_dir / "2_disputed_data.json",
        resolve_stage2_json_path(project_dir, "2_disputed_data.json"),
        project_dir / "2_llm3_verified.yaml",
        project_dir / "2_llm3_verified.json",
        resolve_stage2_json_path(project_dir, "2_llm3_verified.json"),
        project_dir / "2_final_corpus.yaml",
        project_dir / "2_final_corpus.json",
        resolve_stage2_json_path(project_dir, "2_final_corpus.json"),
        project_dir / "2_screening_failed_pieces.yaml",
        project_dir / "2_screening_failed_pieces.json",
        resolve_stage2_json_path(project_dir, "2_screening_failed_pieces.json"),
        project_dir / "2_stage_failure_report.md",
    ]
    for file_path in dict.fromkeys(cleanup_files):
        if file_path.exists():
            file_path.unlink()


def _load_existing_final(project_dir: Path) -> list[dict[str, Any]] | None:
    final_json_path = resolve_stage2_json_path(project_dir, "2_final_corpus.json")
    if not final_json_path.exists():
        return None
    try:
        payload = read_json(final_json_path)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(payload, list) and payload:
        return payload
    return None


def _can_resume(
    *,
    project_dir: Path,
    signature: dict[str, Any],
) -> tuple[bool, int | None]:
    manifest = _read_manifest(project_dir)
    if not manifest:
        return False, None

    if manifest.get("signature") != signature:
        return False, None

    if manifest.get("status") not in {"running", "screening_completed", "arbitrating"}:
        return False, None

    if resolve_stage2_json_path(project_dir, "2_final_corpus.json").exists():
        return False, None

    # Resume requires at least the generated fragments pool and one progress signal.
    has_fragments = (project_dir / "_processed_data" / "kanripo_fragments.jsonl").exists()
    has_batches = (project_dir / "_processed_data" / "kanripo_screening_batches.jsonl").exists()
    has_cursor_or_raw = (
        resolve_stage2_internal_path(project_dir, ".cursor_llm1.json").exists()
        or resolve_stage2_internal_path(project_dir, ".cursor_llm2.json").exists()
        or (project_dir / "2_llm1_raw.jsonl").exists()
        or (project_dir / "2_llm2_raw.jsonl").exists()
    )
    if not (has_fragments and has_batches and has_cursor_or_raw):
        return False, None

    max_fragments = manifest.get("max_fragments")
    if max_fragments is not None:
        try:
            max_fragments = int(max_fragments)
        except Exception:  # noqa: BLE001
            max_fragments = None
    return True, max_fragments


def _write_failure_report(
    *,
    project_dir: Path,
    selected_scopes: list[str],
    target_themes: list[dict[str, str]],
    attempts: int,
    max_fragments: int | None,
    screening_audit: dict[str, Any] | None,
) -> None:
    llm1_rows = read_jsonl(project_dir / "2_llm1_raw.jsonl")
    llm2_rows = read_jsonl(project_dir / "2_llm2_raw.jsonl")

    llm1_true = sum(1 for row in llm1_rows if row.get("is_relevant") is True)
    llm2_true = sum(1 for row in llm2_rows if row.get("is_relevant") is True)

    unique_piece_1 = len({row.get("piece_id") for row in llm1_rows if row.get("piece_id")})
    unique_piece_2 = len({row.get("piece_id") for row in llm2_rows if row.get("piece_id")})

    manual_review = screening_audit.get("manual_review") if isinstance(screening_audit, dict) else None
    manual_review_text = ""
    if isinstance(manual_review, dict):
        manual_review_text = (
            f"- manual_review: piece_count={manual_review.get('piece_count')} "
            f"path={manual_review.get('yaml_path')}"
        )

    audit_text = f"- screening_audit: {screening_audit}\n" if screening_audit else ""

    report = "\n".join(
        [
            "# 阶段二失败报告",
            "",
            "## 失败结论",
            "- 阶段二在重试后仍未产出 `2_final_corpus.yaml`，流程已按严格模式终止。",
            "",
            "## 本次运行统计",
            f"- scopes: {selected_scopes}",
            f"- target_themes: {[item.get('theme') for item in target_themes]}",
            f"- 尝试次数: {attempts}",
            f"- 最终 max_fragments: {max_fragments}",
            f"- llm1 原始记录数: {len(llm1_rows)} | is_relevant=true: {llm1_true} | unique_piece: {unique_piece_1}",
            f"- llm2 原始记录数: {len(llm2_rows)} | is_relevant=true: {llm2_true} | unique_piece: {unique_piece_2}",
            manual_review_text,
            audit_text.strip(),
            "",
            "## 建议动作",
            "1. 更换或扩大检索范围（scope），确保语料与研究主题相关。",
            "2. 调整阶段一 `target_themes`，避免主题过窄导致零命中。",
            "3. 增大 `--max-fragments`，必要时不设上限。",
            "4. 检查阶段二提示词是否过于严格，必要时降低筛选阈值。",
            "",
        ]
    )
    write_text(project_dir / "2_stage_failure_report.md", report + "\n")


async def _run_stage2_data_collection_async(
    *,
    project_dir: Path,
    kanripo_dir: Path,
    selected_scopes: list[str],
    target_themes: list[dict[str, str]],
    llm_client: OpenAICompatClient,
    llm1_endpoint: LLMEndpointConfig,
    llm2_endpoint: LLMEndpointConfig,
    llm3_endpoint: LLMEndpointConfig,
    logger,
    processed_dir: Path,
    signature: dict[str, Any],
    can_resume: bool,
    resume_limit: int | None,
    max_fragments: int | None,
    max_empty_retries: int,
    llm1_concurrency: int | None,
    llm2_concurrency: int | None,
    arbitration_concurrency: int | None,
    sync_headroom: float,
    sync_max_ahead: int,
    sync_mode: str,
    fragment_max_attempts: int,
    retry_backoff_seconds: float,
    screening_batch_max_chars: int,
) -> list[dict[str, Any]]:
    attempt = 0
    current_limit = resume_limit if can_resume else max_fragments
    latest_screening_audit: dict[str, Any] | None = None

    while attempt <= max_empty_retries:
        attempt += 1
        is_resume_attempt = attempt == 1 and can_resume

        if is_resume_attempt:
            logger.info(
                "阶段二尝试 #%s（断点续传），max_fragments=%s",
                attempt,
                current_limit,
            )
            fragments_path = project_dir / "_processed_data" / "kanripo_fragments.jsonl"
            if not fragments_path.exists():
                raise RuntimeError("阶段二断点续传失败：缺少 fragments 文件")
        else:
            if attempt > 1 and current_limit is not None:
                current_limit = current_limit * 2
            logger.info("阶段二尝试 #%s，max_fragments=%s", attempt, current_limit)
            _reset_stage2_artifacts(project_dir)
            fragments_path = parse_kanripo_to_fragments(
                kanripo_dir=kanripo_dir,
                selected_scopes=selected_scopes,
                project_processed_dir=processed_dir,
                logger=logger,
                max_fragments=current_limit,
            )

        _write_manifest(
            project_dir,
            {
                "status": "running",
                "attempt": attempt,
                "max_fragments": current_limit,
                "signature": signature,
            },
        )

        llm1_raw_path, llm2_raw_path, screening_audit = await run_archival_screening(
            project_dir=project_dir,
            fragments_path=fragments_path,
            target_themes=target_themes,
            llm_client=llm_client,
            llm1_endpoint=llm1_endpoint,
            llm2_endpoint=llm2_endpoint,
            logger=logger,
            llm1_concurrency=llm1_concurrency,
            llm2_concurrency=llm2_concurrency,
            sync_headroom=sync_headroom,
            sync_max_ahead=sync_max_ahead,
            sync_mode=sync_mode,
            fragment_max_attempts=fragment_max_attempts,
            retry_backoff_seconds=retry_backoff_seconds,
            screening_batch_max_chars=screening_batch_max_chars,
        )
        latest_screening_audit = screening_audit

        _write_manifest(
            project_dir,
            {
                "status": "screening_completed",
                "attempt": attempt,
                "max_fragments": current_limit,
                "signature": signature,
                "screening_audit": screening_audit,
            },
        )
        _write_manifest(
            project_dir,
            {
                "status": "arbitrating",
                "attempt": attempt,
                "max_fragments": current_limit,
                "signature": signature,
                "screening_audit": screening_audit,
            },
        )

        final_corpus = await run_archival_arbitration(
            project_dir=project_dir,
            llm1_raw_path=llm1_raw_path,
            llm2_raw_path=llm2_raw_path,
            llm_client=llm_client,
            llm3_endpoint=llm3_endpoint,
            logger=logger,
            concurrency=arbitration_concurrency,
            retry_backoff_seconds=retry_backoff_seconds,
        )

        if final_corpus:
            _write_manifest(
                project_dir,
                {
                    "status": "completed",
                    "attempt": attempt,
                    "max_fragments": current_limit,
                    "signature": signature,
                    "final_corpus_count": len(final_corpus),
                    "screening_audit": screening_audit,
                },
            )
            return final_corpus

        logger.warning("阶段二尝试 #%s 未产生有效 final_corpus。", attempt)
        _write_manifest(
            project_dir,
            {
                "status": "empty_final",
                "attempt": attempt,
                "max_fragments": current_limit,
                "signature": signature,
                "screening_audit": screening_audit,
            },
        )

    _write_failure_report(
        project_dir=project_dir,
        selected_scopes=selected_scopes,
        target_themes=target_themes,
        attempts=attempt,
        max_fragments=current_limit,
        screening_audit=latest_screening_audit,
    )
    _write_manifest(
        project_dir,
        {
            "status": "failed",
            "attempt": attempt,
            "max_fragments": current_limit,
            "signature": signature,
            "screening_audit": latest_screening_audit,
            "failure_report": "2_stage_failure_report.md",
        },
    )
    raise RuntimeError(
        "阶段二失败：多次重试后 `2_final_corpus.yaml` 仍为空，已停止流程。"
        "详见 2_stage_failure_report.md。"
    )


def run_stage2_data_collection(
    *,
    project_dir: Path,
    kanripo_dir: Path,
    selected_scopes: list[str],
    target_themes: list[dict[str, str]],
    llm_client: OpenAICompatClient,
    llm1_endpoint: LLMEndpointConfig,
    llm2_endpoint: LLMEndpointConfig,
    llm3_endpoint: LLMEndpointConfig,
    logger,
    max_fragments: int | None = None,
    max_empty_retries: int = 2,
    llm1_concurrency: int | None = None,
    llm2_concurrency: int | None = None,
    arbitration_concurrency: int | None = None,
    sync_headroom: float = 0.85,
    sync_max_ahead: int = 128,
    sync_mode: str = "lowest_shared",
    fragment_max_attempts: int = 3,
    retry_backoff_seconds: float = 2.0,
    screening_batch_max_chars: int = 300,
) -> list[dict[str, Any]]:
    if not selected_scopes:
        raise ValueError("阶段二需要至少一个语料范围（scope）。")

    processed_dir = project_dir / "_processed_data"
    ensure_dir(processed_dir)
    ensure_dir(stage2_internal_dir(project_dir))

    existing_final = _load_existing_final(project_dir)
    if existing_final is not None:
        logger.info("阶段二已存在有效 final corpus，直接复用。")
        return existing_final

    signature = _signature(
        selected_scopes,
        target_themes,
        llm1_endpoint,
        llm2_endpoint,
        llm3_endpoint,
        screening_batch_max_chars,
    )
    can_resume, resume_limit = _can_resume(project_dir=project_dir, signature=signature)
    return asyncio.run(
        _run_stage2_data_collection_async(
            project_dir=project_dir,
            kanripo_dir=kanripo_dir,
            selected_scopes=selected_scopes,
            target_themes=target_themes,
            llm_client=llm_client,
            llm1_endpoint=llm1_endpoint,
            llm2_endpoint=llm2_endpoint,
            llm3_endpoint=llm3_endpoint,
            logger=logger,
            processed_dir=processed_dir,
            signature=signature,
            can_resume=can_resume,
            resume_limit=resume_limit,
            max_fragments=max_fragments,
            max_empty_retries=max_empty_retries,
            llm1_concurrency=llm1_concurrency,
            llm2_concurrency=llm2_concurrency,
            arbitration_concurrency=arbitration_concurrency,
            sync_headroom=sync_headroom,
            sync_max_ahead=sync_max_ahead,
            sync_mode=sync_mode,
            fragment_max_attempts=fragment_max_attempts,
            retry_backoff_seconds=retry_backoff_seconds,
            screening_batch_max_chars=screening_batch_max_chars,
        )
    )


__all__ = [
    "ScopeOption",
    "list_available_scope_dirs",
    "list_available_scope_options",
    "list_available_scopes",
    "run_stage2_data_collection",
    "read_cached_scopes",
]
