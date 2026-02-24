from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from core.llm_client import OpenAICompatClient
from core.utils import ensure_dir, read_jsonl, write_text
from workflow.stage2_data_collection.archival_arbitration import run_archival_arbitration
from workflow.stage2_data_collection.archival_screening import run_archival_screening
from workflow.stage2_data_collection.data_ingestion.parse_kanripo import (
    list_available_scopes,
    parse_kanripo_to_fragments,
)


def _reset_stage2_artifacts(project_dir: Path) -> None:
    cleanup_files = [
        project_dir / "_processed_data" / "kanripo_fragments.jsonl",
        project_dir / "2_llm1_raw.jsonl",
        project_dir / "2_llm2_raw.jsonl",
        project_dir / ".cursor_llm1.json",
        project_dir / ".cursor_llm2.json",
        project_dir / "2_consensus_data.yaml",
        project_dir / "2_consensus_data.json",
        project_dir / "2_disputed_data.yaml",
        project_dir / "2_disputed_data.json",
        project_dir / "2_llm3_verified.yaml",
        project_dir / "2_llm3_verified.json",
        project_dir / "2_final_corpus.yaml",
        project_dir / "2_final_corpus.json",
        project_dir / "2_stage_failure_report.md",
    ]
    for file_path in cleanup_files:
        if file_path.exists():
            file_path.unlink()


def _write_failure_report(
    *,
    project_dir: Path,
    selected_scopes: list[str],
    target_themes: list[dict[str, str]],
    attempts: int,
    max_fragments: int | None,
) -> None:
    llm1_rows = read_jsonl(project_dir / "2_llm1_raw.jsonl")
    llm2_rows = read_jsonl(project_dir / "2_llm2_raw.jsonl")

    llm1_true = sum(1 for row in llm1_rows if row.get("is_relevant") is True)
    llm2_true = sum(1 for row in llm2_rows if row.get("is_relevant") is True)

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
            f"- llm1 原始记录数: {len(llm1_rows)} | is_relevant=true: {llm1_true}",
            f"- llm2 原始记录数: {len(llm2_rows)} | is_relevant=true: {llm2_true}",
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


def run_stage2_data_collection(
    *,
    project_dir: Path,
    kanripo_dir: Path,
    selected_scopes: list[str],
    target_themes: list[dict[str, str]],
    llm_client: OpenAICompatClient,
    model_llm1: str,
    model_llm2: str,
    model_llm3: str,
    logger,
    max_fragments: int | None = None,
    max_empty_retries: int = 2,
) -> list[dict[str, Any]]:
    if not selected_scopes:
        raise ValueError("阶段二需要至少一个语料范围（scope）。")

    processed_dir = project_dir / "_processed_data"
    ensure_dir(processed_dir)

    attempt = 0
    current_limit = max_fragments
    final_corpus: list[dict[str, Any]] = []

    while attempt <= max_empty_retries:
        attempt += 1
        _reset_stage2_artifacts(project_dir)
        logger.info("阶段二尝试 #%s，max_fragments=%s", attempt, current_limit)

        fragments_path = parse_kanripo_to_fragments(
            kanripo_dir=kanripo_dir,
            selected_scopes=selected_scopes,
            project_processed_dir=processed_dir,
            logger=logger,
            max_fragments=current_limit,
        )

        llm1_raw_path, llm2_raw_path = asyncio.run(
            run_archival_screening(
                project_dir=project_dir,
                fragments_path=fragments_path,
                target_themes=target_themes,
                llm_client=llm_client,
                model_llm1=model_llm1,
                model_llm2=model_llm2,
                logger=logger,
            )
        )

        final_corpus = asyncio.run(
            run_archival_arbitration(
                project_dir=project_dir,
                llm1_raw_path=llm1_raw_path,
                llm2_raw_path=llm2_raw_path,
                llm_client=llm_client,
                model_llm3=model_llm3,
                logger=logger,
            )
        )
        if final_corpus:
            return final_corpus

        logger.warning("阶段二尝试 #%s 未产生有效 final_corpus。", attempt)
        if current_limit is not None:
            current_limit = current_limit * 2

    _write_failure_report(
        project_dir=project_dir,
        selected_scopes=selected_scopes,
        target_themes=target_themes,
        attempts=attempt,
        max_fragments=current_limit,
    )
    raise RuntimeError(
        "阶段二失败：多次重试后 `2_final_corpus.yaml` 仍为空，已停止流程。"
        "详见 2_stage_failure_report.md。"
    )


__all__ = ["run_stage2_data_collection", "list_available_scopes"]
