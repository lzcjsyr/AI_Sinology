from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from core import AppConfig, LiteLLMClient, StateManager
from core.cli_ui import CLIUI
from core.logger import setup_logger
from core.state_manager import STAGE_STATUS_COMPLETED, StageProgress
from core.utils import parse_idea_from_proposal, parse_target_themes_from_proposal
from workflow import (
    run_stage1_topic_selection,
    run_stage2_data_collection,
    run_stage3_outlining,
    run_stage4_drafting,
    run_stage5_polishing,
)
from workflow.stage2_data_collection import (
    ScopeOption,
    list_available_scope_dirs,
    list_available_scope_options,
    read_cached_scopes,
)

SCOPE_CODE_PATTERN = re.compile(r"KR[1-4][a-z](?:\d+)?", re.IGNORECASE)
SCOPE_FAMILY_PATTERN = re.compile(r"^(KR[1-4][a-z])", re.IGNORECASE)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多智能体中文学术写作系统 (CLI)")
    parser.add_argument("--new-project", help="创建新项目名")
    parser.add_argument("--continue-project", help="继续已有项目名")
    parser.add_argument("--idea", help="研究意向（阶段一输入）")
    parser.add_argument("--scopes", help="阶段二语料范围，逗号分隔，如 KR1a,KR3j（KR-Catalog 二级类目）")
    parser.add_argument("--scope-dirs", help="阶段二额外目录，逗号分隔，如 KR1a0001,KR3j0160")
    parser.add_argument("--max-fragments", type=int, help="阶段二最多处理的切片数（调试用）")
    parser.add_argument(
        "--stage2-llm1-concurrency",
        type=int,
        help="阶段二 llm1 并发请求数（不传则自动计算）",
    )
    parser.add_argument(
        "--stage2-llm2-concurrency",
        type=int,
        help="阶段二 llm2 并发请求数（不传则自动计算）",
    )
    parser.add_argument(
        "--stage2-arbitration-concurrency",
        type=int,
        help="阶段二争议仲裁并发请求数（不传则自动计算）",
    )
    parser.add_argument(
        "--stage2-sync-headroom",
        type=float,
        help="阶段二同速限额头寸比例（0.01-1.0）",
    )
    parser.add_argument(
        "--stage2-sync-max-ahead",
        type=int,
        help="阶段二双模型最大允许进度差（条）",
    )
    parser.add_argument(
        "--stage2-fragment-max-attempts",
        type=int,
        help="阶段二每条碎片的最大重试次数",
    )
    parser.add_argument(
        "--stage2-screening-batch-max-chars",
        type=int,
        help="阶段二粗筛批次的最大字符数",
    )
    parser.add_argument(
        "--stage2-max-empty-retries",
        type=int,
        help="阶段二最终语料为空时的重跑次数",
    )
    parser.add_argument("--start-stage", type=int, choices=[1, 2, 3, 4, 5], help="从指定阶段开始")
    parser.add_argument("--end-stage", type=int, choices=[1, 2, 3, 4, 5], default=5, help="执行到指定阶段")
    parser.add_argument("--yes", action="store_true", help="自动确认交互提示")
    parser.add_argument(
        "--overwrite-stage1",
        action="store_true",
        help="阶段一已存在时是否覆盖重生成",
    )
    return parser.parse_args()


def _ask_choice(prompt: str, valid: set[str], ui: CLIUI) -> str:
    while True:
        value = ui.prompt(prompt)
        if value in valid:
            return value
        ui.warning(f"请输入有效选项: {sorted(valid)}")


def _choose_project_interactive(state_manager: StateManager, ui: CLIUI) -> tuple[str, bool]:
    ui.menu(
        "请选择模式",
        [
            ("1", "创建新研究项目"),
            ("2", "继续现有项目"),
        ],
    )
    choice = _ask_choice("输入选项 [1/2]:", {"1", "2"}, ui)
    if choice == "1":
        name = ui.prompt("输入新项目名称:")
        return name, True

    projects = state_manager.list_projects()
    if not projects:
        ui.warning("当前没有可继续的项目，请先创建新项目。")
        name = ui.prompt("输入新项目名称:")
        return name, True

    ui.menu("可继续项目", [(str(idx), proj) for idx, proj in enumerate(projects, start=1)])

    while True:
        raw = ui.prompt("输入项目编号或项目名:")
        if raw.isdigit():
            num = int(raw)
            if 1 <= num <= len(projects):
                return projects[num - 1], False
        if raw in projects:
            return raw, False
        ui.warning("输入无效，请重试。")


def _normalize_scope_code(raw_scope: str) -> str:
    token = raw_scope.strip()
    if not token:
        return ""

    match = SCOPE_CODE_PATTERN.search(token)
    if match:
        token = match.group(0)

    if token.lower().startswith("kr") and len(token) > 2:
        return f"KR{token[2:].lower()}"
    return token


def _split_scope_tokens(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    normalized = raw_value.replace("，", ",")
    return [token for token in normalized.split(",")]


def _parse_scope_inputs(raw_value: str | None) -> list[str]:
    scopes: list[str] = []
    for raw in _split_scope_tokens(raw_value):
        scope = _normalize_scope_code(raw)
        if scope and scope not in scopes:
            scopes.append(scope)
    return scopes


def _is_back_navigation_input(raw_value: str) -> bool:
    normalized = raw_value.strip()
    lowered = normalized.lower()
    return lowered in {"b", "back", "return"} or normalized in {"返回", "上一步"}


def _choose_scope_codes_interactive(scope_options: list[ScopeOption], ui: CLIUI) -> list[str] | None:
    if not scope_options:
        return []

    available_scopes = {option.code for option in scope_options}

    ui.section("请选择 Kanripo 检索范围（可输入多个，用逗号分隔）")
    ui.info("可选条目（来自 KR-Catalog/KR/KR1~KR4.txt）。按 ESC / Ctrl+C 或输入 b 可返回上一层。")
    while True:
        picked = ui.multi_select_with_start(
            title="范围多选：方向键移动，Enter 选中/取消，Tab 切到开始按钮",
            options=[(option.code, option.display_label) for option in scope_options],
            start_label="开始",
            cancel_label="取消",
        )
        unavailable_reason = ui.last_multiselect_unavailable_reason()
        if picked is None:
            if unavailable_reason is None:
                return None

            ui.info(f"当前终端暂不能使用交互多选（{unavailable_reason}），自动切换为手动输入模式。")
            ui.list_items([option.display_label for option in scope_options])
            while True:
                raw = ui.prompt("输入 scope 列表，例如 KR1a,KR3j（输入 b 返回）:")
                if _is_back_navigation_input(raw):
                    return None
                if not raw:
                    ui.warning("至少输入一个 scope。")
                    continue
                scopes = _parse_scope_inputs(raw)
                invalid = [s for s in scopes if s not in available_scopes]
                if invalid:
                    ui.warning(f"以下 scope 不存在: {invalid}")
                    continue
                return scopes

        if picked:
            return picked

        ui.warning("至少选择一个 scope。")


def _merge_unique(*groups: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for group in groups:
        for item in group:
            if item in seen:
                continue
            seen.add(item)
            merged.append(item)
    return merged


def _scope_parent_code(scope: str, available_scope_codes: set[str]) -> str | None:
    match = SCOPE_FAMILY_PATTERN.match(scope)
    if not match:
        return None
    parent = _normalize_scope_code(match.group(1))
    if parent == scope or parent not in available_scope_codes:
        return None
    return parent


def _collapse_scope_inputs(scope_inputs: list[str], available_scope_codes: list[str]) -> list[str]:
    scope_code_set = set(available_scope_codes)
    collapsed: list[str] = []

    for scope in scope_inputs:
        if scope in scope_code_set:
            collapsed = [item for item in collapsed if _scope_parent_code(item, scope_code_set) != scope]
            if scope not in collapsed:
                collapsed.append(scope)
            continue

        parent = _scope_parent_code(scope, scope_code_set)
        if parent and parent in collapsed:
            continue

        if scope not in collapsed:
            collapsed.append(scope)

    return collapsed


def _find_missing_scope_inputs(
    scope_inputs: list[str],
    *,
    available_scope_codes: list[str],
    available_scope_dirs: list[str],
) -> tuple[list[str], list[str]]:
    scope_code_set = set(available_scope_codes)
    scope_dir_set = set(available_scope_dirs)
    missing_scopes: list[str] = []
    missing_dirs: list[str] = []

    for scope in scope_inputs:
        if scope in scope_code_set or scope in scope_dir_set:
            continue
        target = missing_dirs if len(scope) > 4 else missing_scopes
        if scope not in target:
            target.append(scope)

    return missing_scopes, missing_dirs


def _format_missing_scope_inputs(missing_scopes: list[str], missing_dirs: list[str]) -> str:
    messages: list[str] = []
    if missing_scopes:
        messages.append(f"以下 scope 不存在: {missing_scopes}")
    if missing_dirs:
        messages.append(f"以下目录不存在: {missing_dirs}")
    return "；".join(messages)


def _resolve_scope_inputs(
    scope_inputs: list[str],
    *,
    available_scope_codes: list[str],
    available_scope_dirs: list[str],
) -> tuple[list[str], str | None]:
    missing_scopes, missing_dirs = _find_missing_scope_inputs(
        scope_inputs,
        available_scope_codes=available_scope_codes,
        available_scope_dirs=available_scope_dirs,
    )
    if missing_scopes or missing_dirs:
        return [], _format_missing_scope_inputs(missing_scopes, missing_dirs)
    return _collapse_scope_inputs(scope_inputs, available_scope_codes), None


def _choose_scope_input_mode(has_scope_options: bool, ui: CLIUI) -> str:
    if not has_scope_options:
        ui.info("未找到 KR-Catalog 二级类目，仅支持手动输入范围。")
        return "2"

    ui.menu(
        "请选择 Kanripo 检索方式",
        [
            ("1", "按 KR1a 级别勾选范围"),
            ("2", "手动输入范围（支持 KR1a 与 KR2e0020 混输）"),
        ],
    )
    return _ask_choice("输入选项 [1/2]:", {"1", "2"}, ui)


def _choose_manual_scope_inputs_interactive(
    scope_options: list[ScopeOption],
    *,
    available_scope_codes: list[str],
    available_scope_dirs: list[str],
    allow_back: bool,
    ui: CLIUI,
) -> list[str] | None:
    ui.section("手动输入 Kanripo 检索范围")
    ui.info("支持输入 KR1a、KR2e0020 等范围，多个值可用逗号分隔。若同时输入父级与子级，自动合并为父级。")
    if allow_back:
        ui.info("输入 b 可返回上一层并切换检索方式。")
    if scope_options:
        ui.info("KR-Catalog 二级类目参考如下：")
        ui.list_items([option.display_label for option in scope_options])

    while True:
        prompt_message = "输入范围列表，例如 KR1a,KR2e0020:"
        if allow_back:
            prompt_message = "输入范围列表，例如 KR1a,KR2e0020（输入 b 返回）:"
        raw = ui.prompt(prompt_message)
        if allow_back and _is_back_navigation_input(raw):
            return None
        if not raw:
            ui.warning("至少输入一个范围。")
            continue

        scope_inputs = _parse_scope_inputs(raw)
        finalized, error_message = _resolve_scope_inputs(
            scope_inputs,
            available_scope_codes=available_scope_codes,
            available_scope_dirs=available_scope_dirs,
        )
        if error_message:
            ui.warning(error_message)
            continue
        return finalized


def _choose_stage2_scope_inputs(
    *,
    args: argparse.Namespace,
    config: AppConfig,
    project_dir: Path,
    ui: CLIUI,
    logger,
) -> list[str]:
    scope_options = list_available_scope_options(config.kanripo_dir)
    available_scope_codes = [option.code for option in scope_options]
    available_scope_dirs = list_available_scope_dirs(config.kanripo_dir)
    if not available_scope_codes and not available_scope_dirs:
        raise RuntimeError(f"未找到 Kanripo 数据目录: {config.kanripo_dir}")

    valid_scope_inputs = sorted(set(_merge_unique(available_scope_codes, available_scope_dirs)))
    scope_display_map = {option.code: option.display_label for option in scope_options}
    raw_scope_inputs = _merge_unique(
        _parse_scope_inputs(args.scopes),
        _parse_scope_inputs(args.scope_dirs),
    )
    scope_inputs, scope_input_error = _resolve_scope_inputs(
        raw_scope_inputs,
        available_scope_codes=available_scope_codes,
        available_scope_dirs=available_scope_dirs,
    )
    if scope_input_error:
        raise RuntimeError(scope_input_error)

    if not scope_inputs:
        cached_scope_inputs = read_cached_scopes(project_dir, valid_scope_inputs)
        scope_inputs, scope_input_error = _resolve_scope_inputs(
            cached_scope_inputs,
            available_scope_codes=available_scope_codes,
            available_scope_dirs=available_scope_dirs,
        )
        if scope_input_error:
            scope_inputs = []

    if not scope_inputs and args.yes:
        scope_inputs = [available_scope_codes[0] if available_scope_codes else available_scope_dirs[0]]
        logger.info(
            "--yes 模式下自动选择 scope: %s (%s)",
            scope_inputs[0],
            scope_display_map.get(scope_inputs[0], scope_inputs[0]),
        )

    while not scope_inputs:
        input_mode = _choose_scope_input_mode(bool(scope_options), ui)
        if input_mode == "1":
            chosen_scopes = _choose_scope_codes_interactive(scope_options, ui)
            if chosen_scopes is None:
                continue
            scope_inputs = _collapse_scope_inputs(chosen_scopes, available_scope_codes)
            continue

        manual_scope_inputs = _choose_manual_scope_inputs_interactive(
            scope_options,
            available_scope_codes=available_scope_codes,
            available_scope_dirs=available_scope_dirs,
            allow_back=bool(scope_options),
            ui=ui,
        )
        if manual_scope_inputs is None:
            continue
        scope_inputs = manual_scope_inputs

    for scope in scope_inputs:
        display = scope_display_map.get(scope, f"自定义目录 [{scope}]")
        logger.info("阶段二检索范围: %s -> %s", scope, display)
    return scope_inputs


def _confirm(message: str, auto_yes: bool, ui: CLIUI) -> bool:
    if auto_yes:
        return True
    answer = ui.prompt(f"{message} [y/N]:").lower()
    return answer in {"y", "yes"}


@dataclass(frozen=True)
class _Stage2Runtime:
    max_fragments: int | None
    llm1_concurrency: int | None
    llm2_concurrency: int | None
    arbitration_concurrency: int | None
    sync_headroom: float
    sync_max_ahead: int
    fragment_max_attempts: int
    max_empty_retries: int
    screening_batch_max_chars: int


def _load_saved_idea(project_dir: Path) -> str:
    proposal_path = project_dir / "1_research_proposal.md"
    if not proposal_path.exists():
        return ""
    try:
        return parse_idea_from_proposal(proposal_path)
    except Exception:  # noqa: BLE001
        return ""


def _resolve_stage2_runtime(args: argparse.Namespace, config: AppConfig) -> _Stage2Runtime:
    llm1_concurrency = (
        args.stage2_llm1_concurrency
        if args.stage2_llm1_concurrency is not None
        else config.stage2_llm1_concurrency
    )
    llm2_concurrency = (
        args.stage2_llm2_concurrency
        if args.stage2_llm2_concurrency is not None
        else config.stage2_llm2_concurrency
    )
    arbitration_concurrency = (
        args.stage2_arbitration_concurrency
        if args.stage2_arbitration_concurrency is not None
        else config.stage2_arbitration_concurrency
    )
    sync_headroom = (
        args.stage2_sync_headroom
        if args.stage2_sync_headroom is not None
        else config.stage2_sync_headroom
    )
    sync_max_ahead = (
        args.stage2_sync_max_ahead
        if args.stage2_sync_max_ahead is not None
        else config.stage2_sync_max_ahead
    )
    fragment_max_attempts = (
        args.stage2_fragment_max_attempts
        if args.stage2_fragment_max_attempts is not None
        else config.stage2_fragment_max_attempts
    )
    max_empty_retries = (
        args.stage2_max_empty_retries
        if args.stage2_max_empty_retries is not None
        else config.stage2_max_empty_retries
    )
    screening_batch_max_chars = (
        args.stage2_screening_batch_max_chars
        if args.stage2_screening_batch_max_chars is not None
        else config.stage2_screening_batch_max_chars
    )

    if llm1_concurrency is not None and llm1_concurrency < 1:
        raise RuntimeError("阶段二 llm1 并发参数必须 >= 1。")
    if llm2_concurrency is not None and llm2_concurrency < 1:
        raise RuntimeError("阶段二 llm2 并发参数必须 >= 1。")
    if arbitration_concurrency is not None and arbitration_concurrency < 1:
        raise RuntimeError("阶段二仲裁并发参数必须 >= 1。")
    if not 0.01 <= float(sync_headroom) <= 1.0:
        raise RuntimeError("阶段二同速 headroom 必须在 [0.01, 1.0]。")
    if sync_max_ahead < 0:
        raise RuntimeError("阶段二同速 max_ahead 必须 >= 0。")
    if screening_batch_max_chars < 1:
        raise RuntimeError("阶段二 screening batch 最大字符数必须 >= 1。")

    return _Stage2Runtime(
        max_fragments=(
            args.max_fragments
            if args.max_fragments is not None
            else config.default_max_fragments
        ),
        llm1_concurrency=llm1_concurrency,
        llm2_concurrency=llm2_concurrency,
        arbitration_concurrency=arbitration_concurrency,
        sync_headroom=sync_headroom,
        sync_max_ahead=sync_max_ahead,
        fragment_max_attempts=fragment_max_attempts,
        max_empty_retries=max_empty_retries,
        screening_batch_max_chars=screening_batch_max_chars,
    )


def _show_project_progress(project_name: str, stage_progress: list[StageProgress], ui: CLIUI) -> None:
    ui.section(f"项目进度 | {project_name}")
    for item in stage_progress:
        ui.aligned_pair(
            f"[{item.stage_index}] {item.stage_name}",
            item.status_display,
            right_tone="green" if item.status == STAGE_STATUS_COMPLETED else None,
        )
    print()


def _choose_existing_project_execution(
    *,
    stage_progress: list[StageProgress],
    suggested_stage: int,
    default_end_stage: int,
    auto_yes: bool,
    ui: CLIUI,
) -> tuple[int, int]:
    fallback_start = 5 if suggested_stage == 6 else suggested_stage
    fallback_end = max(fallback_start, default_end_stage)

    if auto_yes:
        return fallback_start, fallback_end

    max_selectable_stage = 5 if suggested_stage == 6 else suggested_stage
    valid_stages = {
        str(item.stage_index)
        for item in stage_progress
        if item.stage_index <= max_selectable_stage
    }

    if suggested_stage == 6:
        ui.info("当前项目已全部完成。请选择任一阶段序号，从该阶段重新执行。")
    else:
        ui.info(
            f"直接输入阶段序号即可：已完成阶段会重跑，当前阶段（{suggested_stage}）会继续执行，后续阶段暂不可选。"
        )

    stage = int(
        _ask_choice(
            f"输入要执行的阶段序号 [1-{max_selectable_stage}]:",
            valid_stages,
            ui,
        )
    )
    return stage, max(stage, default_end_stage)


def _refresh_project_progress(
    project_name: str,
    state_manager: StateManager,
    ui: CLIUI,
) -> list[StageProgress]:
    stage_progress = state_manager.infer_stage_progress(project_name)
    _show_project_progress(project_name, stage_progress, ui)
    return stage_progress


def _handle_stage_completion(
    *,
    stage: int,
    end_stage: int,
    project_name: str,
    state_manager: StateManager,
    auto_yes: bool,
    ui: CLIUI,
) -> bool:
    _refresh_project_progress(project_name, state_manager, ui)
    if stage >= end_stage:
        return True

    next_stage = stage + 1
    if _confirm(f"请检查阶段 {stage} 的输出成果。确认无误后进入阶段 {next_stage} 吗？", auto_yes, ui):
        return True

    ui.info(f"已暂停在阶段 {stage}。确认成果后，可继续该项目并从阶段 {next_stage} 开始。")
    return False


def _run_stage_sequence(
    start_stage: int,
    end_stage: int,
    *,
    run_stage: Callable[[int], None],
    on_stage_completed: Callable[[int], bool],
) -> bool:
    for stage in range(start_stage, end_stage + 1):
        run_stage(stage)
        if not on_stage_completed(stage):
            return False
    return True


def main() -> int:
    args = _parse_args()
    ui = CLIUI()
    ui.header("AI 汉学论文流水线", "更清晰的 CLI 交互已启用")

    root_dir = Path(__file__).resolve().parent
    config = AppConfig.load(root_dir)
    state_manager = StateManager(config.outputs_dir)
    logger = setup_logger(config.outputs_dir / "system.log")

    if args.new_project and args.continue_project:
        ui.error("--new-project 与 --continue-project 不能同时使用")
        return 1

    try:
        if args.new_project:
            project_name = args.new_project
            is_new = True
        elif args.continue_project:
            project_name = args.continue_project
            is_new = False
        else:
            project_name, is_new = _choose_project_interactive(state_manager, ui)

        stage_progress: list[StageProgress] = []
        rerun_requested = False
        if is_new:
            state = state_manager.create_project(project_name)
            if not args.idea:
                args.idea = ui.prompt("请输入研究意向:")
            if not args.idea:
                ui.error("研究意向不能为空。")
                return 1
            start_stage = args.start_stage or 1
            end_stage = args.end_stage
        else:
            state = state_manager.infer_state(project_name)
            stage_progress = state_manager.infer_stage_progress(project_name)
            _show_project_progress(project_name, stage_progress, ui)

            if args.start_stage:
                start_stage = args.start_stage
                end_stage = args.end_stage
            else:
                start_stage, end_stage = _choose_existing_project_execution(
                    stage_progress=stage_progress,
                    suggested_stage=state.next_stage,
                    default_end_stage=args.end_stage,
                    auto_yes=args.yes,
                    ui=ui,
                )

            highest_completed = state_manager.highest_completed_stage(stage_progress)
            rerun_requested = start_stage <= highest_completed

        if start_stage > end_stage:
            ui.error(f"start-stage({start_stage}) 不能大于 end-stage({end_stage})")
            return 1

        project_dir = state.project_dir
        if not is_new and rerun_requested:
            if start_stage == 1 and not args.idea:
                args.idea = _load_saved_idea(project_dir)

            stale_artifacts = state_manager.collect_artifacts_from_stage(project_dir, start_stage)
            if stale_artifacts:
                preview_items = [str(path.relative_to(project_dir)) for path in stale_artifacts[:8]]
                if len(stale_artifacts) > 8:
                    preview_items.append(f"... 其余 {len(stale_artifacts) - 8} 个文件")
                ui.warning(
                    f"将从阶段 {start_stage} 重跑，并清理该阶段及后续产物（共 {len(stale_artifacts)} 个文件）。"
                )
                ui.list_items(preview_items)
                if not _confirm("是否继续清理并重跑？", args.yes, ui):
                    return 0
                removed = state_manager.clear_artifacts_from_stage(project_dir, start_stage)
                logger.info(
                    "用户请求重跑阶段 %s，已清理旧文件 %s 个",
                    start_stage,
                    len(removed),
                )
                ui.info(f"已清理旧产物 {len(removed)} 个，将重新执行阶段 {start_stage} 及其后续阶段。")

        ui.section("执行配置")
        ui.key_value("项目目录", str(project_dir))
        ui.key_value("执行阶段", f"{start_stage} -> {end_stage}")
        print()
        logger.info("当前项目: %s", project_dir)
        logger.info("执行阶段范围: %s -> %s", start_stage, end_stage)

        config.validate_api()
        llm_client = LiteLLMClient(config, logger)

        if start_stage > 1 and not (project_dir / "1_research_proposal.md").exists():
            ui.error("缺少 1_research_proposal.md，无法从当前阶段继续。")
            return 1

        stage2_runtime = _resolve_stage2_runtime(args, config)
        overwrite_stage1 = args.overwrite_stage1 or (not is_new and rerun_requested)

        def _run_stage(stage: int) -> None:
            stage_name = state_manager.stage_name(stage)
            ui.section(f"阶段 {stage} | {stage_name}")
            logger.info("开始执行第 %s 阶段", stage)

            if stage == 1:
                idea = args.idea
                if not idea:
                    idea = _load_saved_idea(project_dir)
                if not idea:
                    idea = ui.prompt("请输入研究意向:")
                if not idea:
                    raise RuntimeError("阶段一需要研究意向。")

                run_stage1_topic_selection(
                    project_dir=project_dir,
                    idea=idea,
                    llm_client=llm_client,
                    llm_config=config.stage1_llm,
                    logger=logger,
                    overwrite=overwrite_stage1,
                )

            elif stage == 2:
                themes = parse_target_themes_from_proposal(project_dir / "1_research_proposal.md")
                if not themes:
                    raise RuntimeError("阶段二无法继续：未从阶段一提取到 target_themes")
                scope_inputs = _choose_stage2_scope_inputs(
                    args=args,
                    config=config,
                    project_dir=project_dir,
                    ui=ui,
                    logger=logger,
                )

                run_stage2_data_collection(
                    project_dir=project_dir,
                    kanripo_dir=config.kanripo_dir,
                    selected_scopes=scope_inputs,
                    target_themes=themes,
                    llm_client=llm_client,
                    llm1_endpoint=config.stage2_llm1,
                    llm2_endpoint=config.stage2_llm2,
                    llm3_endpoint=config.stage2_llm3,
                    logger=logger,
                    max_fragments=stage2_runtime.max_fragments,
                    max_empty_retries=stage2_runtime.max_empty_retries,
                    llm1_concurrency=stage2_runtime.llm1_concurrency,
                    llm2_concurrency=stage2_runtime.llm2_concurrency,
                    arbitration_concurrency=stage2_runtime.arbitration_concurrency,
                    sync_headroom=stage2_runtime.sync_headroom,
                    sync_max_ahead=stage2_runtime.sync_max_ahead,
                    sync_mode=config.stage2_sync_mode,
                    fragment_max_attempts=stage2_runtime.fragment_max_attempts,
                    retry_backoff_seconds=config.retry_backoff_seconds,
                    screening_batch_max_chars=stage2_runtime.screening_batch_max_chars,
                )

            elif stage == 3:
                run_stage3_outlining(
                    project_dir=project_dir,
                    llm_client=llm_client,
                    llm_config=config.stage3_llm,
                    logger=logger,
                )

            elif stage == 4:
                run_stage4_drafting(
                    project_dir=project_dir,
                    llm_client=llm_client,
                    llm_config=config.stage4_llm,
                    logger=logger,
                )

            elif stage == 5:
                run_stage5_polishing(
                    project_dir=project_dir,
                    llm_client=llm_client,
                    llm_config=config.stage5_llm,
                    logger=logger,
                )

        completed_requested_range = _run_stage_sequence(
            start_stage,
            end_stage,
            run_stage=_run_stage,
            on_stage_completed=lambda stage: _handle_stage_completion(
                stage=stage,
                end_stage=end_stage,
                project_name=project_name,
                state_manager=state_manager,
                auto_yes=args.yes,
                ui=ui,
            ),
        )
        if not completed_requested_range:
            ui.info(f"已暂停。项目目录: {project_dir}")
            return 0

        ui.success(f"流程执行完成。项目目录: {project_dir}")
        return 0

    except KeyboardInterrupt:
        ui.warning("检测到中断。当前进度已写入文件，可直接继续项目。")
        return 130
    except Exception as e:  # noqa: BLE001
        logger.exception("执行失败: %s", e)
        ui.error(f"执行失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
