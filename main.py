from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core import AppConfig, LiteLLMClient, StateManager
from core.logger import setup_logger
from core.utils import parse_target_themes_from_proposal, read_json, write_json
from workflow import (
    run_stage1_topic_selection,
    run_stage2_data_collection,
    run_stage3_outlining,
    run_stage4_drafting,
    run_stage5_polishing,
)
from workflow.stage2_data_collection import list_available_scopes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多智能体中文学术写作系统 (CLI)")
    parser.add_argument("--new-project", help="创建新项目名")
    parser.add_argument("--continue-project", help="继续已有项目名")
    parser.add_argument("--idea", help="研究意向（阶段一输入）")
    parser.add_argument("--scopes", help="阶段二语料范围，逗号分隔，如 KR3j0160,KR1a0001")
    parser.add_argument("--max-fragments", type=int, help="阶段二最多处理的切片数（调试用）")
    parser.add_argument(
        "--stage2-concurrency",
        type=int,
        help="阶段二每个模型的并发请求数",
    )
    parser.add_argument(
        "--stage2-fragment-max-attempts",
        type=int,
        help="阶段二每条碎片的最大重试次数",
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


def _ask_choice(prompt: str, valid: set[str]) -> str:
    while True:
        value = input(prompt).strip()
        if value in valid:
            return value
        print(f"请输入有效选项: {sorted(valid)}")


def _choose_project_interactive(state_manager: StateManager) -> tuple[str, bool]:
    print("\n请选择模式:")
    print("[1] 创建新研究项目")
    print("[2] 继续现有项目")
    choice = _ask_choice("输入选项 [1/2]: ", {"1", "2"})
    if choice == "1":
        name = input("输入新项目名称: ").strip()
        return name, True

    projects = state_manager.list_projects()
    if not projects:
        print("当前没有可继续的项目，请先创建新项目。")
        name = input("输入新项目名称: ").strip()
        return name, True

    print("\n可继续项目:")
    for idx, proj in enumerate(projects, start=1):
        print(f"[{idx}] {proj}")

    while True:
        raw = input("输入项目编号或项目名: ").strip()
        if raw.isdigit():
            num = int(raw)
            if 1 <= num <= len(projects):
                return projects[num - 1], False
        if raw in projects:
            return raw, False
        print("输入无效，请重试。")


def _choose_scopes_interactive(available_scopes: list[str]) -> list[str]:
    print("\n请选择 Kanripo 检索范围（可输入多个，用逗号分隔）")
    preview = available_scopes[:30]
    print("可选示例（前30个）：")
    for name in preview:
        print(f"- {name}")

    while True:
        raw = input("输入 scope 列表，例如 KR3j0160,KR1a0001: ").strip()
        if not raw:
            print("至少输入一个 scope。")
            continue
        scopes = [s.strip() for s in raw.split(",") if s.strip()]
        invalid = [s for s in scopes if s not in available_scopes]
        if invalid:
            print(f"以下 scope 不存在: {invalid}")
            continue
        return scopes


def _parse_scopes_arg(scopes_arg: str | None) -> list[str]:
    if not scopes_arg:
        return []
    return [s.strip() for s in scopes_arg.split(",") if s.strip()]


def _confirm(message: str, auto_yes: bool) -> bool:
    if auto_yes:
        return True
    answer = input(f"{message} [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def main() -> int:
    args = _parse_args()

    root_dir = Path(__file__).resolve().parent
    config = AppConfig.load(root_dir)
    state_manager = StateManager(config.outputs_dir)
    logger = setup_logger(config.outputs_dir / "system.log")

    if args.new_project and args.continue_project:
        print("--new-project 与 --continue-project 不能同时使用")
        return 1

    try:
        if args.new_project:
            project_name = args.new_project
            is_new = True
        elif args.continue_project:
            project_name = args.continue_project
            is_new = False
        else:
            project_name, is_new = _choose_project_interactive(state_manager)

        if is_new:
            state = state_manager.create_project(project_name)
            if not args.idea:
                args.idea = input("请输入研究意向: ").strip()
            if not args.idea:
                print("研究意向不能为空。")
                return 1
            start_stage = args.start_stage or 1
        else:
            state = state_manager.infer_state(project_name)
            if state.next_stage == 6 and not args.start_stage:
                print(f"项目 {project_name} 已完成全部阶段。")
                if not _confirm("是否从阶段5重新执行润色？", args.yes):
                    return 0
                start_stage = 5
            else:
                start_stage = args.start_stage or state.next_stage

        end_stage = args.end_stage
        if start_stage > end_stage:
            print(f"start-stage({start_stage}) 不能大于 end-stage({end_stage})")
            return 1

        project_dir = state.project_dir
        logger.info("当前项目: %s", project_dir)
        logger.info("执行阶段范围: %s -> %s", start_stage, end_stage)

        config.validate_api()
        llm_client = LiteLLMClient(config, logger)

        if start_stage > 1 and not (project_dir / "1_research_proposal.md").exists():
            print("缺少 1_research_proposal.md，无法从当前阶段继续。")
            return 1

        max_fragments = (
            args.max_fragments
            if args.max_fragments is not None
            else config.default_max_fragments
        )
        stage2_concurrency = (
            args.stage2_concurrency
            if args.stage2_concurrency is not None
            else config.stage2_screening_concurrency
        )
        stage2_fragment_max_attempts = (
            args.stage2_fragment_max_attempts
            if args.stage2_fragment_max_attempts is not None
            else config.stage2_fragment_max_attempts
        )
        stage2_max_empty_retries = (
            args.stage2_max_empty_retries
            if args.stage2_max_empty_retries is not None
            else config.stage2_max_empty_retries
        )

        for stage in range(start_stage, end_stage + 1):
            logger.info("开始执行第 %s 阶段", stage)

            if stage == 1:
                idea = args.idea
                if not idea and (project_dir / "1_research_proposal_meta.json").exists():
                    try:
                        idea = str(read_json(project_dir / "1_research_proposal_meta.json").get("idea") or "")
                    except Exception:  # noqa: BLE001
                        idea = ""
                if not idea:
                    idea = input("请输入研究意向: ").strip()
                if not idea:
                    raise RuntimeError("阶段一需要研究意向。")

                run_stage1_topic_selection(
                    project_dir=project_dir,
                    idea=idea,
                    llm_client=llm_client,
                    llm_config=config.stage1_llm,
                    logger=logger,
                    overwrite=args.overwrite_stage1,
                )

            elif stage == 2:
                themes = parse_target_themes_from_proposal(project_dir / "1_research_proposal.md")
                if not themes:
                    raise RuntimeError("阶段二无法继续：未从阶段一提取到 target_themes")

                available_scopes = list_available_scopes(config.kanripo_dir)
                if not available_scopes:
                    raise RuntimeError(f"未找到 Kanripo 数据目录: {config.kanripo_dir}")

                scopes = _parse_scopes_arg(args.scopes)
                scope_record_path = project_dir / "2_scope_selection.json"

                if not scopes and scope_record_path.exists():
                    try:
                        cached = read_json(scope_record_path)
                        cached_scopes = cached.get("scopes")
                        if isinstance(cached_scopes, list):
                            scopes = [s for s in cached_scopes if s in available_scopes]
                    except Exception:  # noqa: BLE001
                        scopes = []

                if not scopes:
                    if args.yes:
                        scopes = [available_scopes[0]]
                        logger.info("--yes 模式下自动选择 scope: %s", scopes)
                    else:
                        scopes = _choose_scopes_interactive(available_scopes)

                invalid_scopes = [s for s in scopes if s not in available_scopes]
                if invalid_scopes:
                    raise RuntimeError(f"存在非法 scope: {invalid_scopes}")

                write_json(scope_record_path, {"scopes": scopes})

                run_stage2_data_collection(
                    project_dir=project_dir,
                    kanripo_dir=config.kanripo_dir,
                    selected_scopes=scopes,
                    target_themes=themes,
                    llm_client=llm_client,
                    llm1_endpoint=config.stage2_llm1,
                    llm2_endpoint=config.stage2_llm2,
                    llm3_endpoint=config.stage2_llm3,
                    logger=logger,
                    max_fragments=max_fragments,
                    max_empty_retries=stage2_max_empty_retries,
                    screening_concurrency=stage2_concurrency,
                    fragment_max_attempts=stage2_fragment_max_attempts,
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

        print(f"\n流程执行完成。项目目录: {project_dir}")
        return 0

    except KeyboardInterrupt:
        print("\n检测到中断。当前进度已写入文件，可直接继续项目。")
        return 130
    except Exception as e:  # noqa: BLE001
        logger.exception("执行失败: %s", e)
        print(f"执行失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
