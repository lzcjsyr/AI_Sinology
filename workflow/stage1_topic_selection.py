from __future__ import annotations

import json
from pathlib import Path

from core.config import LLMEndpointConfig
from core.llm_client import OpenAICompatClient
from core.prompt_loader import PromptSpec, build_messages, load_prompt
from core.utils import (
    markdown_front_matter,
    parse_json_from_text,
    parse_target_themes_from_proposal,
    read_text,
    write_json,
    write_text,
)


def _load_section_specs(spec: PromptSpec) -> list[tuple[str, str]]:
    raw_sections = spec.raw.get("section_plan")
    if not isinstance(raw_sections, list) or not raw_sections:
        raise RuntimeError(f"提示词 `{spec.prompt_id}` 缺少 section_plan 列表")

    sections: list[tuple[str, str]] = []
    for idx, item in enumerate(raw_sections, start=1):
        if not isinstance(item, dict):
            raise RuntimeError(f"提示词 `{spec.prompt_id}` 的 section_plan[{idx}] 不是对象")
        title = str(item.get("section_title") or "").strip()
        instruction = str(item.get("section_instruction") or "").strip()
        if not title or not instruction:
            raise RuntimeError(
                f"提示词 `{spec.prompt_id}` 的 section_plan[{idx}] 缺少 "
                "section_title/section_instruction"
            )
        sections.append((title, instruction))
    return sections


def _generate_target_themes(
    llm_client: OpenAICompatClient,
    llm_config: LLMEndpointConfig,
    idea: str,
    logger,
) -> list[dict[str, str]]:
    prompt_spec = load_prompt("stage1_target_themes")
    last_error: Exception | None = None
    for attempt in range(1, 4):
        messages = build_messages(prompt_spec, idea=idea)
        try:
            response = llm_client.chat(
                messages,
                temperature=0.2,
                **llm_config.as_client_kwargs(),
            )
            payload = parse_json_from_text(response.content)
            items = payload.get("target_themes")
            if not isinstance(items, list):
                raise ValueError("target_themes 不是数组")

            themes: list[dict[str, str]] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                theme = str(item.get("theme", "")).strip()
                desc = str(item.get("description", "")).strip()
                if theme:
                    themes.append({"theme": theme, "description": desc})
            if themes:
                return themes
            raise ValueError("target_themes 为空")
        except Exception as e:  # noqa: BLE001
            last_error = e
            logger.warning("生成 target_themes 失败，attempt=%s error=%s", attempt, e)

    raise RuntimeError(f"阶段一失败：无法生成有效 target_themes。last_error={last_error}")


def _parse_section_content(response_content: str, section_title: str) -> str:
    payload = parse_json_from_text(response_content)
    content = str(payload.get("section_content") or "").strip()
    if not content:
        raise ValueError(f"section_content 为空: {section_title}")
    return content


def _compose_proposal(target_themes: list[dict[str, str]], sections: list[str]) -> str:
    proposal = markdown_front_matter(target_themes)
    if sections:
        proposal += "\n\n" + "\n\n".join(sections)
    return proposal + "\n"


def _restore_completed_sections(
    output_path: Path,
    section_specs: list[tuple[str, str]],
) -> list[str]:
    if not output_path.exists():
        return []

    proposal = read_text(output_path)
    expected_titles = {title for title, _instruction in section_specs}
    section_blocks: dict[str, str] = {}
    current_title: str | None = None
    current_lines: list[str] = []

    for line in proposal.splitlines():
        heading = line.strip()
        matched_title: str | None = None
        if heading.startswith("## "):
            candidate = heading[3:].strip()
            if candidate in expected_titles:
                matched_title = candidate

        if matched_title is not None:
            if current_title and current_title not in section_blocks:
                block = "\n".join(current_lines).strip()
                if block:
                    section_blocks[current_title] = block
            current_title = matched_title
            current_lines = [line]
            continue

        if current_title is not None:
            current_lines.append(line)

    if current_title and current_title not in section_blocks:
        block = "\n".join(current_lines).strip()
        if block:
            section_blocks[current_title] = block

    completed: list[str] = []
    for title, _instruction in section_specs:
        block = section_blocks.get(title)
        if not block:
            break
        lines = block.splitlines()
        if len(lines) <= 1 or not "\n".join(lines[1:]).strip():
            break
        completed.append(block)
    return completed


def run_stage1_topic_selection(
    *,
    project_dir: Path,
    idea: str,
    llm_client: OpenAICompatClient,
    llm_config: LLMEndpointConfig,
    logger,
    overwrite: bool = False,
) -> list[dict[str, str]]:
    output_path = project_dir / "1_research_proposal.md"
    meta_path = project_dir / "1_research_proposal_meta.json"
    section_prompt_spec = load_prompt("stage1_section_writer")
    section_specs = _load_section_specs(section_prompt_spec)
    section_total = len(section_specs)
    target_themes: list[dict[str, str]] = []
    sections: list[str] = []

    if output_path.exists() and not overwrite:
        target_themes = parse_target_themes_from_proposal(output_path)
        if target_themes:
            sections = _restore_completed_sections(output_path, section_specs)
            if len(sections) == section_total:
                logger.info("阶段一已存在，复用: %s", output_path)
                return target_themes
            logger.info(
                "阶段一检测到部分草稿，将续写剩余小节: %s (completed=%s/%s)",
                output_path,
                len(sections),
                section_total,
            )
        else:
            logger.info("阶段一检测到旧文件但缺少 target_themes，将从头重写: %s", output_path)

    if not target_themes:
        target_themes = _generate_target_themes(llm_client, llm_config, idea, logger)
        sections = []

    context = json.dumps(target_themes, ensure_ascii=False, indent=2)
    write_text(output_path, _compose_proposal(target_themes, sections))

    for idx, (title, instruction) in enumerate(section_specs[len(sections) :], start=len(sections) + 1):
        logger.info("阶段一小节生成中: %s/%s %s", idx, section_total, title)
        messages = build_messages(
            section_prompt_spec,
            idea=idea,
            context=context,
            section_title=title,
            section_instruction=instruction,
        )

        response = llm_client.chat(
            messages,
            temperature=0.4,
            **llm_config.as_client_kwargs(),
        )
        try:
            content = _parse_section_content(response.content, title)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"阶段一失败：小节 {title} 返回格式错误。error={exc}") from exc

        sections.append(f"## {title}\n\n{content}")
        write_text(output_path, _compose_proposal(target_themes, sections))
        logger.info("阶段一小节已落盘: %s/%s %s -> %s", idx, section_total, title, output_path)

    write_json(
        meta_path,
        {
            "idea": idea,
            "target_themes": target_themes,
            "section_total": section_total,
            "completed_sections": len(sections),
        },
    )
    logger.info("阶段一完成: %s", output_path)
    return target_themes
