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
            raise RuntimeError(f"提示词 `{spec.prompt_id}` 的 sections[{idx}] 不是对象")
        title = str(item.get("section") or "").strip()
        instruction = str(item.get("goal") or "").strip()
        if not title or not instruction:
            raise RuntimeError(
                f"提示词 `{spec.prompt_id}` 的 sections[{idx}] 缺少 section/goal"
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

    if output_path.exists() and not overwrite:
        themes = parse_target_themes_from_proposal(output_path)
        if themes:
            logger.info("阶段一已存在，复用: %s", output_path)
            return themes

    target_themes = _generate_target_themes(llm_client, llm_config, idea, logger)
    section_prompt_spec = load_prompt("stage1_section_writer")
    section_specs = _load_section_specs(section_prompt_spec)
    context = json.dumps(target_themes, ensure_ascii=False, indent=2)

    sections: list[str] = []
    for title, instruction in section_specs:
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
        content = response.content.strip()
        if not content:
            raise RuntimeError(f"阶段一失败：小节 {title} 返回空内容。")

        sections.append(f"## {title}\n\n{content}")

    proposal = (
        markdown_front_matter(target_themes)
        + "\n\n"
        + "\n\n".join(sections)
        + "\n"
    )

    write_text(output_path, proposal)
    write_json(meta_path, {"idea": idea, "target_themes": target_themes})
    logger.info("阶段一完成: %s", output_path)
    return target_themes
