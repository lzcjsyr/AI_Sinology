from __future__ import annotations

import re
from difflib import SequenceMatcher
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


def _theme_key(theme: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", theme.lower())


def _themes_are_similar(left_key: str, right_key: str) -> bool:
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True

    shorter, longer = (
        (left_key, right_key) if len(left_key) <= len(right_key) else (right_key, left_key)
    )
    if len(shorter) >= 4 and shorter in longer:
        return True
    return SequenceMatcher(None, left_key, right_key).ratio() >= 0.82


def _coalesce_target_themes(raw_items: list[dict[str, str]], logger) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue

        theme = str(item.get("theme", "")).strip()
        desc = str(item.get("description", "")).strip()
        key = _theme_key(theme)
        if not theme or not key:
            continue

        duplicate: dict[str, str] | None = None
        for existing in merged:
            if _themes_are_similar(key, existing["__key"]):
                duplicate = existing
                break

        if duplicate is not None:
            if desc and desc not in duplicate["description"]:
                duplicate["description"] = (
                    f"{duplicate['description']}；{desc}"
                    if duplicate["description"]
                    else desc
                )
            logger.info("阶段一主题合并: `%s` -> `%s`", theme, duplicate["theme"])
            continue

        merged.append(
            {
                "theme": theme,
                "description": desc,
                "__key": key,
            }
        )

    if len(merged) > 3:
        logger.info("阶段一主题超过 3 个，按优先顺序截断为前 3 个。")

    return [
        {
            "theme": item["theme"],
            "description": item["description"],
        }
        for item in merged[:3]
    ]


def _generate_target_themes(
    llm_client: OpenAICompatClient,
    llm_config: LLMEndpointConfig,
    idea: str,
    proposal: str,
    logger,
) -> list[dict[str, str]]:
    prompt_spec = load_prompt("stage1_target_themes")
    last_error: Exception | None = None
    for attempt in range(1, 4):
        messages = build_messages(prompt_spec, idea=idea, proposal=proposal)
        try:
            response = llm_client.chat(
                messages,
                temperature=0.2,
                **llm_config.as_client_kwargs(),
            )
            payload = parse_json_from_text(response.content)
            raw_items = payload.get("target_themes")
            if not isinstance(raw_items, list):
                raise ValueError("target_themes 不是数组")

            themes = _coalesce_target_themes(raw_items, logger)
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


def _compose_proposal(
    sections: list[str],
    *,
    target_themes: list[dict[str, str]] | None = None,
) -> str:
    blocks: list[str] = []
    if target_themes:
        blocks.append(markdown_front_matter(target_themes))
    if sections:
        blocks.append("\n\n".join(sections))
    if not blocks:
        return ""
    return "\n\n".join(blocks) + "\n"


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


def _write_stage1_meta(
    *,
    meta_path: Path,
    idea: str,
    target_themes: list[dict[str, str]],
    section_total: int,
    completed_sections: int,
) -> None:
    write_json(
        meta_path,
        {
            "idea": idea,
            "target_themes": target_themes,
            "section_total": section_total,
            "completed_sections": completed_sections,
        },
    )


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
    sections: list[str] = []
    existing_themes_raw: list[dict[str, str]] = []
    existing_themes: list[dict[str, str]] = []

    if output_path.exists() and not overwrite:
        sections = _restore_completed_sections(output_path, section_specs)
        existing_themes_raw = parse_target_themes_from_proposal(output_path)
        existing_themes = _coalesce_target_themes(existing_themes_raw, logger)

        if len(sections) == section_total and existing_themes:
            if existing_themes != existing_themes_raw:
                write_text(output_path, _compose_proposal(sections, target_themes=existing_themes))
            _write_stage1_meta(
                meta_path=meta_path,
                idea=idea,
                target_themes=existing_themes,
                section_total=section_total,
                completed_sections=len(sections),
            )
            logger.info("阶段一已存在，复用: %s", output_path)
            return existing_themes

        if len(sections) == section_total:
            logger.info("阶段一正文已完成但缺少有效 target_themes，将补生成主题: %s", output_path)
        elif sections:
            logger.info(
                "阶段一检测到部分草稿，将续写剩余小节: %s (completed=%s/%s)",
                output_path,
                len(sections),
                section_total,
            )
        else:
            logger.info("阶段一将从头生成 proposal 草稿: %s", output_path)

    for idx, (title, instruction) in enumerate(section_specs[len(sections) :], start=len(sections) + 1):
        logger.info("阶段一小节生成中: %s/%s %s", idx, section_total, title)
        context = _compose_proposal(sections) or "（暂无已完成草稿）"
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
        write_text(output_path, _compose_proposal(sections))
        logger.info("阶段一小节已落盘: %s/%s %s -> %s", idx, section_total, title, output_path)

    proposal = _compose_proposal(sections)
    target_themes = _generate_target_themes(
        llm_client=llm_client,
        llm_config=llm_config,
        idea=idea,
        proposal=proposal,
        logger=logger,
    )
    write_text(output_path, _compose_proposal(sections, target_themes=target_themes))
    _write_stage1_meta(
        meta_path=meta_path,
        idea=idea,
        target_themes=target_themes,
        section_total=section_total,
        completed_sections=len(sections),
    )
    logger.info("阶段一完成: %s", output_path)
    return target_themes
