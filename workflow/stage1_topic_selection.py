from __future__ import annotations

from pathlib import Path

from core.config import LLMEndpointConfig
from core.llm_client import OpenAICompatClient
from core.utils import (
    markdown_front_matter,
    parse_json_from_text,
    parse_target_themes_from_proposal,
    write_json,
    write_text,
)


SECTION_SPECS = [
    ("研究背景与问题陈述", "请写约 500 字，交代研究背景、学术语境和问题边界。"),
    ("核心研究问题", "请提出 3-5 个层层递进的研究问题，并给出简短解释。"),
    ("学术史述评", "请写约 1200-1500 字，梳理相关研究史并指出空白。"),
    ("研究思路与切入点", "请写约 800 字，说明方法、材料处理和章节思路。"),
    ("预期创新与学术价值", "请写约 500 字，明确创新点和理论/史料价值。"),
    ("史料检索策略说明", "请给出清晰可执行的史料检索策略，覆盖语料范围与筛选标准。"),
]


def _generate_target_themes(
    llm_client: OpenAICompatClient,
    llm_config: LLMEndpointConfig,
    idea: str,
    logger,
) -> list[dict[str, str]]:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        messages = [
            {
                "role": "system",
                "content": "你是中文学术研究选题顾问。只返回 JSON，不要输出任何额外文字。",
            },
            {
                "role": "user",
                "content": (
                    "根据以下研究意向生成 3-5 个检索主题，字段必须为"
                    " target_themes:[{theme,description}]。\n"
                    "要求：主题短语明确，description 可执行。\n"
                    f"研究意向：{idea}"
                ),
            },
        ]
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

    sections: list[str] = []
    for title, instruction in SECTION_SPECS:
        messages = [
            {
                "role": "system",
                "content": "你是严谨的中文学术写作助手，请输出可直接用于研究计划书的正文。",
            },
            {
                "role": "user",
                "content": (
                    f"研究主题：{idea}\n"
                    f"目标检索主题：{target_themes}\n"
                    f"写作任务：{instruction}\n"
                    "请直接输出该小节正文，不要重复标题。"
                ),
            },
        ]

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
