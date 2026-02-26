from __future__ import annotations

from pathlib import Path

from core.config import LLMEndpointConfig
from core.llm_client import OpenAICompatClient
from core.prompt_loader import PromptSpec, build_messages, load_prompt
from core.utils import clamp_text, parse_json_from_text, read_json, read_yaml, write_text


def _format_blockquote(piece_id: str, text: str) -> str:
    lines = text.splitlines() or [""]
    quoted = [f"> [{piece_id}] {lines[0]}"]
    for line in lines[1:]:
        quoted.append(f"> {line}")
    return "\n".join(quoted)


def _generate_sub_section_analysis(
    *,
    llm_client: OpenAICompatClient,
    llm_config: LLMEndpointConfig,
    prompt_spec: PromptSpec,
    subsection_title: str,
    subsection_argument: str,
    evidence: list[dict[str, str]],
) -> dict[str, str]:
    evidence_preview = []
    for item in evidence:
        evidence_preview.append(
            f"- {item['piece_id']}: {clamp_text(item['original_text'], 420)}"
        )

    response = llm_client.chat(
        build_messages(
            prompt_spec,
            subsection_title=subsection_title,
            subsection_argument=subsection_argument,
            evidence_preview="\n".join(evidence_preview),
        ),
        temperature=0.4,
        **llm_config.as_client_kwargs(),
    )
    data = parse_json_from_text(response.content)
    result = {
        "topic_sentence": str(data.get("topic_sentence") or ""),
        "analysis": str(data.get("analysis") or ""),
        "mini_conclusion": str(data.get("mini_conclusion") or ""),
    }
    if not result["topic_sentence"] or not result["analysis"] or not result["mini_conclusion"]:
        raise RuntimeError(f"阶段四失败：小节 {subsection_title} 返回字段不完整。")
    return result


def run_stage4_drafting(
    *,
    project_dir: Path,
    llm_client: OpenAICompatClient,
    llm_config: LLMEndpointConfig,
    logger,
) -> str:
    prompt_spec = load_prompt("stage4_subsection_analysis")
    outline_yaml_path = project_dir / "3_outline_matrix.yaml"
    corpus_path = project_dir / "2_final_corpus.json"
    output_path = project_dir / "4_first_draft.md"

    if not outline_yaml_path.exists():
        raise RuntimeError("阶段四无法开始：缺少 3_outline_matrix.yaml")
    outline = read_yaml(outline_yaml_path)
    if not corpus_path.exists():
        raise RuntimeError("阶段四无法开始：缺少 2_final_corpus.json")

    corpus = read_json(corpus_path)
    if not isinstance(outline, dict):
        raise RuntimeError("3_outline_matrix 结构错误")
    if not isinstance(corpus, list):
        raise RuntimeError("2_final_corpus.json 结构错误")

    corpus_map = {
        str(item.get("piece_id")): {
            "piece_id": str(item.get("piece_id")),
            "source_file": str(item.get("source_file") or ""),
            "original_text": str(item.get("original_text") or ""),
            "matched_theme": str(item.get("matched_theme") or ""),
        }
        for item in corpus
        if item.get("piece_id")
    }

    lines: list[str] = []
    thesis_statement = str(outline.get("thesis_statement") or "本文核心论点待补充。")
    lines.append("# 论文初稿")
    lines.append("")
    lines.append("## 绪论")
    lines.append("")
    lines.append(thesis_statement)
    lines.append("")

    used_piece_ids: list[str] = []

    for chapter in outline.get("chapters", []):
        chapter_title = str(chapter.get("chapter_title") or "未命名章")
        chapter_argument = str(chapter.get("chapter_argument") or "")
        lines.append(f"## {chapter_title}")
        lines.append("")
        lines.append(chapter_argument)
        lines.append("")

        for section in chapter.get("sections", []):
            section_title = str(section.get("section_title") or "未命名节")
            section_transition = str(section.get("section_transition") or "")
            rebuttal = str(section.get("counter_arguments_rebuttals") or "")
            lines.append(f"### {section_title}")
            lines.append("")
            if section_transition:
                lines.append(section_transition)
                lines.append("")

            for sub in section.get("sub_sections", []):
                sub_title = str(sub.get("sub_section_title") or "未命名小节")
                sub_argument = str(sub.get("sub_section_argument") or "")
                anchors = sub.get("evidence_anchors") or []
                anchors = [str(a) for a in anchors if str(a) in corpus_map]

                evidence_items = [corpus_map[a] for a in anchors]
                if evidence_items:
                    used_piece_ids.extend([item["piece_id"] for item in evidence_items])

                analysis = _generate_sub_section_analysis(
                    llm_client=llm_client,
                    llm_config=llm_config,
                    prompt_spec=prompt_spec,
                    subsection_title=sub_title,
                    subsection_argument=sub_argument,
                    evidence=evidence_items,
                )

                lines.append(f"#### {sub_title}")
                lines.append("")
                lines.append(f"**主题句**：{analysis['topic_sentence']}")
                lines.append("")

                for item in evidence_items:
                    lines.append("**史料引文**")
                    lines.append("")
                    lines.append(_format_blockquote(item["piece_id"], item["original_text"]))
                    lines.append("")

                lines.append("**分析**")
                lines.append("")
                lines.append(analysis["analysis"])
                lines.append("")

                lines.append("**段落小结**")
                lines.append("")
                lines.append(analysis["mini_conclusion"])
                lines.append("")

            if rebuttal:
                lines.append("**可能反驳与回应**")
                lines.append("")
                lines.append(rebuttal)
                lines.append("")

    lines.append("## 结论")
    lines.append("")
    lines.append("本文通过结构化史料切片与主题映射，完成了从论纲到论证文本的受控生成。")
    lines.append("")

    unique_used_ids = []
    seen = set()
    for pid in used_piece_ids:
        if pid not in seen:
            unique_used_ids.append(pid)
            seen.add(pid)

    lines.append("## 初排版引注区")
    lines.append("")
    for idx, piece_id in enumerate(unique_used_ids, start=1):
        item = corpus_map.get(piece_id)
        if item is None:
            continue
        lines.append(f"[{idx}] {piece_id} | {item['source_file']} | 主题: {item['matched_theme']}")

    draft = "\n".join(lines).strip() + "\n"
    write_text(output_path, draft)
    logger.info("阶段四完成: %s", output_path)
    return draft
