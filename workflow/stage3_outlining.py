from __future__ import annotations

from pathlib import Path
from typing import Any

from core.llm_client import OpenAICompatClient
from core.utils import parse_json_from_text, parse_target_themes_from_proposal, read_json, write_json


def _collect_piece_ids(corpus: list[dict[str, Any]]) -> set[str]:
    return {str(item.get("piece_id")) for item in corpus if item.get("piece_id")}


def _sanitize_outline(outline: dict[str, Any], valid_piece_ids: set[str]) -> dict[str, Any]:
    thesis = str(outline.get("thesis_statement") or "本文主张仍待补充。").strip()
    chapters = outline.get("chapters")
    if not isinstance(chapters, list):
        chapters = []

    fixed_chapters: list[dict[str, Any]] = []
    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        sections = chapter.get("sections")
        if not isinstance(sections, list):
            sections = []

        fixed_sections: list[dict[str, Any]] = []
        for section in sections:
            if not isinstance(section, dict):
                continue
            sub_sections = section.get("sub_sections")
            if not isinstance(sub_sections, list):
                sub_sections = []

            fixed_sub_sections: list[dict[str, Any]] = []
            for sub in sub_sections:
                if not isinstance(sub, dict):
                    continue
                anchors = sub.get("evidence_anchors")
                if not isinstance(anchors, list):
                    anchors = []
                valid_anchors = [
                    str(anchor) for anchor in anchors if str(anchor) in valid_piece_ids
                ]
                if not valid_anchors and valid_piece_ids:
                    valid_anchors = [next(iter(valid_piece_ids))]

                fixed_sub_sections.append(
                    {
                        "sub_section_title": str(sub.get("sub_section_title") or "未命名小节"),
                        "sub_section_argument": str(sub.get("sub_section_argument") or "待补充论证"),
                        "evidence_anchors": valid_anchors,
                    }
                )

            fixed_sections.append(
                {
                    "section_title": str(section.get("section_title") or "未命名节"),
                    "section_transition": str(section.get("section_transition") or "待补充过渡"),
                    "sub_sections": fixed_sub_sections,
                    "counter_arguments_rebuttals": str(
                        section.get("counter_arguments_rebuttals") or "待补充回应"
                    ),
                }
            )

        fixed_chapters.append(
            {
                "chapter_title": str(chapter.get("chapter_title") or "未命名章"),
                "chapter_argument": str(chapter.get("chapter_argument") or "待补充分论点"),
                "sections": fixed_sections,
            }
        )

    return {"thesis_statement": thesis, "chapters": fixed_chapters}


def run_stage3_outlining(
    *,
    project_dir: Path,
    llm_client: OpenAICompatClient,
    logger,
) -> dict[str, Any]:
    proposal_path = project_dir / "1_research_proposal.md"
    corpus_json_path = project_dir / "2_final_corpus.json"
    output_path = project_dir / "3_outline_matrix.json"

    if not proposal_path.exists():
        raise RuntimeError("阶段三无法开始：缺少 1_research_proposal.md")
    if not corpus_json_path.exists():
        raise RuntimeError("阶段三无法开始：缺少 2_final_corpus.json")

    target_themes = parse_target_themes_from_proposal(proposal_path)
    corpus = read_json(corpus_json_path)
    if not isinstance(corpus, list) or not corpus:
        raise RuntimeError("阶段三无法开始：2_final_corpus 为空")

    valid_piece_ids = _collect_piece_ids(corpus)
    proposal_hint = ""
    if (project_dir / "1_research_proposal_meta.json").exists():
        try:
            meta = read_json(project_dir / "1_research_proposal_meta.json")
            proposal_hint = str(meta.get("idea") or "")
        except Exception:  # noqa: BLE001
            proposal_hint = ""

    corpus_summary_lines = []
    for rec in corpus[:30]:
        corpus_summary_lines.append(
            f"- piece_id={rec.get('piece_id')} | theme={rec.get('matched_theme')}"
        )

    prompt = (
        "请基于以下研究主题和证据列表，生成三级论文大纲 JSON。"
        "必须使用字段：thesis_statement, chapters[].chapter_title, chapter_argument,"
        " sections[].section_title, section_transition, sub_sections[].sub_section_title,"
        " sub_section_argument, evidence_anchors[], counter_arguments_rebuttals。\n"
        "硬约束：evidence_anchors 只能使用给定 piece_id，禁止虚构。\n"
        "只返回 JSON，不要任何额外文本。\n\n"
        f"研究意向：{proposal_hint}\n"
        f"目标主题：{target_themes}\n"
        f"可用证据列表：\n{chr(10).join(corpus_summary_lines)}\n"
    )

    response = llm_client.chat(
        [
            {"role": "system", "content": "你是学术论证架构师。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    outline = parse_json_from_text(response.content)

    outline = _sanitize_outline(outline, valid_piece_ids)

    if not outline.get("chapters"):
        raise RuntimeError("阶段三失败：模型返回大纲为空。")

    write_json(output_path, outline)
    logger.info("阶段三完成: %s", output_path)
    return outline
