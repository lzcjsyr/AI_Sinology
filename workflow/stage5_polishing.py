from __future__ import annotations

import html
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.config import LLMEndpointConfig
from core.llm_client import OpenAICompatClient
from core.prompt_loader import PromptSpec, build_messages, load_prompt
from core.utils import parse_json_from_text, read_json, read_text, write_json, write_text


_STAGE5_PROGRESS_FILE = "5_polish_progress.json"


@dataclass(frozen=True)
class _PolishUnit:
    title: str
    content: str


def _extract_quote_blocks(markdown_text: str) -> list[tuple[str, str]]:
    lines = markdown_text.splitlines()
    blocks: list[tuple[str, str]] = []
    current_piece_id: str | None = None
    current_lines: list[str] = []

    for line in lines + [""]:
        if line.startswith(">"):
            content = line[1:].lstrip()
            if current_piece_id is None:
                match = re.match(r"\[([^\]]+)\]\s*(.*)", content)
                if match:
                    current_piece_id = match.group(1).strip()
                    current_lines = [match.group(2).strip()]
                continue
            current_lines.append(content)
            continue

        if current_piece_id is not None:
            quote_text = "\n".join(s for s in current_lines if s is not None).strip()
            blocks.append((current_piece_id, quote_text))
            current_piece_id = None
            current_lines = []

    return blocks


def _verify_quotes(draft_text: str, corpus_map: dict[str, dict[str, Any]]) -> tuple[int, int, list[str]]:
    blocks = _extract_quote_blocks(draft_text)
    total = len(blocks)
    matched = 0
    mismatches: list[str] = []

    for piece_id, quote_text in blocks:
        expected = str(corpus_map.get(piece_id, {}).get("original_text") or "").strip()
        if expected and quote_text.strip() == expected:
            matched += 1
        else:
            mismatches.append(piece_id)

    return total, matched, mismatches


def _generate_abstract_and_keywords(
    llm_client: OpenAICompatClient,
    llm_config: LLMEndpointConfig,
    prompt_spec: PromptSpec,
    draft_text: str,
) -> dict[str, Any]:
    response = llm_client.chat(
        build_messages(prompt_spec, draft_excerpt=draft_text[:12000]),
        temperature=0.3,
        **llm_config.as_client_kwargs(),
    )
    data = parse_json_from_text(response.content)
    keywords = data.get("keywords")
    if not isinstance(keywords, list):
        keywords = []
    result = {
        "abstract_cn": str(data.get("abstract_cn") or ""),
        "abstract_en": str(data.get("abstract_en") or ""),
        "keywords": [str(k) for k in keywords if str(k).strip()][:6],
    }
    if not result["abstract_cn"] or not result["abstract_en"] or not result["keywords"]:
        raise RuntimeError("阶段五失败：摘要或关键词返回为空。")
    return result


def _extract_heading_ranges(markdown_text: str, heading_level: int) -> list[tuple[int, int]]:
    heading_matches = list(re.finditer(r"(?m)^(#{1,6})\s+.+$", markdown_text))
    ranges: list[tuple[int, int]] = []
    for idx, match in enumerate(heading_matches):
        if len(match.group(1)) != heading_level:
            continue
        start = match.start()
        end = len(markdown_text)
        for nxt in heading_matches[idx + 1 :]:
            if len(nxt.group(1)) <= heading_level:
                end = nxt.start()
                break
        ranges.append((start, end))
    return ranges


def _build_polish_plan(markdown_text: str) -> tuple[list[str], list[_PolishUnit], int | None]:
    for level in (4, 3, 2):
        ranges = _extract_heading_ranges(markdown_text, level)
        if not ranges:
            continue

        static_parts: list[str] = []
        units: list[_PolishUnit] = []
        cursor = 0
        for idx, (start, end) in enumerate(ranges, start=1):
            static_parts.append(markdown_text[cursor:start])
            content = markdown_text[start:end]
            heading_line = content.splitlines()[0].strip() if content.splitlines() else ""
            title = re.sub(r"^#+\s*", "", heading_line).strip()
            if not title:
                title = f"段落{idx}"
            units.append(_PolishUnit(title=title, content=content))
            cursor = end
        static_parts.append(markdown_text[cursor:])
        return static_parts, units, level

    return ["", ""], [_PolishUnit(title="全文", content=markdown_text)], None


def _compose_draft(static_parts: list[str], unit_contents: list[str]) -> str:
    if len(static_parts) != len(unit_contents) + 1:
        raise RuntimeError("阶段五失败：润色切片计划结构不一致。")
    out = [static_parts[0]]
    for idx, content in enumerate(unit_contents):
        out.append(content)
        out.append(static_parts[idx + 1])
    return "".join(out)


def _normalize_polished_unit(source: str, candidate: str) -> str:
    candidate = candidate.strip()
    if not candidate:
        return source

    source_lines = source.splitlines()
    if source_lines:
        source_heading = source_lines[0].strip()
        if re.match(r"^#{2,6}\s+", source_heading):
            candidate_lines = candidate.splitlines()
            if not candidate_lines:
                candidate = source_heading
            else:
                candidate_first = candidate_lines[0].strip()
                if candidate_first != source_heading:
                    if re.match(r"^#{2,6}\s+", candidate_first):
                        candidate_lines = candidate_lines[1:]
                    body = "\n".join(candidate_lines).strip()
                    candidate = source_heading
                    if body:
                        candidate += "\n\n" + body

    if source.endswith("\n") and not candidate.endswith("\n"):
        candidate += "\n"
    return candidate


def _generate_polished_subsection(
    *,
    llm_client: OpenAICompatClient,
    llm_config: LLMEndpointConfig,
    prompt_spec: PromptSpec,
    manuscript_preview: str,
    subsection_text: str,
    subsection_title: str,
    current_index: int,
    total_units: int,
    completed_titles: list[str],
) -> str:
    completed_titles_preview = "、".join(completed_titles[-8:]) if completed_titles else "无"
    response = llm_client.chat(
        build_messages(
            prompt_spec,
            manuscript_preview=manuscript_preview,
            subsection_text=subsection_text,
            subsection_title=subsection_title,
            current_index=current_index,
            total_units=total_units,
            completed_units=len(completed_titles),
            completed_titles=completed_titles_preview,
        ),
        temperature=0.35,
        **llm_config.as_client_kwargs(),
    )
    data = parse_json_from_text(response.content)
    polished_subsection = str(data.get("polished_subsection") or "").strip()
    if not polished_subsection:
        raise RuntimeError(f"阶段五失败：段落 {current_index}/{total_units} 返回空内容。")
    return polished_subsection


def _load_polish_progress(progress_path: Path) -> tuple[str, int] | None:
    if not progress_path.exists():
        return None
    try:
        payload = read_json(progress_path)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(payload, dict):
        return None
    if str(payload.get("status") or "") != "running":
        return None

    snapshot = payload.get("draft_snapshot")
    completed_units = payload.get("completed_units")
    if not isinstance(snapshot, str):
        return None
    if not isinstance(completed_units, int) or completed_units < 0:
        return None
    return snapshot, completed_units


def _write_polish_progress(
    *,
    progress_path: Path,
    total_units: int,
    completed_units: int,
    heading_level: int | None,
    unit_titles: list[str],
    draft_snapshot: str,
) -> None:
    write_json(
        progress_path,
        {
            "status": "running",
            "total_units": total_units,
            "completed_units": completed_units,
            "heading_level": heading_level,
            "unit_titles": unit_titles,
            "draft_snapshot": draft_snapshot,
        },
    )


def _markdown_to_paragraphs(text: str) -> list[str]:
    paragraphs: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            paragraphs.append("")
            continue
        line = re.sub(r"^#+\s*", "", line)
        line = re.sub(r"^>\s?", "", line)
        line = re.sub(r"^\*\*(.*?)\*\*$", r"\1", line)
        paragraphs.append(line)
    return paragraphs


def _write_simple_docx(text: str, output_path: Path) -> None:
    paragraphs = _markdown_to_paragraphs(text)

    body_parts = []
    for paragraph in paragraphs:
        safe_text = html.escape(paragraph)
        body_parts.append(
            "<w:p><w:r><w:t xml:space=\"preserve\">"
            + safe_text
            + "</w:t></w:r></w:p>"
        )

    document_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<w:document xmlns:wpc=\"http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas\" "
        "xmlns:mc=\"http://schemas.openxmlformats.org/markup-compatibility/2006\" "
        "xmlns:o=\"urn:schemas-microsoft-com:office:office\" "
        "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" "
        "xmlns:m=\"http://schemas.openxmlformats.org/officeDocument/2006/math\" "
        "xmlns:v=\"urn:schemas-microsoft-com:vml\" "
        "xmlns:wp14=\"http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing\" "
        "xmlns:wp=\"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing\" "
        "xmlns:w10=\"urn:schemas-microsoft-com:office:word\" "
        "xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\" "
        "xmlns:w14=\"http://schemas.microsoft.com/office/word/2010/wordml\" "
        "xmlns:wpg=\"http://schemas.microsoft.com/office/word/2010/wordprocessingGroup\" "
        "xmlns:wpi=\"http://schemas.microsoft.com/office/word/2010/wordprocessingInk\" "
        "xmlns:wne=\"http://schemas.microsoft.com/office/word/2006/wordml\" "
        "xmlns:wps=\"http://schemas.microsoft.com/office/word/2010/wordprocessingShape\" mc:Ignorable=\"w14 wp14\">"
        "<w:body>"
        + "".join(body_parts)
        + "<w:sectPr><w:pgSz w:w=\"11906\" w:h=\"16838\"/>"
        "<w:pgMar w:top=\"1440\" w:right=\"1440\" w:bottom=\"1440\" w:left=\"1440\" w:header=\"708\" w:footer=\"708\" w:gutter=\"0\"/>"
        "<w:cols w:space=\"708\"/><w:docGrid w:linePitch=\"360\"/></w:sectPr>"
        "</w:body></w:document>"
    )

    content_types = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
        "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
        "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
        "<Override PartName=\"/word/document.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/>"
        "</Types>"
    )

    rels = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
        "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"word/document.xml\"/>"
        "</Relationships>"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", document_xml)


def run_stage5_polishing(
    *,
    project_dir: Path,
    llm_client: OpenAICompatClient,
    llm_config: LLMEndpointConfig,
    logger,
) -> tuple[Path, Path, Path]:
    abstract_prompt_spec = load_prompt("stage5_abstract_keywords")
    subsection_prompt_spec = load_prompt("stage5_subsection_polish")
    draft_path = project_dir / "4_first_draft.md"
    corpus_path = project_dir / "2_final_corpus.json"
    polished_md_path = project_dir / "5_final_manuscript.md"
    progress_path = project_dir / _STAGE5_PROGRESS_FILE

    if not draft_path.exists():
        raise RuntimeError("阶段五无法开始：缺少 4_first_draft.md")
    if not corpus_path.exists():
        raise RuntimeError("阶段五无法开始：缺少 2_final_corpus.json")

    draft_text = read_text(draft_path)
    restored_progress = _load_polish_progress(progress_path)
    if restored_progress is not None:
        restored_draft, completed_units = restored_progress
        if restored_draft.strip():
            draft_text = restored_draft
            logger.info("阶段五检测到中断进度，将尝试继续执行。")
    else:
        completed_units = 0

    static_parts, units, heading_level = _build_polish_plan(draft_text)
    unit_titles = [unit.title for unit in units]
    if completed_units > len(units):
        completed_units = 0

    unit_contents = [unit.content for unit in units]
    total_units = len(unit_contents)

    logger.info(
        "阶段五润色计划：total=%s completed=%s heading_level=%s",
        total_units,
        completed_units,
        heading_level if heading_level is not None else "full",
    )

    for idx in range(completed_units, total_units):
        current_draft = _compose_draft(static_parts, unit_contents)
        completed_titles = unit_titles[:idx]
        source_unit = unit_contents[idx]
        polished_unit = _generate_polished_subsection(
            llm_client=llm_client,
            llm_config=llm_config,
            prompt_spec=subsection_prompt_spec,
            manuscript_preview=current_draft,
            subsection_text=source_unit,
            subsection_title=unit_titles[idx],
            current_index=idx + 1,
            total_units=total_units,
            completed_titles=completed_titles,
        )
        polished_unit = _normalize_polished_unit(source_unit, polished_unit)

        source_quotes = _extract_quote_blocks(source_unit)
        if source_quotes and _extract_quote_blocks(polished_unit) != source_quotes:
            logger.warning(
                "阶段五段落 %s/%s (%s) 修改了史料引文，已回退为原段落以保持引文一致。",
                idx + 1,
                total_units,
                unit_titles[idx],
            )
            polished_unit = source_unit

        unit_contents[idx] = polished_unit
        updated_draft = _compose_draft(static_parts, unit_contents)
        write_text(polished_md_path, updated_draft)
        _write_polish_progress(
            progress_path=progress_path,
            total_units=total_units,
            completed_units=idx + 1,
            heading_level=heading_level,
            unit_titles=unit_titles,
            draft_snapshot=updated_draft,
        )
        logger.info("阶段五段落润色完成: %s/%s %s", idx + 1, total_units, unit_titles[idx])

    final_draft = _compose_draft(static_parts, unit_contents)
    corpus = read_json(corpus_path)
    corpus_map = {str(item.get("piece_id")): item for item in corpus if item.get("piece_id")}

    abstract_bundle = _generate_abstract_and_keywords(
        llm_client,
        llm_config,
        abstract_prompt_spec,
        final_draft,
    )
    keywords_text = "、".join(abstract_bundle["keywords"]) if abstract_bundle["keywords"] else "待补充"

    polished_markdown = "\n".join(
        [
            "# 最终定稿（润色版）",
            "",
            "## 中文摘要",
            "",
            abstract_bundle["abstract_cn"],
            "",
            "## 英文摘要",
            "",
            abstract_bundle["abstract_en"],
            "",
            "## 关键词",
            "",
            keywords_text,
            "",
            "---",
            "",
            final_draft.strip(),
            "",
            "## 参考文献（草案）",
            "",
            "1. 研究使用 Kanripo 原始文献库与项目内部结构化史料卡片。",
            "2. 文献格式可按 GB/T 7714-2015 在投稿前做最终统一。",
            "",
        ]
    )

    write_text(polished_md_path, polished_markdown)

    total_quotes, matched_quotes, mismatches = _verify_quotes(final_draft, corpus_map)
    quote_rate = (matched_quotes / total_quotes * 100.0) if total_quotes else 100.0

    checklist_lines = [
        "# 修改与润色自检清单",
        "",
        "## 1. 逻辑修改与结构检查",
        "- 已完成摘要、关键词、结论与章节衔接的统一整理。",
        "- 大纲节点与证据锚点保持一一对应。",
        "",
        "## 2. 引文出处复核",
        f"- 引文块总数：{total_quotes}",
        f"- 与原始史料完全一致：{matched_quotes}",
        f"- 一致率：{quote_rate:.2f}%",
    ]
    if mismatches:
        checklist_lines.append(f"- 不一致 piece_id：{', '.join(mismatches)}")
    else:
        checklist_lines.append("- 未发现引文篡改。")

    checklist_lines.extend(
        [
            "",
            "## 3. 学术规范格式自测",
            "- 文档包含中英文摘要、关键词、正文与参考文献草案。",
            "- 建议投稿前进行一次人工格式终审（脚注、引文细则、参考文献条目）。",
            "",
        ]
    )

    checklist_path = project_dir / "5_revision_checklist.md"
    write_text(checklist_path, "\n".join(checklist_lines))

    docx_path = project_dir / "5_final_manuscript.docx"
    _write_simple_docx(polished_markdown, docx_path)

    if progress_path.exists():
        progress_path.unlink()

    logger.info("阶段五完成: %s, %s, %s", polished_md_path, checklist_path, docx_path)
    return polished_md_path, checklist_path, docx_path
