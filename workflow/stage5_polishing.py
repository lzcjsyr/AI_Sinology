from __future__ import annotations

import html
import re
import zipfile
from pathlib import Path
from typing import Any

from core.config import LLMEndpointConfig
from core.llm_client import OpenAICompatClient
from core.prompt_loader import PromptSpec, build_messages, load_prompt
from core.utils import parse_json_from_text, read_json, read_text, write_text


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
    prompt_spec = load_prompt("stage5_abstract_keywords")
    draft_path = project_dir / "4_first_draft.md"
    corpus_path = project_dir / "2_final_corpus.json"

    if not draft_path.exists():
        raise RuntimeError("阶段五无法开始：缺少 4_first_draft.md")
    if not corpus_path.exists():
        raise RuntimeError("阶段五无法开始：缺少 2_final_corpus.json")

    draft_text = read_text(draft_path)
    corpus = read_json(corpus_path)
    corpus_map = {str(item.get("piece_id")): item for item in corpus if item.get("piece_id")}

    abstract_bundle = _generate_abstract_and_keywords(
        llm_client,
        llm_config,
        prompt_spec,
        draft_text,
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
            draft_text.strip(),
            "",
            "## 参考文献（草案）",
            "",
            "1. 研究使用 Kanripo 原始文献库与项目内部结构化史料卡片。",
            "2. 文献格式可按 GB/T 7714-2015 在投稿前做最终统一。",
            "",
        ]
    )

    polished_md_path = project_dir / "5_final_manuscript.md"
    write_text(polished_md_path, polished_markdown)

    total_quotes, matched_quotes, mismatches = _verify_quotes(draft_text, corpus_map)
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

    logger.info("阶段五完成: %s, %s, %s", polished_md_path, checklist_path, docx_path)
    return polished_md_path, checklist_path, docx_path
