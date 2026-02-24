from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from core.utils import append_jsonl, ensure_dir

PB_PATTERN = re.compile(r"<pb:([^>]+)>")
TITLE_PATTERN = re.compile(r"^#\+TITLE:\s*(.+)$", re.MULTILINE)


def list_available_scopes(kanripo_dir: Path) -> list[str]:
    if not kanripo_dir.exists():
        return []
    scopes = [
        p.name
        for p in kanripo_dir.iterdir()
        if p.is_dir() and p.name.startswith("KR") and not p.name.startswith(".")
    ]
    return sorted(scopes)


def _normalize_title(raw_title: str) -> str:
    text = raw_title.strip()
    if "/" in text:
        text = text.split("/", 1)[0].strip()
    return text


def _clean_fragment_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.startswith("#+"):
            continue
        cleaned = line.replace("¶", "").strip()
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines).strip()


def _split_file_to_fragments(file_path: Path) -> Iterable[dict[str, str]]:
    raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
    title_match = TITLE_PATTERN.search(raw_text)
    source_file = _normalize_title(title_match.group(1)) if title_match else file_path.stem

    matches = list(PB_PATTERN.finditer(raw_text))
    if not matches:
        cleaned = _clean_fragment_text(raw_text)
        if cleaned:
            yield {
                "piece_id": f"{file_path.stem}_fallback_0001",
                "source_file": source_file,
                "original_text": cleaned,
            }
        return

    for idx, match in enumerate(matches):
        piece_id = match.group(1).strip()
        content_start = match.end()
        content_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_text)
        chunk = raw_text[content_start:content_end]
        cleaned = _clean_fragment_text(chunk)
        if not cleaned:
            continue
        yield {
            "piece_id": piece_id,
            "source_file": source_file,
            "original_text": cleaned,
        }


def parse_kanripo_to_fragments(
    *,
    kanripo_dir: Path,
    selected_scopes: list[str],
    project_processed_dir: Path,
    logger,
    max_fragments: int | None = None,
) -> Path:
    ensure_dir(project_processed_dir)
    output_path = project_processed_dir / "kanripo_fragments.jsonl"

    # Full regeneration avoids stale fragments when scope changes.
    if output_path.exists():
        output_path.unlink()

    written = 0
    for scope in selected_scopes:
        scope_dir = kanripo_dir / scope
        if not scope_dir.exists() or not scope_dir.is_dir():
            logger.warning("忽略不存在的 scope: %s", scope)
            continue

        txt_files = sorted(p for p in scope_dir.iterdir() if p.suffix == ".txt")
        for txt_file in txt_files:
            for fragment in _split_file_to_fragments(txt_file):
                append_jsonl(output_path, fragment)
                written += 1
                if max_fragments is not None and written >= max_fragments:
                    logger.info("达到 max_fragments=%s，提前结束切片。", max_fragments)
                    logger.info("阶段2.1完成: %s (records=%s)", output_path, written)
                    return output_path

    logger.info("阶段2.1完成: %s (records=%s)", output_path, written)
    return output_path
