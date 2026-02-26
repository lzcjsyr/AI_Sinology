from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from core.utils import append_jsonl, ensure_dir

PB_PATTERN = re.compile(r"<pb:([^>]+)>")
TITLE_PATTERN = re.compile(r"^#\+TITLE:\s*(.+)$", re.MULTILINE)
CATALOG_SECTION_PATTERN = re.compile(r"^\*\s+KR[1-4]\s+(.+)$")
CATALOG_ENTRY_PATTERN = re.compile(r"^\*\*\s+\[\[file:(KR[1-4][a-z])\.txt\]\[([^\]]+)\]\]$")
TECH_MARKER_PATTERN = re.compile(
    r"\bKR\d+[a-z]?\d*(?:_[A-Za-z0-9\-]+)*\b|_tls_|^pb:",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ScopeOption:
    code: str
    section: str
    label: str

    @property
    def display_label(self) -> str:
        return f"{self.section} [{self.label}]"


def _normalize_scope(scope: str) -> str:
    cleaned = scope.strip()
    if cleaned.lower().startswith("kr") and len(cleaned) > 2:
        return f"KR{cleaned[2:].lower()}"
    return cleaned


def _catalog_root(kanripo_dir: Path) -> Path:
    return kanripo_dir / "KR-Catalog" / "KR"


def list_available_scope_dirs(kanripo_dir: Path) -> list[str]:
    if not kanripo_dir.exists():
        return []
    return sorted(
        p.name
        for p in kanripo_dir.iterdir()
        if p.is_dir()
        and p.name.startswith("KR")
        and not p.name.startswith(".")
        and p.name != "KR-Catalog"
    )


def list_available_scope_options(kanripo_dir: Path) -> list[ScopeOption]:
    if not kanripo_dir.exists():
        return []

    catalog_root = _catalog_root(kanripo_dir)
    options: list[ScopeOption] = []
    seen_codes: set[str] = set()

    for idx in range(1, 5):
        catalog_file = catalog_root / f"KR{idx}.txt"
        if not catalog_file.exists():
            continue

        section_name = ""
        for raw_line in catalog_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line:
                continue

            section_match = CATALOG_SECTION_PATTERN.match(line)
            if section_match:
                section_name = section_match.group(1).strip()
                continue

            entry_match = CATALOG_ENTRY_PATTERN.match(line)
            if not entry_match:
                continue

            scope_code = _normalize_scope(entry_match.group(1))
            if scope_code in seen_codes:
                continue

            options.append(
                ScopeOption(
                    code=scope_code,
                    section=section_name or f"KR{idx}",
                    label=entry_match.group(2).strip(),
                )
            )
            seen_codes.add(scope_code)

    if options:
        return options

    # Fallback for environments without catalog files.
    scopes = list_available_scope_dirs(kanripo_dir)
    return [ScopeOption(code=scope, section="未分类", label=scope) for scope in sorted(scopes)]


def list_available_scopes(kanripo_dir: Path) -> list[str]:
    return [option.code for option in list_available_scope_options(kanripo_dir)]


def _resolve_scope_dirs(kanripo_dir: Path, scope: str) -> list[Path]:
    normalized_scope = _normalize_scope(scope)
    exact_dir = kanripo_dir / normalized_scope
    if exact_dir.exists() and exact_dir.is_dir():
        return [exact_dir]

    if normalized_scope == "KR-Catalog":
        return []

    return sorted(
        p
        for p in kanripo_dir.iterdir()
        if p.is_dir()
        and not p.name.startswith(".")
        and p.name != "KR-Catalog"
        and p.name.startswith(normalized_scope)
    )


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
        stripped = line.strip()
        if stripped.startswith("#"):
            comment_payload = stripped.lstrip("#").strip()
            if TECH_MARKER_PATTERN.search(comment_payload or ""):
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
    parsed_dirs: set[str] = set()
    for scope in selected_scopes:
        scope_dirs = _resolve_scope_dirs(kanripo_dir, scope)
        if not scope_dirs:
            logger.warning("忽略不存在的 scope: %s", scope)
            continue

        logger.info("scope=%s 匹配到 %s 个目录。", scope, len(scope_dirs))
        for scope_dir in scope_dirs:
            if scope_dir.name in parsed_dirs:
                continue
            parsed_dirs.add(scope_dir.name)
            txt_files = sorted(p for p in scope_dir.iterdir() if p.suffix == ".txt")
            for txt_file in txt_files:
                for fragment in _split_file_to_fragments(txt_file):
                    append_jsonl(output_path, fragment)
                    written += 1
                    if max_fragments is not None and written >= max_fragments:
                        logger.info("达到 max_fragments=%s，提前结束切片。", max_fragments)
                        logger.info(
                            "阶段2.1完成: %s (records=%s, dirs=%s)",
                            output_path,
                            written,
                            len(parsed_dirs),
                        )
                        return output_path

    logger.info("阶段2.1完成: %s (records=%s, dirs=%s)", output_path, written, len(parsed_dirs))
    return output_path
