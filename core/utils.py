from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_JSON_TAG_RE = re.compile(r"<json>\s*(.*?)\s*</json>", re.IGNORECASE | re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(?=\s*[}\]])")
_SMART_QUOTE_TRANSLATION = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
    }
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^a-z0-9\-_\u4e00-\u9fff]", "", text)
    return text or "research-project"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> Any:
    ruby_code = (
        "require 'yaml'; require 'json'; "
        "data = YAML.safe_load(File.read(ARGV[0]), permitted_classes: [], aliases: false); "
        "puts JSON.generate(data)"
    )
    try:
        result = subprocess.run(
            ["ruby", "-e", ruby_code, str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("读取 YAML 失败：系统未安装 ruby") from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"读取 YAML 失败: {path} error={detail}")

    try:
        return json.loads(result.stdout)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"YAML 解码失败: {path} error={exc}") from exc


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                rows.append(data)
    return rows


def _strip_outer_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) < 2:
        return stripped
    if not lines[0].startswith("```"):
        return stripped

    tail = lines[-1].strip()
    if tail == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _normalize_json_candidate(text: str) -> str:
    return text.strip().lstrip("\ufeff").translate(_SMART_QUOTE_TRANSLATION)


def _extract_tagged_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for pattern in (_JSON_FENCE_RE, _JSON_TAG_RE):
        for match in pattern.finditer(text):
            candidate = _normalize_json_candidate(match.group(1))
            if candidate:
                candidates.append(candidate)
    return candidates


def _iter_balanced_json_snippets(text: str) -> list[str]:
    snippets: list[str] = []
    length = len(text)
    for start, ch in enumerate(text):
        if ch not in "{[":
            continue
        stack = [ch]
        in_string = False
        string_quote = ""
        escaped = False
        for idx in range(start + 1, length):
            current = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                    continue
                if current == "\\":
                    escaped = True
                    continue
                if current == string_quote:
                    in_string = False
                continue

            if current in {'"', "'"}:
                in_string = True
                string_quote = current
                continue

            if current in "{[":
                stack.append(current)
                continue
            if current in "}]":
                if not stack:
                    break
                opening = stack[-1]
                if (opening, current) not in {("{", "}"), ("[", "]")}:
                    break
                stack.pop()
                if not stack:
                    snippet = _normalize_json_candidate(text[start : idx + 1])
                    if snippet:
                        snippets.append(snippet)
                    break
    return snippets


def _replace_json_literals_outside_strings(text: str) -> str:
    replacements = {"true": "True", "false": "False", "null": "None"}
    result: list[str] = []
    idx = 0
    length = len(text)
    in_string = False
    string_quote = ""
    escaped = False

    while idx < length:
        ch = text[idx]
        if in_string:
            result.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == string_quote:
                in_string = False
            idx += 1
            continue

        if ch in {'"', "'"}:
            in_string = True
            string_quote = ch
            result.append(ch)
            idx += 1
            continue

        matched = False
        for source, target in replacements.items():
            end = idx + len(source)
            if not text.startswith(source, idx):
                continue
            prev = text[idx - 1] if idx > 0 else ""
            next_char = text[end] if end < length else ""
            if (prev.isalnum() or prev == "_") or (next_char.isalnum() or next_char == "_"):
                continue
            result.append(target)
            idx = end
            matched = True
            break
        if matched:
            continue

        result.append(ch)
        idx += 1

    return "".join(result)


def _repair_json_candidate(text: str) -> str:
    repaired = _normalize_json_candidate(text)
    repaired = _TRAILING_COMMA_RE.sub("", repaired)
    return repaired


def _try_parse_python_like_value(text: str) -> Any:
    normalized = _replace_json_literals_outside_strings(text)
    value = ast.literal_eval(normalized)
    if isinstance(value, (dict, list)):
        return value
    raise ValueError("Python 风格结构的根节点不是对象或数组")


def _try_decode_candidate(text: str, *, allow_python_like: bool) -> Any:
    decoder = json.JSONDecoder()
    candidates = [_normalize_json_candidate(text)]
    repaired = _repair_json_candidate(text)
    if repaired not in candidates:
        candidates.append(repaired)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        for idx, ch in enumerate(candidate):
            if ch not in "{[":
                continue
            try:
                value, _end = decoder.raw_decode(candidate[idx:])
                return value
            except json.JSONDecodeError:
                continue

        if allow_python_like:
            try:
                return _try_parse_python_like_value(candidate)
            except Exception:  # noqa: BLE001
                pass

    raise ValueError("模型返回中未找到可解析的 JSON 结构")


def _extract_first_json_value(text: str) -> Any:
    seen: set[str] = set()
    candidates: list[str] = []
    for candidate in [
        _normalize_json_candidate(text),
        _normalize_json_candidate(_strip_outer_code_fence(text)),
        *_extract_tagged_json_candidates(text),
    ]:
        if candidate and candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)

    for candidate in list(candidates):
        for snippet in _iter_balanced_json_snippets(candidate):
            if snippet and snippet not in seen:
                seen.add(snippet)
                candidates.append(snippet)

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return _try_decode_candidate(candidate, allow_python_like=True)
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    if last_error is not None:
        raise ValueError("模型返回中未找到可解析的 JSON 结构") from last_error
    raise ValueError("模型返回中未找到可解析的 JSON 结构")


def extract_json_block(text: str) -> str:
    data = _extract_first_json_value(text)
    if not isinstance(data, dict):
        raise ValueError("模型返回 JSON 根节点不是对象")
    return json.dumps(data, ensure_ascii=False)


def parse_json_from_text(text: str) -> dict[str, Any]:
    data = _extract_first_json_value(text)
    if not isinstance(data, dict):
        raise ValueError("模型返回 JSON 根节点不是对象")
    return data


def parse_target_themes_from_proposal(proposal_path: Path) -> list[dict[str, str]]:
    text = read_text(proposal_path)
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return []

    front_matter_lines: list[str] = []
    for line in lines[1:]:
        if line.strip() == "---":
            break
        front_matter_lines.append(line.rstrip())

    themes: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    in_target_themes = False

    for line in front_matter_lines:
        stripped = line.strip()
        if stripped.startswith("target_themes:"):
            in_target_themes = True
            continue
        if not in_target_themes:
            continue

        if stripped.startswith("- theme:"):
            if current and current.get("theme"):
                themes.append(current)
            theme_value = stripped.split(":", 1)[1].strip().strip('"')
            current = {"theme": theme_value, "description": ""}
            continue

        if stripped.startswith("description:") and current is not None:
            desc_value = stripped.split(":", 1)[1].strip().strip('"')
            current["description"] = desc_value
            continue

    if current and current.get("theme"):
        themes.append(current)

    return themes


def parse_idea_from_proposal(proposal_path: Path) -> str:
    text = read_text(proposal_path)
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return ""

    for line in lines[1:]:
        stripped = line.strip()
        if stripped == "---":
            break
        if stripped.startswith("idea:"):
            return stripped.split(":", 1)[1].strip().strip('"')
    return ""


def _yaml_escape(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _dump_yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)

    s = str(value)
    if "\n" in s:
        return "|"
    return _yaml_escape(s)


def _dump_yaml_node(data: Any, indent: int) -> list[str]:
    prefix = " " * indent
    lines: list[str] = []

    if isinstance(data, list):
        if not data:
            lines.append(prefix + "[]")
            return lines

        for item in data:
            if isinstance(item, (dict, list)):
                lines.append(prefix + "-")
                lines.extend(_dump_yaml_node(item, indent + 2))
            else:
                scalar = _dump_yaml_scalar(item)
                if scalar == "|":
                    lines.append(prefix + "- |")
                    for line in str(item).splitlines() or [""]:
                        lines.append(" " * (indent + 2) + line)
                else:
                    lines.append(prefix + "- " + scalar)
        return lines

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                lines.append(prefix + f"{key}:")
                lines.extend(_dump_yaml_node(value, indent + 2))
            else:
                scalar = _dump_yaml_scalar(value)
                if scalar == "|":
                    lines.append(prefix + f"{key}: |")
                    for line in str(value).splitlines() or [""]:
                        lines.append(" " * (indent + 2) + line)
                else:
                    lines.append(prefix + f"{key}: {scalar}")
        return lines

    lines.append(prefix + _dump_yaml_scalar(data))
    return lines


def dump_yaml(data: Any) -> str:
    return "\n".join(_dump_yaml_node(data, 0)) + "\n"


def write_yaml(path: Path, data: Any) -> None:
    write_text(path, dump_yaml(data))


def markdown_front_matter(
    target_themes: list[dict[str, str]],
    *,
    idea: str | None = None,
) -> str:
    lines = ["---"]
    if idea and idea.strip():
        lines.append(f'idea: "{idea.strip()}"')
    lines.append("target_themes:")
    for item in target_themes:
        theme = item.get("theme", "").strip()
        desc = item.get("description", "").strip()
        lines.append(f'  - theme: "{theme}"')
        lines.append(f'    description: "{desc}"')
    lines.append("---")
    return "\n".join(lines)


def clamp_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."
