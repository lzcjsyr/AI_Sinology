from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any


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


def _extract_first_json_value(text: str) -> Any:
    candidates = [text.strip(), _strip_outer_code_fence(text)]
    decoder = json.JSONDecoder()

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


def markdown_front_matter(target_themes: list[dict[str, str]]) -> str:
    lines = ["---", "target_themes:"]
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
