from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any


@dataclass(frozen=True)
class PromptSpec:
    prompt_id: str
    file_path: Path
    purpose: str
    variables: dict[str, Any]
    expected_output: dict[str, Any]
    system_prompt: str
    user_template: str
    raw: dict[str, Any]


def _default_prompts_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "prompts"


def _require_non_empty_string(value: Any, *, field: str, file_path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"提示词文件 `{file_path}` 缺少必填字符串字段: `{field}`")
    return value.strip()


def _validate_prompt_block(prompt: dict[str, Any], *, file_path: Path) -> None:
    _require_non_empty_string(
        prompt.get("system"),
        field="prompt.system",
        file_path=file_path,
    )
    _require_non_empty_string(
        prompt.get("user_template"),
        field="prompt.user_template",
        file_path=file_path,
    )

def load_prompt(prompt_id: str, prompts_dir: Path | None = None) -> PromptSpec:
    if not prompt_id.strip():
        raise RuntimeError("prompt_id 不能为空")

    root = prompts_dir or _default_prompts_dir()
    file_path = root / f"{prompt_id}.yaml"
    if not file_path.exists():
        raise RuntimeError(f"缺少提示词文件: {file_path}")

    payload = _read_yaml_as_object(file_path)

    if not isinstance(payload, dict):
        raise RuntimeError(f"提示词文件根节点必须是对象: {file_path}")

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError(f"提示词文件缺少 `metadata` 对象: {file_path}")

    meta_step = _require_non_empty_string(
        metadata.get("step"),
        field="metadata.step",
        file_path=file_path,
    )
    if meta_step != prompt_id:
        raise RuntimeError(
            f"提示词文件ID不匹配: 文件={file_path.name} metadata.step={meta_step}"
        )
    purpose = _require_non_empty_string(
        metadata.get("purpose"),
        field="metadata.purpose",
        file_path=file_path,
    )

    variables = payload.get("variables")
    if not isinstance(variables, dict) or not variables:
        raise RuntimeError(f"提示词文件缺少 `variables` 对象: {file_path}")

    expected_output = payload.get("expected_output")
    if not isinstance(expected_output, dict) or not expected_output:
        raise RuntimeError(f"提示词文件缺少 `expected_output` 对象: {file_path}")

    prompt = payload.get("prompt")
    if not isinstance(prompt, dict):
        raise RuntimeError(f"提示词文件缺少 `prompt` 对象: {file_path}")
    _validate_prompt_block(prompt, file_path=file_path)

    system_prompt = _require_non_empty_string(
        prompt.get("system"),
        field="prompt.system",
        file_path=file_path,
    )
    user_template = _require_non_empty_string(
        prompt.get("user_template"),
        field="prompt.user_template",
        file_path=file_path,
    )

    return PromptSpec(
        prompt_id=prompt_id,
        file_path=file_path,
        purpose=purpose,
        variables=variables,
        expected_output=expected_output,
        system_prompt=system_prompt,
        user_template=user_template,
        raw=payload,
    )


def _read_yaml_as_object(file_path: Path) -> dict[str, Any]:
    ruby_code = (
        "require 'yaml'; require 'json'; "
        "data = YAML.safe_load(File.read(ARGV[0]), permitted_classes: [], aliases: false); "
        "puts JSON.generate(data)"
    )
    try:
        result = subprocess.run(
            ["ruby", "-e", ruby_code, str(file_path)],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("读取 YAML 失败：系统未安装 ruby，无法解析 prompts 文件") from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"读取提示词文件失败: {file_path} error={detail}")

    try:
        payload = json.loads(result.stdout)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"提示词文件 JSON 解码失败: {file_path} error={exc}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"提示词文件根节点必须是对象: {file_path}")
    return payload


def _to_template_value(name: str, value: Any, *, prompt_id: str) -> str:
    if value is None:
        raise RuntimeError(f"提示词 `{prompt_id}` 变量 `{name}` 不能为 None")
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def render_user_template(spec: PromptSpec, **variables: Any) -> str:
    safe_variables = {
        name: _to_template_value(name, value, prompt_id=spec.prompt_id)
        for name, value in variables.items()
    }
    try:
        return Template(spec.user_template).substitute(safe_variables).strip()
    except KeyError as exc:
        missing = str(exc).strip("'")
        raise RuntimeError(
            f"提示词 `{spec.prompt_id}` 缺少模板变量 `{missing}`。文件: {spec.file_path}"
        ) from exc


def build_messages(spec: PromptSpec, **variables: Any) -> list[dict[str, str]]:
    user_content = render_user_template(spec, **variables)
    return [
        {"role": "system", "content": spec.system_prompt},
        {"role": "user", "content": user_content},
    ]
