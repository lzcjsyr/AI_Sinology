from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.config import AppConfig

try:
    from litellm import acompletion, completion
except Exception:  # noqa: BLE001
    acompletion = None
    completion = None


@dataclass
class LLMResponse:
    raw: dict[str, Any]
    content: str
    usage: dict[str, Any] | None = None


def _response_to_dict(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    return dict(response)


def _normalize_usage(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if hasattr(usage, "dict"):
        return usage.dict()
    return None


def _extract_content(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"LLM 返回缺少 choices 字段: {data}")

    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
            else:
                parts.append(str(part))
        content = "\n".join(parts)
    return str(content)


class LiteLLMClient:
    def __init__(self, config: AppConfig, logger) -> None:
        self.config = config
        self.logger = logger

        if completion is None or acompletion is None:
            raise RuntimeError(
                "未安装 litellm。请先安装后再运行：`python3 -m pip install litellm`"
            )

    def _build_payload(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None,
        api_key: str | None,
        api_base: str | None,
        temperature: float,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model or self.config.model_default,
            "messages": messages,
            "temperature": temperature,
            "custom_llm_provider": "openai",
            "api_key": api_key or self.config.api_key,
            "api_base": api_base or self.config.base_url,
            "num_retries": self.config.max_retries,
            "timeout": self.config.request_timeout,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return payload

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        payload = self._build_payload(
            messages,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response_obj = completion(**payload)
        data = _response_to_dict(response_obj)
        content = _extract_content(data)
        usage = _normalize_usage(data.get("usage"))
        return LLMResponse(raw=data, content=content, usage=usage)

    async def achat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        payload = self._build_payload(
            messages,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response_obj = await acompletion(**payload)
        data = _response_to_dict(response_obj)
        content = _extract_content(data)
        usage = _normalize_usage(data.get("usage"))
        return LLMResponse(raw=data, content=content, usage=usage)


# Backward compatibility for existing imports.
OpenAICompatClient = LiteLLMClient
