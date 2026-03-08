from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from core.config import AppConfig

try:
    from litellm import Router, acompletion, completion
except Exception:  # noqa: BLE001
    Router = None
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
        self._router_cache: dict[tuple[str, str, tuple[str, ...]], tuple[Any, str]] = {}

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
        response_format: dict[str, Any] | None,
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
        if response_format is not None:
            payload["response_format"] = response_format
        return payload

    def _normalize_api_keys(
        self,
        *,
        api_key: str | None,
        api_keys: list[str] | tuple[str, ...] | None,
    ) -> tuple[str, ...]:
        seen: set[str] = set()
        normalized: list[str] = []
        for key in api_keys or ():
            raw = str(key or "").strip()
            if not raw or raw in seen:
                continue
            seen.add(raw)
            normalized.append(raw)
        single = (api_key or "").strip()
        if single and single not in seen:
            normalized.append(single)
            seen.add(single)
        if not normalized and str(self.config.api_key or "").strip():
            normalized.append(str(self.config.api_key).strip())
        return tuple(normalized)

    def _get_router(
        self,
        *,
        model: str,
        api_base: str,
        api_keys: tuple[str, ...],
    ) -> tuple[Any, str]:
        cache_key = (model, api_base, api_keys)
        cached = self._router_cache.get(cache_key)
        if cached is not None:
            return cached
        if Router is None:
            raise RuntimeError("当前 litellm 不支持 Router，无法启用多 key 负载均衡。")

        seed = "|".join((model, api_base, *api_keys))
        model_group = f"pool_{hashlib.sha1(seed.encode('utf-8')).hexdigest()[:12]}"
        model_list = [
            {
                "model_name": model_group,
                "litellm_params": {
                    "model": model,
                    "custom_llm_provider": "openai",
                    "api_key": key,
                    "api_base": api_base,
                    "num_retries": self.config.max_retries,
                    "timeout": self.config.request_timeout,
                },
            }
            for key in api_keys
        ]
        router = Router(
            model_list=model_list,
            routing_strategy="simple-shuffle",
            num_retries=self.config.max_retries,
            timeout=self.config.request_timeout,
        )
        cached_pair = (router, model_group)
        self._router_cache[cache_key] = cached_pair
        return cached_pair

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        api_key: str | None = None,
        api_keys: list[str] | tuple[str, ...] | None = None,
        api_base: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        chosen_model = model or self.config.model_default
        chosen_api_base = api_base or self.config.base_url
        normalized_keys = self._normalize_api_keys(api_key=api_key, api_keys=api_keys)
        if len(normalized_keys) > 1 and Router is not None:
            router, model_group = self._get_router(
                model=chosen_model,
                api_base=chosen_api_base,
                api_keys=normalized_keys,
            )
            router_kwargs: dict[str, Any] = {
                "model": model_group,
                "messages": messages,
                "temperature": temperature,
                "timeout": self.config.request_timeout,
                "num_retries": self.config.max_retries,
            }
            if max_tokens is not None:
                router_kwargs["max_tokens"] = max_tokens
            if response_format is not None:
                router_kwargs["response_format"] = response_format
            response_obj = router.completion(**router_kwargs)
            data = _response_to_dict(response_obj)
            content = _extract_content(data)
            usage = _normalize_usage(data.get("usage"))
            return LLMResponse(raw=data, content=content, usage=usage)

        payload = self._build_payload(
            messages,
            model=chosen_model,
            api_key=normalized_keys[0] if normalized_keys else api_key,
            api_base=chosen_api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
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
        api_keys: list[str] | tuple[str, ...] | None = None,
        api_base: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        chosen_model = model or self.config.model_default
        chosen_api_base = api_base or self.config.base_url
        normalized_keys = self._normalize_api_keys(api_key=api_key, api_keys=api_keys)
        if len(normalized_keys) > 1 and Router is not None:
            router, model_group = self._get_router(
                model=chosen_model,
                api_base=chosen_api_base,
                api_keys=normalized_keys,
            )
            router_kwargs: dict[str, Any] = {
                "model": model_group,
                "messages": messages,
                "temperature": temperature,
                "timeout": self.config.request_timeout,
                "num_retries": self.config.max_retries,
            }
            if max_tokens is not None:
                router_kwargs["max_tokens"] = max_tokens
            if response_format is not None:
                router_kwargs["response_format"] = response_format
            response_obj = await router.acompletion(**router_kwargs)
            data = _response_to_dict(response_obj)
            content = _extract_content(data)
            usage = _normalize_usage(data.get("usage"))
            return LLMResponse(raw=data, content=content, usage=usage)

        payload = self._build_payload(
            messages,
            model=chosen_model,
            api_key=normalized_keys[0] if normalized_keys else api_key,
            api_base=chosen_api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        response_obj = await acompletion(**payload)
        data = _response_to_dict(response_obj)
        content = _extract_content(data)
        usage = _normalize_usage(data.get("usage"))
        return LLMResponse(raw=data, content=content, usage=usage)


# Backward compatibility for existing imports.
OpenAICompatClient = LiteLLMClient
