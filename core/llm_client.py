from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from core.config import AppConfig


@dataclass
class LLMResponse:
    raw: dict[str, Any]
    content: str
    usage: dict[str, Any] | None = None


class OpenAICompatClient:
    def __init__(self, config: AppConfig, logger) -> None:
        self.config = config
        self.logger = logger
        self.endpoint = f"{config.base_url.rstrip('/')}/chat/completions"

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": model or self.config.model_default,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        data = self._post_json(payload)
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM 返回缺少 choices 字段: {data}")

        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        if isinstance(content, list):
            # Some gateways return list-of-parts.
            content = "\n".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )

        return LLMResponse(raw=data, content=str(content), usage=data.get("usage"))

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }

        backoff = self.config.retry_backoff_seconds
        last_error: Exception | None = None

        for attempt in range(1, self.config.max_retries + 1):
            req = urllib.request.Request(
                self.endpoint,
                data=request_body,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.config.request_timeout) as response:
                    raw = response.read().decode("utf-8")
                    return json.loads(raw)
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="ignore")
                self.logger.warning(
                    "LLM HTTPError attempt=%s status=%s body=%s",
                    attempt,
                    e.code,
                    body[:500],
                )
                if e.code in {429, 500, 502, 503, 504} and attempt < self.config.max_retries:
                    time.sleep(backoff * attempt)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                last_error = e
                self.logger.warning("LLM request failed attempt=%s error=%s", attempt, e)
                if attempt < self.config.max_retries:
                    time.sleep(backoff * attempt)
                    continue
                break

        raise RuntimeError(f"LLM 请求失败，重试耗尽: {last_error}")
