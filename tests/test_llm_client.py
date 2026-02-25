from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core import llm_client as llm_client_module
from core.llm_client import LiteLLMClient


def _mock_response(content: str = "ok") -> dict:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


class LiteLLMClientTests(unittest.TestCase):
    def _build_client(self) -> LiteLLMClient:
        config = SimpleNamespace(
            model_default="default-model",
            api_key="default-key",
            base_url="https://default.example.com/v1",
            max_retries=3,
            request_timeout=180,
        )
        return LiteLLMClient(config=config, logger=None)

    def test_chat_uses_request_level_api_overrides(self) -> None:
        captured: dict[str, str] = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return _mock_response("sync-ok")

        async def fake_acompletion(**kwargs):
            return _mock_response("async-ok")

        with patch.object(llm_client_module, "completion", fake_completion), patch.object(
            llm_client_module,
            "acompletion",
            fake_acompletion,
        ):
            client = self._build_client()
            result = client.chat(
                [{"role": "user", "content": "hello"}],
                model="stage-model",
                api_key="stage-key",
                api_base="https://stage.example.com/v1",
                temperature=0.0,
            )

        self.assertEqual(result.content, "sync-ok")
        self.assertEqual(captured["model"], "stage-model")
        self.assertEqual(captured["api_key"], "stage-key")
        self.assertEqual(captured["api_base"], "https://stage.example.com/v1")

    def test_chat_falls_back_to_default_api_settings(self) -> None:
        captured: dict[str, str] = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return _mock_response("sync-ok")

        async def fake_acompletion(**kwargs):
            return _mock_response("async-ok")

        with patch.object(llm_client_module, "completion", fake_completion), patch.object(
            llm_client_module,
            "acompletion",
            fake_acompletion,
        ):
            client = self._build_client()
            result = client.chat([{"role": "user", "content": "hello"}], temperature=0.1)

        self.assertEqual(result.content, "sync-ok")
        self.assertEqual(captured["model"], "default-model")
        self.assertEqual(captured["api_key"], "default-key")
        self.assertEqual(captured["api_base"], "https://default.example.com/v1")

    def test_achat_supports_request_level_overrides(self) -> None:
        captured: dict[str, str] = {}

        def fake_completion(**kwargs):
            return _mock_response("sync-ok")

        async def fake_acompletion(**kwargs):
            captured.update(kwargs)
            return _mock_response("async-ok")

        with patch.object(llm_client_module, "completion", fake_completion), patch.object(
            llm_client_module,
            "acompletion",
            fake_acompletion,
        ):
            client = self._build_client()
            result = asyncio.run(
                client.achat(
                    [{"role": "user", "content": "hello"}],
                    model="async-model",
                    api_key="async-key",
                    api_base="https://async.example.com/v1",
                    temperature=0.0,
                )
            )

        self.assertEqual(result.content, "async-ok")
        self.assertEqual(captured["model"], "async-model")
        self.assertEqual(captured["api_key"], "async-key")
        self.assertEqual(captured["api_base"], "https://async.example.com/v1")


if __name__ == "__main__":
    unittest.main()
