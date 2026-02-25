from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from core.config import AppConfig


ENV_KEYS_TO_CLEAR = [
    "API_KEY",
    "BASE_URL",
    "MODEL",
    "MODEL_STAGE1",
    "MODEL_STAGE3",
    "MODEL_STAGE4",
    "MODEL_STAGE5",
    "MODEL_LLM1",
    "MODEL_LLM2",
    "MODEL_LLM3",
    "SILICONFLOW_API_KEY",
    "OPENROUTER_API_KEY",
    "VOLCENGINE_API_KEY",
    "SILICONFLOW_BASE_URL",
    "OPENROUTER_BASE_URL",
    "VOLCENGINE_BASE_URL",
    "STAGE2_CONCURRENCY",
    "STAGE2_FRAGMENT_MAX_ATTEMPTS",
    "STAGE2_MAX_EMPTY_RETRIES",
]


class AppConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env_backup = {key: os.environ.get(key) for key in ENV_KEYS_TO_CLEAR}
        for key in ENV_KEYS_TO_CLEAR:
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        for key in ENV_KEYS_TO_CLEAR:
            value = self._env_backup.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _load_with_env_file(self, env_content: str) -> AppConfig:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".env").write_text(env_content.strip() + "\n", encoding="utf-8")
            return AppConfig.load(root)

    def test_loads_stage_endpoints_and_defaults(self) -> None:
        config = self._load_with_env_file(
            """
            SILICONFLOW_API_KEY=sf_key
            OPENROUTER_API_KEY=or_key
            VOLCENGINE_API_KEY=ve_key
            """
        )

        self.assertEqual(config.stage1_llm.provider, "siliconflow")
        self.assertEqual(config.stage2_llm1.provider, "siliconflow")
        self.assertEqual(config.stage2_llm2.provider, "openrouter")
        self.assertEqual(config.stage2_llm3.provider, "volcengine")
        self.assertEqual(config.stage2_llm1.api_key, "sf_key")
        self.assertEqual(config.stage2_llm2.api_key, "or_key")
        self.assertEqual(config.stage2_llm3.api_key, "ve_key")
        self.assertEqual(config.provider_base_urls["siliconflow"], "https://api.siliconflow.cn/v1")
        self.assertEqual(config.provider_base_urls["openrouter"], "https://openrouter.ai/api/v1")
        self.assertEqual(
            config.provider_base_urls["volcengine"],
            "https://ark.cn-beijing.volces.com/api/v3",
        )
        self.assertEqual(config.stage2_screening_concurrency, 4)
        self.assertEqual(config.stage2_fragment_max_attempts, 3)
        self.assertEqual(config.stage2_max_empty_retries, 2)

    def test_validate_api_reports_missing_provider_keys(self) -> None:
        config = self._load_with_env_file(
            """
            SILICONFLOW_API_KEY=sf_key_only
            """
        )
        with self.assertRaises(ValueError) as ctx:
            config.validate_api()
        msg = str(ctx.exception)
        self.assertIn("stage2_llm2", msg)
        self.assertIn("OPENROUTER_API_KEY", msg)
        self.assertIn("stage2_llm3", msg)
        self.assertIn("VOLCENGINE_API_KEY", msg)

    def test_legacy_api_key_fallback_allows_validation(self) -> None:
        config = self._load_with_env_file(
            """
            API_KEY=legacy_shared_key
            """
        )
        config.validate_api()
        self.assertEqual(config.stage2_llm1.api_key, "legacy_shared_key")
        self.assertEqual(config.stage2_llm2.api_key, "legacy_shared_key")
        self.assertEqual(config.stage2_llm3.api_key, "legacy_shared_key")

    def test_stage2_runtime_values_can_be_overridden(self) -> None:
        config = self._load_with_env_file(
            """
            SILICONFLOW_API_KEY=sf_key
            OPENROUTER_API_KEY=or_key
            VOLCENGINE_API_KEY=ve_key
            STAGE2_CONCURRENCY=9
            STAGE2_FRAGMENT_MAX_ATTEMPTS=5
            STAGE2_MAX_EMPTY_RETRIES=4
            """
        )

        self.assertEqual(config.stage2_screening_concurrency, 9)
        self.assertEqual(config.stage2_fragment_max_attempts, 5)
        self.assertEqual(config.stage2_max_empty_retries, 4)

    def test_stage2_max_empty_retries_allows_zero(self) -> None:
        config = self._load_with_env_file(
            """
            SILICONFLOW_API_KEY=sf_key
            OPENROUTER_API_KEY=or_key
            VOLCENGINE_API_KEY=ve_key
            STAGE2_MAX_EMPTY_RETRIES=0
            """
        )
        self.assertEqual(config.stage2_max_empty_retries, 0)


if __name__ == "__main__":
    unittest.main()
