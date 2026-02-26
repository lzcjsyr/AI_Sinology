from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from unittest.mock import patch

from core.config import AppConfig, PIPELINE_LLM_CONFIG


ENV_KEYS_TO_CLEAR = [
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
    "ALIYUN_API_KEY",
    "SILICONFLOW_BASE_URL",
    "OPENROUTER_BASE_URL",
    "VOLCENGINE_BASE_URL",
    "ALIYUN_BASE_URL",
    "STAGE2_LLM1_CONCURRENCY",
    "STAGE2_LLM2_CONCURRENCY",
    "STAGE2_ARBITRATION_CONCURRENCY",
    "STAGE1_RPM",
    "STAGE1_TPM",
    "STAGE2_LLM1_RPM",
    "STAGE2_LLM1_TPM",
    "STAGE2_LLM2_RPM",
    "STAGE2_LLM2_TPM",
    "STAGE2_LLM3_RPM",
    "STAGE2_LLM3_TPM",
    "STAGE3_RPM",
    "STAGE3_TPM",
    "STAGE4_RPM",
    "STAGE4_TPM",
    "STAGE5_RPM",
    "STAGE5_TPM",
    "STAGE2_SYNC_HEADROOM",
    "STAGE2_SYNC_MAX_AHEAD",
    "STAGE2_SYNC_MODE",
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
            ALIYUN_API_KEY=aliyun_key
            """
        )

        self.assertEqual(config.stage1_llm.provider, "siliconflow")
        self.assertEqual(config.stage2_llm1.provider, "aliyun")
        self.assertEqual(config.stage2_llm2.provider, "volcengine")
        self.assertEqual(config.stage2_llm3.provider, "volcengine")
        self.assertEqual(config.stage2_llm1.api_key, "aliyun_key")
        self.assertEqual(config.stage2_llm2.api_key, "ve_key")
        self.assertEqual(config.stage2_llm3.api_key, "ve_key")
        self.assertEqual(config.provider_base_urls["siliconflow"], "https://api.siliconflow.cn/v1")
        self.assertEqual(config.provider_base_urls["openrouter"], "https://openrouter.ai/api/v1")
        self.assertEqual(
            config.provider_base_urls["volcengine"],
            "https://ark.cn-beijing.volces.com/api/v3",
        )
        self.assertEqual(
            config.provider_base_urls["aliyun"],
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.assertIsNone(config.stage2_llm1_concurrency)
        self.assertIsNone(config.stage2_llm2_concurrency)
        self.assertIsNone(config.stage2_arbitration_concurrency)
        self.assertAlmostEqual(config.stage2_sync_headroom, 0.85)
        self.assertEqual(config.stage2_sync_max_ahead, 128)
        self.assertEqual(config.stage2_sync_mode, "lowest_shared")
        self.assertEqual(config.stage2_llm1.rpm, 30000)
        self.assertEqual(config.stage2_llm1.tpm, 10000000)
        self.assertEqual(config.stage2_llm2.rpm, 30000)
        self.assertEqual(config.stage2_llm2.tpm, 5000000)
        self.assertEqual(config.stage2_fragment_max_attempts, 3)
        self.assertEqual(config.stage2_max_empty_retries, 2)

    def test_validate_api_reports_missing_provider_keys(self) -> None:
        config = self._load_with_env_file(
            """
            OPENROUTER_API_KEY=or_key_only
            """
        )
        with self.assertRaises(ValueError) as ctx:
            config.validate_api()
        msg = str(ctx.exception)
        self.assertIn("stage1", msg)
        self.assertIn("SILICONFLOW_API_KEY", msg)

    def test_stage2_runtime_values_can_be_overridden(self) -> None:
        config = self._load_with_env_file(
            """
            SILICONFLOW_API_KEY=sf_key
            OPENROUTER_API_KEY=or_key
            VOLCENGINE_API_KEY=ve_key
            ALIYUN_API_KEY=aliyun_key
            STAGE2_LLM1_CONCURRENCY=11
            STAGE2_LLM2_CONCURRENCY=13
            STAGE2_ARBITRATION_CONCURRENCY=7
            STAGE2_SYNC_HEADROOM=0.9
            STAGE2_SYNC_MAX_AHEAD=64
            STAGE2_FRAGMENT_MAX_ATTEMPTS=5
            STAGE2_MAX_EMPTY_RETRIES=4
            STAGE2_LLM1_RPM=1200
            STAGE2_LLM1_TPM=240000
            STAGE2_LLM2_RPM=32000
            STAGE2_LLM2_TPM=5200000
            """
        )

        self.assertEqual(config.stage2_llm1_concurrency, 11)
        self.assertEqual(config.stage2_llm2_concurrency, 13)
        self.assertEqual(config.stage2_arbitration_concurrency, 7)
        self.assertAlmostEqual(config.stage2_sync_headroom, 0.9)
        self.assertEqual(config.stage2_sync_max_ahead, 64)
        self.assertEqual(config.stage2_fragment_max_attempts, 5)
        self.assertEqual(config.stage2_max_empty_retries, 4)
        self.assertEqual(config.stage2_llm1.rpm, 1200)
        self.assertEqual(config.stage2_llm1.tpm, 240000)
        self.assertEqual(config.stage2_llm2.rpm, 32000)
        self.assertEqual(config.stage2_llm2.tpm, 5200000)

    def test_stage2_max_empty_retries_allows_zero(self) -> None:
        config = self._load_with_env_file(
            """
            SILICONFLOW_API_KEY=sf_key
            OPENROUTER_API_KEY=or_key
            VOLCENGINE_API_KEY=ve_key
            ALIYUN_API_KEY=aliyun_key
            STAGE2_MAX_EMPTY_RETRIES=0
            """
        )
        self.assertEqual(config.stage2_max_empty_retries, 0)

    def test_missing_stage_rate_limit_in_pipeline_config_raises(self) -> None:
        with patch.dict(
            PIPELINE_LLM_CONFIG,
            {"stage2_llm1": {"provider": "siliconflow", "model": "deepseek-ai/DeepSeek-V3.2"}},
            clear=False,
        ):
            with self.assertRaises(ValueError) as ctx:
                self._load_with_env_file(
                    """
                    SILICONFLOW_API_KEY=sf_key
                    OPENROUTER_API_KEY=or_key
                    VOLCENGINE_API_KEY=ve_key
                    """
                )
        self.assertIn("stage2_llm1", str(ctx.exception))
        self.assertIn("rpm", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
