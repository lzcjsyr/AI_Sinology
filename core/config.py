from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


def _load_dotenv(dotenv_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not dotenv_path.exists():
        return values

    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        values[key] = value
    return values


# ==============================
# 用户可编辑配置区（优先修改这里）
# ==============================
#
# 说明：
# 1) 这里负责“各阶段用哪个 provider + 哪个 model”。
# 2) API Key 不写在这里，而写在 .env（见 .env.example）。
# 3) 三个 provider 的默认 base_url 已内置，可按需改成私有网关。
#
PROVIDER_DEFAULT_BASE_URLS: dict[str, str] = {
    "siliconflow": "https://api.siliconflow.cn/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "volcengine": "https://ark.cn-beijing.volces.com/api/v3",
}

PIPELINE_LLM_CONFIG: dict[str, dict[str, str]] = {
    "stage1": {
        "provider": "siliconflow",
        "model": "deepseek-ai/DeepSeek-V3.2",
    },
    "stage2_llm1": {
        "provider": "siliconflow",
        "model": "deepseek-ai/DeepSeek-V3.2",
    },
    "stage2_llm2": {
        "provider": "siliconflow",
        "model": "deepseek-ai/DeepSeek-V3.2",
    },
    "stage2_llm3": {
        "provider": "siliconflow",
        "model": "deepseek-ai/DeepSeek-V3.2",
    },
    "stage3": {
        "provider": "siliconflow",
        "model": "deepseek-ai/DeepSeek-V3.2",
    },
    "stage4": {
        "provider": "siliconflow",
        "model": "deepseek-ai/DeepSeek-V3.2",
    },
    "stage5": {
        "provider": "siliconflow",
        "model": "deepseek-ai/DeepSeek-V3.2",
    },
}

STAGE2_RUNTIME_DEFAULTS: dict[str, int] = {
    "screening_concurrency": 4,
    "fragment_max_attempts": 3,
    "max_empty_retries": 2,
}

# ==============================
# 非用户配置：解析与兼容逻辑
# ==============================

PROVIDER_API_KEY_ENV_NAMES: dict[str, str] = {
    "siliconflow": "SILICONFLOW_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "volcengine": "VOLCENGINE_API_KEY",
}

STAGE_MODEL_ENV_OVERRIDES: dict[str, str] = {
    "stage1": "MODEL_STAGE1",
    "stage2_llm1": "MODEL_LLM1",
    "stage2_llm2": "MODEL_LLM2",
    "stage2_llm3": "MODEL_LLM3",
    "stage3": "MODEL_STAGE3",
    "stage4": "MODEL_STAGE4",
    "stage5": "MODEL_STAGE5",
}


@dataclass(frozen=True)
class LLMEndpointConfig:
    stage: str
    provider: str
    model: str
    base_url: str
    api_key: str

    def as_client_kwargs(self) -> dict[str, str]:
        return {
            "model": self.model,
            "api_key": self.api_key,
            "api_base": self.base_url,
        }


def _as_int_with_min(value: str | None, default: int, min_value: int) -> int:
    if value is None or not str(value).strip():
        return default
    parsed = int(value)
    if parsed < min_value:
        raise ValueError(f"配置值必须 >= {min_value}，实际为: {value}")
    return parsed


def _normalize_provider(name: str) -> str:
    provider = (name or "").strip().lower()
    aliases = {
        "volcengine-ark": "volcengine",
        "ark": "volcengine",
        "火山引擎": "volcengine",
    }
    return aliases.get(provider, provider)


def _build_stage_endpoint(
    *,
    stage: str,
    pick,
    provider_base_urls: dict[str, str],
    provider_api_keys: dict[str, str],
    fallback_model: str,
    legacy_api_key: str,
) -> LLMEndpointConfig:
    spec: dict[str, Any] = dict(PIPELINE_LLM_CONFIG.get(stage) or {})

    provider = _normalize_provider(str(spec.get("provider") or ""))
    if provider not in provider_base_urls:
        available = ", ".join(sorted(provider_base_urls.keys()))
        raise ValueError(f"阶段 `{stage}` 的 provider 无效: `{provider}`。可选: {available}")

    default_model = str(spec.get("model") or fallback_model).strip() or fallback_model
    model_override_env = STAGE_MODEL_ENV_OVERRIDES.get(stage)
    model = pick(model_override_env, default_model) if model_override_env else default_model
    model = (model or default_model).strip()

    return LLMEndpointConfig(
        stage=stage,
        provider=provider,
        model=model,
        base_url=provider_base_urls[provider],
        api_key=provider_api_keys.get(provider) or legacy_api_key,
    )


@dataclass(frozen=True)
class AppConfig:
    root_dir: Path
    data_dir: Path
    outputs_dir: Path
    kanripo_dir: Path
    request_timeout: int
    max_retries: int
    retry_backoff_seconds: float
    default_max_fragments: Optional[int]
    provider_base_urls: dict[str, str]
    provider_api_keys: dict[str, str]
    api_key: str
    base_url: str
    model_default: str
    stage1_llm: LLMEndpointConfig
    stage2_llm1: LLMEndpointConfig
    stage2_llm2: LLMEndpointConfig
    stage2_llm3: LLMEndpointConfig
    stage3_llm: LLMEndpointConfig
    stage4_llm: LLMEndpointConfig
    stage5_llm: LLMEndpointConfig
    stage2_screening_concurrency: int
    stage2_fragment_max_attempts: int
    stage2_max_empty_retries: int

    @classmethod
    def load(cls, root_dir: Path) -> "AppConfig":
        env_file_values = _load_dotenv(root_dir / ".env")

        def pick(name: str, default: Optional[str] = None) -> Optional[str]:
            return os.getenv(name) or env_file_values.get(name) or default

        legacy_api_key = pick("API_KEY", "")
        model_default = pick("MODEL", PIPELINE_LLM_CONFIG["stage1"]["model"]) or PIPELINE_LLM_CONFIG[
            "stage1"
        ]["model"]

        provider_base_urls = {
            provider: (
                pick(f"{provider.upper()}_BASE_URL", default_url) or default_url
            ).rstrip("/")
            for provider, default_url in PROVIDER_DEFAULT_BASE_URLS.items()
        }
        provider_api_keys = {
            provider: pick(env_name, "") or ""
            for provider, env_name in PROVIDER_API_KEY_ENV_NAMES.items()
        }

        legacy_base_url = (
            pick("BASE_URL", provider_base_urls[PIPELINE_LLM_CONFIG["stage1"]["provider"]])
            or provider_base_urls[PIPELINE_LLM_CONFIG["stage1"]["provider"]]
        ).rstrip("/")

        raw_limit = pick("MAX_FRAGMENTS", "")
        if raw_limit:
            try:
                default_max_fragments: Optional[int] = int(raw_limit)
            except ValueError:
                default_max_fragments = None
        else:
            default_max_fragments = None

        stage1_llm = _build_stage_endpoint(
            stage="stage1",
            pick=pick,
            provider_base_urls=provider_base_urls,
            provider_api_keys=provider_api_keys,
            fallback_model=model_default,
            legacy_api_key=legacy_api_key,
        )
        stage2_llm1 = _build_stage_endpoint(
            stage="stage2_llm1",
            pick=pick,
            provider_base_urls=provider_base_urls,
            provider_api_keys=provider_api_keys,
            fallback_model=model_default,
            legacy_api_key=legacy_api_key,
        )
        stage2_llm2 = _build_stage_endpoint(
            stage="stage2_llm2",
            pick=pick,
            provider_base_urls=provider_base_urls,
            provider_api_keys=provider_api_keys,
            fallback_model=model_default,
            legacy_api_key=legacy_api_key,
        )
        stage2_llm3 = _build_stage_endpoint(
            stage="stage2_llm3",
            pick=pick,
            provider_base_urls=provider_base_urls,
            provider_api_keys=provider_api_keys,
            fallback_model=model_default,
            legacy_api_key=legacy_api_key,
        )
        stage3_llm = _build_stage_endpoint(
            stage="stage3",
            pick=pick,
            provider_base_urls=provider_base_urls,
            provider_api_keys=provider_api_keys,
            fallback_model=model_default,
            legacy_api_key=legacy_api_key,
        )
        stage4_llm = _build_stage_endpoint(
            stage="stage4",
            pick=pick,
            provider_base_urls=provider_base_urls,
            provider_api_keys=provider_api_keys,
            fallback_model=model_default,
            legacy_api_key=legacy_api_key,
        )
        stage5_llm = _build_stage_endpoint(
            stage="stage5",
            pick=pick,
            provider_base_urls=provider_base_urls,
            provider_api_keys=provider_api_keys,
            fallback_model=model_default,
            legacy_api_key=legacy_api_key,
        )

        return cls(
            root_dir=root_dir,
            data_dir=root_dir / "data",
            outputs_dir=root_dir / "outputs",
            kanripo_dir=(root_dir / "data" / "kanripo_repos"),
            request_timeout=int(pick("REQUEST_TIMEOUT", "180") or "180"),
            max_retries=int(pick("MAX_RETRIES", "3") or "3"),
            retry_backoff_seconds=float(pick("RETRY_BACKOFF_SECONDS", "2.0") or "2.0"),
            default_max_fragments=default_max_fragments,
            provider_base_urls=provider_base_urls,
            provider_api_keys=provider_api_keys,
            api_key=legacy_api_key or stage1_llm.api_key,
            base_url=legacy_base_url,
            model_default=model_default,
            stage1_llm=stage1_llm,
            stage2_llm1=stage2_llm1,
            stage2_llm2=stage2_llm2,
            stage2_llm3=stage2_llm3,
            stage3_llm=stage3_llm,
            stage4_llm=stage4_llm,
            stage5_llm=stage5_llm,
            stage2_screening_concurrency=_as_int_with_min(
                pick(
                    "STAGE2_CONCURRENCY",
                    str(STAGE2_RUNTIME_DEFAULTS["screening_concurrency"]),
                ),
                STAGE2_RUNTIME_DEFAULTS["screening_concurrency"],
                1,
            ),
            stage2_fragment_max_attempts=_as_int_with_min(
                pick(
                    "STAGE2_FRAGMENT_MAX_ATTEMPTS",
                    str(STAGE2_RUNTIME_DEFAULTS["fragment_max_attempts"]),
                ),
                STAGE2_RUNTIME_DEFAULTS["fragment_max_attempts"],
                1,
            ),
            stage2_max_empty_retries=_as_int_with_min(
                pick(
                    "STAGE2_MAX_EMPTY_RETRIES",
                    str(STAGE2_RUNTIME_DEFAULTS["max_empty_retries"]),
                ),
                STAGE2_RUNTIME_DEFAULTS["max_empty_retries"],
                0,
            ),
        )

    def validate_api(self) -> None:
        stage_endpoints = [
            self.stage1_llm,
            self.stage2_llm1,
            self.stage2_llm2,
            self.stage2_llm3,
            self.stage3_llm,
            self.stage4_llm,
            self.stage5_llm,
        ]
        problems: list[str] = []
        for endpoint in stage_endpoints:
            if not endpoint.model.strip():
                problems.append(f"阶段 `{endpoint.stage}` 缺少 model。")
            if not endpoint.base_url.strip():
                problems.append(f"阶段 `{endpoint.stage}` 缺少 base_url。")
            if not endpoint.api_key.strip():
                key_name = PROVIDER_API_KEY_ENV_NAMES.get(endpoint.provider, "API_KEY")
                problems.append(
                    f"阶段 `{endpoint.stage}` 使用 provider `{endpoint.provider}`，"
                    f"但缺少密钥 `{key_name}`（或通用 `API_KEY`）。"
                )
        if problems:
            raise ValueError("LLM 配置不完整：\n- " + "\n- ".join(problems))
