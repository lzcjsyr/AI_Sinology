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
# 3) provider 的默认 base_url 已内置，可按需改成私有网关。
#
PROVIDER_DEFAULT_BASE_URLS: dict[str, str] = {
    "siliconflow": "https://api.siliconflow.cn/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "volcengine": "https://ark.cn-beijing.volces.com/api/v3",
    "aliyun": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}

PIPELINE_LLM_CONFIG: dict[str, dict[str, Any]] = {
    "stage1": {
        "provider": "siliconflow",
        "model": "deepseek-ai/DeepSeek-V3.2",
        "rpm": 1000,
        "tpm": 100000,
    },
    "stage2_llm1": {
        "provider": "volcengine",
        "model": "doubao-seed-2-0-lite-260215",
        "rpm": 30000,
        "tpm": 5000000,
    },
    "stage2_llm2": {
        "provider": "volcengine",
        "model": "doubao-seed-2-0-mini-260215",
        "rpm": 30000,
        "tpm": 5000000,
    },
    "stage2_llm3": {
        "provider": "volcengine",
        "model": "doubao-seed-2-0-pro-260215",
        "rpm": 30000,
        "tpm": 5000000,
    },
    "stage3": {
        "provider": "siliconflow",
        "model": "deepseek-ai/DeepSeek-V3.2",
        "rpm": 1000,
        "tpm": 100000,
    },
    "stage4": {
        "provider": "siliconflow",
        "model": "deepseek-ai/DeepSeek-V3.2",
        "rpm": 1000,
        "tpm": 100000,
    },
    "stage5": {
        "provider": "siliconflow",
        "model": "deepseek-ai/DeepSeek-V3.2",
        "rpm": 1000,
        "tpm": 100000,
    },
}

STAGE2_RUNTIME_DEFAULTS: dict[str, Any] = {
    # ---------------- 2.2 阶段：并发与资源控制 ----------------
    # LLM1 在 2.2 阶段执行片段筛选时的并发请求数；留空表示自动计算。
    "llm1_concurrency": None,
    # LLM2 在 2.2 阶段执行片段筛选时的并发请求数；留空表示自动计算。
    "llm2_concurrency": None,
    
    # ---------------- 2.3 阶段：并发与资源控制 ----------------
    # LLM3 在 2.3 阶段执行交叉验证（针对差异结果进行仲裁）时的并发请求数；
    # 留空表示自动计算。
    "arbitration_concurrency": None,
    
    # ---------------- 容错与重试机制 ----------------
    # 单个片段在单次遇到 LLM 请求失败（网络错误、超时等）时的最大重试次数
    "fragment_max_attempts": 3,
    # 阶段级最大空跑重试次数：如果整个阶段二跑完未产生有效结果集（2_final_corpus），
    # 系统允许自动扩增碎片池并重跑的最大次数
    "max_empty_retries": 2,
    
    # ---------------- 交叉验证同步机制 ----------------
    # 速率同步池余量 (0~1)：设定为 0.85 即真实使用双方限制下限（如较弱平台的 RPM）的 85%，
    # 余下 15% 用来防止突然触发 Provider 端真实硬限流
    "sync_headroom": 0.85,
    # 双模型同步时，允许领先方（进度较快的 LLM）最多领先落后方多少个片段
    # 超过此数值领先方将会阻塞等待，确保双模型处理进度相近
    "sync_max_ahead": 128,
    # 交叉并发运行时的速率同步模式：
    # "lowest_shared" 表示双模型统一被限制在两者中最弱的 RPM/TPM 标准下
    "sync_mode": "lowest_shared",
}

# ==============================
# 非用户配置：解析与兼容逻辑
# ==============================

PROVIDER_API_KEY_ENV_NAMES: dict[str, str] = {
    "siliconflow": "SILICONFLOW_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "volcengine": "VOLCENGINE_API_KEY",
    "aliyun": "ALIYUN_API_KEY",
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

STAGE_RATE_LIMIT_ENV_OVERRIDES: dict[str, tuple[str, str]] = {
    "stage1": ("STAGE1_RPM", "STAGE1_TPM"),
    "stage2_llm1": ("STAGE2_LLM1_RPM", "STAGE2_LLM1_TPM"),
    "stage2_llm2": ("STAGE2_LLM2_RPM", "STAGE2_LLM2_TPM"),
    "stage2_llm3": ("STAGE2_LLM3_RPM", "STAGE2_LLM3_TPM"),
    "stage3": ("STAGE3_RPM", "STAGE3_TPM"),
    "stage4": ("STAGE4_RPM", "STAGE4_TPM"),
    "stage5": ("STAGE5_RPM", "STAGE5_TPM"),
}


@dataclass(frozen=True)
class LLMEndpointConfig:
    stage: str
    provider: str
    model: str
    base_url: str
    api_key: str
    rpm: int
    tpm: int

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


def _as_optional_int_with_min(value: str | None, min_value: int) -> int | None:
    if value is None or not str(value).strip():
        return None
    parsed = int(value)
    if parsed < min_value:
        raise ValueError(f"配置值必须 >= {min_value}，实际为: {value}")
    return parsed


def _as_float_in_range(value: str | None, default: float, min_value: float, max_value: float) -> float:
    if value is None or not str(value).strip():
        return default
    parsed = float(value)
    if parsed < min_value or parsed > max_value:
        raise ValueError(
            f"配置值必须在 [{min_value}, {max_value}] 范围内，实际为: {value}"
        )
    return parsed


def _normalize_provider(name: str) -> str:
    provider = (name or "").strip().lower()
    aliases = {
        "volcengine-ark": "volcengine",
        "ark": "volcengine",
        "火山引擎": "volcengine",
        "dashscope": "aliyun",
        "alibaba": "aliyun",
        "aliyun-dashscope": "aliyun",
        "阿里云": "aliyun",
    }
    return aliases.get(provider, provider)


def _build_stage_endpoint(
    *,
    stage: str,
    pick,
    provider_base_urls: dict[str, str],
    provider_api_keys: dict[str, str],
) -> LLMEndpointConfig:
    spec: dict[str, Any] = dict(PIPELINE_LLM_CONFIG.get(stage) or {})

    provider = _normalize_provider(str(spec.get("provider") or ""))
    if provider not in provider_base_urls:
        available = ", ".join(sorted(provider_base_urls.keys()))
        raise ValueError(f"阶段 `{stage}` 的 provider 无效: `{provider}`。可选: {available}")

    default_model = str(spec.get("model") or "").strip()
    if not default_model:
        raise ValueError(f"阶段 `{stage}` 缺少 `model` 配置。")
    model_override_env = STAGE_MODEL_ENV_OVERRIDES.get(stage)
    model = pick(model_override_env, default_model) if model_override_env else default_model
    model = (model or default_model).strip()

    if "rpm" not in spec:
        raise ValueError(f"阶段 `{stage}` 缺少 `rpm` 配置。")
    if "tpm" not in spec:
        raise ValueError(f"阶段 `{stage}` 缺少 `tpm` 配置。")

    default_rpm = _as_int_with_min(str(spec.get("rpm")), 1000, 1)
    default_tpm = _as_int_with_min(str(spec.get("tpm")), 100000, 1)

    stage_rate_env = STAGE_RATE_LIMIT_ENV_OVERRIDES.get(stage)
    rpm_raw: str | None = pick(stage_rate_env[0]) if stage_rate_env else None
    tpm_raw: str | None = pick(stage_rate_env[1]) if stage_rate_env else None
    rpm = _as_int_with_min(rpm_raw, default_rpm, 1)
    tpm = _as_int_with_min(tpm_raw, default_tpm, 1)

    return LLMEndpointConfig(
        stage=stage,
        provider=provider,
        model=model,
        base_url=provider_base_urls[provider],
        api_key=provider_api_keys.get(provider) or "",
        rpm=rpm,
        tpm=tpm,
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
    stage2_llm1_concurrency: int | None
    stage2_llm2_concurrency: int | None
    stage2_arbitration_concurrency: int | None
    stage2_sync_headroom: float
    stage2_sync_max_ahead: int
    stage2_sync_mode: str
    stage2_fragment_max_attempts: int
    stage2_max_empty_retries: int

    @classmethod
    def load(cls, root_dir: Path) -> "AppConfig":
        env_file_values = _load_dotenv(root_dir / ".env")

        def pick(name: str, default: Optional[str] = None) -> Optional[str]:
            return os.getenv(name) or env_file_values.get(name) or default

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
        )
        stage2_llm1 = _build_stage_endpoint(
            stage="stage2_llm1",
            pick=pick,
            provider_base_urls=provider_base_urls,
            provider_api_keys=provider_api_keys,
        )
        stage2_llm2 = _build_stage_endpoint(
            stage="stage2_llm2",
            pick=pick,
            provider_base_urls=provider_base_urls,
            provider_api_keys=provider_api_keys,
        )
        stage2_llm3 = _build_stage_endpoint(
            stage="stage2_llm3",
            pick=pick,
            provider_base_urls=provider_base_urls,
            provider_api_keys=provider_api_keys,
        )
        stage3_llm = _build_stage_endpoint(
            stage="stage3",
            pick=pick,
            provider_base_urls=provider_base_urls,
            provider_api_keys=provider_api_keys,
        )
        stage4_llm = _build_stage_endpoint(
            stage="stage4",
            pick=pick,
            provider_base_urls=provider_base_urls,
            provider_api_keys=provider_api_keys,
        )
        stage5_llm = _build_stage_endpoint(
            stage="stage5",
            pick=pick,
            provider_base_urls=provider_base_urls,
            provider_api_keys=provider_api_keys,
        )

        stage2_llm1_concurrency = _as_optional_int_with_min(
            pick("STAGE2_LLM1_CONCURRENCY"),
            1,
        )
        stage2_llm2_concurrency = _as_optional_int_with_min(
            pick("STAGE2_LLM2_CONCURRENCY"),
            1,
        )
        stage2_arbitration_concurrency = _as_optional_int_with_min(
            pick("STAGE2_ARBITRATION_CONCURRENCY"),
            1,
        )
        stage2_sync_headroom = _as_float_in_range(
            pick("STAGE2_SYNC_HEADROOM", str(STAGE2_RUNTIME_DEFAULTS["sync_headroom"])),
            float(STAGE2_RUNTIME_DEFAULTS["sync_headroom"]),
            0.01,
            1.0,
        )
        stage2_sync_max_ahead = _as_int_with_min(
            pick("STAGE2_SYNC_MAX_AHEAD", str(STAGE2_RUNTIME_DEFAULTS["sync_max_ahead"])),
            int(STAGE2_RUNTIME_DEFAULTS["sync_max_ahead"]),
            0,
        )
        stage2_sync_mode = str(
            pick("STAGE2_SYNC_MODE", str(STAGE2_RUNTIME_DEFAULTS["sync_mode"]))
            or STAGE2_RUNTIME_DEFAULTS["sync_mode"]
        ).strip()
        if stage2_sync_mode not in {"lowest_shared"}:
            raise ValueError(
                f"STAGE2_SYNC_MODE 仅支持 `lowest_shared`，实际为: {stage2_sync_mode}"
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
            api_key=stage1_llm.api_key,
            base_url=stage1_llm.base_url,
            model_default=stage1_llm.model,
            stage1_llm=stage1_llm,
            stage2_llm1=stage2_llm1,
            stage2_llm2=stage2_llm2,
            stage2_llm3=stage2_llm3,
            stage3_llm=stage3_llm,
            stage4_llm=stage4_llm,
            stage5_llm=stage5_llm,
            stage2_llm1_concurrency=stage2_llm1_concurrency,
            stage2_llm2_concurrency=stage2_llm2_concurrency,
            stage2_arbitration_concurrency=stage2_arbitration_concurrency,
            stage2_sync_headroom=stage2_sync_headroom,
            stage2_sync_max_ahead=stage2_sync_max_ahead,
            stage2_sync_mode=stage2_sync_mode,
            stage2_fragment_max_attempts=_as_int_with_min(
                pick(
                    "STAGE2_FRAGMENT_MAX_ATTEMPTS",
                    str(STAGE2_RUNTIME_DEFAULTS["fragment_max_attempts"]),
                ),
                int(STAGE2_RUNTIME_DEFAULTS["fragment_max_attempts"]),
                1,
            ),
            stage2_max_empty_retries=_as_int_with_min(
                pick(
                    "STAGE2_MAX_EMPTY_RETRIES",
                    str(STAGE2_RUNTIME_DEFAULTS["max_empty_retries"]),
                ),
                int(STAGE2_RUNTIME_DEFAULTS["max_empty_retries"]),
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
                key_name = PROVIDER_API_KEY_ENV_NAMES.get(endpoint.provider, "对应_PROVIDER_API_KEY")
                problems.append(
                    f"阶段 `{endpoint.stage}` 使用 provider `{endpoint.provider}`，"
                    f"但缺少密钥 `{key_name}`。"
                )
        if problems:
            raise ValueError("LLM 配置不完整：\n- " + "\n- ".join(problems))
