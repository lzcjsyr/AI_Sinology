from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


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
    api_key: str
    base_url: str
    model_default: str
    model_llm1: str
    model_llm2: str
    model_llm3: str

    @classmethod
    def load(cls, root_dir: Path) -> "AppConfig":
        env_file_values = _load_dotenv(root_dir / ".env")

        def pick(name: str, default: Optional[str] = None) -> Optional[str]:
            return os.getenv(name) or env_file_values.get(name) or default

        api_key = pick("API_KEY", "")
        base_url = pick("BASE_URL", "https://api.openai.com/v1")
        model_default = pick("MODEL", "gpt-4o-mini")

        raw_limit = pick("MAX_FRAGMENTS", "")
        if raw_limit:
            try:
                default_max_fragments: Optional[int] = int(raw_limit)
            except ValueError:
                default_max_fragments = None
        else:
            default_max_fragments = None

        return cls(
            root_dir=root_dir,
            data_dir=root_dir / "data",
            outputs_dir=root_dir / "outputs",
            kanripo_dir=(root_dir / "data" / "kanripo_repos"),
            request_timeout=int(pick("REQUEST_TIMEOUT", "180") or "180"),
            max_retries=int(pick("MAX_RETRIES", "3") or "3"),
            retry_backoff_seconds=float(pick("RETRY_BACKOFF_SECONDS", "2.0") or "2.0"),
            default_max_fragments=default_max_fragments,
            api_key=api_key,
            base_url=(base_url or "https://api.openai.com/v1").rstrip("/"),
            model_default=model_default or "gpt-4o-mini",
            model_llm1=pick("MODEL_LLM1", model_default) or "gpt-4o-mini",
            model_llm2=pick("MODEL_LLM2", model_default) or "gpt-4o-mini",
            model_llm3=pick("MODEL_LLM3", model_default) or "gpt-4o-mini",
        )

    def validate_api(self) -> None:
        if not self.api_key:
            raise ValueError(
                "缺少 API_KEY。请在环境变量或 .env 中配置 API_KEY、BASE_URL、MODEL。"
            )
