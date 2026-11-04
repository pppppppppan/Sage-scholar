"""Opt-in metadata-only telemetry. No prices or credentials in source code."""
import json
import math
import os
from dataclasses import dataclass, field
from urllib.parse import urlsplit


@dataclass(frozen=True)
class ObservabilityConfig:
    enabled: bool = False
    host: str = "https://cloud.langfuse.com"
    public_key: str = field(default="", repr=False)
    secret_key: str = field(default="", repr=False)
    queue_size: int = 512
    batch_size: int = 64
    delay_ms: int = 2000
    timeout_seconds: float = 3
    prices: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls):
        if os.getenv("OBSERVABILITY_ENABLED", "false").lower() != "true":
            return cls()  # Disabled mode does not parse prices or import the SDK.
        config = cls(enabled=True,
            host=(os.getenv("LANGFUSE_BASE_URL", "").strip() or os.getenv("LANGFUSE_HOST", "").strip() or "https://cloud.langfuse.com").rstrip("/"),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""), secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            queue_size=int(os.getenv("OBSERVABILITY_QUEUE_SIZE", "512")),
            batch_size=int(os.getenv("OBSERVABILITY_BATCH_SIZE", "64")),
            delay_ms=int(os.getenv("OBSERVABILITY_DELAY_MS", "2000")),
            timeout_seconds=float(os.getenv("OBSERVABILITY_TIMEOUT_SECONDS", "3")),
            prices=json.loads(os.getenv("OBSERVABILITY_MODEL_PRICES_JSON", "{}")))
        url = urlsplit(config.host)
        if url.scheme not in {"https", "http"} or not url.hostname or url.username or url.password:
            raise ValueError("Invalid Langfuse host")
        if not config.public_key or not config.secret_key:
            raise ValueError("Missing Langfuse credentials")
        if not 1 <= config.batch_size <= config.queue_size <= 8192:
            raise ValueError("Invalid telemetry queue limits")
        if not 100 <= config.delay_ms <= 60000 or not 0 < config.timeout_seconds <= 10:
            raise ValueError("Invalid telemetry timeout")
        if not isinstance(config.prices, dict):
            raise ValueError("Prices must map model names to USD per million tokens")
        for price in config.prices.values():
            if not isinstance(price, dict) or not {"input", "output"} <= price.keys():
                raise ValueError("Each price needs input and output")
            if set(price) - {"input", "output", "cache_read", "cache_write"}:
                raise ValueError("Unknown price component")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in price.values()):
                raise ValueError("Invalid model price")
        return config
