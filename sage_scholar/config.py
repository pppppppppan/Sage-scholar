import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")


@dataclass
class LLMConfig:
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL_ID", "deepseek-v4-flash"))
    review_model: str = field(default_factory=lambda: os.getenv("REVIEW_MODEL_ID", ""))
    temperature: float = 0.0
    research_temperature: float = 0.6
    writer_temperature: float = 0.3
    timeout: int = 120


@dataclass
class ContextConfig:
    max_tokens: int = 30000
    reserve_ratio: float = 0.15


@dataclass
class Config:
    config_dir: str = field(default_factory=lambda: os.getenv("SAGE_CONFIG_DIR", str(PROJECT_DIR / "configs")))
    data_dir: str = field(default_factory=lambda: os.getenv("SAGE_DATA_DIR", str(PROJECT_DIR / "data")))
    mcp_connect_timeout: float = field(default_factory=lambda: float(os.getenv("MCP_CONNECT_TIMEOUT", "12")))
    max_rounds: int = 3
    max_steps: int = 3
    llm: LLMConfig = field(default_factory=LLMConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
