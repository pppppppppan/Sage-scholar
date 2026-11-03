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
    max_tokens: int = field(default_factory=lambda: int(os.getenv("CONTEXT_MAX_TOKENS", "30000")))
    reserve_ratio: float = 0.15
    recent_turns: int = field(default_factory=lambda: int(os.getenv("CONTEXT_RECENT_TURNS", "6")))
    history_tokens: int = field(default_factory=lambda: int(os.getenv("CONTEXT_HISTORY_TOKENS", "6000")))
    report_tokens: int = field(default_factory=lambda: int(os.getenv("CONTEXT_REPORT_THRESHOLD", "2000")))
    agent_tokens: dict = field(default_factory=lambda: {
        name: int(os.getenv(f"CONTEXT_{name.upper()}_TOKENS", str(default))) for name, default in (
            ("supervisor", 20000), ("researcher", 30000), ("review", 48000), ("writer", 64000))})


@dataclass
class GenerationConfig:
    researcher_tokens: int = field(default_factory=lambda: int(os.getenv("RESEARCHER_MAX_OUTPUT_TOKENS", "4096")))
    note_tokens: int = field(default_factory=lambda: int(os.getenv("RESEARCH_NOTE_MAX_OUTPUT_TOKENS", "4096")))
    extended_note_tokens: int = field(default_factory=lambda: int(os.getenv("RESEARCH_NOTE_EXTENDED_OUTPUT_TOKENS", "6144")))
    writer_tokens: int = field(default_factory=lambda: int(os.getenv("WRITER_MAX_OUTPUT_TOKENS", "16384")))

    def __post_init__(self):
        if min(self.researcher_tokens, self.note_tokens, self.writer_tokens) < 1:
            raise ValueError("Agent 输出 token 上限必须大于零")
        if self.extended_note_tokens < self.note_tokens:
            raise ValueError("扩展笔记上限不能小于普通笔记上限")


@dataclass
class RouterConfig:
    confidence_gate: bool = field(default_factory=lambda: os.getenv("ROUTER_CONFIDENCE_GATE", "false").lower() == "true")
    base_url: str = field(default_factory=lambda: os.getenv("ROUTER_BASE_URL", "http://127.0.0.1:11434"))
    model: str = field(default_factory=lambda: os.getenv("ROUTER_MODEL", "qwen3.5:2b"))
    timeout: float = field(default_factory=lambda: float(os.getenv("ROUTER_TIMEOUT", "60")))
    confidence_threshold: float = field(default_factory=lambda: float(os.getenv("ROUTER_CONFIDENCE_THRESHOLD", "0.65")))


@dataclass
class Config:
    config_dir: str = field(default_factory=lambda: os.getenv("SAGE_CONFIG_DIR", str(PROJECT_DIR / "configs")))
    data_dir: str = field(default_factory=lambda: os.getenv("SAGE_DATA_DIR", str(PROJECT_DIR / "data")))
    mcp_connect_timeout: float = field(default_factory=lambda: float(os.getenv("MCP_CONNECT_TIMEOUT", "12")))
    max_rounds: int = 3
    max_steps: int = 3
    llm: LLMConfig = field(default_factory=LLMConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
