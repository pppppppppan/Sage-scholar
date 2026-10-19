import os
from dataclasses import dataclass, field


@dataclass
class VisualConfig:
    enabled: bool = field(default_factory=lambda: os.getenv("PDF_VISUAL_ENABLED", "true").lower() == "true")
    parser: str = field(default_factory=lambda: os.getenv("PDF_VISUAL_PARSER", "docling"))
    embedding_model: str = field(default_factory=lambda: os.getenv("VISUAL_EMBEDDING_MODEL", "Qwen/Qwen3-VL-Embedding-2B"))
    embedding_revision: str = field(default_factory=lambda: os.getenv("VISUAL_EMBEDDING_REVISION", "main"))
    device: str = field(default_factory=lambda: os.getenv("VISUAL_EMBEDDING_DEVICE", "cpu"))
    local_only: bool = field(default_factory=lambda: os.getenv("VISUAL_EMBEDDING_LOCAL_ONLY", "true").lower() == "true")
    model: str = field(default_factory=lambda: os.getenv("VISION_MODEL_ID", "deepseek-flash"))
    max_pages: int = field(default_factory=lambda: int(os.getenv("PDF_VISUAL_MAX_PAGES", "200")))
    max_side: int = field(default_factory=lambda: int(os.getenv("PDF_VISUAL_MAX_SIDE", "1600")))
    max_images: int = field(default_factory=lambda: int(os.getenv("VISION_MAX_IMAGES", "3")))
    tokens_per_image: int = field(default_factory=lambda: int(os.getenv("VISION_TOKENS_PER_IMAGE", "1024")))
    run_tokens: int = field(default_factory=lambda: int(os.getenv("VISION_RUN_IMAGE_TOKENS", "60000")))

    def __post_init__(self):
        if self.parser not in {"docling", "page"}:
            raise ValueError("PDF_VISUAL_PARSER must be docling or page")
        if not 1 <= self.max_pages <= 1000 or not 256 <= self.max_side <= 4096:
            raise ValueError("Invalid PDF visual page/resolution limit")
        if not 1 <= self.max_images <= 12 or min(self.tokens_per_image, self.run_tokens) < 1:
            raise ValueError("Invalid visual budget")
