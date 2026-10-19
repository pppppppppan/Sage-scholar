from typing import Literal
from pydantic import BaseModel, Field


class VisualElement(BaseModel):
    element_id: str
    kind: Literal["page", "figure", "table"]
    page: int = Field(ge=1)
    bbox: tuple[float, float, float, float]
    image: str
    image_sha256: str
    caption: str = ""
    text: str = ""
    label: str = ""
    width: int
    height: int
    vector: list[float] | None = None


class VisualManifest(BaseModel):
    schema_version: int = 1
    source_hash: str
    parser: str
    page_count: int
    status: Literal["parsed", "ready", "failed"] = "parsed"
    warning: str = ""
    embedding_identity: dict = Field(default_factory=dict)
    elements: list[VisualElement] = Field(default_factory=list)
