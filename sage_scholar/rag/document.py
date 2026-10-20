from __future__ import annotations
import hashlib
import json
from pathlib import Path
from pydantic import BaseModel, Field


class SourceElement(BaseModel):
    content: str
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    element_type: str = 'paragraph'
    element_id: str | None = None
    metadata: dict = Field(default_factory=dict)


class Chunk(SourceElement):
    chunk_idx: int
    file_path: str
    embedding: list[float] | None = None
    tokens: list[str] | None = None
    schema_version: int = 1  # Missing fields in old indexes retain legacy semantics.
    parser_version: str = 'legacy-markdown'


class Document(BaseModel):
    file_path: str
    text: str
    elements: list[SourceElement] | None = None
    parser_version: str = 'markdown-structure-v2'

    @classmethod
    def read_file(cls, file_path: str) -> Document:
        path = Path(file_path)
        text = path.read_text(encoding='utf-8')
        sidecar = path.with_suffix('.structure.json')
        if not sidecar.exists():
            return cls(file_path=str(path), text=text)
        data = json.loads(sidecar.read_text())
        if data.get('schema_version') != 2 or data.get('converted_sha256') != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('结构化解析版本或内容校验不一致，请重新导入文献')
        return cls(file_path=str(path), text=text, parser_version=data['parser_version'],
                   elements=[SourceElement.model_validate(item) for item in data['elements']])
