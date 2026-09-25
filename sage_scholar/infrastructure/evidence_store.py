from __future__ import annotations

import json
from pathlib import Path
import re

from sage_scholar.domain.evidence import EvidenceRef

from .json_store import file_lock, write_json


class JsonEvidenceStore:
    def __init__(self, data_dir: str):
        self.directory = Path(data_dir) / "evidence"

    def _path(self, evidence_id: str) -> Path:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", evidence_id):
            raise ValueError("invalid evidence ID")
        return self.directory / f"{evidence_id}.json"

    def save(self, evidence: EvidenceRef) -> None:
        path = self._path(evidence.evidence_id)
        with file_lock(path):
            write_json(path, evidence.model_dump(mode="json", exclude={"text"}))

    def get(self, evidence_id: str) -> EvidenceRef | None:
        path = self._path(evidence_id)
        with file_lock(path):
            return EvidenceRef.model_validate(json.loads(path.read_text())) if path.exists() else None
