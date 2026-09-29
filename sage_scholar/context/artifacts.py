"""Large untrusted context is persisted outside prompts, without becoming Evidence."""
import hashlib
import json
import re
from pathlib import Path
from sage_scholar.infrastructure.json_store import write_json


class ArtifactStore:
    def __init__(self, data_dir):
        self.directory = Path(data_dir) / "context_artifacts"

    def put(self, text: str, owner: str, kind="tool_result") -> str:
        identity = "art_" + hashlib.sha256((owner + "\0" + kind + "\0" + text).encode()).hexdigest()
        write_json(self.directory / f"{identity}.json", {"artifact_id": identity, "owner": owner, "kind": kind, "text": text})
        return identity

    def read(self, identity, owner, offset=0, limit=4000):
        if not re.fullmatch(r"art_[a-f0-9]{64}", identity):
            raise ValueError("invalid artifact ID")
        path = self.directory / f"{identity}.json"
        if not path.exists():
            raise ValueError("artifact not found")
        item = json.loads(path.read_text())
        if item["owner"] != owner:
            raise ValueError("artifact belongs to another request/run")
        offset, limit = max(0, offset), min(12000, max(1, limit))
        return {"artifact_id": identity, "offset": offset, "text": item["text"][offset:offset + limit],
                "total_chars": len(item["text"]), "kind": item["kind"], "is_evidence": False}
