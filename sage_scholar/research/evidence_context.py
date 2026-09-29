"""Run-scoped evidence access; rereads never perform searches."""
from pathlib import Path
import threading


class ResearchEvidenceContext:
    def __init__(self, store, policy, documents=None, context_manager=None):
        self.store, self.policy, self.documents = store, policy, documents
        self.context_manager = context_manager
        self._ids = set()
        self._lock = threading.Lock()

    def register(self, evidence):
        with self._lock:
            self._ids.add(evidence.evidence_id)

    def ids(self):
        with self._lock:
            return sorted(self._ids)

    def get(self, identity):
        with self._lock:
            if identity not in self._ids:
                raise ValueError(f"Evidence 不属于本次 Run: {identity}")
        item = self.store.get(identity)
        if item is None:
            raise ValueError(f"Evidence 不存在: {identity}")
        if not item.original_text.strip() or not item.source_version or not item.parser_version:
            raise ValueError(f"Evidence 内容或版本缺失: {identity}")
        if item.document_id:
            if not self.policy.permits_document(item.document_id):
                raise ValueError(f"文档不在资料范围内: {identity}")
            path = item.metadata.get("converted_path")
            if not path or not Path(path).is_file():
                raise ValueError(f"文档版本原文不可定位: {identity}")
            if self.documents is not None:
                doc = self.documents.get(item.document_id)
                if doc is None or not any(
                    version.document_version == item.source_version
                    and version.parser_version == item.parser_version
                    and version.converted_path == path for version in doc.versions
                ):
                    raise ValueError(f"文档版本不匹配: {identity}")
        else:
            if not self.policy.permits_web or item.metadata.get("kind") != "source_content":
                raise ValueError(f"公共 Observation 不能直接作为来源证据: {identity}")
            if not item.source_id.startswith(("https://", "http://", "arxiv:")):
                raise ValueError(f"来源标识无法定位: {identity}")
        return item

    def read(self, identities, *, max_chars=24000):
        """Bound context per call and disclose omitted portions rather than silently cut."""
        if self.context_manager:
            # Retain the legacy upper bound as a caller hint; allocation now uses tokens.
            tokens = min(max_chars // 2, int(self.context_manager.max_tokens * 0.35))
            return self.context_manager.read_evidence(self, identities, budget_tokens=tokens)
        records, remaining = [], max_chars
        for identity in dict.fromkeys(identities):
            item = self.get(identity)
            size = min(len(item.original_text), max(0, remaining), max_chars // max(1, len(set(identities))))
            records.append({
                "evidence_id": identity, "source_id": item.source_id, "document_id": item.document_id,
                "source_version": item.source_version, "parser_version": item.parser_version,
                "section": item.section, "page": item.page, "location": item.location,
                "original_text": item.original_text[:size],
                "truncated": size < len(item.original_text),
            })
            remaining -= size
        return records
