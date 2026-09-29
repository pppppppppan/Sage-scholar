"""Resolve explicit identities, never turn search snippets or memories into Evidence."""
import re

from sage_scholar.research.evidence_context import ResearchEvidenceContext
from .models import CitationIdentity


class AuditSources:
    def __init__(self, retrieval, policy, fetch=None):
        self.retrieval, self.policy, self.fetch = retrieval, policy, fetch
        self.context = ResearchEvidenceContext(retrieval.evidence_store, policy, retrieval.documents)
        self._sections = None

    def resolve(self, citation, source):
        items = []
        # Reference entries may contain authors/title around an explicit URL or DOI.
        original_source = source
        url = re.search(r"https?://[^\s<>\]]+", source)
        doi = re.search(r"\b10\.\d{4,9}/[^\s]+", source)
        if url:
            source = url.group().rstrip(".,;)")
        elif doi:
            source = "https://doi.org/" + doi.group().rstrip(".,;)")
        reason = "在当前资料范围内未定位到原文；未查到不等于伪造。"
        try:
            if source.startswith("ev_"):
                item = self.retrieval.evidence_store.get(source)
                items = [item] if item else []
            else:
                if self.policy.permits_local:
                    if self._sections is None:
                        self._sections = self.retrieval.read_documents_sync(self.policy)
                    docs = self.retrieval.permitted_documents(self.policy)
                    ids = {d.document_id for d in docs if source in {
                        d.document_id, d.title, d.source_identifier, d.metadata.get("arxiv_id")}}
                    items = [e for e in self._sections if e.document_id in ids or e.source_id == source]
                if not items and self.policy.permits_web:
                    items = self.retrieval.evidence_store.find_source(source)
                    if not items and self.fetch and source.startswith(("https://", "http://", "arxiv:")):
                        items = self.fetch(source)
            valid = []
            for item in items:
                self.context.register(item)
                self.context.get(item.evidence_id)
                valid.append(item.evidence_id)
            return CitationIdentity(citation=citation, source=original_source,
                status="source_found" if valid else "source_not_found", evidence_ids=valid, reason="原文与版本可定位" if valid else reason)
        except (ValueError, OSError, RuntimeError) as exc:
            return CitationIdentity(citation=citation, source=original_source, status="source_not_found", reason=f"{reason} {exc}")
