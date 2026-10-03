"""Resolve citation identities first, then obtain authorized original-text Evidence."""
from sage_scholar.research.evidence_context import ResearchEvidenceContext
from sage_scholar.runtime.journal import WorkflowInterrupted
from .bibliography import parse_reference, match_reference, match_local_reference, AmbiguousLocalReference
from .models import CitationIdentity


class AuditSources:
    def __init__(self, retrieval, policy, fetch=None):
        self.retrieval, self.policy, self.fetch = retrieval, policy, fetch
        self.context = ResearchEvidenceContext(retrieval.evidence_store, policy, retrieval.documents)
        self._sections = None
        self._resolved = {}

    def _local(self, reference):
        if not self.policy.permits_local:
            return None, []
        doc = match_local_reference(reference, self.retrieval.permitted_documents(self.policy))
        if doc is None:
            return None, []
        if self._sections is None:
            self._sections = self.retrieval.read_documents_sync(self.policy)
        return doc, [e for e in self._sections if e.document_id == doc.document_id]

    def resolve(self, citation, source):
        if source in self._resolved:
            return self._resolved[source].model_copy(update={"citation": citation}, deep=True)
        original = source
        reference = parse_reference(source)
        source = reference.locator or source
        metadata, items, matched = {}, [], False
        reason = "在当前资料范围内未定位到来源；当前无法确认，不等于伪造。"
        try:
            if source.startswith("ev_"):
                item = self.retrieval.evidence_store.get(source)
                items = [item] if item else []
            else:
                doc, items = self._local(reference)
                if doc:
                    matched = True
                    source = doc.document_id
                    metadata = {"document_id": doc.document_id, "title": doc.title,
                                "resolution": "local_exact"}
                    reason = "已通过本地标识或规范化精确题名匹配；已有作者/年份元数据无冲突。"
                if not matched and not items and self.policy.permits_web:
                    if (not reference.locator and not source.startswith("doc_")
                            and any(c.isalpha() for c in reference.title)
                            and self.fetch and hasattr(self.fetch, "search")):
                        metadata, reason = match_reference(reference, self.fetch.search(reference))
                        if metadata:
                            matched = True
                            source = metadata["source"]
                    if reference.locator or matched:
                        items = self.retrieval.evidence_store.find_source(source)
                        if not items and self.fetch:
                            items = self.fetch(source)
            valid = []
            for item in items:
                self.context.register(item)
                self.context.get(item.evidence_id)
                valid.append(item.evidence_id)
            if valid:
                reason = (reason + " " if matched else "") + "已定位允许读取的原文与版本。"
            elif matched:
                reason += " 文献身份已匹配，但尚无可读取原文，不能以检索摘要核验支持关系。"
            result = CitationIdentity(citation=citation, source=original,
                status="source_found" if valid or matched else "source_not_found", evidence_ids=valid,
                resolved_source=source if valid or matched else None, metadata=metadata or {}, reason=reason)
        except AmbiguousLocalReference as exc:
            result = CitationIdentity(citation=citation, source=original, status="source_not_found",
                metadata={"resolution": "ambiguous"}, reason=str(exc) + " 当前无法确认来源，不等于伪造。")
        except WorkflowInterrupted:
            raise
        except Exception as exc:
            result = CitationIdentity(citation=citation, source=original,
                status="source_found" if matched else "source_not_found",
                resolved_source=source if matched else None, metadata=metadata or {},
                reason=("文献身份已匹配，但原文暂不可读取。" if matched else reason) + f" 核验未完成：{exc}")
        self._resolved[original] = result
        return result
