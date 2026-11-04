"""Resolve semantic document references without granting source permissions."""
from dataclasses import dataclass

from .classifier import SemanticIntent
from .models import SourceScope


@dataclass
class ResolvedSources:
    scope: SourceScope
    document_ids: list[str]
    available_local: bool
    question: str | None = None
    unknown: tuple[str, ...] = ()
    boundary_conflict: bool = False


class SourceResolver:
    def resolve(
        self,
        intent: SemanticIntent,
        *,
        requested_scope: SourceScope | None,
        selected: list[str],
        catalog: list[dict] | None,
        resolution_message: str | None = None,
    ) -> ResolvedSources:
        scope = SourceScope(requested_scope or intent.source_scope)
        # Public/conversation do not touch stale or unrelated UI-selected documents.
        if scope in {SourceScope.PUBLIC, SourceScope.CONVERSATION}:
            return ResolvedSources(scope, [], False)
        by_id = {d['document_id']: d for d in catalog or ()}
        # None means the caller did not supply a catalog, not an empty library.
        # The application supplies an authoritative catalog, including an empty one.
        known = set(by_id) if catalog is not None else set(selected)
        ready = (
            {i for i, document in by_id.items() if document.get('ready', False)}
            if catalog is not None else set(selected)
        )
        references = list(dict.fromkeys(intent.document_ids))
        question = None
        boundary_conflict = False
        if scope == SourceScope.WORKSPACE:
            docs = []  # Workspace is the whole permitted library, never a title-filtered subset.
        else:
            docs = references or list(selected)
            if requested_scope == SourceScope.SELECTED_DOCUMENTS:
                if references and not set(references) <= set(selected):
                    question = '所指论文超出明确选择的资料范围，请确认本次允许使用的论文。'
                    boundary_conflict = True
                docs = [i for i in docs if i in selected]
        unknown = tuple(i for i in docs if i not in known)
        if unknown:
            question = question or '所指论文不在当前文献库，请上传或重新选择。'
        docs = [i for i in docs if i in known]
        if any(i not in ready for i in docs):
            question = question or '所指论文尚未完成解析和索引，请在文献库查看处理状态。'
        if resolution_message and set(docs) & set(selected):
            question = question or resolution_message
        if scope == SourceScope.SELECTED_DOCUMENTS and not docs:
            question = question or '请明确选择或提供本次要使用的论文。'
        available = bool(set(docs) & ready) if docs else bool(ready)
        if not available and catalog is not None:
            question = question or '当前没有可用的本地资料，请上传论文并等待解析完成，或调整资料范围。'
        return ResolvedSources(scope, docs, available, question, unknown, boundary_conflict)
