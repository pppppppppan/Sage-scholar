from __future__ import annotations

import re

from .models import ExecutionMode, RouteDecision, SourceScope, Workflow


class QueryRouter:
    """Deterministic first router.

    It keeps source policy separate from workflow depth. An LLM classifier can be
    added behind this contract later without changing callers.
    """

    _PAPER2CODE = re.compile(r"(实现|复现|写出|生成).{0,12}(论文|方法|算法|代码)|paper\s*2\s*code", re.I)
    _CITATION = re.compile(r"(核验|检查|审计|验证).{0,12}(引用|参考文献|文献|结论)|虚假引用|错引")
    _RESEARCH = re.compile(r"调研|综述|研究现状|研究空白|主要方法|系统比较|横向比较|deep\s*research", re.I)
    _SEARCH = re.compile(r"最近|最新|新论文|查找|搜索|搜一下|有哪些论文")
    _QA = re.compile(r"这篇论文|该论文|文献库|知识库|上传的|pdf|论文中|哪一页|概括.{0,4}论文", re.I)
    _NO_WEB = re.compile(r"不要联网|不联网|仅使用|只用|仅限")

    def route(
        self,
        query: str,
        *,
        requested_workflow: Workflow | None = None,
        requested_scope: SourceScope | None = None,
        document_ids: list[str] | None = None,
        allow_web: bool | None = None,
    ) -> RouteDecision:
        text = query.strip()
        docs = list(dict.fromkeys(document_ids or []))

        workflow = requested_workflow or self._infer_workflow(text, bool(docs))
        scope = requested_scope or self._infer_scope(text, workflow, bool(docs))
        web_allowed = allow_web if allow_web is not None else not bool(self._NO_WEB.search(text))
        if scope in {SourceScope.SELECTED_DOCUMENTS, SourceScope.WORKSPACE}:
            web_allowed = False

        needs_document = workflow in {Workflow.QA, Workflow.PAPER2CODE} and self._refers_to_unspecified_document(text, docs)
        execution_mode = (
            ExecutionMode.RUN
            if workflow in {Workflow.DEEP_RESEARCH, Workflow.CITATION_AUDIT, Workflow.PAPER2CODE}
            else ExecutionMode.INLINE
        )

        return RouteDecision(
            workflow=workflow,
            source_scope=scope,
            document_ids=docs,
            allow_web=web_allowed,
            execution_mode=execution_mode,
            needs_clarification=needs_document,
            clarification_message="请先选择或上传你所说的论文。" if needs_document else None,
            reason_code=self._reason_code(workflow, requested_workflow is not None),
        )

    def _infer_workflow(self, query: str, has_documents: bool) -> Workflow:
        if self._PAPER2CODE.search(query):
            return Workflow.PAPER2CODE
        if self._CITATION.search(query):
            return Workflow.CITATION_AUDIT
        if self._RESEARCH.search(query):
            return Workflow.DEEP_RESEARCH
        if self._QA.search(query) or has_documents:
            return Workflow.QA
        if self._SEARCH.search(query):
            return Workflow.SEARCH
        return Workflow.DIRECT

    @staticmethod
    def _infer_scope(query: str, workflow: Workflow, has_documents: bool) -> SourceScope:
        if has_documents:
            return SourceScope.SELECTED_DOCUMENTS
        if re.search(r"文献库|知识库|我的.{0,8}(论文|文献)|上传", query):
            return SourceScope.WORKSPACE
        if workflow in {Workflow.SEARCH, Workflow.DEEP_RESEARCH, Workflow.CITATION_AUDIT}:
            return SourceScope.PUBLIC
        return SourceScope.CONVERSATION

    @staticmethod
    def _refers_to_unspecified_document(query: str, docs: list[str]) -> bool:
        return not docs and bool(re.search(r"这篇|该论文|选中的论文|这份论文", query))

    @staticmethod
    def _reason_code(workflow: Workflow, explicit: bool) -> str:
        return "explicit_workflow" if explicit else f"rule_{workflow.value}"
