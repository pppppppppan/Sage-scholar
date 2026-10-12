"""Semantic intent plus deterministic permission validation, never keyword routing."""
from .classifier import LocalClassifier
from .models import ExecutionMode, RouteDecision, SourceScope, Workflow


class QueryRouter:
    def __init__(self, classifier=None, *, confidence_threshold=0.65):
        self.classifier = classifier or LocalClassifier()
        self.confidence_threshold = confidence_threshold

    def route(self, query, *, requested_workflow=None, requested_scope=None, document_ids=None,
              allow_web=None, history=(), catalog=(), resolution_message=None):
        if requested_workflow in {Workflow.CITATION_AUDIT, Workflow.PAPER2CODE, Workflow.SEARCH}:
            raise ValueError("该独立执行类型未提供，请选择 Direct、QA 或 Deep Research")
        selected = list(dict.fromkeys(document_ids or []))
        intent = self.classifier.classify({
            "query": query, "requested_workflow": requested_workflow, "requested_scope": requested_scope,
            "selected_document_ids": selected, "allow_web": allow_web,
            "recent_conversation": [{"role": m["role"], "content": m["content"][:1500]} for m in history[-6:]],
            "catalog": list(catalog)[:50],
        })
        workflow = Workflow(requested_workflow or intent.workflow)
        scope = SourceScope(requested_scope or intent.source_scope)
        known = {d["document_id"] for d in catalog} | set(selected)
        unknown = set(intent.document_ids) - known
        docs = selected or [i for i in dict.fromkeys(intent.document_ids) if i in known]
        question = None if workflow == Workflow.DIRECT else (resolution_message or (
            "无法确认所指论文，请上传或明确选择文档。" if unknown else None))
        if intent.workflow == "clarify" or intent.confidence < self.confidence_threshold:
            question = question or intent.clarification or "请补充你希望回答的问题、所指资料或研究范围。"
        # Explicit source boundaries always win; the model cannot widen them.
        if requested_scope is None and docs and scope not in {SourceScope.MIXED, SourceScope.CONVERSATION}:
            scope = SourceScope.SELECTED_DOCUMENTS
        web = bool(intent.allow_web and allow_web is not False and scope in {SourceScope.PUBLIC, SourceScope.MIXED})
        if workflow == Workflow.DIRECT:
            scope, docs, web = SourceScope.CONVERSATION, [], False
        if workflow == Workflow.QA:
            if scope not in {SourceScope.SELECTED_DOCUMENTS, SourceScope.WORKSPACE, SourceScope.MIXED}:
                question = question or "请上传或选择要问的论文，或者说明要查询文献库中的哪些资料。"
            if not docs and not any(d.get("ready", False) for d in catalog):
                question = question or "文献库中没有可用资料，请先上传论文并等待解析完成。"
        if scope == SourceScope.SELECTED_DOCUMENTS and not docs:
            question = question or "请先选择本次使用的论文。"
        if workflow == Workflow.DEEP_RESEARCH and (scope == SourceScope.CONVERSATION or (scope == SourceScope.PUBLIC and not web)):
            question = question or "当前不允许联网，请选择本地资料，或明确允许公网资料检索。"
        if question:
            workflow = Workflow.CLARIFY
        return RouteDecision(workflow=workflow, source_scope=scope, document_ids=docs, allow_web=web,
            execution_mode=ExecutionMode.RUN, needs_clarification=bool(question), clarification_message=question,
            confidence=intent.confidence, reason_code="semantic_clarify" if question else "local_semantic")
