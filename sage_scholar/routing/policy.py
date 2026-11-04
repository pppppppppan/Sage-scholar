"""Deterministic permission validator. No keyword-based workflow classifier."""
from .classifier import SemanticIntent
from .models import RouteDecision, Workflow, SourceScope, ExecutionMode
from .sources import ResolvedSources


class PolicyValidator:
    def validate(
        self,
        intent: SemanticIntent,
        sources: ResolvedSources,
        *,
        requested_workflow: Workflow | None,
        requested_scope: SourceScope | None,
        allow_web: bool | None,
        confidence_gate: bool = False,
        confidence_threshold: float = .65,
    ) -> tuple[RouteDecision, str | None]:
        workflow = Workflow(requested_workflow or intent.workflow)
        scope, docs = sources.scope, sources.document_ids
        question, trigger = sources.question, 'source_resolution' if sources.question else None
        if sources.boundary_conflict:
            trigger = 'selected_boundary'
        # A direct operation does not need source access, even when UI context exists.
        if workflow == Workflow.DIRECT:
            scope, docs, question, trigger = SourceScope.CONVERSATION, [], None, None
        if intent.workflow == 'clarify':
            question = question or intent.clarification or '请明确本次任务的研究对象或所指论文。'
            trigger = trigger or 'model'
        if confidence_gate and intent.confidence < confidence_threshold:
            question = question or intent.clarification or '当前意图判断把握不足，请补充任务对象或资料范围。'
            trigger = trigger or 'confidence'
        if (workflow != Workflow.DIRECT and requested_scope and intent.source_explicit
                and intent.source_scope != requested_scope):
            question = '当前文字要求的资料范围与界面设置冲突，请确认以哪个范围为准。'
            trigger = 'scope_conflict'
        web = (
            workflow != Workflow.DIRECT
            and scope in {SourceScope.PUBLIC, SourceScope.MIXED}
            and allow_web is not False
        )
        if workflow != Workflow.DIRECT and scope in {SourceScope.PUBLIC, SourceScope.MIXED} and allow_web is False:
            question = '本次资料范围需要公网，但联网已被禁止；请调整资料范围或明确允许联网。'
            trigger = 'web_forbidden'
        if workflow == Workflow.QA and scope not in {SourceScope.SELECTED_DOCUMENTS,SourceScope.WORKSPACE,SourceScope.MIXED}:
            question = question or '文档问答需要明确的本地论文或文献库，请选择资料或修改执行类型。'
            trigger = trigger or 'qa_sources_missing'
        if workflow == Workflow.DEEP_RESEARCH and scope == SourceScope.CONVERSATION:
            question = question or '研究需要资料来源，请指定本地文献或允许公网检索。'
            trigger = trigger or 'research_sources_missing'
        if question:
            workflow, web = Workflow.CLARIFY, False  # Waiting must not dispatch source tools.
        decision = RouteDecision(
            workflow=workflow,
            source_scope=scope,
            document_ids=docs,
            allow_web=web,
            execution_mode=ExecutionMode.RUN,
            needs_clarification=bool(question),
            clarification_message=question,
            confidence=intent.confidence,
            reason_code=f'clarify_{trigger}' if question else 'local_semantic',
        )
        return decision, trigger
