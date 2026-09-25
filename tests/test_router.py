from sage_scholar.routing import ExecutionMode, QueryRouter, SourceScope, Workflow


def test_simple_question_uses_direct_path():
    route = QueryRouter().route("今天是星期几")
    assert route.workflow is Workflow.DIRECT
    assert route.execution_mode is ExecutionMode.INLINE


def test_research_question_uses_long_running_path():
    route = QueryRouter().route("帮我调研 AI4CFD 领域的主要方法")
    assert route.workflow is Workflow.DEEP_RESEARCH
    assert route.execution_mode is ExecutionMode.RUN
    assert route.source_scope is SourceScope.PUBLIC


def test_selected_documents_enforce_no_web():
    route = QueryRouter().route(
        "只用这篇论文解释实验设置",
        document_ids=["paper-a"],
        allow_web=True,
    )
    assert route.workflow is Workflow.QA
    assert route.source_scope is SourceScope.SELECTED_DOCUMENTS
    assert route.allow_web is False


def test_unspecified_paper_requires_clarification():
    route = QueryRouter().route("这篇论文用了什么数据集？")
    assert route.workflow is Workflow.QA
    assert route.needs_clarification is True


def test_explicit_no_web_in_text_cannot_be_overridden_by_default_allow_flag():
    route = QueryRouter().route("不要联网，调研已有资料", allow_web=True)
    assert route.allow_web is False
