"""Deterministic model boundaries; the workflow tests still execute real LangGraph."""
import json
from types import SimpleNamespace
from sage_scholar.domain.execution import ExecutionRequest
from sage_scholar.domain.research import ResearchNote, SubQuestion, SupervisorReview
from sage_scholar.routing import ExecutionMode, SourceScope, Workflow
from sage_scholar.runtime.limits import execution_budget
from sage_scholar.research.references import finalize_report, reference_catalog


def research_request(**limits):
    return ExecutionRequest(query="research topic", workflow=Workflow.DEEP_RESEARCH,
        source_scope=SourceScope.PUBLIC, allow_web=True, execution_mode=ExecutionMode.RUN,
        budget=execution_budget(Workflow.DEEP_RESEARCH, {"max_rounds": 0, **limits}))


class ScriptedLLM:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def invoke(self, **kwargs):
        self.calls.append(kwargs)
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        if callable(value):
            value = value(kwargs)
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        if isinstance(value, dict):
            value = json.dumps(value)
        return SimpleNamespace(content=value, tool_calls=None)


class FakeSupervisor:
    def plan(self, request):
        return [SubQuestion(label="方法", objective="有哪些主要方法？")]

    def review(self, request, tasks, notes):
        return SupervisorReview()


class FakeResearcher:
    def run(self, task, request):
        return ResearchNote(task_id=task.task_id, question=task.objective,
            content="有证据支持的研究笔记。[ev_a]", source_ids=["ev_a"])


class FakeContext:
    def get(self, identity):
        if identity != "ev_a":
            raise ValueError(identity)
        return SimpleNamespace(source_id="https://example.org/paper", document_id=None,
            source_version="v1", original_text="有证据支持的研究笔记。", metadata={"title": "研究论文"},
            section="方法", page=None)


class FakeWriter:
    def __init__(self, *args, evidence_context=None, **kwargs):
        self.context = evidence_context or FakeContext()

    def run(self, request, notes, gaps=()):
        ids = list(dict.fromkeys(i for note in notes for i in note.source_ids))
        return finalize_report("# report\n\n有证据支持的研究笔记。[1]",
                               reference_catalog(self.context, ids), self.context, gaps)
