"""Deterministic model/tool boundaries; tests still execute the real LangGraph."""
import json
from types import SimpleNamespace

from sage_scholar.domain.execution import ExecutionRequest
from sage_scholar.domain.research import (
    Claim, ClaimVerification, Finding, GapReview, ReportDraft, ResearchContract,
    ResearchNote, ResearchTask, VerificationResult,
)
from sage_scholar.routing import ExecutionMode, SourceScope, Workflow


def contract(**limits):
    request = ExecutionRequest(query="research topic", workflow=Workflow.DEEP_RESEARCH,
        source_scope=SourceScope.PUBLIC, allow_web=True, execution_mode=ExecutionMode.RUN,
        budget={"max_rounds": 0, **limits})
    return ResearchContract.from_request(request)


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
    def plan_tasks(self, contract):
        return [ResearchTask(task_id="T1", label="方法", objective="有哪些主要方法？")]


class FakeResearcher:
    def run(self, task, contract, dependency_notes=()):
        return ResearchNote(task_id=task.task_id, question=task.objective, revision=task.revision,
            findings=[Finding(text="有证据支持的研究笔记。", evidence_ids=["ev_a"])], evidence_ids=["ev_a"])


class FakeWriter:
    def run(self, contract, notes, **kwargs):
        assert all(isinstance(note, ResearchNote) for note in notes)
        return ReportDraft(title="最终报告", claims=[Claim(claim_id="C1",
            text="有证据支持的研究笔记。", evidence_ids=["ev_a"])])


class CompleteReview:
    def review(self, contract, tasks, notes):
        return GapReview()


class SupportedVerifier:
    def verify(self, draft):
        return VerificationResult(claims=[ClaimVerification(claim_id=c.claim_id,
            status="supported", identity_valid=True, reason="source supports claim", evidence_ids=c.evidence_ids)
            for c in draft.claims])
