"""Deterministic model/tool boundaries; tests still execute the real LangGraph."""
import json
from types import SimpleNamespace

from sage_scholar.domain.execution import ExecutionRequest
from sage_scholar.domain.research import (
    Finding, GapReview, ResearchContract, ResearchNote, ResearchTask,
    Synthesis, SynthesisBlock, ReportOutline, SectionPlan, WrittenSection, ReportReview,
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


class FakeContext:
    def get(self, identity):
        if identity != "ev_a":
            raise ValueError(identity)
        return SimpleNamespace(source_id="https://example.org/paper", document_id=None,
            source_version="v1", original_text="有证据支持的研究笔记。", metadata={"title": "研究论文"}, section="方法", page=None)


class FakeWriter:
    def __init__(self, *args, evidence_context=None, **kwargs):
        self.context = evidence_context or FakeContext()

    def outline(self, contract, synthesis):
        return ReportOutline(title="最终报告", sections=[SectionPlan(section_id="S1", heading="方法与适用条件",
            objective="解释机制与局限", block_indices=[0])])

    def write_section(self, contract, synthesis, outline, plan, references, previous=None, **kwargs):
        return WrittenSection(section_id=plan.section_id, heading=plan.heading,
            markdown=f"有证据支持的研究笔记。[{next(iter(references))}]",
            revision=previous.revision + 1 if previous else 0)


class FakeSynthesis:
    def __init__(self, *args): pass
    def run(self, contract, materials, allowed_ids):
        return Synthesis(blocks=[SynthesisBlock(heading="方法", analysis="比较方法的机制及适用范围。",
            evidence_ids=allowed_ids)])


class CompleteReview:
    def review(self, contract, tasks, notes):
        return GapReview()


class SupportedReview:
    def __init__(self, *args): pass
    def review(self, *args):
        return ReportReview(acceptable=True)
