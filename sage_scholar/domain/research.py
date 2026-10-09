"""Research contracts contain identities and findings; source text stays in EvidenceStore."""
from __future__ import annotations

import re
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .execution import ExecutionRequest


class ResearchLimits(BaseModel):
    max_rounds: int = Field(default=3, ge=0, le=20)
    max_steps: int = Field(default=3, ge=1, le=30)
    max_tasks: int = Field(default=20, ge=1, le=100)
    max_tool_calls: int = Field(default=60, ge=0, le=1000)
    max_writer_revisions: int = Field(default=1, ge=0, le=3)


class ResearchContract(BaseModel):
    model_config = ConfigDict(frozen=True)
    execution: ExecutionRequest
    objective: str
    requested_dimensions: list[str] = Field(default_factory=list)
    deliverables: list[str] = Field(default_factory=lambda: ["研究报告"])
    constraints: list[str] = Field(default_factory=list)
    limits: ResearchLimits = Field(default_factory=ResearchLimits)

    @classmethod
    def from_request(cls, request, *, max_rounds=3, max_steps=3):
        dimensions = request.metadata.get("requested_dimensions")
        if not isinstance(dimensions, list):
            patterns = {
                "methods": r"方法|method", "datasets": r"数据集|dataset",
                "metrics": r"指标|metric", "compute": r"算力|计算成本|训练成本|compute",
                "ablation": r"消融|ablation", "limitations": r"局限|limitation",
            }
            dimensions = [name for name, pattern in patterns.items() if re.search(pattern, request.query, re.I)]
        limits = {"max_rounds": max_rounds, "max_steps": max_steps}
        limits.update({key: value for key, value in request.budget.items() if key in ResearchLimits.model_fields})
        return cls(
            execution=request.model_copy(deep=True), objective=request.query,
            requested_dimensions=dimensions,
            deliverables=request.metadata.get("requested_deliverables", ["研究报告"]),
            constraints=request.metadata.get("constraints", []), limits=ResearchLimits(**limits),
        )

    @property
    def source_policy(self):
        return self.execution.source_policy


class ResearchTask(BaseModel):
    task_id: str
    objective: str
    label: str = ""
    dependencies: list[str] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    status: Literal["pending", "completed", "failed", "blocked"] = "pending"
    revision: int = 0
    completion_criteria: list[str] = Field(default_factory=list)


class Finding(BaseModel):
    text: str = Field(max_length=3000)
    evidence_ids: list[str] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    basis: Literal["primary", "secondary", "inference", "unknown"] = "unknown"
    location: str = ""


class NoteDraft(BaseModel):
    findings: list[Finding] = Field(default_factory=list, max_length=24)
    evidence_ids: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class ResearchNote(NoteDraft):
    # A persisted note can merge several bounded model drafts after targeted follow-up.
    findings: list[Finding] = Field(default_factory=list)
    note_id: str = Field(default_factory=lambda: "note_" + uuid4().hex)
    task_id: str
    question: str
    revision: int = 0
    tool_usage: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FollowUp(BaseModel):
    objective: str
    task_id: str | None = None
    dimensions: list[str] = Field(default_factory=list)


class GapReview(BaseModel):
    covered: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    next_tasks: list[FollowUp] = Field(default_factory=list)

    @property
    def gaps(self):
        return list(dict.fromkeys([*self.missing, *self.conflicts, *self.unknowns]))


class SynthesisBlock(BaseModel):
    heading: str = Field(max_length=160)
    analysis: str = Field(max_length=6000)
    evidence_ids: list[str] = Field(default_factory=list, max_length=40)
    limitations: list[str] = Field(default_factory=list, max_length=12)


class Synthesis(BaseModel):
    blocks: list[SynthesisBlock] = Field(default_factory=list, max_length=8)


class SectionPlan(BaseModel):
    section_id: str
    heading: str = Field(max_length=160)
    objective: str = Field(max_length=1000)
    block_indices: list[int] = Field(default_factory=list)


class ReportOutline(BaseModel):
    title: str = Field(max_length=200)
    sections: list[SectionPlan] = Field(min_length=1, max_length=8)


class WrittenSection(BaseModel):
    section_id: str
    heading: str
    markdown: str
    revision: int = 0


class ReportReview(BaseModel):
    acceptable: bool
    issues: list[str] = Field(default_factory=list, max_length=8)
    revision_instructions: str = Field(default="", max_length=3000)
    source_request: str = Field(default="", max_length=1000)


class ResearchReport(BaseModel):
    report: str
    citations: dict[int, str] = Field(default_factory=dict)
    citation_evidence: dict[int, list[str]] = Field(default_factory=dict)
    remaining_gaps: list[str] = Field(default_factory=list)
    section_reviews: dict[str, ReportReview] = Field(default_factory=dict)
    partial: bool = False
