"""Small research records. ExecutionRequest owns the objective, policy and budget."""
from typing import Literal
from pydantic import BaseModel, Field


class SubQuestion(BaseModel):
    objective: str = Field(min_length=1, max_length=1600)
    label: str = Field(default="", max_length=120)
    # Optional defaults keep existing checkpoints readable. New model plans require these.
    scope_keys: list[str] = Field(default_factory=list, max_length=12)
    expected_output: str = Field(default="", max_length=1200)
    covers: list[str] = Field(default_factory=list, max_length=12)
    research_gap: str = Field(default="", max_length=1200)


class ResearchTask(SubQuestion):
    task_id: str
    status: Literal["pending", "completed", "partial", "failed", "blocked"] = "pending"
    # Explicit retries get a new journal/cache key; completed tasks stay intact.
    revision: int = 0


class ResearchNote(BaseModel):
    task_id: str
    question: str
    content: str
    source_ids: list[str] = Field(default_factory=list)
    status: Literal["completed", "partial", "failed"] = "completed"


class NoteReview(BaseModel):
    task_id: str
    verdict: Literal["approved", "retry"] = "approved"
    reason: str = Field(default="", max_length=800)


class SupervisorReview(BaseModel):
    note_reviews: list[NoteReview] = Field(default_factory=list)
    remaining_gaps: list[str] = Field(default_factory=list, max_length=6)
    follow_up_questions: list[SubQuestion] = Field(default_factory=list, max_length=3)


class ResearchReport(BaseModel):
    report: str
    citations: dict[int, str] = Field(default_factory=dict)
    citation_evidence: dict[int, list[str]] = Field(default_factory=dict)
    remaining_gaps: list[str] = Field(default_factory=list)
    partial: bool = False
