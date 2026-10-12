"""Read durable Markdown research notes as traceable leads, never as factual Evidence."""
from pydantic import BaseModel
from .conversation_source import conversation_messages
from sage_scholar.context.protection import prompt_brief
from sage_scholar.research.evidence_context import ResearchEvidenceContext


def plain(value):
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


class ResearchMemorySource:
    def __init__(self, run_store, retrieval, checkpoints=None):
        self.store, self.retrieval, self.checkpoints = run_store, retrieval, checkpoints

    def load(self, run_id):
        run = self.store.get(run_id)
        if not run or str(run.workflow) != "deep_research" or str(run.status) not in {"completed", "partial"}:
            raise ValueError("请选择已完成或 PARTIAL 的持久化研究任务")
        values, notes, review = {}, {}, {}
        if self.checkpoints and self.checkpoints.compatible(run):
            checkpoint = self.checkpoints.saver.get_tuple(self.checkpoints.research_config(run.execution))
            values = checkpoint.checkpoint.get("channel_values", {}) if checkpoint else {}
        for event in self.store.events_for(run_id):
            if event.type == "research_task_completed":
                note = event.data.get("note", {})
                if note.get("content") and note.get("task_id"):
                    notes[note["task_id"]] = note
            elif event.type == "supervisor_reviewed":
                review = event.data
        if "research_notes" in values:
            notes = {key: plain(note) for key, note in values["research_notes"].items() if note is not None}
        review = plain(values.get("supervisor_review") or review)
        request = values.get("execution") or run.execution
        if isinstance(request, dict):
            from sage_scholar.domain.execution import ExecutionRequest
            request = ExecutionRequest.model_validate(request)
        messages = conversation_messages(request)
        access = ResearchEvidenceContext(self.retrieval.evidence_store, request.source_policy, self.retrieval.documents)
        for note in notes.values():
            if note.get("status") == "failed" or not note.get("content"):
                continue
            ids = []
            for identity in note.get("source_ids", []):
                item = self.retrieval.evidence_store.get(identity)
                if item is None:
                    continue
                try:
                    access.register(item)
                    access.get(identity)
                    ids.append(identity)
                except (ValueError, KeyError):
                    continue
            if ids:
                # Full Markdown retains qualifications and unknowns for exact-quote extraction.
                # run_id + task_id identifies the persisted note; no second note model is needed.
                messages.append({"role": "assistant", "content": note["content"],
                    "evidence_ids": list(dict.fromkeys(ids)), "research_note_id": f"{run_id}:{note['task_id']}",
                    "task_id": note["task_id"]})
        background = {"brief": prompt_brief(request), "supervisor_review": review,
            "remaining_gaps": run.remaining_gaps, "research_notes": [
                {key: note.get(key) for key in ("task_id", "question", "status")} for note in notes.values()]}
        return run, messages, background
