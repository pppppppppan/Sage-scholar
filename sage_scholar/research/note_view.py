"""Small, evidence-free views of ResearchNote diagnostics for model prompts."""
from __future__ import annotations

import re


_SOURCE_DUMP = re.compile(
    r"\[ev_[^\]]+\]\s*(?:contents\s+of\s+https?://|document=)|"
    r"(?:^|\n)#{2,6}\s*(?:abstract|introduction)\b",
    re.IGNORECASE,
)


def compact_diagnostics(values, *, max_items=16, max_chars=800):
    """Keep actionable diagnoses; source bodies belong in EvidenceStore."""
    result = []
    for value in values or ():
        text = str(value).strip()
        if not text or _SOURCE_DUMP.search(text) or len(text) > 4000:
            continue
        text = " ".join(text.split())
        if len(text) > max_chars:
            text = text[: max_chars - 1].rstrip() + "…"
        if text not in result:
            result.append(text)
        if len(result) >= max_items:
            break
    return result


def sanitize_note(note):
    """Enforce the ResearchNote boundary before persistence and downstream use."""
    note = note.model_copy(deep=True)
    note.unknowns = compact_diagnostics(note.unknowns)
    note.conflicts = compact_diagnostics(note.conflicts)
    note.limitations = compact_diagnostics(note.limitations)
    metadata = dict(note.metadata)
    metadata["tool_errors"] = compact_diagnostics(metadata.get("tool_errors", ()))
    metadata["validation_issues"] = compact_diagnostics(metadata.get("validation_issues", ()))
    note.metadata = metadata
    return note


def note_prompt_view(note):
    """Serialize research conclusions without execution history or raw tool output."""
    clean = sanitize_note(note)
    return {
        "task_id": clean.task_id,
        "question": clean.question,
        "revision": clean.revision,
        "findings": [item.model_dump(mode="json") for item in clean.findings],
        "evidence_ids": clean.evidence_ids,
        "unknowns": clean.unknowns,
        "conflicts": clean.conflicts,
        "limitations": clean.limitations,
        "validation_issues": clean.metadata.get("validation_issues", []),
    }
