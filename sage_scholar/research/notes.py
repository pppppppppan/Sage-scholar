from sage_scholar.domain.research import ResearchNote
from .note_view import sanitize_note


def validate_note(note: ResearchNote, context, allowed_ids) -> ResearchNote:
    note = note.model_copy(deep=True)
    allowed = set(allowed_ids)
    issues = list(note.metadata.get("validation_issues", []))
    candidates = list(dict.fromkeys([*note.evidence_ids, *[
        identity for finding in note.findings for identity in finding.evidence_ids
    ]]))
    valid = []
    for identity in candidates:
        try:
            if identity not in allowed:
                raise ValueError(f"Evidence 未由此任务或其依赖读取: {identity}")
            context.get(identity)
            valid.append(identity)
        except ValueError as exc:
            issues.append(str(exc))
    note.evidence_ids = valid
    for finding in note.findings:
        finding.evidence_ids = [identity for identity in finding.evidence_ids if identity in valid]
        if not finding.evidence_ids:
            note.unknowns.append(f"缺少来源支持：{finding.text}")
    note.unknowns = list(dict.fromkeys(note.unknowns))
    note.metadata["validation_issues"] = list(dict.fromkeys(issues))
    return sanitize_note(note)
