"""Stable task-aware recall input, excluding growing observations and source bodies."""
import json


def recall_query(request, messages, purpose=""):
    objectives, dimensions = [], []
    def brief(value):
        if not isinstance(value, dict):
            return
        for key in ("objective", "question", "query"):
            if isinstance(value.get(key), str):
                objectives.append(value[key][:1200])
        for key in ("requested_dimensions", "dimensions", "comparison_dimensions"):
            if isinstance(value.get(key), list):
                dimensions.extend(str(v)[:150] for v in value[key][:12])
        for key in ("brief", "task", "execution"):
            brief(value.get(key))
    first = next((m.get("content", "") for m in messages if m.get("role") == "user" and not m.get("_context_kind")), "")
    try:
        payload = json.loads(first)
        brief(payload)
        # A Writer's scope is the questions it must synthesize, not each source body.
        if isinstance(payload, dict):
            for note in payload.get("notes", [])[:20]:
                brief({"question": note.get("question", "")})
    except (ValueError, TypeError):
        objectives.append(str(first)[:1200])
    dimensions.extend(str(v)[:150] for v in request.metadata.get("requested_dimensions", [])[:12])
    terms = [request.query[:2000], purpose, *objectives, *dimensions]
    if purpose.endswith((":react", ":forced_summary")):
        # Put the assigned subquestion first; avoid letting the global topic dominate recall.
        terms = [*objectives, purpose, request.query[:600], *dimensions]
    return "\n".join(dict.fromkeys(v for v in terms if v))[:6000]
