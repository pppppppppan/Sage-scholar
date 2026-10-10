"""Validate source identity without rewriting a Markdown note through another LLM."""
import re
from sage_scholar.domain.research import ResearchNote

SOURCE_ID = re.compile(r"\bev_[a-zA-Z0-9_-]+\b")


def validate_note(note: ResearchNote, context, allowed_ids) -> ResearchNote:
    valid = []
    referenced = list(dict.fromkeys(SOURCE_ID.findall(note.content)))
    allowed = set(allowed_ids)
    for identity in referenced:
        if identity not in allowed:
            continue
        try:
            context.get(identity)
            valid.append(identity)
        except ValueError:
            continue
    invalid = set(referenced) - set(valid)
    content = SOURCE_ID.sub(lambda m: "来源未确认" if m.group() in invalid else m.group(), note.content)
    status = note.status
    if invalid or not valid:
        content += "\n\n> 来源说明：部分发现尚无可回读原文支持，不能作为已核实结论。"
        status = "partial" if status == "completed" else status
    return note.model_copy(update={"content": content, "source_ids": valid, "status": status})
