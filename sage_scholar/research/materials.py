"""Bounded views shared by synthesis, writing and report review."""
import json
from sage_scholar.context.token_counter import TokenCounter
from .note_view import note_prompt_view


def encoded(value):
    return json.dumps(value, ensure_ascii=False)


def batches(values, limit=6000):
    counter, batch, size = TokenCounter(), [], 0
    for value in values:
        cost = counter.count(encoded(value))
        if cost > limit:
            raise ValueError("单条研究材料超出阶段预算，需要重新整理笔记")
        if batch and size + cost > limit:
            yield batch
            batch, size = [], 0
        batch.append(value)
        size += cost
    if batch:
        yield batch


def note_units(notes):
    # Split large notes into individual findings, keeping provenance and contrary data.
    for note in notes:
        view = note_prompt_view(note)
        yield {"task_id": note.task_id, "question": note.question}
        for field in ("unknowns", "conflicts", "limitations"):
            for item in view[field]:
                yield {"task_id": note.task_id, field: item}
        for finding in view["findings"]:
            yield {"task_id": note.task_id, "finding": finding}


def source_read(context, identities, tokens=4500):
    return context.read(list(dict.fromkeys(identities)), max_chars=tokens * 2)


def selected_blocks(synthesis, plan):
    return [synthesis.blocks[i] for i in plan.block_indices]
