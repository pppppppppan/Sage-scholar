"""Deterministic consolidation of the Supervisor's semantic scope decisions.

The model assigns the same scope_keys to equivalent research outcomes in its
existing plan/review call. We do not infer semantics from string similarity.
"""
from sage_scholar.domain.research import SubQuestion, SupervisorReview


def key(text):
    return " ".join(text.split()).casefold()


def identities(question):
    values = {("objective", key(question.objective))}
    values.update(("scope", key(item)) for item in question.scope_keys if item.strip())
    return values


def unique(values):
    found, result = set(), []
    for value in values:
        if value.strip() and key(value) not in found:
            found.add(key(value))
            result.append(value.strip())
    return result


def merge(questions):
    """Retain every distinct objective/output/coverage item; never slice text."""
    return SubQuestion(
        objective="；".join(unique(q.objective for q in questions)),
        label=questions[0].label,
        scope_keys=unique(item for q in questions for item in q.scope_keys),
        expected_output="；".join(unique(q.expected_output for q in questions)),
        covers=unique(item for q in questions for item in q.covers),
        research_gap="；".join(unique(q.research_gap for q in questions)),
    )


def consolidate(questions):
    # Keep all aliases while grouping so A/B and B/C overlaps merge transitively.
    groups = []
    for question in questions:
        aliases = identities(question)
        matching = [i for i, (keys, _) in enumerate(groups) if keys & aliases]
        if not matching:
            groups.append((aliases, [question]))
            continue
        first = matching[0]
        members = []
        for i in matching:
            aliases |= groups[i][0]
            members.extend(groups[i][1])
        members.append(question)
        groups[first] = aliases, members
        for i in reversed(matching[1:]):
            groups.pop(i)
    result = [merge(members) for _, members in groups]
    check_coverage(unique(item for q in questions for item in q.covers), result)
    return result


def check_coverage(requirements, questions):
    covered = {key(item) for q in questions for item in q.covers}
    missing = [item for item in requirements if key(item) not in covered]
    if missing:
        raise ValueError("任务合并后遗漏研究目标：" + "；".join(missing))


def within_budget(questions, maximum):
    if maximum <= 0:
        return []
    if len(questions) <= maximum:
        return questions
    # Combine the overflow instead of silently dropping user requirements.
    result = [*questions[:maximum - 1], merge(questions[maximum - 1:])]
    check_coverage(unique(item for q in questions for item in q.covers), result)
    return result


def already_planned(question, known):
    if ("objective", key(question.objective)) in known:
        return True
    scopes = {(kind, value) for kind, value in identities(question) if kind == "scope"}
    # Shared background must not cause a new outcome to be discarded.
    return bool(scopes) and scopes <= known


def novel_followups(questions, tasks):
    known = set().union(*(identities(task) for task in tasks)) if tasks else set()
    # Completed, partial, pending and failed tasks all retain their scope identity.
    # A follow-up must describe a new, narrower outcome rather than rename that task.
    return consolidate([q for q in questions if not already_planned(q, known)])


def compact_gaps(gaps):
    gaps = unique(gaps)
    # Keep the persisted schema bound without silently discarding unresolved requirements.
    return gaps if len(gaps) <= 6 else [*gaps[:5], "；".join(gaps[5:])]


def prune_review(review, tasks):
    known = set().union(*(identities(task) for task in tasks)) if tasks else set()
    withheld = [q.research_gap for q in review.follow_up_questions if already_planned(q, known) and q.research_gap]
    return SupervisorReview(
        note_reviews=review.note_reviews,
        follow_up_questions=novel_followups(review.follow_up_questions, tasks),
        remaining_gaps=compact_gaps([*review.remaining_gaps, *withheld]),
    )
