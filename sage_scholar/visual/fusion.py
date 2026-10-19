"""Rank fusion across independent text and image embedding spaces."""


def fuse(text, images, limit):
    if not images:
        return text
    scores, records = {}, {}
    for ranking in (text, images):
        for rank, item in enumerate(ranking, 1):
            records[item.evidence_id] = item
            scores[item.evidence_id] = scores.get(item.evidence_id, 0) + 1 / (60 + rank)
    return [records[key] for key in sorted(scores, key=scores.get, reverse=True)[:limit]]
