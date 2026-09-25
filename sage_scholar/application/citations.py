"""Citation identity stays in evidence IDs; numeric labels are a view."""
from __future__ import annotations

from dataclasses import dataclass
import re

from sage_scholar.domain import EvidenceRef


@dataclass
class CitationRendering:
    text: str
    citations: dict[int, str]
    issues: list[str]


def render_citations(answer: str, evidence: list[EvidenceRef]) -> CitationRendering:
    identities = list(dict.fromkeys(item.evidence_id for item in evidence))
    labels = {identity: index for index, identity in enumerate(identities, 1)}
    issues = []

    def replace(match):
        token = match.group(1)
        if token in labels:
            return f"[{labels[token]}]"
        # Handle legacy numeric output and grouped numeric citations, never invent an identity.
        if re.fullmatch(r"\d+(?:\s*[,，-]\s*\d+)*", token):
            numbers = []
            for part in re.split(r"[,，]", token):
                bounds = [int(value) for value in part.split("-")]
                if len(bounds) == 2:
                    lo, hi = bounds
                    if hi < lo or hi - lo > 1000:
                        issues.append(f"无效引用范围 [{token}]")
                        return "【引用无效】"
                    numbers.extend(range(lo, hi + 1))
                else:
                    numbers.extend(bounds)
            invalid = [value for value in numbers if not 1 <= value <= len(identities)]
            if not invalid:
                return "".join(f"[{value}]" for value in numbers)
        elif not token.startswith("ev"):
            return match.group(0)
        issues.append(f"引用 [{token}] 不在本次提供的证据中")
        return "【引用无效】"

    rendered = re.sub(r"\[([^\]\n]+)\](?!\()", replace, answer)
    return CitationRendering(rendered, dict(enumerate(identities, 1)), list(dict.fromkeys(issues)))


def canonicalize_citations(answer: str, evidence: list[EvidenceRef]) -> CitationRendering:
    rendered = render_citations(answer, evidence)
    rendered.text = re.sub(
        r"\[(\d+)\](?!\()", lambda match: f"[{rendered.citations[int(match.group(1))]}]",
        rendered.text,
    )
    return rendered
