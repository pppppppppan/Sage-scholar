from sage_scholar.domain.research import ResearchReport


class CitationRenderer:
    """Deterministic rendering from Claim objects; no URL dedup or citation guessing."""
    def render(self, draft, verification, *, remaining_gaps=()):
        judgments = {item.claim_id: item for item in verification.claims}
        claims = {claim.claim_id: claim for claim in draft.claims}
        labels, lines, removed, kept = {}, [f"# {draft.title}"], [], []
        seen_claims = set()
        # Unplaced claims are not silently dropped from verification or disclosure.
        sections = [(section.heading, section.claim_ids) for section in draft.sections]
        unplaced = [claim_id for claim_id in claims if all(claim_id not in ids for _, ids in sections)]
        if unplaced:
            sections.append(("其他发现", unplaced))
        for heading, identities in sections:
            body = []
            for identity in identities:
                if identity in seen_claims:
                    continue
                seen_claims.add(identity)
                claim, judgment = claims[identity], judgments.get(identity)
                if judgment is None or judgment.status != "supported" or not judgment.identity_valid:
                    removed.append(identity)
                    continue
                kept.append(claim)
                for evidence_id in claim.evidence_ids:
                    labels.setdefault(evidence_id, len(labels) + 1)
                citations = "".join(f"[{labels[item]}]" for item in dict.fromkeys(claim.evidence_ids))
                body.append(f"{claim.text} {citations}")
            if body:
                lines.append("## " + heading + "\n\n" + "\n\n".join(body))
        gaps = list(dict.fromkeys([*remaining_gaps, *verification.issues]))
        for identity in removed:
            judgment = judgments.get(identity)
            gaps.append(f"已移除未充分支持的主张 {identity}：" + (judgment.reason if judgment else "缺少核验结果"))
        if not kept:
            gaps.append("没有获得经核验支持的报告主张")
        if gaps:
            lines.append("## 未解决事项\n\n" + "\n".join(f"- {gap}" for gap in gaps))
        if labels:
            lines.append("## 证据索引\n\n" + "\n".join(
                f"[{number}] {identity} — /api/evidence/{identity}" for identity, number in labels.items()
            ))
        return ResearchReport(
            report="\n\n".join(lines), citations={number: identity for identity, number in labels.items()},
            claims=kept, verification=verification, removed_claim_ids=removed,
            remaining_gaps=list(dict.fromkeys(gaps)),
        )
