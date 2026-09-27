import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from sage_scholar.agents.writer import Writer
from sage_scholar.domain.research import (
    Claim, ClaimVerification, Finding, ReportDraft, ReportSection, ResearchNote, VerificationResult,
)
from sage_scholar.graph.graph_builder import build_graph
from sage_scholar.research.budget import BudgetedTool, ResearchBudget
from sage_scholar.research.renderer import CitationRenderer
from sage_scholar.research.verification import CitationVerifier
from sage_scholar.tools import ToolCallError
from research_fakes import ScriptedLLM, contract, FakeSupervisor, FakeResearcher, FakeWriter, CompleteReview
from test_research_evidence import evidence_context, SourceReader


def draft(ids, text="Method A beats B on Dataset X but costs more"):
    return ReportDraft(title="Comparison", sections=[ReportSection(heading="Results", claim_ids=["C1"])],
        claims=[Claim(claim_id="C1", text=text, evidence_ids=ids)])


def test_writer_reads_structured_notes_and_only_selected_originals(evidence_context):
    a, b = evidence_context.ids()
    note = ResearchNote(task_id="T1", question="method", evidence_ids=[a],
                        findings=[Finding(text="summary", evidence_ids=[a])])
    llm = ScriptedLLM(draft([a]))
    result = Writer(llm, evidence_context=evidence_context).run(contract(), [note])
    payload = json.loads(llm.calls[0]["messages"][1]["content"])
    assert payload["notes"][0]["task_id"] == "T1"
    assert [item["evidence_id"] for item in payload["evidence"]] == [a]
    assert payload["evidence"][0]["original_text"] == evidence_context.get(a).original_text
    assert result.claims[0].evidence_ids == [a]
    llm = ScriptedLLM(draft([b]))
    with pytest.raises(ValueError, match="outside"):
        Writer(llm, evidence_context=evidence_context).run(contract(), [note])


def test_writer_rejects_embedded_citations(evidence_context):
    ids = evidence_context.ids()
    note = ResearchNote(task_id="T1", question="q", evidence_ids=ids)
    with pytest.raises(ValueError, match="embedded citations"):
        Writer(ScriptedLLM(draft(ids, "result [1]")), evidence_context=evidence_context).run(contract(), [note])


def test_nonexistent_evidence_fails_identity_without_model_call(evidence_context):
    llm = ScriptedLLM()
    result = CitationVerifier(llm, evidence_context).verify(draft(["ev_invented"]))
    assert not result.claims[0].identity_valid and result.claims[0].status == "unsupported"
    assert llm.calls == []


@pytest.mark.parametrize("status", ["supported", "partially_supported", "unsupported", "uncertain"])
def test_semantic_statuses_remain_distinct_and_joint_sources_are_passed(evidence_context, status):
    llm = ScriptedLLM({"status": status, "reason": "Both result and cost must support the comparison."})
    result = CitationVerifier(llm, evidence_context).verify(draft(evidence_context.ids()))
    assert result.claims[0].status == status and result.claims[0].identity_valid
    payload = json.loads(llm.calls[0]["messages"][1]["content"])
    assert len(payload["evidence"]) == 2
    assert {item["original_text"] for item in payload["evidence"]} == {
        evidence_context.get(identity).original_text for identity in evidence_context.ids()}
    assert bool(result.unresolved) == (status != "supported")


def test_verifier_failure_does_not_become_supported(evidence_context):
    result = CitationVerifier(ScriptedLLM(RuntimeError("provider unavailable")), evidence_context).verify(
        draft(evidence_context.ids()))
    assert result.claims[0].status == "uncertain"
    assert "provider unavailable" in result.claims[0].reason


def test_numbering_is_deterministic_and_unsupported_claims_are_removed():
    report = ReportDraft(title="Report", sections=[ReportSection(heading="Findings", claim_ids=["C2", "C1", "C3"])],
        claims=[Claim(claim_id="C1", text="first", evidence_ids=["ev_a", "ev_b"]),
                Claim(claim_id="C2", text="second", evidence_ids=["ev_b"]),
                Claim(claim_id="C3", text="UNSUPPORTED BODY", evidence_ids=["ev_a"])])
    judgments = VerificationResult(claims=[ClaimVerification(claim_id=claim.claim_id,
        status="unsupported" if claim.claim_id == "C3" else "supported", identity_valid=True,
        reason="missing experimental conditions", evidence_ids=claim.evidence_ids) for claim in report.claims])
    renderer = CitationRenderer()
    result = renderer.render(report, judgments)
    assert result == renderer.render(report, judgments)
    assert result.citations == {1: "ev_b", 2: "ev_a"}
    assert "second [1]" in result.report and "first [2][1]" in result.report
    assert "UNSUPPORTED BODY" not in result.report
    assert result.removed_claim_ids == ["C3"] and result.remaining_gaps
    assert "/api/evidence/ev_b" in result.report


@pytest.mark.parametrize("success", [True, False])
def test_revision_is_bounded_and_only_supported_claims_survive(evidence_context, success):
    ids = evidence_context.ids()
    class RevisingWriter:
        calls = 0
        def run(self, contract, notes, previous=None, verification=None):
            self.calls += 1
            if previous:
                assert verification.unresolved
            return draft(ids, "scoped result" if previous else "overclaim")
    writer = RevisingWriter()
    llm = ScriptedLLM({"status": "unsupported", "reason": "overclaim"},
                      {"status": "supported" if success else "partially_supported", "reason": "joint evidence"})
    graph = build_graph(FakeSupervisor(), lambda _: FakeResearcher(), writer,
        gap_reviewer=CompleteReview(), verifier=CitationVerifier(llm, evidence_context))
    report = graph.invoke({"research_contract": contract(max_writer_revisions=1)})["report_result"]
    assert writer.calls == 2 and len(report.verification_history) == 2
    assert bool(report.remaining_gaps) != success
    assert bool(report.claims) == success
    assert "overclaim [" not in report.report
    if not success:
        assert report.removed_claim_ids == ["C1"]


def test_claim_deleted_during_revision_is_disclosed(evidence_context):
    ids = evidence_context.ids()
    class DeletingWriter:
        def run(self, contract, notes, previous=None, **kwargs):
            return ReportDraft(title="Report") if previous else draft(ids)
    graph = build_graph(FakeSupervisor(), lambda _: FakeResearcher(), DeletingWriter(),
        gap_reviewer=CompleteReview(), verifier=CitationVerifier(
            ScriptedLLM({"status": "unsupported", "reason": "no result"}), evidence_context))
    report = graph.invoke({"research_contract": contract()})["report_result"]
    assert report.removed_claim_ids == ["C1"]
    assert "修订中移除" in report.report and report.remaining_gaps


def test_shared_tool_budget_is_atomic_and_exhaustion_produces_gap():
    budget = ResearchBudget(3)
    tool = BudgetedTool(SourceReader(), budget)
    def call(_):
        try:
            tool.run_tool({})
            return True
        except ToolCallError:
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(call, range(20)))
    assert sum(results) == budget.used == 3 and budget.exhausted
    class Never:
        def run(self, *args):
            pytest.fail("No research tool work after budget exhausted")
    from research_fakes import SupportedVerifier
    graph = build_graph(FakeSupervisor(), lambda _: Never(), FakeWriter(),
                        gap_reviewer=CompleteReview(), verifier=SupportedVerifier(), budget=budget)
    result = graph.invoke({"research_contract": contract()})
    assert "工具调用预算已耗尽" in result["report_result"].remaining_gaps
    assert result["tasks"]["T1"].status == "blocked"


def test_truncated_joint_evidence_is_not_fully_supported(tmp_path):
    from sage_scholar.infrastructure.evidence_store import JsonEvidenceStore
    from sage_scholar.research.evidence_context import ResearchEvidenceContext
    from sage_scholar.tools.policy import EvidenceTool
    store = JsonEvidenceStore(str(tmp_path))
    context = ResearchEvidenceContext(store, contract().source_policy)
    EvidenceTool(SourceReader("x" * 60000), store, context.register).run_tool({"url": "https://example.org/long"})
    result = CitationVerifier(ScriptedLLM({"status": "supported", "reason": "claims match"}), context).verify(draft(context.ids()))
    assert result.claims[0].status == "uncertain"
    assert "预算" in result.claims[0].reason
