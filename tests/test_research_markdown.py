import json
from sage_scholar.agents.writer import Writer
from sage_scholar.domain.research import ResearchNote
from sage_scholar.research.references import finalize_report
from research_fakes import research_request, ScriptedLLM, FakeContext


def test_global_writer_uses_all_notes_and_one_call_without_raw_source_dump():
    context = FakeContext()
    notes = [ResearchNote(task_id=f"T{i}", question=f"q{i}",
        content=f"发现 {i}，边界未知。[ev_a]", source_ids=["ev_a"]) for i in range(3)]
    llm = ScriptedLLM("# 调研报告\n\n综合比较。[1]\n\n$x_{[1]} = 2$")
    result = Writer(llm, evidence_context=context).run(research_request(), notes)
    assert len(llm.calls) == 1 and llm.calls[0]["tag"] == "writer:report"
    payload = json.loads(llm.calls[0]["messages"][-1]["content"])
    assert len(payload["notes"]) == 3 and "[1]" in payload["notes"][0]["content"]
    assert "original_text" not in json.dumps(payload)
    assert result.citations == {1: "ev_a"} and "$x_{[1]} = 2$" in result.report
    assert "## 参考文献" in result.report


def test_unknown_citation_is_disclosed_without_dropping_entire_report():
    report = finalize_report("# Report\n\nKnown [1], unconfirmed [9].", {1: "ev_a"}, FakeContext())
    assert report.partial and "[来源未确认]" in report.report
    assert "Known [1]" in report.report and report.citation_evidence == {1: ["ev_a"]}
