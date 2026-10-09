"""Fixed synthetic evidence, real stage implementations, no external models."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from research_fakes import contract, FakeSupervisor, FakeResearcher, FakeWriter, FakeSynthesis, CompleteReview, SupportedReview
from sage_scholar.domain.research import ResearchNote, Finding, ReportReview, WrittenSection, ReportOutline, SectionPlan
from sage_scholar.domain.evidence import EvidenceRef
from sage_scholar.infrastructure.evidence_store import JsonEvidenceStore
from sage_scholar.research.evidence_context import ResearchEvidenceContext
from sage_scholar.agents.supervisor import Supervisor
from sage_scholar.agents.writer import Writer
from sage_scholar.research.synthesis import ResearchSynthesis
from sage_scholar.research.report_review import ReportReviewer
from sage_scholar.research.references import reference_catalog, check_section, finalize_report
from sage_scholar.research.materials import batches, note_units
from sage_scholar.graph.graph_builder import build_graph
from sage_scholar.runtime.checkpoints import Checkpoints
from sage_scholar.research.continuation import continue_graph
from sage_scholar.runtime.journal import WorkflowInterrupted
from sage_scholar.context.manager import ContextBudgetExceeded

CASES = json.loads((Path(__file__).parent / "fixtures/research_reports.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_fixed_evidence_to_markdown_all_stages(case, tmp_path):
    request = contract().model_copy(update={"objective": case["query"]})
    store = JsonEvidenceStore(str(tmp_path))
    context = ResearchEvidenceContext(store, request.source_policy)
    ids = []
    for i, text in enumerate(case["sources"]):
        item = EvidenceRef.create(source_id=f"https://example.org/{case['name']}/{i}", source_version="v1",
            parser_version="fixture", original_text=text, section="Methods / Limitations",
            metadata={"kind": "source_content", "title": case["titles"][i]})
        store.save(item); context.register(item); ids.append(item.evidence_id)
    class Model:
        calls = []
        def invoke(self, **kwargs):
            self.calls.append(kwargs)
            tag = kwargs["tag"]
            payload = json.loads(kwargs["messages"][-1]["content"])
            if tag == "supervisor:plan":
                result = {"analysis": case["analysis"], "tasks": [{"task_id": "T1",
                    "objective": case["query"], "dimensions": case["headings"],
                    "completion_criteria": ["解释机制、比较条件及局限"]}]}
            elif tag == "synthesis:combine":
                assert set(e["evidence_id"] for e in payload["evidence"]) == set(ids)
                result = {"blocks": [{"heading": "跨论文比较", "analysis": case["analysis"], "evidence_ids": ids}]}
            elif tag == "writer:outline":
                result = {"title": case["query"], "sections": [{"section_id": f"s{i}", "heading": h,
                    "objective": h, "block_indices": [0]} for i, h in enumerate(case["headings"])]}
            elif tag == "writer:section":
                # Gold text checks transport/format/scoping, not actual LLM generation quality.
                assert set(s["evidence_id"] for s in payload["sources"]) == set(ids)
                n = int(payload["section"]["section_id"][1:])
                result = case["paragraph"] if n == 0 else (
                    "| 比较维度 | 条件 |\n|---|---|\n| 实验 | 对齐数据与预算 [1][2] |\n\n" + case["formula"]
                    if n == 1 else "材料未提供统一实验，尚不能判断全面优劣；需补充相同条件下的比较。[1][2]")
            elif tag == "report_review:section":
                assert payload["sources"]
                result = {"acceptable": True}
            else:
                raise AssertionError(tag)
            return SimpleNamespace(content=json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else result, tool_calls=None)
    model = Model()
    class Researcher:
        def run(self, task, request, dependency_notes):
            return ResearchNote(task_id=task.task_id, question=task.objective, evidence_ids=ids,
                findings=[Finding(text=case["analysis"], evidence_ids=ids, basis="inference", dimensions=case["headings"])])
    cp = Checkpoints(tmp_path)
    graph = build_graph(Supervisor(model, 4), lambda _: Researcher(), Writer(model, evidence_context=context),
        gap_reviewer=CompleteReview(), synthesizer=ResearchSynthesis(model, context),
        report_reviewer=ReportReviewer(model, context), checkpointer=cp.saver)
    result = graph.invoke({"research_contract": request}, cp.config("fixture"))["report_result"]
    assert not result.partial and len(result.section_reviews) == 3
    assert all(f"## {h}" in result.report for h in case["headings"])
    assert "| 比较维度 |" in result.report and "## 参考文献" in result.report
    assert "ev_" not in result.report and "prior_findings" not in result.report
    assert "Synthetic fixture" not in result.report and "其他发现" not in result.report
    assert set(i for values in result.citation_evidence.values() for i in values) <= set(ids)
    assert len(model.calls) == 9
    assert max(len(c["messages"][-1]["content"]) for c in model.calls) < 15000
    (tmp_path / (case["name"] + ".md")).write_text(result.report)
    cp.close()


class TwoSectionWriter(FakeWriter):
    def outline(self, request, synthesis):
        return ReportOutline(title="比较", sections=[SectionPlan(section_id=k, heading=k,
            objective="比较方法", block_indices=[0]) for k in ("one", "two")])


def test_local_revision_does_not_rewrite_other_sections():
    calls = []
    class WriterDouble(TwoSectionWriter):
        def write_section(self, *args, previous=None, **kwargs):
            calls.append((args[3].section_id, bool(previous)))
            return super().write_section(*args, previous=previous, **kwargs)
    class Review(SupportedReview):
        def review(self, request, outline, section, *args):
            return ReportReview(acceptable=section.section_id != "two" or section.revision > 0,
                                issues=["收窄优劣结论"] if section.section_id == "two" and not section.revision else [])
    graph = build_graph(FakeSupervisor(), lambda _: FakeResearcher(), WriterDouble(),
        gap_reviewer=CompleteReview(), synthesizer=FakeSynthesis(), report_reviewer=Review())
    result = graph.invoke({"research_contract": contract()}, {"recursion_limit": 100})
    assert calls == [("one", False), ("two", False), ("two", True)]
    assert not result["report_result"].partial


@pytest.mark.parametrize("error", [WorkflowInterrupted("cancel"), ContextBudgetExceeded("context")])
def test_control_errors_never_turn_into_report(error):
    class Review(SupportedReview):
        def review(self, *args): raise error
    graph = build_graph(FakeSupervisor(), lambda _: FakeResearcher(), FakeWriter(),
        gap_reviewer=CompleteReview(), synthesizer=FakeSynthesis(), report_reviewer=Review())
    with pytest.raises(type(error), match=str(error)):
        graph.invoke({"research_contract": contract()})


def test_writer_failure_never_publishes_notes_or_exception():
    class Broken(FakeWriter):
        def write_section(self, *args, **kwargs):
            raise RuntimeError("SECRET provider diagnostic " + "Contents of source " * 200)
    graph = build_graph(FakeSupervisor(), lambda _: FakeResearcher(), Broken(),
        gap_reviewer=CompleteReview(), synthesizer=FakeSynthesis(), report_reviewer=SupportedReview())
    with pytest.raises(ValueError, match="尚未形成可发布"):
        graph.invoke({"research_contract": contract()})


def test_report_checkpoint_resume_keeps_finished_sections(tmp_path):
    calls = []
    class WriterDouble(TwoSectionWriter):
        def write_section(self, *args, **kwargs):
            calls.append(args[3].section_id)
            return super().write_section(*args, **kwargs)
    cp = Checkpoints(tmp_path)
    def graph():
        return build_graph(FakeSupervisor(), lambda _: FakeResearcher(), WriterDouble(),
            gap_reviewer=CompleteReview(), synthesizer=FakeSynthesis(), report_reviewer=SupportedReview(),
            checkpointer=cp.saver)
    list(graph().stream({"research_contract": contract()}, cp.config("r"),
        interrupt_after=["advance_review_node"], durability="sync"))
    assert calls == ["one"]
    cp.close()
    cp = Checkpoints(tmp_path)
    result = graph().invoke(None, cp.config("r"))
    assert calls == ["one", "two"] and not result["report_result"].partial
    cp.close()


def test_explicit_continue_only_retries_unfinished_writing(tmp_path):
    calls, fail = [], [True]
    class WriterDouble(TwoSectionWriter):
        def write_section(self, *args, **kwargs):
            calls.append(args[3].section_id)
            if args[3].section_id == "two" and fail[0]: raise RuntimeError("unavailable")
            return super().write_section(*args, **kwargs)
    cp = Checkpoints(tmp_path)
    graph = build_graph(FakeSupervisor(), lambda _: FakeResearcher(), WriterDouble(),
        gap_reviewer=CompleteReview(), synthesizer=FakeSynthesis(), report_reviewer=SupportedReview(), checkpointer=cp.saver)
    conf, request = cp.config("r"), contract()
    assert graph.invoke({"research_contract": request}, conf)["report_result"].partial
    fail[0] = False
    execution = request.execution.model_copy(update={"metadata": {"continuation": 1}})
    continue_graph(graph, conf, graph.get_state(conf), execution, request.limits)
    assert not graph.invoke(None, conf)["report_result"].partial
    assert calls == ["one", "two", "two"]
    cp.close()


def test_synthesis_inputs_exclude_poisoned_diagnostics():
    dump = "[ev_original] Contents of https://example.org/paper:\n" + "paper full text " * 20000
    note = FakeResearcher().run(SimpleNamespace(task_id="T1", objective="q", revision=0), contract())
    note.limitations.append(dump)
    note.conflicts.append("不同数据条件导致结论不同")
    payload = json.dumps(list(batches(note_units([note]))), ensure_ascii=False)
    assert len(payload) < 2000 and "paper full text" not in payload and "不同数据条件" in payload


def test_shared_source_number_preserves_fragments_and_math(tmp_path):
    context = ResearchEvidenceContext(JsonEvidenceStore(str(tmp_path)), contract().source_policy)
    ids = []
    for n in range(2):
        item = EvidenceRef.create(source_id="https://example.org/paper", source_version="v1", parser_version="test",
            original_text=f"Source fragment {n}", chunk_index=n, metadata={"kind":"source_content", "title":"One Paper"})
        context.store.save(item); context.register(item); ids.append(item.evidence_id)
    mapping = reference_catalog(context, ids)
    outline = ReportOutline(title="report", sections=[SectionPlan(section_id="s", heading="方法", objective="q", block_indices=[0])])
    section = WrittenSection(section_id="s", heading="方法", markdown="比较 [2][1]。公式 $x[2]$ 和代码 " + chr(96) + "a[2]" + chr(96))
    result = finalize_report(outline, {"s": section}, mapping, context, {"s":ReportReview(acceptable=True)}, [])
    assert "比较 [1]。" in result.report and "$x[2]$" in result.report
    assert chr(96) + "a[2]" + chr(96) in result.report
    assert len(result.citations) == 1 and set(result.citation_evidence[1]) == set(ids)
    assert check_section(section.model_copy(update={"markdown": "事实 [99]"}), mapping, context)


def test_stage_metrics_survive_restart_unknown_cache_not_miss(tmp_path):
    from sage_scholar.runtime.store import RunStore
    store = RunStore(tmp_path)
    store.record_stage("r", "research", prompt=100, completion=20, cached=70, seconds=2)
    store.record_stage("r", "research", prompt=200, completion=10, seconds=3)
    store.record_stage("r", "tool:fetch", failed=True, seconds=1)
    store.close()
    store = RunStore(tmp_path)
    data = store.stages("r")
    assert data["research"]["calls"] == 2 and data["research"]["input_tokens"] == 300
    assert data["research"]["cache_miss_tokens"] == 30 and data["research"]["seconds"] == 5
    assert data["tool:fetch"]["failures"] == 1
    store.purge_run("r")
    assert store.stages("r") == {}
    store.close()


def test_audit_requests_rejected_old_schema_not_resumed(tmp_path):
    from sage_scholar.routing import QueryRouter, Workflow
    from sage_scholar.runtime.models import RunSnapshot, RunStatus, utc_now
    with pytest.raises(ValueError, match="已移除"):
        QueryRouter().route("check", requested_workflow=Workflow.CITATION_AUDIT)
    assert QueryRouter().route("检查论文引用", document_ids=["doc"]).workflow == Workflow.QA
    cp = Checkpoints(tmp_path)
    request = contract().execution
    run = RunSnapshot(run_id="old", workflow=request.workflow, query="q", execution=request,
        status=RunStatus.RUNNING, created_at=utc_now(), updated_at=utc_now(), checkpoint={"schema":"research-v4-1"})
    assert not cp.compatible(run)
    cp.close()

