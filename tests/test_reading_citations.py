import asyncio

from sage_scholar.application.citations import render_citations
from sage_scholar.application.qa_service import QAService
from sage_scholar.domain import EvidenceRef
from sage_scholar.domain.execution import ExecutionRequest
from sage_scholar.routing import ExecutionMode, SourceScope, Workflow
from test_document_evidence import library  # pytest fixture
from test_query_service import FakeModel


def request(query, document_id):
    return ExecutionRequest(query=query, workflow=Workflow.QA, source_scope=SourceScope.SELECTED_DOCUMENTS,
                            document_ids=(document_id,), allow_web=False, execution_mode=ExecutionMode.INLINE)


def test_targeted_qa_calls_retrieval(library):
    _, adapter, a, _, _, pipeline = library
    result = asyncio.run(QAService(adapter, FakeModel("Dataset Alpha [1]")).answer(
        request("这篇论文用了什么数据集？", a.document_id)
    ))
    assert pipeline.allowed is not None
    assert result.reading_strategy == "targeted"
    assert result.evidence and result.citations[1] == result.evidence[0].evidence_id


def test_overview_reads_every_section_without_top_k(library):
    _, adapter, a, _, _, pipeline = library
    model = FakeModel("section summary")
    result = asyncio.run(QAService(adapter, model).answer(request("概括这篇论文", a.document_id)))
    assert pipeline.allowed is None
    assert result.reading_strategy == "overview"
    sections = [messages[-1]["content"] for messages, tag in model.calls if tag == "qa_section"]
    assert len(sections) == 3
    assert any("Needs more testing" in text for text in sections)
    assert {item.section for item in result.evidence} == {"Method", "Experiments", "Limitations"}


def test_citation_identity_and_invalid_numbers_are_checked():
    evidence = [EvidenceRef(evidence_id="ev_test", source_id="paper", text="fact")]
    rendering = render_citations("Supported [ev_test]. Wrong [7] [1,9] [ev_fake].", evidence)
    assert rendering.citations == {1: "ev_test"}
    assert "Supported [1]" in rendering.text
    assert "[7]" not in rendering.text
    assert len(rendering.issues) == 3


def test_qa_returns_citation_warning_in_api_contract(library):
    _, adapter, a, _, _, _ = library
    result = asyncio.run(QAService(adapter, FakeModel("Dataset Alpha [99]")).answer(
        request("这篇论文用了什么数据集？", a.document_id)
    ))
    assert result.citation_issues
    assert "【引用无效】" in result.answer
    assert "引用检查" in result.answer


def test_overview_section_citations_are_canonical_before_synthesis(library):
    _, adapter, a, _, _, _ = library
    model = FakeModel("summary [1]")
    result = asyncio.run(QAService(adapter, model).answer(request("概括全文", a.document_id)))
    final_context = next(messages[-1]["content"] for messages, tag in model.calls if tag == "qa")
    assert all(f"[{item.evidence_id}]" in final_context for item in result.evidence)
    assert "summary [1]" not in final_context


def test_overview_covers_more_than_top_five_chunks(library):
    manager, adapter, a, _, source, _ = library
    source.write_text("\n".join(f"# Section {i}\nDistinct finding {i}" for i in range(12)))
    uploaded = manager.ingest(str(source))
    model = FakeModel("summary")
    result = asyncio.run(QAService(adapter, model).answer(request("概括全文", uploaded.document_id)))
    assert len(result.evidence) == 12
    assert sum(tag == "qa_section" for _, tag in model.calls) == 12
    assert any("Distinct finding 11" in messages[-1]["content"] for messages, tag in model.calls if tag == "qa_section")
