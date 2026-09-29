"""V4.1: discover identities, read original sources, never infer missing citations."""
import json
from types import SimpleNamespace

import pytest

from research_fakes import ScriptedLLM, contract
from sage_scholar.audit.bibliography import parse_reference
from sage_scholar.audit.fetch import AuditFetch, AuditSourceTools
from sage_scholar.audit.models import AuditClaim
from sage_scholar.audit.service import CitationAuditService
from sage_scholar.audit.sources import AuditSources
from sage_scholar.domain import EvidenceRef
from sage_scholar.infrastructure.evidence_store import JsonEvidenceStore
from sage_scholar.research.verification import CitationVerifier
from sage_scholar.routing import SourceScope
from sage_scholar.tools.base_tool import BaseTool

CITATION = "Vaswani et al. Attention Is All You Need. NeurIPS 2017."
PAPER = {"title": "Attention Is All You Need", "authors": "Ashish Vaswani; Noam Shazeer",
         "published_date": "2017-06-12", "paper_id": "1706.03762", "source": "arxiv",
         "url": "https://arxiv.org/abs/1706.03762", "abstract": "SEARCH SNIPPET, NOT EVIDENCE"}


class Tool(BaseTool):
    source_access = "public"
    def __init__(self, name, result, kind="observation"):
        super().__init__(name, name, [])
        self.result, self.evidence_kind, self.calls = result, kind, []
    def run_tool(self, parameters):
        self.calls.append(parameters)
        return self.result


@pytest.fixture
def setup(tmp_path):
    store = JsonEvidenceStore(str(tmp_path))
    retrieval = SimpleNamespace(evidence_store=store, documents=None)
    search = Tool("mcp__paper-search__search_papers", json.dumps({"papers": [PAPER]}))
    read = Tool("read_arxiv_paper", "ORIGINAL: A beats B on Dataset X only.", "source_content")
    fetch = Tool("mcp__fetch__fetch", "ORIGINAL web text.", "source_content")
    tools = {t.name: t for t in (search, read, fetch)}
    starts = []
    def get(require_external):
        starts.append(require_external)
        return SimpleNamespace(registry=SimpleNamespace(tools=tools), _initialize_research_tools=lambda: [])
    request = contract().execution
    adapter = AuditFetch(SimpleNamespace(get=get), retrieval).for_request(request)
    assert starts == []  # Lazy construction, no MCP initialization.
    return retrieval, request, adapter, search, read, fetch, starts


@pytest.mark.parametrize("reference", [CITATION, "Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS.",
                                      'Vaswani et al. "Attention Is All You Need". 2017.'])
def test_bibliographic_citation_searches_matches_then_reads_original(setup, reference):
    retrieval, request, adapter, search, read, _, _ = setup
    sources = AuditSources(retrieval, request.source_policy, adapter)
    identity = sources.resolve("1", reference)
    assert identity.status == "source_found", identity.reason
    assert identity.source == reference and identity.citation == "1"
    assert identity.metadata["title"] == PAPER["title"]
    assert len(search.calls) == len(read.calls) == 1
    assert search.calls[0]["query"] == PAPER["title"]
    assert identity.evidence_ids
    for eid in identity.evidence_ids:
        text = sources.context.get(eid).original_text
        assert "ORIGINAL" in text and "SNIPPET" not in text
    # The same reference used by another claim reuses its identity and source content.
    assert sources.resolve("2", reference).citation == "2"
    assert len(search.calls) == len(read.calls) == 1


@pytest.mark.parametrize("reference,locator", [
    ("DOI:10.1000/ABC", "https://doi.org/10.1000/abc"),
    (CITATION + " DOI:10.1000/ABC https://example.org/other", "https://doi.org/10.1000/abc"),
    ("arXiv:1706.03762", "arxiv:1706.03762"),
    ("https://arxiv.org/pdf/1706.03762v2.pdf", "arxiv:1706.03762v2"),
    ("1706.03762", "arxiv:1706.03762"),
    ("https://example.org/source", "https://example.org/source"),
])
def test_explicit_identifiers_bypass_title_search(setup, reference, locator):
    retrieval, request, adapter, search, read, fetch, _ = setup
    identity = AuditSources(retrieval, request.source_policy, adapter).resolve(reference, reference)
    assert identity.status == "source_found", identity.reason
    assert identity.resolved_source == locator
    assert search.calls == []
    assert len(read.calls) + len(fetch.calls) == 1


@pytest.mark.parametrize("papers", [[], [{**PAPER, "authors": "Other Author"}],
    [{**PAPER, "published_date": "2018-01-01"}], [{**PAPER, "title": "Attention Is NOT All You Need"}],
    [PAPER, {**PAPER, "paper_id": "1801.12345", "url": "https://arxiv.org/abs/1801.12345"}],
])
def test_missing_mismatched_or_ambiguous_metadata_is_not_guessed(setup, papers):
    retrieval, request, adapter, search, read, fetch, _ = setup
    search.result = json.dumps({"papers": papers})
    identity = AuditSources(retrieval, request.source_policy, adapter).resolve("1", CITATION)
    assert identity.status == "source_not_found" and identity.evidence_ids == []
    assert "fabricated" not in identity.model_dump_json()
    assert not read.calls and not fetch.calls


def test_duplicate_hits_with_same_identifiers_are_one_identity(setup):
    retrieval, request, adapter, search, _, _, _ = setup
    search.result = json.dumps({"papers": [PAPER, {**PAPER, "doi": "10.48550/arXiv.1706.03762"}]})
    assert AuditSources(retrieval, request.source_policy, adapter).resolve("1", CITATION).status == "source_found"


def test_identity_found_without_fulltext_stays_uncertain_and_has_no_snippet_evidence(setup):
    retrieval, request, adapter, _, read, _, _ = setup
    read.result = "Error: source temporarily unavailable"
    sources = AuditSources(retrieval, request.source_policy, adapter)
    llm = ScriptedLLM()  # No text, no semantic verification call.
    result = CitationAuditService(retrieval, lambda: llm).verify_claim(1,
        AuditClaim(text="A always beats B", citations=[CITATION]), sources, {}, CitationVerifier(llm, sources.context))
    assert result.source_identity_status == "source_found" and result.support_status == "uncertain"
    assert result.evidence_ids == [] and llm.calls == []


@pytest.mark.parametrize("extracted", [True, False])
def test_missing_citation_never_binds_selected_documents(setup, extracted):
    retrieval, request, adapter, search, read, fetch, starts = setup
    claim = {"text": "方法 A 在所有任务上都优于 B。", "citations": []}
    metadata = {"audit": {"text": claim["text"]}} if extracted else {"audit": {"claims": [claim]}}
    request = request.model_copy(update={"metadata": metadata, "source_scope": SourceScope.SELECTED_DOCUMENTS,
        "document_ids": ("doc_a", "doc_b", "doc_c"), "allow_web": False})
    llm = ScriptedLLM({"claims": [claim]}) if extracted else ScriptedLLM()
    service = CitationAuditService(retrieval, lambda: llm)
    claims, mapping = service.parse(request, llm)
    assert claims[0].citations == []
    sources = AuditSources(retrieval, request.source_policy, adapter)
    result = service.verify_claim(1, claims[0], sources, mapping, CitationVerifier(llm, sources.context))
    assert result.citation_status == "missing" and result.source_identity_status == "not_applicable"
    assert result.support_status == "uncertain" and result.evidence_ids == result.sources == []
    assert "missing_citation" in service.outcome([result]).result["report"]
    assert starts == [] and not search.calls and not read.calls and not fetch.calls


def test_no_web_policy_prevents_bibliographic_network_discovery(setup):
    retrieval, request, adapter, _, _, _, starts = setup
    request = request.model_copy(update={"allow_web": False})
    result = AuditSources(retrieval, request.source_policy, adapter).resolve("1", CITATION)
    assert result.status == "source_not_found" and starts == []


@pytest.mark.parametrize("status", ["supported", "partially_supported", "unsupported"])
def test_discovered_source_still_uses_existing_claim_verifier(setup, status):
    retrieval, request, adapter, _, _, _, _ = setup
    sources = AuditSources(retrieval, request.source_policy, adapter)
    llm = ScriptedLLM({"status": status, "reason": "The experiment only concerns Dataset X"})
    result = CitationAuditService(retrieval, lambda: llm).verify_claim(1,
        AuditClaim(text="A always beats B", citations=["1"]), sources, {"1": CITATION}, CitationVerifier(llm, sources.context))
    assert result.citation_status == "present" and result.source_identity_status == "source_found"
    assert result.support_status == status and result.evidence_ids
    payload = json.loads(llm.calls[0]["messages"][-1]["content"])
    assert "ORIGINAL" in payload["evidence"][0]["original_text"]
    assert "SNIPPET" not in json.dumps(payload)


def test_explicit_evidence_and_local_documents_keep_existing_policy_checks(tmp_path):
    from test_document_evidence import LocalPipeline
    from sage_scholar.config import Config
    from sage_scholar.infrastructure.retrieval import LegacyRetrievalAdapter
    from sage_scholar.library.library_manager import LibraryManager
    config = Config(data_dir=str(tmp_path / "data"))
    pipeline = LocalPipeline()
    manager = LibraryManager(config.data_dir, paper_tool=None, pdfmux_tool=None, pipeline=pipeline)
    paper = tmp_path / "paper.md"; paper.write_text("# Method\nA beats B on X only")
    doc = manager.ingest(str(paper), custom_title="paper")
    retrieval = LegacyRetrievalAdapter(config, pipeline)
    request = contract().execution.model_copy(update={"source_scope": SourceScope.SELECTED_DOCUMENTS,
        "document_ids": (doc.document_id,), "allow_web": False})
    sources = AuditSources(retrieval, request.source_policy)
    found = sources.resolve(doc.document_id, doc.document_id)
    assert found.status == "source_found" and found.evidence_ids
    assert sources.resolve("1", found.evidence_ids[0]).status == "source_found"
    other = request.model_copy(update={"document_ids": ("doc_other",)})
    assert AuditSources(retrieval, other.source_policy).resolve("1", found.evidence_ids[0]).status == "source_not_found"


def test_mcp_metadata_decoder_handles_separate_json_content_blocks():
    assert AuditSourceTools._papers(json.dumps(PAPER) + "\n" + json.dumps(PAPER)) == [PAPER, PAPER]


def test_multiple_results_are_disambiguated_by_metadata_not_order(setup):
    retrieval, request, adapter, search, read, _, _ = setup
    search.result = json.dumps({"papers": [
        {**PAPER, "title": "A Different Paper", "paper_id": "1801.12345", "url": "https://arxiv.org/abs/1801.12345"},
        {**PAPER, "authors": "Other Author"}, PAPER,
    ]})
    found = AuditSources(retrieval, request.source_policy, adapter).resolve("1", CITATION)
    assert found.status == "source_found" and found.resolved_source == "arxiv:1706.03762"
    assert read.calls == [{"paper_id": "1706.03762"}]


def test_one_unread_citation_cannot_be_hidden_by_another_supported_source(setup):
    retrieval, request, adapter, _, read, _, _ = setup
    existing = EvidenceRef.create(source_id="https://example.org/source", source_version="v1", parser_version="test",
        original_text="A beats B on X", metadata={"kind": "source_content"})
    retrieval.evidence_store.save(existing)
    read.result = "Error: inaccessible fulltext"
    sources = AuditSources(retrieval, request.source_policy, adapter)
    llm = ScriptedLLM({"status": "supported", "reason": "The first Evidence supports the claim"})
    result = CitationAuditService(retrieval, lambda: llm).verify_claim(1,
        AuditClaim(text="A beats B on X", citations=[existing.evidence_id, CITATION]), sources, {}, CitationVerifier(llm, sources.context))
    assert result.source_identity_status == "source_found"
    assert result.support_status == "uncertain" and result.evidence_ids == [existing.evidence_id]
