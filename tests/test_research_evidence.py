import asyncio
import json
from pathlib import Path

import pytest
from openai.types.chat import ChatCompletionMessage

from sage_scholar.agents.researcher import Researcher
from sage_scholar.domain.evidence import EvidenceRef
from sage_scholar.domain.execution import SourcePolicy
from sage_scholar.domain.research import Finding, ResearchNote, ResearchTask
from sage_scholar.infrastructure.evidence_store import JsonEvidenceStore
from sage_scholar.research.evidence_context import ResearchEvidenceContext
from sage_scholar.research.notes import validate_note
from sage_scholar.routing import SourceScope
from sage_scholar.tools import BaseTool
from sage_scholar.tools.policy import EvidenceTool
from research_fakes import ScriptedLLM, contract
from test_document_evidence import library


class SourceReader(BaseTool):
    source_access = "public"
    evidence_kind = "source_content"

    def __init__(self, result="Method A scores 90 on Dataset X.", name="fetch"):
        super().__init__(name, "source reader", [])
        self.result = result

    def run_tool(self, parameters):
        return self.result


@pytest.fixture
def evidence_context(tmp_path):
    store = JsonEvidenceStore(str(tmp_path))
    context = ResearchEvidenceContext(store, SourcePolicy(source_scope=SourceScope.PUBLIC, allow_web=True))
    for url, text in [("https://example.org/result", "Method A scores 90; Method B scores 80 on Dataset X."),
                      ("https://example.org/cost", "Method A costs 20 GPU hours; B costs 10 GPU hours.")]:
        EvidenceTool(SourceReader(text), store, context.register).run_tool({"url": url})
    return context


@pytest.mark.parametrize("result", ["Error: no access", "404 Not Found", '{"error":"blocked"}',
                                    '{"status":"downloaded","path":"paper.pdf"}', ""])
def test_tool_error_and_metadata_are_observations(tmp_path, result):
    store, captured, observations = JsonEvidenceStore(str(tmp_path)), [], []
    output = EvidenceTool(SourceReader(result), store, captured.append, observations.append).run_tool(
        {"url": "https://example.org"})
    assert "Observation" in output
    assert captured == [] and observations[0]["evidence_ids"] == []


def test_unopened_search_snippets_are_not_evidence(tmp_path):
    tool = SourceReader("Search result: Method A scores 90")
    tool.evidence_kind = "observation"
    captured = []
    result = EvidenceTool(tool, JsonEvidenceStore(str(tmp_path)), captured.append).run_tool(
        {"url": "https://example.org"})
    assert captured == [] and "Observation" in result


def test_source_text_is_preserved_with_stable_version_and_chunk_locations(tmp_path):
    text = "exact original text " * 1100
    store, captured = JsonEvidenceStore(str(tmp_path)), []
    tool = EvidenceTool(SourceReader(text), store, captured.append)
    tool.run_tool({"url": "https://example.org/source"})
    assert len(captured) > 1
    assert "".join(item.original_text for item in captured) == text
    assert len({item.source_version for item in captured}) == 1
    first_ids = [item.evidence_id for item in captured]
    for item in captured:
        assert store.get(item.evidence_id).original_text == text[item.location["char_start"]:item.location["char_end"]]
        assert item.metadata["kind"] == "source_content"
    captured.clear()
    tool.run_tool({"url": "https://example.org/source"})
    assert [item.evidence_id for item in captured] == first_ids


def test_note_validation_cleans_fabricated_and_other_task_ids(evidence_context):
    a, b = evidence_context.ids()
    note = ResearchNote(task_id="T1", question="compare", evidence_ids=[a, b, "ev_fake"],
        findings=[Finding(text="fact", evidence_ids=[a, "ev_fake"])],
        unknowns=["preprocessing unclear"], conflicts=["dataset size differs"])
    cleaned = validate_note(note, evidence_context, [a])
    assert cleaned.evidence_ids == cleaned.findings[0].evidence_ids == [a]
    assert cleaned.metadata["validation_issues"]
    assert cleaned.unknowns == note.unknowns and cleaned.conflicts == note.conflicts
    assert note.evidence_ids == [a, b, "ev_fake"]


def test_actual_react_loop_submits_typed_note(tmp_path):
    store = JsonEvidenceStore(str(tmp_path))
    context = ResearchEvidenceContext(store, contract().source_policy)
    tool = EvidenceTool(SourceReader(), store, context.register)
    class Model(ScriptedLLM):
        def invoke(self, **kwargs):
            if kwargs["tag"].endswith(":react") and not context.ids():
                return ChatCompletionMessage(role="assistant", content=None, tool_calls=[{
                    "id": "call_1", "type": "function", "function": {"name": "fetch",
                        "arguments": '{"url":"https://example.org/paper"}'}}])
            return super().invoke(**kwargs)
    def submit(kwargs):
        assert "allowed_evidence_ids" in kwargs["messages"][1]["content"]
        return {"findings": [{"text": "A scores 90", "evidence_ids": context.ids()}],
                "evidence_ids": context.ids() + ["ev_fabricated"],
                "unknowns": ["hardware unspecified"], "conflicts": ["other experiment differs"]}
    researcher = Researcher("T1", Model("summary with source IDs", submit), [tool],
                            evidence_context=context, observed_ids=context.ids)
    note = researcher.run(ResearchTask(task_id="T1", objective="compare", revision=2), contract())
    assert isinstance(note, ResearchNote)
    assert note.evidence_ids == context.ids()
    assert note.task_id == "T1" and note.revision == 2 and note.tool_usage == {"fetch": 1}
    assert note.unknowns == ["hardware unspecified"] and note.conflicts == ["other experiment differs"]
    assert note.metadata["validation_issues"]
    assert "exact original" not in note.model_dump_json()


def test_reread_rejects_missing_or_other_run_evidence(evidence_context):
    identity = evidence_context.ids()[0]
    context = ResearchEvidenceContext(evidence_context.store, contract().source_policy)
    with pytest.raises(ValueError, match="不属于"):
        context.get(identity)
    item = evidence_context.get(identity)
    context.register(item)
    evidence_context.store._path(identity).unlink()
    with pytest.raises(ValueError, match="不存在"):
        context.get(identity)


def test_local_version_and_scope_remain_verifiable_after_reingestion(library):
    manager, adapter, a, b, source, _ = library
    policy = SourcePolicy(source_scope=SourceScope.SELECTED_DOCUMENTS, document_ids=(a.document_id,))
    old = asyncio.run(adapter.read_documents(policy=policy))[0]
    context = ResearchEvidenceContext(adapter.evidence_store, policy, adapter.documents)
    context.register(old)
    source.write_text("updated paper")
    manager.ingest(str(source))
    assert context.get(old.evidence_id).source_version == old.source_version
    foreign = EvidenceRef.create(**{**old.model_dump(exclude={"evidence_id", "text"}), "document_id": b.document_id})
    adapter.evidence_store.save(foreign)
    context.register(foreign)
    with pytest.raises(ValueError, match="范围"):
        context.get(foreign.evidence_id)
    Path(old.metadata["converted_path"]).unlink()
    with pytest.raises(ValueError, match="不可定位"):
        context.get(old.evidence_id)


def test_public_evidence_never_bypasses_no_web(evidence_context):
    item = evidence_context.get(evidence_context.ids()[0])
    context = ResearchEvidenceContext(evidence_context.store,
        SourcePolicy(source_scope=SourceScope.PUBLIC, allow_web=False))
    context.register(item)
    with pytest.raises(ValueError):
        context.get(item.evidence_id)


def test_success_status_is_not_a_source_failure(tmp_path):
    observations = []
    EvidenceTool(SourceReader('{"status":"downloaded","path":"paper.pdf"}'),
                 JsonEvidenceStore(str(tmp_path)), lambda _: pytest.fail("status is not evidence"),
                 observations.append).run_tool({"url": "https://example.org/paper"})
    assert observations[0]["kind"] == "discovery_or_status"


def test_tool_exception_is_externalized_but_never_promoted(tmp_path):
    class Failing(SourceReader):
        def run_tool(self, parameters):
            raise RuntimeError("source unavailable")
    observations = []
    with pytest.raises(RuntimeError):
        EvidenceTool(Failing(), JsonEvidenceStore(str(tmp_path)),
                     lambda _: pytest.fail("error is not evidence"), observations.append).run_tool({})
    assert observations[0]["kind"] == "error" and observations[0]["evidence_ids"] == []
