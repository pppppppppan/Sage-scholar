"""Whole HTTP -> Run -> real research agents/graph -> citations, without paid APIs."""
import json

import pytest
from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletionMessage

from sage_scholar.api.app import create_app
from sage_scholar.application.direct_service import DirectService
from sage_scholar.application.query_service import QueryService
from sage_scholar.application.research_service import ResearchService
from sage_scholar.config import Config
from sage_scholar.research.runtime import ResearchRuntime
from sage_scholar.routing import QueryRouter, Workflow
from sage_scholar.runtime import RunManager
from test_document_evidence import library
from test_query_service import FakeModel


@pytest.mark.parametrize("unsupported", [False, True])
@pytest.mark.parametrize("tool_limit", [1, 60])
def test_research_api_citations_progress_and_partial(monkeypatch, tmp_path, library, unsupported, tool_limit):
    _, adapter, a, _, _, pipeline = library
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "agents.json").write_text('{"researcher":{"allowed_tools":["rag_search"]}}')
    config = Config(data_dir=adapter._config.data_dir, config_dir=str(config_dir), max_rounds=0)
    config.llm.review_model = None
    class Model:
        total_calls = total_prompt_tokens = total_completion_tokens = 0
        def invoke(self, messages, tag, **kwargs):
            self.total_calls += 1
            if tag == "supervisor:plan":
                result = {"sub_questions": [{"task_id": "T1", "label": "方法", "question": "Read Method",
                    "rationale": "read original", "dimensions": ["methods"]}]}
            elif tag.endswith(":react"):
                if not any(message["role"] == "tool" for message in messages):
                    return ChatCompletionMessage(role="assistant", content=None, tool_calls=[{
                        "id": "call1", "type": "function", "function": {"name": "rag_search",
                            "arguments": '{"query":"method"}'}}])
                result = "Method A is described in the paper."
            elif tag.endswith(":note"):
                payload = json.loads(messages[-1]["content"])
                result = {"findings": [{"text": "Method A", "evidence_ids": payload["allowed_evidence_ids"],
                    "dimensions": ["methods"]}], "evidence_ids": payload["allowed_evidence_ids"]}
            elif tag == "research:gap_review":
                result = {"covered": ["methods"]}
            elif tag.startswith("writer:"):
                payload = json.loads(messages[-1]["content"])
                ids = [item["evidence_id"] for item in payload["evidence"]]
                assert all(item["document_id"] == a.document_id for item in payload["evidence"])
                result = {"title": "Research Report", "claims": [{"claim_id": "C1", "text": "Method A", "evidence_ids": ids}]}
                if unsupported:
                    result["claims"].append({"claim_id": "C2", "text": "A scores 100%", "evidence_ids": ids})
            elif tag == "research:verify_claim":
                payload = json.loads(messages[-1]["content"])
                text = payload["claim"]["text"]
                assert payload["evidence"] and "SECRET B" not in str(payload["evidence"])
                result = {"status": "supported" if text == "Method A" else "unsupported",
                          "reason": "method is named" if text == "Method A" else "no numeric result in source"}
            else:
                raise AssertionError(tag)
            return ChatCompletionMessage(role="assistant", content=json.dumps(result) if isinstance(result, dict) else result)
    def forbid_mcp(*args):
        raise AssertionError("local-only research attempted external connection")
    monkeypatch.setattr("sage_scholar.research.runtime.create_mcp_clients", forbid_mcp)
    runtime = ResearchRuntime(config, llm_client=Model(), retrieval=adapter, connect_external=False)
    class Provider:
        def get(self, require_external):
            assert require_external is False
            return runtime
    manager = RunManager()
    class Container:
        async def close(self):
            await manager.shutdown()
    container = Container()
    container.config, container.run_manager, container.evidence_store = config, manager, adapter.evidence_store
    container.query_service = QueryService(QueryRouter(), DirectService(FakeModel()), None, manager,
        {Workflow.DEEP_RESEARCH: ResearchService(Provider())})
    before_chunks = len(pipeline.vector_store.chunks)
    with TestClient(create_app(container)) as client:
        response = client.post("/api/queries", json={"query": "调研指定论文的方法", "workflow": "deep_research",
            "source_scope": "selected_documents", "document_ids": [a.document_id], "allow_web": False,
            "budget": {"max_tool_calls": tool_limit}})
        assert response.status_code == 202, response.text
        run_id = response.json()["run_id"]
        sse = client.get(f"/api/runs/{run_id}/events").text
        assert "event: research_planned" in sse and "event: research_task_completed" in sse
        assert "event: citation_verifying" in sse and "event: write" in sse and "event: run_finished" in sse
        assert '"phase": "started"' in sse
        snapshot = client.get(f"/api/runs/{run_id}").json()["run"]
        assert snapshot["status"] == ("partial" if unsupported else "completed"), snapshot
        result = snapshot["result"]
        assert result["citations"] and "Method A [1]" in result["report"]
        assert "A scores 100%" not in result["report"]
        assert not any("工具调用预算" in gap for gap in snapshot["remaining_gaps"])
        assert '"blocked_by_budget": false' in sse
        assert bool(snapshot["remaining_gaps"]) == unsupported
        assert bool(result["removed_claim_ids"]) == unsupported
        for identity in result["citations"].values():
            evidence = client.get("/api/evidence/" + identity)
            assert evidence.status_code == 200
            assert evidence.json()["document_id"] == a.document_id
            assert "Method A" in evidence.json()["original_text"]
    assert len(pipeline.vector_store.chunks) == before_chunks  # No report -> RAG write.
