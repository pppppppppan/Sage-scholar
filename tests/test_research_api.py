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
                result = {"analysis": "read original", "tasks": [{"task_id": "T1", "label": "方法", "objective": "Read Method", "dimensions": ["methods"], "completion_criteria": ["解释 Method A"]}]}
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
            elif tag == "supervisor:gap_review":
                result = {"covered": ["methods"]}
            elif tag == "synthesis:combine":
                payload = json.loads(messages[-1]["content"])
                result = {"blocks": [{"heading": "方法", "analysis": "原文介绍 Method A，实验细节未提供。",
                    "evidence_ids": [item["evidence_id"] for item in payload["evidence"]]}]}
            elif tag == "writer:outline":
                result = {"title": "Research Report", "sections": [{"section_id": "s1", "heading": "方法",
                    "objective": "解释方法", "block_indices": [0]}]}
            elif tag in {"writer:section", "writer:revise_section"}:
                payload = json.loads(messages[-1]["content"])
                assert all(item["document_id"] == a.document_id for item in payload["sources"])
                result = "Method A [1]。实验结果未提供。"
                if unsupported and tag == "writer:section":
                    result += "A scores 100%"
            elif tag == "report_review:section":
                payload = json.loads(messages[-1]["content"])
                assert payload["sources"] and "SECRET B" not in str(payload["sources"])
                bad = "A scores 100%" in payload["section"]["markdown"]
                result = {"acceptable": not bad, "issues": ["实验数字缺少支持"] if bad else [],
                          "revision_instructions": "删除没有来源的数字"}
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
        assert "event: report_reviewing" in sse and "event: write" in sse and "event: run_finished" in sse
        assert '"phase": "started"' in sse
        snapshot = client.get(f"/api/runs/{run_id}").json()["run"]
        assert snapshot["status"] == "completed", snapshot
        result = snapshot["result"]
        assert result["citations"] and "Method A [1]" in result["report"]
        assert "A scores 100%" not in result["report"]
        assert not any("工具调用预算" in gap for gap in snapshot["remaining_gaps"])
        assert '"blocked_by_budget": false' in sse
        assert snapshot["remaining_gaps"] == []
        assert ("event: report_revising" in sse) == unsupported
        assert "removed_claim_ids" not in result
        for identity in result["citations"].values():
            evidence = client.get("/api/evidence/" + identity)
            assert evidence.status_code == 200
            assert evidence.json()["document_id"] == a.document_id
            assert "Method A" in evidence.json()["original_text"]
    assert len(pipeline.vector_store.chunks) == before_chunks  # No report -> RAG write.
