from types import SimpleNamespace

from fastapi.testclient import TestClient

from sage_scholar.api.app import create_app
from sage_scholar.application.query_service import QueryResult, QueryStatus
from sage_scholar.routing import QueryRouter


class FakeQueryService:
    def __init__(self):
        self.last_command = None

    async def execute(self, command):
        self.last_command = command
        return QueryResult(
            request_id="request-1",
            route=QueryRouter().route(command.query),
            status=QueryStatus.INLINE,
            answer="今天是星期四。",
        )


class FakeRunManager:
    def get(self, run_id):
        return None

    async def shutdown(self):
        return None


class FakeDocumentService:
    async def list_documents(self):
        return []


class FakeContainer:
    def __init__(self):
        self.query_service = FakeQueryService()
        self.run_manager = FakeRunManager()
        self.document_service = FakeDocumentService()
        self.config = SimpleNamespace(data_dir="/tmp")

    async def close(self):
        return None


def test_query_endpoint_uses_query_service():
    container = FakeContainer()
    with TestClient(create_app(container)) as client:
        response = client.post(
            "/api/queries",
            json={
                "query": "它有什么缺点？",
                "history": [
                    {"role": "user", "content": "介绍一下方法 A"},
                    {"role": "assistant", "content": "方法 A 是一种数值方法。"},
                ],
            },
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["route"]["workflow"] == "direct"
    assert payload["answer"] == "今天是星期四。"
    assert container.query_service.last_command.history[1]["role"] == "assistant"


def test_existing_api_paths_and_methods_remain():
    expected = {
        ("/api/health", "get"), ("/api/queries", "post"),
        ("/api/runs/{run_id}", "get"), ("/api/runs/{run_id}/events", "get"),
        ("/api/runs/{run_id}/cancel", "post"), ("/api/library", "get"),
        ("/api/library/ingest", "post"), ("/api/library/upload", "post"),
        ("/api/library/save-report", "post"), ("/api/library/{title}", "delete"),
    }
    paths = create_app(FakeContainer()).openapi()["paths"]
    assert expected <= {(path, method) for path, methods in paths.items() for method in methods}


def test_library_upload_qa_evidence_and_sse_compatibility(tmp_path):
    from sage_scholar.application.direct_service import DirectService
    from sage_scholar.application.document_service import DocumentService
    from sage_scholar.application.qa_service import QAService
    from sage_scholar.application.query_service import QueryService
    from sage_scholar.config import Config
    from sage_scholar.domain.execution import RunOutcome
    from sage_scholar.infrastructure.retrieval import LegacyRetrievalAdapter
    from sage_scholar.routing import Workflow
    from sage_scholar.runtime import RunManager
    from test_document_evidence import LocalPipeline
    from test_query_service import FakeModel

    config = Config(data_dir=str(tmp_path / "data"))
    pipeline = LocalPipeline()
    retrieval = LegacyRetrievalAdapter(config, pipeline)
    model = FakeModel("Dataset Alpha [1]")
    manager = RunManager()
    class PartialRunner:
        async def run(self, request, emit, cancelled):
            await emit("write", {"report": "report"})
            return RunOutcome(status="partial", result={"report": "report"},
                              reason="missing evidence", remaining_gaps=["ablation"])
    class Container:
        async def close(self):
            await manager.shutdown()
    container = Container()
    container.config = config
    container.run_manager = manager
    container.evidence_store = retrieval.evidence_store
    container.document_service = DocumentService(None, config.data_dir, lambda: pipeline)
    container.query_service = QueryService(QueryRouter(), DirectService(model), QAService(retrieval, model),
                                            manager, {Workflow.DEEP_RESEARCH: PartialRunner()})
    with TestClient(create_app(container)) as client:
        assert client.get("/api/health").status_code == 200
        response = client.post("/api/library/upload", files={
            "file": ("paper.md", b"# Method\nDataset Alpha", "text/markdown"),
        })
        assert response.status_code == 200, response.text
        document = response.json()
        assert document["ingestion_status"] == "ready"
        assert client.get("/api/library").json()[0]["document_id"] == document["document_id"]
        response = client.post("/api/queries", json={
            "query": "论文用了什么数据集", "workflow": "qa",
            "source_scope": "selected_documents", "document_ids": [document["document_id"]],
            "allow_web": False,
        })
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["citations"]["1"] == payload["evidence"][0]["evidence_id"]
        evidence_id = payload["evidence"][0]["evidence_id"]
        saved = client.get("/api/evidence/" + evidence_id).json()
        assert saved["document_id"] == document["document_id"]
        assert saved["page"] is None
        assert saved["original_text"] == saved["text"]
        response = client.post("/api/queries", json={"query": "调研 AI"})
        assert response.status_code == 202
        run_id = response.json()["run_id"]
        events = client.get(f"/api/runs/{run_id}/events")
        assert events.status_code == 200
        assert "event: run_finished" in events.text and '"status": "partial"' in events.text
        snapshot = client.get(f"/api/runs/{run_id}").json()["run"]
        assert snapshot["status"] == "partial"
        assert snapshot["execution"]["source_scope"] == "public"
        sequence = int(events.text.split("id: ")[-1].splitlines()[0])
        assert client.get(f"/api/runs/{run_id}/events?after={sequence}").text == ""
        assert client.post(f"/api/runs/{run_id}/cancel").json()["run"]["status"] == "partial"
        assert client.delete("/api/library/" + document["document_id"]).status_code == 200
        assert client.get("/api/library").json() == []
        assert client.get("/api/evidence/" + evidence_id).status_code == 200
