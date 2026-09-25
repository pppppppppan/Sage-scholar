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
