import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

from sage_scholar.application.direct_service import DirectService
from sage_scholar.application.qa_service import QAService
from sage_scholar.application.query_service import QueryCommand, QueryService, QueryStatus
from sage_scholar.domain import EvidenceRef
from sage_scholar.domain.execution import RunOutcome
from sage_scholar.routing import QueryRouter, Workflow
from sage_scholar.runtime import RunManager, RunStatus
from router_fakes import FixedClassifier


class FakeModel:
    def __init__(self, response: str = "model answer"):
        self.response = response
        self.calls = []

    async def complete(self, messages, *, temperature=0, tag=""):
        self.calls.append((messages, tag))
        return self.response


class FakeRetrieval:
    async def retrieve(self, query, *, policy, top_k=5):
        return [
            EvidenceRef(
                evidence_id="ev-1",
                source_id="paper-a",
                text="The experiments use Dataset A.",
            )
        ]


class FakeRunner:
    async def run(self, request, emit, cancelled):
        await emit("plan", {"sub_questions": [{"label": "方法", "question": request.query}]})
        await emit("write", {"report": "# report"})
        return RunOutcome(result={"report": "# report"})


def make_service(model=None, workflow="direct"):
    model = model or FakeModel()
    manager = RunManager()
    direct = DirectService(
        model,
        clock=lambda zone: datetime(2026, 9, 24, 12, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    service = QueryService(
        router=QueryRouter(FixedClassifier(workflow)),
        direct=direct,
        qa=QAService(FakeRetrieval(), model),
        run_manager=manager,
        long_runners={Workflow.DEEP_RESEARCH: FakeRunner()},
    )
    return service, manager, model


def test_direct_date_does_not_call_model():
    async def scenario():
        service, manager, model = make_service()
        result = await service.execute(QueryCommand(query="今天是星期几"))
        assert result.status is QueryStatus.ACCEPTED
        await manager._tasks[result.run_id]
        assert manager.get(result.run_id).result["answer"] == "今天是2026年09月24日，星期四。"
        assert model.calls == []
        await manager.shutdown()

    asyncio.run(scenario())


def test_direct_follow_up_forwards_conversation_history():
    async def scenario():
        service, manager, model = make_service(FakeModel("它的主要缺点是成本较高。"))
        history = [
            {"role": "user", "content": "介绍一下方法 A"},
            {"role": "assistant", "content": "方法 A 是一种数值方法。"},
        ]
        result = await service.execute(QueryCommand(query="它有什么缺点？", history=history))
        await manager._tasks[result.run_id]
        assert manager.get(result.run_id).result["answer"] == "它的主要缺点是成本较高。"
        messages, tag = model.calls[0]
        assert tag == "direct"
        assert messages[-3:] == [
            *history,
            {"role": "user", "content": "它有什么缺点？"},
        ]
        await manager.shutdown()

    asyncio.run(scenario())


def test_qa_returns_structured_evidence():
    async def scenario():
        service, manager, model = make_service(FakeModel("使用 Dataset A [1]。"), "qa")
        result = await service.execute(
            QueryCommand(query="论文中用了什么数据集？", document_ids=["paper-a"])
        )
        assert result.status is QueryStatus.ACCEPTED
        await manager._tasks[result.run_id]
        answer = manager.get(result.run_id).result
        assert answer["answer"] == "使用 Dataset A [1]。"
        assert answer["evidence"][0]["source_id"] == "paper-a"
        assert model.calls[0][1] == "qa"
        await manager.shutdown()

    asyncio.run(scenario())


def test_research_is_owned_by_run_manager():
    async def scenario():
        service, manager, _model = make_service(workflow="deep_research")
        result = await service.execute(QueryCommand(query="调研 AI4CFD 的主要方法"))
        assert result.status is QueryStatus.ACCEPTED
        assert result.run_id
        events = [event async for event in manager.events(result.run_id)]
        assert any(event.type == "write" for event in events)
        assert manager.get(result.run_id).status is RunStatus.COMPLETED
        await manager.shutdown()

    asyncio.run(scenario())
