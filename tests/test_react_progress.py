"""Committed model/tool steps survive Stop, exceptions and reopening SQLite."""
import json
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletionMessage

from research_fakes import FakeContext, research_request
from test_agent_context import provider
from sage_scholar.agents.researcher import Researcher
from sage_scholar.context.manager import ContextManager, request_context
from sage_scholar.domain.research import ResearchTask
from sage_scholar.research.progress import ResearchProgress
from sage_scholar.runtime.control import RunControl, run_control
from sage_scholar.runtime.journal import JournaledTool, WorkflowInterrupted
from sage_scholar.runtime.store import RunStore
from sage_scholar.tools import BaseTool


def answer(text="## 核心发现\n方法及局限 [ev_a]", reason="stop"):
    value = ChatCompletionMessage(role="assistant", content=text)
    value.finish_reason = reason
    return value


def calls(*indices):
    return ChatCompletionMessage(role="assistant", content=None, tool_calls=[{
        "id": f"call_{i}", "type": "function", "function": {
            "name": "read", "arguments": json.dumps({"page": i})}} for i in indices])


class Model:
    def __init__(self, *responses):
        self.responses, self.calls = iter(responses), []

    def invoke(self, **kwargs):
        self.calls.append(kwargs)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class Reader(BaseTool):
    replay_policy = "safe_retry"

    def __init__(self, observed, stop_at=None):
        super().__init__("read", "Read source", [])
        self.observed, self.calls, self.stop_at = observed, [], stop_at

    def run_tool(self, parameters):
        page = parameters["page"]
        if page == self.stop_at:
            raise WorkflowInterrupted("stopped before dispatch")
        self.calls.append(page)
        self.observed.add("ev_a")
        return f"[ev_a] source page {page}"


def setup(tmp_path, model, *, stop_at=None, task_id="T1", run_id="r1", store=None):
    store = store or RunStore(tmp_path)
    request = research_request().model_copy(update={"run_id": run_id})
    task = ResearchTask(task_id=task_id, objective="Read methods and limitations")
    observed = set()
    raw = Reader(observed, stop_at)
    tool = JournaledTool(raw, store, run_id, task_id, replay=lambda text: observed.add("ev_a"))
    progress = ResearchProgress(store, run_id, task_id)
    agent = Researcher(task_id, model, [tool], progress=progress, evidence_context=FakeContext(), observed_ids=lambda: observed)
    return agent, raw, task, request, progress, store


def test_eight_steps_then_markdown_summary_and_early_finish(tmp_path):
    llm = Model(*[calls(i) for i in range(8)], answer())
    agent, raw, task, request, _, _ = setup(tmp_path, llm)
    note = agent.run(task, request)
    assert raw.calls == list(range(8)) and len(llm.calls) == 9
    assert all(c["max_tokens"] == 4096 for c in llm.calls)
    assert llm.calls[-1]["tag"] == "T1:forced_summary"
    assert note.status == "completed" and note.source_ids == ["ev_a"]
    early = Model(calls(1), answer())
    agent, raw, task, request, _, _ = setup(tmp_path / "early", early)
    agent.run(task, request)
    assert len(early.calls) == 2 and raw.calls == [1]


def test_length_finish_gets_only_one_extended_note_attempt(tmp_path):
    llm = Model(calls(1), answer("unfinished [ev_a]", "length"), answer("still unfinished [ev_a]", "length"))
    agent, _, task, request, _, _ = setup(tmp_path, llm)
    agent.max_steps = 1
    note = agent.run(task, request)
    assert [c["max_tokens"] for c in llm.calls] == [4096, 4096, 6144]
    assert note.status == "partial"


def test_resume_pending_batch_without_repeating_completed_model_or_tool(tmp_path):
    first = Model(calls(1, 2))
    agent, raw, task, request, progress, store = setup(tmp_path, first, stop_at=2)
    with pytest.raises(WorkflowInterrupted):
        agent.run(task, request)
    saved = progress.load(progress.fingerprint(task, request))
    assert raw.calls == [1] and len(first.calls) == 1
    assert [m["role"] for m in saved["history"]] == ["user", "assistant", "tool"]
    store.conn.close()
    resumed_model = Model(answer())
    agent, raw, task, request, progress, store = setup(tmp_path, resumed_model)
    note = agent.run(task, request)
    assert raw.calls == [2] and len(resumed_model.calls) == 1
    assert note.source_ids == ["ev_a"] and note.status == "completed"
    ContextManager.validate_pairs(progress.load(progress.fingerprint(task, request))["history"])


def test_journal_replays_result_after_crash_before_react_commit(tmp_path):
    agent, raw, task, request, progress, store = setup(tmp_path, Model(calls(1)))
    save = agent._save
    def crash_after_tool():
        if agent._history[-1].role == "tool":
            raise WorkflowInterrupted("crash before React result commit")
        save()
    agent._save = crash_after_tool
    with pytest.raises(WorkflowInterrupted):
        agent.run(task, request)
    assert raw.calls == [1]
    store.conn.close()
    agent, raw, task, request, _, _ = setup(tmp_path, Model(answer()))
    note = agent.run(task, request)
    assert raw.calls == [] and note.source_ids == ["ev_a"]


def test_stop_after_provider_response_preserves_model_message(tmp_path):
    manager = ContextManager(tmp_path)
    def create(**kwargs):
        control.stopped.set()
        return SimpleNamespace(choices=[SimpleNamespace(message=calls(1), finish_reason="tool_calls")],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))
    llm = provider(manager, create)
    agent, raw, task, request, progress, store = setup(tmp_path, llm)
    control = RunControl(request, None, lambda: False)
    with request_context(request), run_control(control), pytest.raises(WorkflowInterrupted):
        agent.run(task, request)
    assert raw.calls == []
    assert progress.load(progress.fingerprint(task, request))["steps_taken"] == 1
    agent, raw, task, request, _, _ = setup(tmp_path, Model(answer()), store=store)
    assert agent.run(task, request).status == "completed" and raw.calls == [1]


def test_model_failure_keeps_previous_steps_and_scopes_are_isolated(tmp_path):
    agent, raw, task, request, progress, store = setup(tmp_path, Model(calls(1), RuntimeError("connection lost")))
    with pytest.raises(WorkflowInterrupted, match="已保存 ReAct"):
        agent.run(task, request)
    assert raw.calls == [1]
    assert ResearchProgress(store, "r1", "T2").load("any") is None
    assert ResearchProgress(store, "r2", "T1").load("any") is None
    assert ResearchProgress(store, "r1", "T1", 1).load("any") is None
    with pytest.raises(WorkflowInterrupted, match="不兼容"):
        progress.load("different-policy")
    agent, raw, task, request, _, _ = setup(tmp_path, Model(answer()), store=store)
    assert agent.run(task, request).source_ids == ["ev_a"] and raw.calls == []


def test_empty_response_does_not_poison_committed_history(tmp_path):
    agent, _, task, request, progress, store = setup(tmp_path, Model(calls(1), answer("")))
    with pytest.raises(WorkflowInterrupted, match="空响应"):
        agent.run(task, request)
    saved = progress.load(progress.fingerprint(task, request))
    assert saved["phase"] == "collect" and saved["steps_taken"] == 1
    agent, raw, task, request, _, _ = setup(tmp_path, Model(answer()), store=store)
    assert agent.run(task, request).status == "completed" and not raw.calls


def test_large_tool_text_is_externalized_in_durable_history(tmp_path):
    manager = ContextManager(tmp_path)
    model = Model(calls(1), RuntimeError("offline"))
    model.context_manager = manager
    agent, raw, task, request, progress, _ = setup(tmp_path, model)
    original = json.dumps({"body": "[ev_a] original methods and experiments " * 2000,
                           "unknowns": ["Dataset split unknown"], "negative_evidence": ["Failed on shifted boundaries"]})
    def read(parameters):
        raw.observed.add("ev_a")
        return original
    raw.run_tool = read
    with request_context(request), pytest.raises(WorkflowInterrupted):
        agent.run(task, request)
    saved = progress.load(progress.fingerprint(task, request))
    pointer = json.loads(saved["history"][-1]["content"])
    assert "artifact_id" in pointer and len(json.dumps(saved)) < len(original) // 4
    assert manager.artifacts.read(pointer["artifact_id"], request.run_id, limit=1000)["text"] == original[:1000]
    assert saved["source_ids"] == ["ev_a"]
    assert pointer["protected_findings"] == [{"unknowns": ["Dataset split unknown"]},
                                              {"negative_evidence": ["Failed on shifted boundaries"]}]


def test_langgraph_restart_keeps_completed_sibling_and_private_react(tmp_path):
    from research_fakes import FakeSupervisor, FakeWriter
    from sage_scholar.domain.research import SubQuestion
    from sage_scholar.graph.graph_builder import build_graph
    from sage_scholar.runtime.checkpoints import Checkpoints
    from sage_scholar.runtime.journal import DurableResearcher

    class Supervisor(FakeSupervisor):
        def plan(self, request):
            return [SubQuestion(objective="question A"), SubQuestion(objective="question B")]

    class Sources(FakeContext):
        def get(self, identity):
            if identity not in {"ev_a", "ev_b"}:
                raise ValueError(identity)
            return super().get("ev_a")

    store, checkpoints = RunStore(tmp_path), Checkpoints(tmp_path)
    request = research_request().model_copy(update={"run_id": "graph-run"})
    config = checkpoints.config(request.run_id)
    models = {"T1": Model(calls(1), answer()), "T2": Model(calls(2), RuntimeError("disconnect"))}
    dispatched = []
    def create(task_id):
        def factory(task):
            observed = set()
            identity = "ev_a" if task_id == "T1" else "ev_b"
            tool = Reader(observed)
            def read(parameters):
                observed.add(identity)
                dispatched.append((task_id, parameters["page"]))
                return f"[{identity}] source for {task_id}"
            tool.run_tool = read
            wrapped = JournaledTool(tool, store, request.run_id, task_id,
                                    replay=lambda result: observed.add(identity))
            return Researcher(task_id, models[task_id], [wrapped], evidence_context=Sources(),
                observed_ids=lambda: observed, progress=ResearchProgress(store, request.run_id, task_id, task.revision))
        return DurableResearcher(factory, store, request.run_id, lambda: False)

    graph = build_graph(Supervisor(), create, FakeWriter(evidence_context=Sources()), checkpointer=checkpoints.saver)
    with pytest.raises(WorkflowInterrupted):
        graph.invoke({"execution": request}, config)
    assert store.result(request.run_id, "note:markdown:T1:0")["source_ids"] == ["ev_a"]
    before = len(models["T1"].calls)
    checkpoints.close()
    store.close()
    store, checkpoints = RunStore(tmp_path), Checkpoints(tmp_path)
    models["T2"] = Model(answer("Private B [ev_b]"))
    graph = build_graph(Supervisor(), create, FakeWriter(evidence_context=Sources()), checkpointer=checkpoints.saver)
    result = graph.invoke(None, config)
    assert len(models["T1"].calls) == before and len(models["T2"].calls) == 1
    assert sorted(dispatched) == [("T1", 1), ("T2", 2)]
    assert result["research_notes"]["T1"].source_ids == ["ev_a"]
    assert result["research_notes"]["T2"].source_ids == ["ev_b"]
    t2_prompt = json.dumps(models["T2"].calls[0]["messages"])
    assert "question A" not in t2_prompt and "ev_a" not in t2_prompt
    checkpoints.close()
    store.close()
