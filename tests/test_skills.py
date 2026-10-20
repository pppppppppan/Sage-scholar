"""Real tool/Context/SQLite boundaries with scripted model decisions; no paid API."""
import asyncio
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from uuid import uuid4

import pytest
from openai.types.chat import ChatCompletionMessage

from sage_scholar.skills import SkillService, SkillRegistry, SkillError
from sage_scholar.skills.service import STATE_KEY
from sage_scholar.context.manager import ContextManager, request_context
from sage_scholar.runtime.store import RunStore
from sage_scholar.runtime.checkpoints import Checkpoints
from sage_scholar.runtime.models import RunSnapshot, RunEvent, utc_now
from sage_scholar.domain.execution import ExecutionRequest
from sage_scholar.tools.calling import invoke_tools
from sage_scholar.tools.tool_conversation import ConversationReadTool
from sage_scholar.application.skill_support import SkillSupport
from sage_scholar.application.direct_service import DirectService
from sage_scholar.application.qa_service import QAService
from sage_scholar.agents.supervisor import Supervisor
from sage_scholar.agents.researcher import Researcher
from sage_scholar.domain.research import ResearchTask
from sage_scholar.graph.graph_builder import build_graph
from research_fakes import research_request, FakeSupervisor, FakeWriter, FakeResearcher
from test_document_evidence import library


def request(query="你好", **updates):
    return ExecutionRequest.model_validate({"query": query, "workflow": "direct", "source_scope": "conversation",
        "allow_web": False, "execution_mode": "run", "run_id": "r1", **updates})


def answer(text):
    return ChatCompletionMessage(role="assistant", content=text)


def call(tool_name, **params):
    return ChatCompletionMessage(role="assistant", content=None, tool_calls=[{
        "id": uuid4().hex, "type": "function", "function": {"name": tool_name, "arguments": json.dumps(params)}}])


class Model:
    def __init__(self, manager, *responses):
        self.manager, self.responses, self.calls = manager, iter(responses), []

    def invoke(self, messages, **kwargs):
        prepared = self.manager.prepare(messages, tools=kwargs.get("tools"), tool_choice=kwargs.get("tool_choice"),
            output_reserve=kwargs.get("max_tokens", 4096), purpose=kwargs.get("tag", ""))
        self.calls.append({**kwargs, "messages": prepared.messages, "accounting": prepared.accounting})
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response(self.calls[-1]) if callable(response) else response


@pytest.fixture
def setup(tmp_path):
    registry_root = tmp_path / "skills"
    shutil.copytree(Path(__file__).resolve().parents[1] / "skills", registry_root)
    store = RunStore(tmp_path / "data")
    skills = SkillService(store, SkillRegistry(registry_root))
    manager = ContextManager(tmp_path / "data")
    manager.run_store, manager.skills = store, skills
    yield skills, manager, store, registry_root
    store.close()


def test_discovery_only_reads_frontmatter_and_refs_are_lazy(setup, monkeypatch):
    skills, _, _, root = setup
    (root / "paper-review/SKILL.md").write_text((root / "paper-review/SKILL.md").read_text() + "\nUNREAD_BODY")
    original = Path.read_text
    def forbidden(path, *args, **kwargs):
        raise AssertionError("metadata discovery must not read full text")
    monkeypatch.setattr(Path, "read_text", forbidden)
    registry = SkillRegistry(root)
    assert "UNREAD_BODY" not in json.dumps(registry.catalog())
    monkeypatch.setattr(Path, "read_text", original)
    from sage_scholar.skills import loader
    reads, original_read = [], loader.read_bounded
    def tracked(path, expected=None):
        reads.append(path.name)
        return original_read(path, expected)
    monkeypatch.setattr(loader, "read_bounded", tracked)
    req = request()
    skills.view(req, "paper-review")
    assert reads == ["SKILL.md"]
    skills.view(req, "paper-review")
    skills.view(req, "paper-review", "references/review-checklist.md")
    skills.view(req, "paper-review", "references/review-checklist.md")
    assert reads == ["SKILL.md", "review-checklist.md"]


@pytest.mark.parametrize("name,path", [("../paper-review", None), ("missing-skill", None),
    ("paper-review", "../citation-check/SKILL.md"), ("paper-review", "/etc/passwd"),
    ("paper-review", "C:\\secret.md"), ("paper-review", "references/../../secrets.md")])
def test_invalid_names_and_paths(setup, name, path):
    with pytest.raises(SkillError):
        setup[0].view(request(), name, path)


def test_symlink_escape_rejected(setup, tmp_path):
    skills, _, _, root = setup
    outside = tmp_path / "secret.md"
    outside.write_text("secret")
    (root / "paper-review/references/escape.md").symlink_to(outside)
    with pytest.raises(SkillError):
        skills.view(request(), "paper-review")


def test_pinned_body_and_read_refs_survive_restart_unread_changes_refused(setup):
    skills, _, store, root = setup
    req = request()
    body = skills.view(req, "paper-review")
    ref = skills.view(req, "paper-review", "references/review-checklist.md")
    skills.view(req, "citation-check")
    (root / "paper-review/SKILL.md").write_text(body["content"] + "\nchanged")
    (root / "paper-review/references/review-checklist.md").write_text("changed")
    (root / "citation-check/references/verification-rules.md").write_text("changed")
    # No re-reading catalog needed for an already-started service; use the prior registry
    # and a fresh SQLite connection to demonstrate recovery independent of memory cache.
    fresh_store = RunStore(root.parent / "data")
    restored = SkillService(fresh_store, SkillRegistry(root))
    assert restored.view(req, "paper-review")["content"] == body["content"]
    assert restored.view(req, "paper-review")["version"] == body["version"]
    assert restored.view(req, "paper-review", "references/review-checklist.md")["content"] == ref["content"]
    with pytest.raises(SkillError, match="改变"):
        restored.view(req, "citation-check", "references/verification-rules.md")
    assert store.result(req.run_id, STATE_KEY)
    store.purge_run(req.run_id)
    assert not restored.selected(req)
    fresh_store.close()


@pytest.mark.parametrize("query", ["你好", "帮我调研 OPD"])
def test_catalog_does_not_force_selection_or_extra_model_call(setup, query):
    skills, manager, store, _ = setup
    req = request(query)
    model = Model(manager, answer("回答"))
    with request_context(req):
        result = invoke_tools(model, [{"role": "user", "content": query}], skills.tools(req), tag="direct")
    assert result.content == "回答" and len(model.calls) == 1
    assert not store.result(req.run_id, STATE_KEY)
    prompt = json.dumps(model.calls[0]["messages"], ensure_ascii=False)
    assert "paper-review" in prompt and "# Paper review" not in prompt


def test_explicit_and_automatic_loading_are_separate(setup):
    skills, manager, _, _ = setup
    req = request("$paper-review 检查实验")
    skills.explicit(req)
    assert [s["name"] for s in skills.selected(req)] == ["paper-review"]
    with pytest.raises(SkillError, match="不存在"):
        skills.explicit(request("使用 $unknown-skill"))
    ordinary = request("公式 $x$ 和 $PATH", run_id="formula")
    skills.explicit(ordinary)
    assert skills.selected(ordinary) == []


def test_model_loads_skill_and_reference_tools_enter_managed_history(setup):
    skills, manager, _, _ = setup
    req = request("检查这篇论文的实验设计")
    model = Model(manager, call("skill_view", name="paper-review"),
                  call("skill_view", name="paper-review", file_path="references/review-checklist.md"), answer("需查验实验"))
    with request_context(req):
        invoke_tools(model, [{"role": "user", "content": req.query}], skills.tools(req), tag="qa")
    prompt = json.dumps(model.calls[-1]["messages"], ensure_ascii=False)
    assert "按论文实际内容选择检查项" in prompt
    assert prompt.count("# Paper review") == 1
    assert model.calls[-1]["accounting"]["categories"]["skills"] > 0
    manager.validate_pairs(model.calls[-1]["messages"])
    # Writer receives selected requirements without calling skill_view again.
    prepared = manager.prepare([{"role": "user", "content": "write"}], request=req, purpose="writer:report")
    assert "Strengths" in json.dumps(prepared.messages)
    assert "# Paper review" not in json.dumps(manager.prepare(
        [{"role": "user", "content": "write"}], request=request(run_id="another"), purpose="writer:report").messages)


def test_supervisor_loads_before_submit_without_new_planning_agent(setup):
    skills, manager, _, _ = setup
    req = research_request().model_copy(update={"run_id": "supervisor", "query": "深入审稿"})
    model = Model(manager, call("skill_view", name="paper-review"), call("submit_result",
        requirements=["评审实验"], questions=[{"objective": "独立检查实验设计", "scope_keys": ["experiments"],
            "expected_output": "证据支持的优缺点和疑点", "covers": ["评审实验"]}]))
    with request_context(req):
        questions = Supervisor(model, 8, skills=skills).plan(req)
    assert len(questions) == 1 and len(model.calls) == 2
    assert "# Paper review" in json.dumps(model.calls[-1]["messages"])


def test_direct_and_qa_have_real_skill_execution_paths(setup, library):
    skills, manager, _, _ = setup
    _, adapter, doc, _, _, _ = library
    model = Model(manager, call("skill_view", name="paper-review"), call("document_sections"),
                  answer("需复核实验统计，请见实验节。"))
    gateway = SimpleNamespace(client=model)
    support = SkillSupport(skills, gateway, adapter, manager)
    req = request("检查这篇论文的实验设计", workflow="qa", source_scope="selected_documents", document_ids=(doc.document_id,))
    result = asyncio.run(QAService(adapter, gateway, context_manager=manager, skill_support=support).answer(req))
    assert result.answer and any(e.section == "Experiments" for e in result.evidence)
    prompt = json.dumps(model.calls[-1]["messages"])
    assert "document_sections" in prompt and "SECRET B" not in prompt
    assert all(t["function"]["name"] not in {"fetch", "search"} for t in model.calls[0]["tools"])
    model2 = Model(manager, call("skill_view", name="citation-check"), answer("无法联网核实原文，unverifiable。"))
    support2 = SkillSupport(skills, SimpleNamespace(client=model2), adapter, manager)
    direct = DirectService(SimpleNamespace(), context_manager=manager, skill_support=support2)
    result = asyncio.run(direct.answer(request("$citation-check 检查已给文本", run_id="d")))
    assert "unverifiable" in result
    assert {s["function"]["name"] for s in model2.calls[0]["tools"]} == {
        "skills_list", "skill_view", "read_conversation", "read_context_artifact"}


def test_conversation_target_and_mapping_bound_to_frozen_history(setup):
    _, _, store, _ = setup
    conversation = store.conversations.create("default", identity="conv")
    old = request("旧问题", run_id="old", conversation_id="conv")
    now = utc_now()
    snap = RunSnapshot(run_id="old", workflow="direct", query=old.query, status="completed", execution=old,
        created_at=now, updated_at=now, result={"report": "原报告 Claim [1]", "citations": {1: "ev_original"}})
    store.append(snap, RunEvent(sequence=1, type="run_created"))
    req = request("检查刚才报告的引用", conversation_id="conv", history=store.conversations.history("conv"))
    reader = ConversationReadTool(req, store)
    entries = json.loads(reader.run_tool({}))["messages"]
    index = next(m["index"] for m in entries if m["role"] == "assistant")
    result = json.loads(reader.run_tool({"index": index}))
    assert result["content"] == "原报告 Claim [1]" and result["citations"] == {"1": "ev_original"}
    assert result["is_evidence"] is False
    with pytest.raises(ValueError, match="工作区"):
        ConversationReadTool(req.model_copy(update={"workspace_id": "other"}), store).run_tool({"index": index})


def test_parallel_researcher_history_is_private_and_selection_shared(setup):
    skills, manager, _, _ = setup
    req = research_request().model_copy(update={"run_id": "parallel"})
    first = Model(manager, call("skill_view", name="paper-review"), answer("笔记甲：尚未读取来源。"))
    second = Model(manager, answer("笔记乙：尚未读取来源。"))
    researchers = [Researcher("T1", first, skills.tools(req)), Researcher("T2", second, skills.tools(req))]
    with request_context(req):
        researchers[0].run(ResearchTask(task_id="T1", objective="私有问题甲"), req)
        researchers[1].run(ResearchTask(task_id="T2", objective="私有问题乙"), req)
    prompt = json.dumps(second.calls[0]["messages"], ensure_ascii=False)
    assert "# Paper review" in prompt and "私有问题甲" not in prompt and "笔记甲" not in prompt
    assert researchers[0]._history is not researchers[1]._history


def test_skills_remain_pinned_with_native_checkpoint_resume(setup):
    skills, manager, store, root = setup
    req = research_request().model_copy(update={"run_id": "checkpoint"})
    class Planner(FakeSupervisor):
        def plan(self, request):
            skills.view(request, "paper-review")
            return super().plan(request)
    checkpoints = Checkpoints(root.parent / "data")
    graph = build_graph(Planner(), lambda _: FakeResearcher(), FakeWriter(), checkpointer=checkpoints.saver)
    cfg = checkpoints.config(req.run_id)
    list(graph.stream({"execution": req}, cfg, interrupt_after=["research_node"], durability="sync"))
    version = skills.selected(req)[0]["version"]
    checkpoints.close()
    checkpoints = Checkpoints(root.parent / "data")
    restored = SkillService(store, skills.registry)
    class Writer(FakeWriter):
        def run(self, request, notes, gaps=()):
            assert restored.selected(request)[0]["version"] == version
            return super().run(request, notes, gaps)
    def no_repeat(_):
        raise AssertionError("已完成研究员不应重跑")
    graph = build_graph(Planner(), no_repeat, Writer(), checkpointer=checkpoints.saver)
    assert graph.invoke(None, cfg)["final_report"]
    checkpoints.close()


def test_actual_research_chain_loads_review_guidance_and_writer_reuses_it(setup, library):
    from sage_scholar.config import Config
    from sage_scholar.research.runtime import ResearchRuntime
    skills, manager, store, _ = setup
    _, adapter, doc, _, _, _ = library
    req = research_request().model_copy(update={"run_id": "whole", "query": "仔细审查这篇论文",
        "source_scope": "selected_documents", "document_ids": (doc.document_id,), "allow_web": False})
    source = adapter.read_documents_sync(req.source_policy)[0]
    model = Model(manager,
        call("skill_view", name="paper-review"),
        call("submit_result", requirements=["检查方法和实验"], questions=[{
            "objective": "检查方法和实验", "scope_keys": ["design"], "expected_output": "有证据的优缺点",
            "covers": ["检查方法和实验"]}]),
        call("document_sections"),
        call("read_context_artifact", identity=source.evidence_id),
        answer("方法：Method A。仍需核实实验对比。 [" + source.evidence_id + "]"),
        call("submit_result", note_reviews=[], remaining_gaps=[], follow_up_questions=[]),
        answer("# 论文审查\n## Strengths\n方法明确。[1]\n## Questions\n还需核对实验对比。"))
    model.total_calls = model.total_prompt_tokens = model.total_completion_tokens = 0
    runtime = ResearchRuntime(Config(data_dir=adapter._config.data_dir), llm_client=model,
        retrieval=adapter, connect_external=False, store=store, context_manager=manager)
    with request_context(req):
        events = list(runtime.run_research(req))
    report = next(e["finalize_node"]["report_result"] for e in events if "finalize_node" in e)
    assert "论文审查" in report.report and report.citations
    writer = next(c for c in model.calls if c["tag"] == "writer:report")
    assert "# Paper review" in json.dumps(writer["messages"], ensure_ascii=False)
    assert writer.get("tools") is None
    assert len(skills.selected(req)) == 1
    runtime.close()


def test_citation_check_reads_original_claim_then_authorized_source(setup, library):
    from sage_scholar.research.evidence_context import ResearchEvidenceContext
    from sage_scholar.context.tools import ContextReadTool
    skills, manager, store, _ = setup
    _, adapter, doc, _, _, _ = library
    store.conversations.create(identity="cite")
    old = request("研究论文", run_id="old", conversation_id="cite", workflow="qa",
                  source_scope="selected_documents", document_ids=(doc.document_id,))
    source = adapter.read_documents_sync(old.source_policy)[0]
    now = utc_now()
    snapshot = RunSnapshot(run_id="old", workflow="qa", query=old.query, status="completed", execution=old,
        created_at=now, updated_at=now, result={"report": "A 在所有任务上优于 B。[1]", "citations": {1: source.evidence_id},
                                             "evidence_ids": [source.evidence_id]})
    store.append(snapshot, RunEvent(sequence=1, type="run_created"))
    req = old.model_copy(update={"run_id": "new", "query": "检查刚才报告中的引用是否真实",
                                "history": store.conversations.history("cite")})
    context = ResearchEvidenceContext(adapter.evidence_store, req.source_policy, adapter.documents, manager)
    reader = ConversationReadTool(req, store, context)
    tools = [*skills.tools(req), reader, ContextReadTool(manager, context, req.run_id)]
    model = Model(manager, call("skill_view", name="citation-check"), call("read_conversation", index=1),
                  call("read_context_artifact", identity=source.evidence_id),
                  answer("Claim: 所有任务优于 B。身份可追溯；insufficient：所读方法段没有支持该比较。"))
    with request_context(req):
        result = invoke_tools(model, [{"role": "user", "content": req.query}], tools, tag="qa")
    assert "insufficient" in result.content
    prompt = json.dumps(model.calls[-1]["messages"], ensure_ascii=False)
    assert "A 在所有任务上优于 B" in prompt and "Method A" in prompt
    # A stricter new request cannot import that evidence merely by loading a Skill.
    denied = req.model_copy(update={"source_scope": "conversation", "document_ids": ()})
    denied_context = ResearchEvidenceContext(adapter.evidence_store, denied.source_policy, adapter.documents, manager)
    data = json.loads(ConversationReadTool(denied, store, denied_context).run_tool({"index": 1}))
    assert data["readable_evidence_ids"] == []
    with pytest.raises(ValueError):
        denied_context.get(source.evidence_id)


def test_tool_budget_cancel_and_fixed_skill_reserve_are_enforced(setup):
    from sage_scholar.runtime.control import RunControl, run_control
    from sage_scholar.runtime.journal import WorkflowInterrupted
    from sage_scholar.context.manager import ContextBudgetExceeded
    skills, manager, store, _ = setup
    req = request("检查", budget={"max_tool_calls": 1, "max_seconds": 100})
    model = Model(manager, call("skill_view", name="paper-review"), call("skill_view", name="citation-check"),
                  answer("工具预算不足，部分项目未核实"))
    with request_context(req), run_control(RunControl(req, store, lambda: False)):
        invoke_tools(model, [{"role": "user", "content": req.query}], skills.tools(req), tag="qa")
    assert [s["name"] for s in skills.selected(req)] == ["paper-review"]
    assert store.budget(req.run_id)["tool_calls"] == 1
    with request_context(req), run_control(RunControl(req, store, lambda: True)), pytest.raises(WorkflowInterrupted):
        invoke_tools(model, [], skills.tools(req), tag="qa")
    small = ContextManager(Path(store.conn.execute('PRAGMA database_list').fetchone()[2]).parent, max_tokens=300)
    small.skills = skills
    with pytest.raises(ContextBudgetExceeded):
        small.prepare([{"role": "user", "content": "test"}], request=req, purpose="writer:report", output_reserve=100)


def test_unavailable_citation_sources_return_partial_not_fabricated_success(setup):
    from sage_scholar.agents.writer import Writer
    from sage_scholar.domain.research import ResearchNote
    skills, manager, _, _ = setup
    req = research_request().model_copy(update={"run_id": "unverifiable"})
    skills.view(req, "citation-check")
    model = Model(manager, answer("# 引用检查\n原 Claim 无法查到原文，unverifiable，不能认定伪造。"))
    writer = Writer(model, evidence_context=SimpleNamespace(), skills=skills)
    class Unavailable(FakeResearcher):
        def run(self, task, request):
            return ResearchNote(task_id=task.task_id, question=task.objective,
                content="未访问原文，不能判断结论是否得到支持。", status="partial")
    graph = build_graph(FakeSupervisor(), lambda _: Unavailable(), writer)
    result = graph.invoke({"execution": req})["report_result"]
    assert result.partial and not result.citations and "unverifiable" in result.report
    assert "所有支持性判断必须为 unverifiable" in json.dumps(model.calls[-1]["messages"], ensure_ascii=False)
    ordinary = req.model_copy(update={"run_id": "ordinary"})
    with pytest.raises(ValueError, match="未取得"):
        graph.invoke({"execution": ordinary})


def test_qa_can_use_sections_when_initial_retrieval_has_no_hit(setup, library, monkeypatch):
    skills, manager, _, _ = setup
    _, adapter, doc, _, _, _ = library
    monkeypatch.setattr(adapter, "retrieve_with_warnings", lambda *args: ([], []))
    source = adapter.read_documents_sync(request(source_scope="selected_documents", document_ids=(doc.document_id,)).source_policy)[0]
    model = Model(manager, call("skill_view", name="paper-review"), call("document_sections"),
                  call("read_context_artifact", identity=source.evidence_id), answer("方法说明待验证。[" + source.evidence_id + "]"))
    gateway = SimpleNamespace(client=model)
    support = SkillSupport(skills, gateway, adapter, manager)
    req = request("审查论文方法", workflow="qa", source_scope="selected_documents", document_ids=(doc.document_id,))
    result = asyncio.run(QAService(adapter, gateway, context_manager=manager, skill_support=support).answer(req))
    assert result.evidence and result.citations and "方法说明待验证" in result.answer
    assert "Method A" in json.dumps(model.calls[-1]["messages"])
