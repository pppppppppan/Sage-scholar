"""Context correctness at provider/retrieval boundaries; no network/model downloads."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from research_fakes import contract, ScriptedLLM
from sage_scholar.application.qa_service import QAService
from sage_scholar.base.llm_client import LLMClient
from sage_scholar.context.conversation import HistorySummary
from sage_scholar.context.manager import ContextBudgetExceeded, ContextManager, request_context
from sage_scholar.context.protection import prompt_contract, user_constraints
from sage_scholar.context.tools import ContextReadTool
from sage_scholar.domain import EvidenceRef
from sage_scholar.domain.execution import ExecutionRequest
from sage_scholar.domain.research import Claim, ReportDraft, Finding, ResearchNote
from sage_scholar.infrastructure.evidence_store import JsonEvidenceStore
from sage_scholar.research.evidence_context import ResearchEvidenceContext
from sage_scholar.routing import SourceScope


def request(**changes):
    return contract().execution.model_copy(update={"request_id": "r42", **changes})


def summary(previous, batch, budget):
    return HistorySummary(goals=["比较论文方法"], conclusions=["先前助手介绍了两个模块，尚待原文确认"],
                          unresolved_questions=["泛化能力待确认"], referents=["第一个是局部编码器，第二个是全局聚合器"]).model_dump()


def long_history():
    return [{"role": "user", "content": "不要考虑微调；实验条件不同不要直接排名"},
            {"role": "assistant", "content": "收到限制。"}] + [
        message for i in range(30) for message in [
            {"role": "user", "content": f"旧问题{i}：什么是 KL divergence？"},
            {"role": "assistant", "content": f"第{i}轮回答。" * 20},
        ]
    ]


def test_recent_summary_constraints_and_complete_budget(tmp_path):
    manager = ContextManager(tmp_path, max_tokens=9000)
    req = request(history=long_history(), query="继续比较", allow_web=False,
                  source_scope=SourceScope.SELECTED_DOCUMENTS, document_ids=("doc_a", "doc_b"))
    tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]
    prepared = manager.prepare([{"role": "system", "content": "answer"}, *req.history,
                                {"role": "user", "content": req.query}], request=req, tools=tools,
                               output_reserve=1200, summarize=summary)
    text = json.dumps(prepared.messages, ensure_ascii=False)
    pins = prepared.messages[0]["content"]
    assert "prior_user_instructions" not in text
    assert "旧问题0" not in text and text.count("旧问题29") == 1
    assert "旧问题" not in pins and "不要考虑微调" in pins and "不要直接排名" in pins
    assert '"allow_web": false' in pins and "doc_a" in pins and "doc_b" in pins
    assert "compressed_history" in text and "全局聚合器" in text
    assert any(m["role"] == "assistant" and "第29轮" in m["content"] for m in prepared.messages)
    accounting = prepared.accounting
    assert accounting["input_tokens"] < accounting["before_compression"]
    assert accounting["total_tokens"] == accounting["input_tokens"] + accounting["output_reserve"] <= 9000
    assert sum(accounting["categories"].values()) == accounting["input_tokens"]
    assert accounting["categories"]["conversation"] > 0 and accounting["tools_tokens"] > 0
    assert not any(key.startswith("_context_") for m in prepared.messages for key in m)


def test_questions_are_not_promoted_to_permanent_constraints():
    values = ["为什么不能使用互联网？", "什么是 KL divergence？", "再举个例子", "好的", "不要使用互联网", "只比较这三篇论文"]
    assert user_constraints([{"role": "user", "content": v} for v in values]) == values[-2:]


def test_summary_is_durable_and_resume_does_not_expand_history(tmp_path):
    req = request(run_id="same-run", history=long_history(), budget={"used_tokens": 2345})
    calls = []
    def summarize(*args):
        calls.append(1)
        return summary(*args)
    first = ContextManager(tmp_path, max_tokens=9000).prepare(
        [{"role": "user", "content": "继续"}], request=req, output_reserve=1000, summarize=summarize)
    assert calls
    restored = ExecutionRequest.model_validate_json(req.model_dump_json())
    def never(*args):
        pytest.fail("Restored context must reuse durable summaries, not call the model again")
    second = ContextManager(tmp_path, max_tokens=9000).prepare(
        [{"role": "user", "content": "继续"}], request=restored, output_reserve=1000, summarize=never)
    assert second.messages == first.messages
    assert "2345" in second.messages[0]["content"] and restored.run_id == "same-run"


def test_dependency_notes_and_negative_findings_survive_tool_externalization(tmp_path):
    manager = ContextManager(tmp_path, max_tokens=5000)
    observation = json.dumps({"negative_evidence": ["A 在 X 失败"], "conflicts": ["实验条件不一致"],
        "unknowns": ["测试划分未知"], "remaining_gaps": ["成本未知"], "source_id": "paper:1",
        "evidence_id": "ev_one", "location": {"page": 4}, "body": "long observation " * 5000}, ensure_ascii=False)
    prepared = manager.prepare([
        {"role": "user", "content": json.dumps({"dependency_notes": [{"task_id": "T1", "findings": ["已完成编码器比较"]}]}, ensure_ascii=False)},
        {"role": "assistant", "content": "", "tool_calls": [call("a")]},
        {"role": "tool", "tool_call_id": "a", "content": observation},
        {"role": "user", "content": "继续"},
    ], request=request(), output_reserve=1000)
    text = json.dumps(prepared.messages, ensure_ascii=False)
    for value in ("A 在 X 失败", "实验条件不一致", "测试划分未知", "成本未知", "已完成编码器比较"):
        assert value in text
    pointer = json.loads(next(m["content"] for m in prepared.messages if m["role"] == "tool"))
    assert pointer["locators"][0]["location"] == {"page": 4}
    assert pointer["is_evidence"] is False
    assert manager.artifacts.read(pointer["artifact_id"], "r42")["text"] == observation[:4000]
    manager.validate_pairs(prepared.messages)


def call(identity):
    return {"id": identity, "type": "function", "function": {"name": "fetch", "arguments": "{}"}}


def test_pressure_retires_whole_old_tool_rounds_and_preserves_latest(tmp_path):
    manager = ContextManager(tmp_path, max_tokens=4000, reserve_ratio=0)
    messages = [{"role": "user", "content": "research"}]
    for i in range(10):
        messages.extend([
            {"role": "assistant", "content": "", "tool_calls": [call(str(i))]},
            {"role": "tool", "tool_call_id": str(i), "content": "observation " * 450},
        ])
    messages.append({"role": "user", "content": "summarize"})
    prepared = manager.prepare(messages, output_reserve=800)
    manager.validate_pairs(prepared.messages)
    remaining = [m["tool_call_id"] for m in prepared.messages if m["role"] == "tool"]
    assert "9" in remaining and len(remaining) < 10
    assert "tool_round" in json.dumps(prepared.messages)
    assert prepared.accounting["total_tokens"] <= 4000
    with pytest.raises(ContextBudgetExceeded):
        manager.prepare([{"role": "tool", "tool_call_id": "orphan", "content": "text"}], output_reserve=100)


def test_source_window_reread_preserves_identity_and_original(tmp_path):
    manager = ContextManager(tmp_path, max_tokens=5000)
    store = JsonEvidenceStore(str(tmp_path))
    item = EvidenceRef.create(source_id="https://example.org/paper", source_version="v3", parser_version="pdf-test",
        section="Experiments", location={"page": 4}, original_text="original evidence " * 3000,
        metadata={"kind": "source_content"})
    store.save(item)
    context = ResearchEvidenceContext(store, contract().source_policy, context_manager=manager)
    context.register(item)
    payload = {"evidence": [{**item.model_dump(), "truncated": False}]}
    prepared = manager.prepare([{"role": "user", "content": json.dumps(payload)}], request=request(), output_reserve=1000)
    window = json.loads(prepared.messages[-1]["content"])["evidence"][0]
    assert window["truncated"] and window["evidence_id"] == item.evidence_id
    assert window["source_version"] == "v3" and window["location"] == {"page": 4}
    assert manager.artifacts.read(window["artifact_id"], "r42")["text"] == item.original_text[:4000]
    tool = ContextReadTool(manager, context, "r42")
    reread = json.loads(tool.run_tool({"identity": item.evidence_id, "offset": 700, "limit": 300}))
    assert reread["original_text"] == item.original_text[700:1000]
    assert reread["parser_version"] == "pdf-test" and reread["source_version"] == "v3"
    with pytest.raises(ValueError):
        tool.run_tool({"identity": "ev_unauthorized"})


def test_verifier_never_silently_shortens_originals(tmp_path):
    manager = ContextManager(tmp_path, max_tokens=4500)
    messages = [{"role": "user", "content": json.dumps({"evidence": [{"original_text": "original " * 5000, "truncated": False}]})}]
    with pytest.raises(ContextBudgetExceeded):
        manager.prepare(messages, request=request(), output_reserve=1000, purpose="research:verify_claim")
    assert json.loads(messages[0]["content"])["evidence"][0]["truncated"] is False


def test_oversized_schema_and_hard_constraints_fail_instead_of_disappearing(tmp_path):
    manager = ContextManager(tmp_path, max_tokens=2000, reserve_ratio=0)
    req = request(metadata={"constraints": ["必须保留的约束 " * 2000]})
    with pytest.raises(ContextBudgetExceeded):
        manager.prepare([{"role": "user", "content": "go"}], request=req, output_reserve=200)
    with pytest.raises(ContextBudgetExceeded):
        manager.prepare([{"role": "user", "content": "go"}], tools=[{"description": "schema " * 3000}], output_reserve=200)


@pytest.mark.parametrize("clarification", [False, True])
def test_qa_followup_resolves_both_retrieval_and_answer_from_same_context(tmp_path, clarification):
    manager = ContextManager(tmp_path, max_tokens=9000)
    req = request(query="第二个相比第一个有什么优势？", allow_web=False, source_scope=SourceScope.SELECTED_DOCUMENTS,
                  document_ids=("doc_A",), history=[
                      {"role": "user", "content": "这篇论文有哪两个核心模块？"},
                      {"role": "assistant", "content": "第一个是局部编码器，第二个是全局聚合器。"}])
    item = EvidenceRef.create(source_id="local:paper", document_id="doc_A", original_text="全局聚合器整合长距离信息。",
                              source_version="v1", parser_version="test")
    class Retrieval:
        calls = []
        async def retrieve(self, query, *, policy):
            self.calls.append((query, policy))
            return [item]
    class Model:
        calls = []
        async def complete(self, messages, *, tag="", **kwargs):
            self.calls.append((messages, tag))
            assert "局部编码器" in json.dumps(messages, ensure_ascii=False)
            if tag == "qa:resolve_reference":
                return json.dumps({"query": "全局聚合器相比局部编码器有什么优势？", "needs_clarification": clarification})
            assert "全局聚合器相比局部编码器" in messages[-1]["content"]
            return f"全局聚合器整合长距离信息 [{item.evidence_id}]"
    retrieval, model = Retrieval(), Model()
    result = asyncio.run(QAService(retrieval, model, context_manager=manager).answer(req))
    if clarification:
        assert not retrieval.calls and "补充名称" in result.answer
    else:
        assert retrieval.calls[0][0] == "全局聚合器相比局部编码器有什么优势？"
        assert retrieval.calls[0][1].document_ids == ("doc_A",) and not retrieval.calls[0][1].allow_web
        assert [tag for _, tag in model.calls] == ["qa:resolve_reference", "qa"]
        assert result.evidence == [item] and not result.citation_issues


def test_same_provider_compresses_without_recursion_and_records_usage(tmp_path):
    manager = ContextManager(tmp_path, max_tokens=7000)
    llm = LLMClient(model="test-model", api_key="test", base_url="http://localhost:1/v1", context_manager=manager)
    sent, usage = [], []
    manager.run_store = SimpleNamespace(record_usage=lambda *args: usage.append(args))
    def create(**kwargs):
        sent.append(kwargs)
        assert all(not any(k.startswith("_context_") for k in m) for m in kwargs["messages"])
        compression = "压缩历史对话" in kwargs["messages"][0]["content"]
        if compression:
            assert not any("source_policy" in m["content"] for m in kwargs["messages"])
        body = json.dumps(summary(None, None, None), ensure_ascii=False) if compression else "answer"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=body))],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    req = request(run_id="run-v42", history=long_history())
    with request_context(req):
        assert llm.invoke([{"role": "user", "content": req.query}], max_tokens=1000).content == "answer"
    assert 1 < len(sent) < 30 and len(usage) == len(sent)
    assert all(record == ("run-v42", 100, 20) for record in usage)
    before = len(sent)
    with request_context(req):
        llm.invoke([{"role": "user", "content": req.query}], max_tokens=1000)
    assert len(sent) == before + 1  # persisted summary reused, paid compression not repeated


def test_contract_history_is_excluded_only_from_prompt_not_checkpoint():
    source = contract().model_copy(update={"execution": request(history=long_history())})
    assert "history" not in prompt_contract(source)["execution"]
    assert source.model_dump()["execution"]["history"] == long_history()


def test_writer_selects_findings_and_counterevidence_not_incidental_observations(tmp_path):
    from sage_scholar.agents.writer import Writer
    store = JsonEvidenceStore(str(tmp_path))
    context = ResearchEvidenceContext(store, contract().source_policy)
    items = [EvidenceRef.create(source_id=f"https://example.org/{i}", original_text=f"Source {i}",
        source_version="v1", parser_version="test", metadata={"kind": "source_content"}) for i in range(3)]
    for item in items:
        store.save(item)
        context.register(item)
    a, b, incidental = [item.evidence_id for item in items]
    note = ResearchNote(task_id="T1", question="q", evidence_ids=[a, b, incidental],
                        findings=[Finding(text="finding", evidence_ids=[a])], conflicts=[f"反例来自 {b}"])
    llm = ScriptedLLM(ReportDraft(title="Report", claims=[Claim(claim_id="C1", text="claim", evidence_ids=[a])]))
    Writer(llm, evidence_context=context).run(contract(), [note])
    payload = json.loads(llm.calls[0]["messages"][-1]["content"])
    assert {e["evidence_id"] for e in payload["evidence"]} == {a, b}


def test_summary_failure_is_explicit_and_entire_original_remains_reloadable(tmp_path):
    manager = ContextManager(tmp_path, max_tokens=4000)
    req = request(history=long_history())
    def unavailable(*args):
        raise RuntimeError("offline")
    bundle = manager.conversation(req, summarize=unavailable)
    assert "incomplete_verbatim_extract" in bundle.summary["modes"]
    assert bundle.summary["conclusions"] == []  # don't manufacture confirmed conclusions
    original = manager.artifacts.read(bundle.summary["artifact_id"], "r42", limit=12000)
    assert "旧问题0" in original["text"] and "不要考虑微调" in bundle.constraints[0]
    # Changing the context schema/owner cannot reuse someone else's artifact.
    with pytest.raises(ValueError):
        manager.artifacts.read(bundle.summary["artifact_id"], "another-run")


def test_small_budget_can_compress_recent_turn_without_losing_constraints(tmp_path):
    manager = ContextManager(tmp_path, max_tokens=3200, reserve_ratio=0)
    history = [{"role": "user", "content": "不要使用微调，只比较成本。" + "普通问题描述 " * 2000},
               {"role": "assistant", "content": "很长的解释 " * 2000}]
    req = request(history=history, allow_web=False, query="比较第二个模块")
    prepared = manager.prepare([{"role": "user", "content": req.query}], request=req,
                               output_reserve=500, summarize=summary)
    text = json.dumps(prepared.messages, ensure_ascii=False)
    assert "全局聚合器" in text and "不要使用微调" in text
    assert '"allow_web": false' in prepared.messages[0]["content"]
    assert "普通问题描述" not in text
    assert prepared.accounting["total_tokens"] <= 3200


def test_restart_reuses_actual_persisted_consumed_budget(tmp_path):
    from sage_scholar.runtime.store import RunStore
    ledger = RunStore(str(tmp_path))
    ledger.record_usage("run42", 800, 120)
    ledger.save_result("run42", "note:T1", {"evidence_ids": ["ev_a"], "findings": ["completed"]})
    ledger.close()
    ledger = RunStore(str(tmp_path))
    manager = ContextManager(tmp_path, max_tokens=5000)
    manager.run_store = ledger
    result = manager.prepare([{"role": "user", "content": "继续"}],
        request=request(run_id="run42"), output_reserve=1000)
    assert '"prompt_tokens": 800' in result.messages[0]["content"]
    assert '"completion_tokens": 120' in result.messages[0]["content"]
    assert ledger.result("run42", "note:T1")["findings"] == ["completed"]
    ledger.close()


def test_direct_uses_shared_conversation_and_arbitrary_token_literals_are_data(tmp_path):
    from sage_scholar.application.direct_service import DirectService
    manager = ContextManager(tmp_path, max_tokens=9000)
    class Model:
        summarize_history = staticmethod(summary)
        async def complete(self, messages, **kwargs):
            kinds = [m.get("_context_kind") for m in messages]
            assert "compressed_history" in kinds and "recent_conversation" in kinds
            prepared = manager.prepare(messages, request=req, output_reserve=1000, summarize=summary)
            text = json.dumps(prepared.messages, ensure_ascii=False)
            assert text.count("旧问题29") == 1 and "旧问题0" not in text
            return "ok"
    req = request(history=long_history(), query="解释字面文本 <|endoftext|>")
    assert asyncio.run(DirectService(Model(), context_manager=manager).answer(req)) == "ok"


def test_failed_incremental_summary_retains_previous_conclusions_and_unknowns(tmp_path):
    manager = ContextManager(tmp_path, max_tokens=9000)
    previous = HistorySummary(conclusions=["用户确认仅比较成本"], unresolved_questions=["unknown: 未报告显存"],
                              referents=["第一个模块是 encoder"]).model_dump()
    fallback = manager.conversations._fallback(previous, [{"role": "assistant", "content": "new observation " * 5000}])
    assert fallback["conclusions"] == previous["conclusions"]
    assert fallback["unresolved_questions"] == previous["unresolved_questions"]
    assert "第一个模块是 encoder" in fallback["referents"]
    assert manager.counter.count(json.dumps(fallback, ensure_ascii=False)) <= manager.conversations.summary_tokens
