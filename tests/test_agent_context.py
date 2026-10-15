"""Per-agent provider payloads, dynamic complete turns and output ceilings, offline."""
import json
from copy import deepcopy
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import pytest
from openai.types.chat import ChatCompletionMessage

from research_fakes import research_request, ScriptedLLM, FakeContext
from sage_scholar.config import ContextConfig
from sage_scholar.context.manager import ContextManager, request_context
from sage_scholar.context.conversation import HistorySummary
from sage_scholar.base.llm_client import LLMClient
from sage_scholar.base.model_limits import limits_for
from sage_scholar.domain.research import ResearchNote
from sage_scholar.agents.writer import Writer


def history(turns=8):
    return [m for i in range(turns) for m in (
        {"role": "user", "content": f"第{i}篇论文 Paper-{i} 方法是什么？"},
        {"role": "assistant", "content": f"Paper-{i} 采用 FNO-{i}，效果尚待原文确认。"})]


def summary(previous, batch, budget):
    return HistorySummary(referents=["上一篇论文是 Paper-7，采用 FNO-7"],
                          conclusions=["先前助手声称有优势，尚未确认"],
                          unresolved_questions=["跨边界条件泛化 unknown/conflict"]).model_dump()


def test_six_complete_turns_and_current_query_not_counted(tmp_path):
    manager = ContextManager(tmp_path, reserve_ratio=0)
    original = history()
    req = research_request().model_copy(update={"history": original, "query": "它呢"})
    prepared = manager.prepare([{"role": "user", "content": req.query}], request=req,
                               purpose="direct", output_reserve=4096, summarize=summary)
    raw_users = [m["content"] for m in prepared.messages if m["role"] == "user" and m["content"].startswith("第")]
    assert raw_users == [original[i]["content"] for i in range(4, 16, 2)]
    assert prepared.messages[-1] == {"role": "user", "content": "它呢"}
    assert req.history == original
    small = manager.conversation(req, token_budget=250, summarize=summary)
    assert len(small.recent) < 12 and small.summary


def test_long_report_summary_is_cached_across_runs_and_original_is_unchanged(tmp_path):
    manager = ContextManager(tmp_path, report_tokens=300)
    report = "# OPD 调研报告\n方法和实验说明 [1]。" * 500
    original = [{"role": "user", "content": "调研 OPD"}, {"role": "assistant", "content": report}]
    req = research_request().model_copy(update={"history": original, "run_id": "r1", "conversation_id": "conversation"})
    calls = []
    def compress(*args):
        calls.append(1)
        return summary(*args)
    bundle = manager.conversation(req, summarize=compress)
    assert calls and len(bundle.recent[1]["content"]) < len(report)
    cached_calls = len(calls)
    next_req = req.model_copy(update={"run_id": "r2"})
    again = manager.conversation(next_req, summarize=compress)
    assert len(calls) == cached_calls
    pointer = json.loads(again.recent[1]["content"])["previous_report"]["artifact_id"]
    original_bytes = manager.artifacts.read(pointer, "r2", limit=12000)["text"]
    assert "OPD" in original_bytes and original[1]["content"] == report


@pytest.mark.parametrize("tag", ["T1:react", "supervisor:review", "writer:report"])
def test_research_agents_do_not_receive_full_conversation(tmp_path, tag):
    manager = ContextManager(tmp_path, agent_tokens=ContextConfig().agent_tokens)
    req = research_request().model_copy(update={"history": [
        {"role": "user", "content": "不要微调。"},
        {"role": "assistant", "content": "UNRELATED-REPORT " * 10000}], "query": "比较方法"})
    prepared = manager.prepare([{"role": "user", "content": '{"question":"子任务 A","unknowns":["待核验"]}'}],
        request=req, purpose=tag, output_reserve=4096,
        summarize=lambda *a: pytest.fail("This agent must not summarize/reinject global history"))
    text = json.dumps(prepared.messages, ensure_ascii=False)
    assert "UNRELATED-REPORT" not in text and "不要微调" in text and "待核验" in text
    assert prepared.accounting["categories"]["conversation"] == 0


def test_parallel_agent_budgets_do_not_mutate_shared_manager(tmp_path):
    manager = ContextManager(tmp_path, agent_tokens={"researcher": 20000, "writer": 64000})
    def prepare(tag, content, reserve):
        return manager.prepare([{"role": "user", "content": content}], request=research_request(), purpose=tag, output_reserve=reserve)
    with ThreadPoolExecutor(2) as pool:
        a = pool.submit(prepare, "T1:react", "private A", 4096)
        b = pool.submit(prepare, "writer:report", "private B", 16384)
        a, b = a.result(), b.result()
    assert a.accounting["max_tokens"] == 20000 and b.accounting["max_tokens"] == 64000
    assert "private B" not in json.dumps(a.messages)
    assert a.accounting["output_reserve"] >= 4096 and b.accounting["output_reserve"] >= 16384
    assert manager.max_tokens == 30000


def provider(manager, create):
    llm = LLMClient.__new__(LLMClient)
    llm.model, llm.context_manager, llm._is_claude, llm.extra_body = "deepseek-v4-flash", manager, False, None
    llm.total_calls = llm.total_prompt_tokens = llm.total_completion_tokens = 0
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client.with_options = lambda **kwargs: client
    llm.client = client
    return llm


def test_actual_provider_receives_prepared_copy_and_output_reserve(tmp_path):
    manager = ContextManager(tmp_path, agent_tokens={"researcher": 12000})
    sent = []
    def create(**kwargs):
        sent.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=ChatCompletionMessage(role="assistant", content="done"), finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10))
    llm = provider(manager, create)
    req = research_request().model_copy(update={"run_id": "r", "history": history()})
    messages = [{"role": "user", "content": "my subtask"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "huge original [ev_a] " * 4000}]
    before = deepcopy(messages)
    with request_context(req):
        llm.invoke(messages, max_tokens=4096, tag="T1:react")
    assert messages == before
    assert "artifact_id" in sent[0]["messages"][-1]["content"]
    assert "Paper-7" not in json.dumps(sent[0]["messages"])
    assert sent[0]["max_tokens"] == 4096
    assert manager._payload_tokens(sent[0]["messages"], None, None) + 4096 <= 12000


def test_writer_long_output_and_truncation_are_explicit(monkeypatch):
    monkeypatch.delenv("WRITER_MAX_OUTPUT_TOKENS", raising=False)
    text = "# 完整报告\n" + "按用户要求解释方法 [1]。\n" * 1200
    llm = ScriptedLLM(text)
    report = Writer(llm, evidence_context=FakeContext()).run(research_request(), [ResearchNote(task_id="T1", question="方法", content="发现 [ev_a]", source_ids=["ev_a"])])
    assert llm.calls[0]["max_tokens"] == 16384 and len(report.report) >= len(text.strip())
    response = ChatCompletionMessage(role="assistant", content="# 截断报告\n发现 [1]")
    response.finish_reason = "length"
    truncated = Writer(SimpleNamespace(invoke=lambda **kwargs: response), evidence_context=FakeContext()).run(
        research_request(), [ResearchNote(task_id="T1", question="方法", content="发现 [ev_a]", source_ids=["ev_a"])])
    assert any("输出上限" in gap for gap in truncated.remaining_gaps)
    limits_for("deepseek-v4-flash").validate_output(16384)
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "8192")
    with pytest.raises(ValueError, match="超过"):
        limits_for("deepseek-v4-flash").validate_output(16384)
