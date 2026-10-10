import json
from sage_scholar.agents.supervisor import Supervisor
from sage_scholar.context.protection import prompt_brief
from sage_scholar.domain.research import ResearchTask, ResearchNote, SupervisorReview
from research_fakes import research_request, ScriptedLLM


def test_one_supervisor_reads_markdown_and_does_not_promote_every_unknown():
    llm = ScriptedLLM(SupervisorReview())
    request = research_request()
    task = ResearchTask(task_id="T1", objective="方法")
    note = ResearchNote(task_id="T1", question="方法", content="发现：方法有效。\n局限：跨数据集未知。[ev_a]", source_ids=["ev_a"])
    result = Supervisor(llm, 4).review(request, [task], [note])
    assert not result.remaining_gaps and not result.follow_up_questions
    assert json.loads(llm.calls[0]["messages"][-1]["content"])["notes"][0]["content"] == note.content
    assert llm.calls[0]["tag"] == "supervisor:review"


def test_brief_has_no_duplicate_execution_or_keyword_dimensions():
    request = research_request(max_tool_calls=7).model_copy(update={
        "query": "比较数据集和方法", "history": [{"role": "user", "content": "old"}],
        "metadata": {"constraints": ["不要微调"]}})
    brief = prompt_brief(request)
    assert set(brief) == {"objective", "constraints", "source_policy", "budget"}
    assert brief["constraints"] == ["不要微调"] and "execution" not in brief
    assert request.history and request.budget["max_tool_calls"] == 7
