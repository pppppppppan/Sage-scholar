import threading
from sage_scholar.domain.research import SubQuestion, SupervisorReview
from sage_scholar.graph.graph_builder import build_graph
from research_fakes import research_request, FakeSupervisor, FakeResearcher, FakeWriter


def test_parallel_research_then_one_global_writer():
    barrier, reviewed, written = threading.Barrier(2, timeout=3), [], []
    class Supervisor(FakeSupervisor):
        def plan(self, request):
            return [SubQuestion(objective="方法"), SubQuestion(objective="实验")]
        def review(self, request, tasks, notes):
            reviewed.append({n.task_id for n in notes})
            return SupervisorReview()
    class Researcher(FakeResearcher):
        def run(self, *args):
            barrier.wait()
            return super().run(*args)
    class Writer(FakeWriter):
        def run(self, request, notes, gaps):
            written.append([n.content for n in notes])
            return super().run(request, notes, gaps)
    graph = build_graph(Supervisor(), lambda _: Researcher(), Writer())
    result = graph.invoke({"execution": research_request()})
    assert reviewed == [{"T1", "T2"}] and len(written) == 1 and len(written[0]) == 2
    assert result["report_result"].citations == {1: "ev_a"}
    assert not result["report_result"].partial
    assert not {"synthesis_node", "outline_node", "report_review_node"} & set(graph.nodes)


def test_followup_adds_question_without_repeating_completed_work():
    calls = []
    class Supervisor(FakeSupervisor):
        def review(self, request, tasks, notes):
            return (SupervisorReview(remaining_gaps=["缺实验"],
                follow_up_questions=[SubQuestion(objective="仅补查共同数据集实验")])
                if len(notes) == 1 else SupervisorReview())
    class Researcher(FakeResearcher):
        def run(self, task, request):
            calls.append(task.task_id)
            return super().run(task, request)
    graph = build_graph(Supervisor(), lambda _: Researcher(), FakeWriter())
    result = graph.invoke({"execution": research_request(max_rounds=1)})
    assert calls == ["T1", "T2"] and len(result["research_notes"]) == 2
    assert not result["report_result"].partial


def test_task_limit_still_writes_available_material():
    class Supervisor(FakeSupervisor):
        def review(self, *args):
            return SupervisorReview(remaining_gaps=["缺实验"], follow_up_questions=[SubQuestion(objective="实验")])
    graph = build_graph(Supervisor(), lambda _: FakeResearcher(), FakeWriter())
    result = graph.invoke({"execution": research_request(max_tasks=1, max_rounds=10)})
    assert result["report_result"].partial and result["report_result"].report
    assert "缺实验" in result["report_result"].remaining_gaps
