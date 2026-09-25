from sage_scholar.agents.supervisor import NoteReview, ReviewResult, SubQuestion
from sage_scholar.graph.graph_builder import build_graph


class FakeSupervisor:
    def plan(self, research_brief, approved_pairs=None, review_result=None):
        return [
            SubQuestion(
                label="方法",
                question="有哪些主要方法？",
                rationale="覆盖核心方法分类",
            )
        ]

    def review(self, research_brief, pending_review_pairs, approved_pairs=None):
        return ReviewResult(
            note_reviews=[
                NoteReview(
                    relevance="PASS",
                    depth="PASS",
                    citations="PASS",
                    sources="PASS",
                    completeness="PASS",
                    verdict="approved",
                )
            ],
            missing_dimensions="",
        )


class FakeResearcher:
    def run(self, sub_question, note_feedback):
        return "有证据支持的研究笔记。", {}


class FakeWriter:
    def run(self, research_brief, notes):
        assert notes == ["有证据支持的研究笔记。"]
        return "# 最终报告"


def test_research_returns_report_without_rag_write_dependency():
    graph = build_graph(
        FakeSupervisor(),
        lambda _researcher_id: FakeResearcher(),
        FakeWriter(),
    )

    result = graph.invoke({"research_brief": "调研测试主题"})

    assert result == {"final_report": "# 最终报告"}


def test_missing_review_entries_are_reported_as_gaps():
    class MissingReview(FakeSupervisor):
        def review(self, **kwargs):
            return ReviewResult(note_reviews=[], missing_dimensions="")
    class Writer:
        def run(self, **kwargs):
            return "partial report"
    graph = build_graph(MissingReview(), lambda _: FakeResearcher(), Writer(), max_rounds=0)
    events = list(graph.stream({"research_brief": "topic"}))
    review = next(event["review_node"] for event in events if "review_node" in event)
    assert "未完成审查" in review["missing_dimensions"]
