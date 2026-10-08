"""Audit has its own small graph; semantic verification stays in CitationVerifier."""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from sage_scholar.research.verification import CitationVerifier
from sage_scholar.runtime.journal import WorkflowInterrupted


class AuditState(TypedDict):
    human_question: dict | None
    human_answer: dict | None
    claims: list
    citations: dict
    items: list
    cursor: int
    outcome: dict


def build_audit_graph(service, request, llm, sources, *, checkpointer=None, cancelled=lambda: False):
    verifier = CitationVerifier(llm, sources.context)
    graph = StateGraph(AuditState)
    def assess_node(state):
        from sage_scholar.runtime.interaction import assess
        return {"human_question": assess(request, llm) if checkpointer and (request.metadata.get("human_question") or not request.metadata.get("audit", {}).get("claims")) else None}
    def human_node(state):
        from sage_scholar.runtime.interaction import ask
        return {"human_answer": ask(state.get("human_question"))}
    def parse(state):
        from sage_scholar.runtime.control import check_control
        check_control()
        clarification = (state.get("human_answer") or {}).get("answer", "")
        effective = request.model_copy(update={"query": request.query + "\n用户补充：" + clarification}) if clarification else request
        claims, citations = service.parse(effective, llm)
        return {"claims": claims, "citations": citations, "items": [], "cursor": 0}
    def verify(state):
        if cancelled():
            raise WorkflowInterrupted("引用核验已中断")
        i = state["cursor"]
        item = service.verify_claim(i + 1, state["claims"][i], sources, state["citations"], verifier)
        return {"items": [*state["items"], item], "cursor": i + 1}
    def route(state):
        return "verify" if state["cursor"] < len(state["claims"]) else "render"
    def render(state):
        return {"outcome": service.outcome(state["items"]).model_dump(mode="json")}
    graph.add_node("parse", parse)
    graph.add_node("verify", verify)
    graph.add_node("render", render)
    graph.add_node("assess", assess_node)
    graph.add_node("human", human_node)
    graph.add_edge(START, "assess")
    graph.add_edge("assess", "human")
    graph.add_edge("human", "parse")
    graph.add_conditional_edges("parse", route)
    graph.add_conditional_edges("verify", route)
    graph.add_edge("render", END)
    return graph.compile(checkpointer=checkpointer)
