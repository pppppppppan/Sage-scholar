"""Audit has its own small graph; semantic verification stays in CitationVerifier."""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from sage_scholar.research.verification import CitationVerifier
from sage_scholar.runtime.journal import WorkflowInterrupted


class AuditState(TypedDict):
    claims: list
    citations: dict
    items: list
    cursor: int
    outcome: dict


def build_audit_graph(service, request, llm, sources, *, checkpointer=None, cancelled=lambda: False):
    verifier = CitationVerifier(llm, sources.context)
    graph = StateGraph(AuditState)
    def parse(state):
        claims, citations = service.parse(request, llm)
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
    graph.add_edge(START, "parse")
    graph.add_conditional_edges("parse", route)
    graph.add_conditional_edges("verify", route)
    graph.add_edge("render", END)
    return graph.compile(checkpointer=checkpointer)
