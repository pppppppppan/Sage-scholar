"""Immutable per-call allocations; a shared manager never mutates its global budget."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ContextPolicy:
    agent: str
    history: str = "window"  # window, summary, none
    memory_items: int = 5


def policy_for(purpose):
    if purpose == "supervisor:review":
        return ContextPolicy("review", "none", 0)
    if purpose.startswith("supervisor:") or purpose == "interaction:assess":
        return ContextPolicy("supervisor", "summary", 3)
    if purpose.endswith((":react", ":forced_summary")):
        return ContextPolicy("researcher", "none", 3)
    if purpose.startswith("writer:"):
        return ContextPolicy("writer", "none", 3)
    if purpose.startswith(("memory:", "context:")):
        return ContextPolicy("background", "none", 0)
    return ContextPolicy("conversation")
