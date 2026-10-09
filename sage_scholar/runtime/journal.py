"""Durable tool journal with explicit retry safety for ambiguous executions."""
import hashlib
import json
import logging
from sage_scholar.tools.base_tool import BaseTool, ToolCallError
from sage_scholar.research.budget import ToolBudgetExceeded

logger = logging.getLogger(__name__)

# Compatibility for calls persisted before replay_policy was added.  Keep this
# list exact: a newly added MCP mutation must remain manual until explicitly
# reviewed and declared safe.
_LEGACY_SAFE_RETRY_TOOLS = frozenset({
    "search",
    "rag_search",
    "read_arxiv_paper",
    "mcp__fetch__fetch",
    "mcp__tavily__tavily_search",
    "mcp__brave-search__brave_web_search",
    "mcp__paper-search__search_arxiv",
    "mcp__paper-search__search_google_scholar",
    "mcp__paper-search__download_arxiv",
    "mcp__paper-search__read_arxiv_paper",
    "mcp__github__search_repositories",
    "mcp__github__search_code",
})


def is_safe_retry_intent(intent):
    """Return true only for explicitly idempotent/read-only persisted calls."""
    intent = intent or {}
    policy = intent.get("replay_policy")
    if policy is not None:
        return policy == "safe_retry"
    return intent.get("tool") in _LEGACY_SAFE_RETRY_TOOLS


class WorkflowInterrupted(Exception):
    pass


class UnknownToolExecution(WorkflowInterrupted):
    pass


class JournaledTool(BaseTool):
    def __init__(self, tool, store, run_id, task_key, cancelled=lambda: False, replay=lambda result: None):
        super().__init__(tool.name, tool.description, tool.parameters)
        self.source_access = tool.source_access
        self.replay_policy = tool.replay_policy
        self.tool, self.store, self.run_id, self.task_key = tool, store, run_id, task_key
        self.cancelled, self.replay = cancelled, replay

    def to_schema(self):
        return self.tool.to_schema()

    def run_tool(self, parameters):
        from sage_scholar.runtime.control import check_control
        check_control()
        if self.cancelled():
            raise WorkflowInterrupted("任务已停止派发工具")
        key = self.task_key + ":" + hashlib.sha256(json.dumps([self.name, parameters], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        cached = self.store.call(self.run_id, key)
        if cached:
            status, payload = cached
            if status != "completed":
                intent = {"tool": self.name, "parameters": parameters, "task": self.task_key,
                          "source_access": self.source_access, "replay_policy": self.replay_policy}
                if self.replay_policy != "safe_retry":
                    raise UnknownToolExecution("外部工具执行状态未知，不能自动重放；请查看记录后明确重新开始")
                self.store.call_restarted(self.run_id, key, intent)
                logger.info("Retrying read-only tool with unknown prior outcome: %s (%s)", self.name, key)
            else:
                if payload.get("error"):
                    raise ToolCallError(payload["error"], self.name)
                self.replay(payload["result"])
                return payload["result"]
        else:
            intent = {"tool": self.name, "parameters": parameters, "task": self.task_key,
                      "source_access": self.source_access, "replay_policy": self.replay_policy}
            self.store.call_started(self.run_id, key, intent)
        try:
            result = self.tool.run_tool(parameters)
        except ToolBudgetExceeded as exc:
            # Dispatch was denied, so there is no ambiguous external side effect to replay.
            with self.store.transaction() as db:
                db.execute("DELETE FROM calls WHERE run_id=? AND call_key=?", (self.run_id, key))
            raise
        except WorkflowInterrupted:
            raise
        except Exception as exc:
            # Any external exception can hide a successful remote operation. Keep the started
            # record unknown until restart/manual review, rather than treating it as never run.
            if self.source_access != "public":
                self.store.call_finished(self.run_id, key, {"error": str(exc)})
            raise
        self.store.call_finished(self.run_id, key, {"result": result})
        return result


class DurableResearcher:
    def __init__(self, factory, store, run_id, cancelled):
        self.factory, self.store, self.run_id, self.cancelled = factory, store, run_id, cancelled

    def run(self, task, contract, dependency_notes=()):
        from sage_scholar.domain.research import ResearchNote
        key = f"note:{task.task_id}:{task.revision}"
        cached = self.store.result(self.run_id, key)
        if cached is not None:
            return ResearchNote.model_validate(cached)
        if self.cancelled():
            raise WorkflowInterrupted("任务已中断")
        note = self.factory(task).run(task, contract, dependency_notes)
        self.store.save_result(self.run_id, key, note.model_dump(mode="json"))
        return note
