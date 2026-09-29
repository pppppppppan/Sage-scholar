"""Replay committed read results; refuse calls whose external execution outcome is unknown."""
import hashlib
import json
from sage_scholar.tools.base_tool import BaseTool, ToolCallError
from sage_scholar.research.budget import ToolBudgetExceeded


class WorkflowInterrupted(Exception):
    pass


class UnknownToolExecution(WorkflowInterrupted):
    pass


class JournaledTool(BaseTool):
    def __init__(self, tool, store, run_id, task_key, cancelled=lambda: False, replay=lambda result: None):
        super().__init__(tool.name, tool.description, tool.parameters)
        self.source_access = tool.source_access
        self.tool, self.store, self.run_id, self.task_key = tool, store, run_id, task_key
        self.cancelled, self.replay = cancelled, replay

    def to_schema(self):
        return self.tool.to_schema()

    def run_tool(self, parameters):
        if self.cancelled():
            raise WorkflowInterrupted("任务已停止派发工具")
        key = self.task_key + ":" + hashlib.sha256(json.dumps([self.name, parameters], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        cached = self.store.call(self.run_id, key)
        if cached:
            status, payload = cached
            if status != "completed":
                raise UnknownToolExecution("外部工具执行状态未知，不能自动重放；请查看记录后明确重新开始")
            if payload.get("error"):
                raise ToolCallError(payload["error"], self.name)
            self.replay(payload["result"])
            return payload["result"]
        self.store.call_started(self.run_id, key, {"tool": self.name, "parameters": parameters,
            "task": self.task_key, "source_access": self.source_access})
        try:
            result = self.tool.run_tool(parameters)
        except ToolBudgetExceeded as exc:
            self.store.call_finished(self.run_id, key, {"error": str(exc)})
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
