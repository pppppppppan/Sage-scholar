"""Small ReAct resume record in the existing RunStore.results table.

LangGraph owns task scheduling; this only stores an individual task's committed
messages. Tool side-effect/replay decisions remain in JournaledTool.
"""
import hashlib
import json
from sage_scholar.runtime.journal import WorkflowInterrupted


class ResearchProgress:
    VERSION = 1

    def __init__(self, store, run_id, task_id, revision=0):
        self.store, self.run_id = store, run_id
        self.key = f"react:{task_id}:{revision}"

    @staticmethod
    def fingerprint(task, request):
        payload = {"question": task.objective, "query": request.query,
                   "policy": request.source_policy.model_dump(mode="json"),
                   "constraints": request.metadata.get("constraints", [])}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def load(self, fingerprint):
        value = self.store.result(self.run_id, self.key)
        if value is not None and (value.get("schema") != self.VERSION or value.get("fingerprint") != fingerprint):
            raise WorkflowInterrupted("ReAct 恢复记录与当前子任务/资料范围不兼容，请明确重新开始任务")
        return value

    def save(self, fingerprint, value):
        self.store.save_result(self.run_id, self.key, {"schema": self.VERSION, "fingerprint": fingerprint, **value})
