import threading
from sage_scholar.tools.base_tool import BaseTool, ToolCallError


class ToolBudgetExceeded(ToolCallError):
    """Rejected before dispatch; the external tool did not execute."""


class ResearchBudget:
    """Only a shared tool-call cap; not a token/cost budget manager."""
    def __init__(self, limit, store=None, run_id=None):
        self.limit, self._used = limit, 0
        self.store, self.run_id = store, run_id
        self._denied_calls = 0
        self._blocked = False
        self._lock = threading.Lock()

    @property
    def used(self):
        return self.store.budget(self.run_id)["tool_calls"] if self.store else self._used

    def reserve(self):
        if self.store:
            if not self.store.reserve(self.run_id, self.limit):
                raise ToolBudgetExceeded("Research tool-call budget exhausted")
            return
        with self._lock:
            if self.used >= self.limit:
                self._denied_calls += 1
                self._blocked = True
                raise ToolBudgetExceeded("Research tool-call budget exhausted")
            self._used += 1

    @property
    def exhausted(self):
        """Capacity reached, not a judgment of research completeness."""
        with self._lock:
            return self.used >= self.limit

    @property
    def remaining(self):
        with self._lock:
            return max(0, self.limit - self.used)

    @property
    def denied_calls(self):
        with self._lock:
            return self.store.budget(self.run_id)["denied_calls"] if self.store else self._denied_calls

    @property
    def blocked_by_budget(self):
        with self._lock:
            return bool(self.store.budget(self.run_id)["blocked_by_budget"]) if self.store else self._blocked

    def mark_blocked(self):
        """Workflow reports work that could not be dispatched; no business inference here."""
        with self._lock:
            self._blocked = True
            if self.store:
                self.store.mark_blocked(self.run_id)


class BudgetedTool(BaseTool):
    def __init__(self, tool, budget):
        super().__init__(tool.name, tool.description, tool.parameters)
        self.source_access = tool.source_access
        self._tool, self._budget = tool, budget

    def to_schema(self):
        return self._tool.to_schema()

    def run_tool(self, parameters):
        from sage_scholar.runtime.control import check_control
        check_control()
        self._budget.reserve()
        return self._tool.run_tool(parameters)
