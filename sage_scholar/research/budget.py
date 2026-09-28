import threading
from sage_scholar.tools.base_tool import BaseTool, ToolCallError


class ResearchBudget:
    """Only a shared tool-call cap; not a token/cost budget manager."""
    def __init__(self, limit):
        self.limit, self.used = limit, 0
        self._denied_calls = 0
        self._blocked = False
        self._lock = threading.Lock()

    def reserve(self):
        with self._lock:
            if self.used >= self.limit:
                self._denied_calls += 1
                self._blocked = True
                raise ToolCallError("Research tool-call budget exhausted")
            self.used += 1

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
            return self._denied_calls

    @property
    def blocked_by_budget(self):
        with self._lock:
            return self._blocked

    def mark_blocked(self):
        """Workflow reports work that could not be dispatched; no business inference here."""
        with self._lock:
            self._blocked = True


class BudgetedTool(BaseTool):
    def __init__(self, tool, budget):
        super().__init__(tool.name, tool.description, tool.parameters)
        self.source_access = tool.source_access
        self._tool, self._budget = tool, budget

    def to_schema(self):
        return self._tool.to_schema()

    def run_tool(self, parameters):
        self._budget.reserve()
        return self._tool.run_tool(parameters)
