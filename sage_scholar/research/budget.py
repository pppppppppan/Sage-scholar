import threading
from sage_scholar.tools.base_tool import BaseTool, ToolCallError


class ResearchBudget:
    """Only a shared tool-call cap; not a token/cost budget manager."""
    def __init__(self, limit):
        self.limit, self.used = limit, 0
        self._lock = threading.Lock()

    def reserve(self):
        with self._lock:
            if self.used >= self.limit:
                raise ToolCallError("Research tool-call budget exhausted")
            self.used += 1

    @property
    def exhausted(self):
        with self._lock:
            return self.used >= self.limit


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
