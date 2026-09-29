"""Audit can read an explicitly cited public source via existing policy/evidence tools."""
from sage_scholar.tools.policy import EvidenceTool, PolicyTool
from sage_scholar.research.budget import ResearchBudget, BudgetedTool
from sage_scholar.runtime.journal import JournaledTool


class AuditFetch:
    def __init__(self, provider, retrieval, store=None):
        self.provider, self.retrieval, self.store = provider, retrieval, store

    def for_request(self, request):
        budget = ResearchBudget(int(request.budget.get("max_tool_calls", 20)), self.store if request.run_id else None, request.run_id)
        def fetch(source):
            runtime = self.provider.get(require_external=True)
            list(runtime._initialize_research_tools())
            parameters = {"url": source}
            name = "mcp__fetch__fetch"
            if source.startswith("arxiv:"):
                name, parameters = "read_arxiv_paper", {"paper_id": source.removeprefix("arxiv:")}
            tool = runtime.registry.tools.get(name)
            if tool is None:
                raise RuntimeError("引用原文读取工具不可用")
            found = []
            def record(item):
                found.append(item)
                if self.store and request.run_id:
                    self.store.register_evidence(request.run_id, item.evidence_id)
            tool = PolicyTool(EvidenceTool(tool, self.retrieval.evidence_store, record), request.source_policy)
            tool = BudgetedTool(tool, budget)
            if self.store and request.run_id:
                tool = JournaledTool(tool, self.store, request.run_id, "audit:source")
            tool.run_tool(parameters)
            return found or self.retrieval.evidence_store.find_source(source)
        return fetch
