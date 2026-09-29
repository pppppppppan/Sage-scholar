"""Audit reuses existing academic discovery and source readers, budget and tool journal."""
import json

from sage_scholar.tools.policy import EvidenceTool, PolicyTool
from sage_scholar.research.budget import ResearchBudget, BudgetedTool
from sage_scholar.runtime.journal import JournaledTool


class AuditFetch:
    def __init__(self, provider, retrieval, store=None):
        self.provider, self.retrieval, self.store = provider, retrieval, store

    def for_request(self, request):
        return AuditSourceTools(self.provider, self.retrieval, self.store, request)


class AuditSourceTools:
    """Callable source reader plus metadata-only search; constructed without starting MCP."""
    def __init__(self, provider, retrieval, store, request):
        self.provider, self.retrieval, self.store, self.request = provider, retrieval, store, request
        self.budget = ResearchBudget(int(request.budget.get("max_tool_calls", 20)),
                                    store if request.run_id else None, request.run_id)

    def _tools(self):
        if not self.request.source_policy.permits_web:
            raise ValueError("当前资料策略禁止联网核验")
        runtime = self.provider.get(require_external=True)
        list(runtime._initialize_research_tools())
        return runtime.registry.tools

    def _invoke(self, tool, parameters, *, kind):
        tool = BudgetedTool(PolicyTool(tool, self.request.source_policy), self.budget)
        if self.store and self.request.run_id:
            tool = JournaledTool(tool, self.store, self.request.run_id, "audit:" + kind)
        return tool.run_tool(parameters)

    def search(self, reference):
        """Search snippets/abstracts never pass through EvidenceTool."""
        tools = self._tools()
        # Some providers interpret years/author names as required body-text terms.
        # Search the title, then enforce author/year constraints on returned metadata.
        query = reference.title
        unified = tools.get("mcp__paper-search__search_papers")
        if unified:
            result = self._invoke(unified, {"query": query, "sources": "arxiv,semantic,crossref",
                                           "max_results_per_source": 5}, kind="identity")
            return self._papers(result)
        # Compatibility with paper-search versions that expose only per-provider searches.
        papers = []
        for name in ("search_semantic", "search_arxiv", "search_crossref"):
            tool = tools.get("mcp__paper-search__" + name)
            if tool:
                result = self._invoke(tool, {"query": query, "max_results": 5}, kind="identity")
                papers.extend(self._papers(result))
        if not any("mcp__paper-search__" + name in tools for name in ("search_semantic", "search_arxiv", "search_crossref")):
            raise RuntimeError("学术文献检索工具不可用")
        return papers

    @staticmethod
    def _papers(result):
        # Some MCP versions serialize a list as separate JSON text blocks.
        decoder, offset, papers = json.JSONDecoder(), 0, []
        while result[offset:].strip():
            offset += len(result[offset:]) - len(result[offset:].lstrip())
            data, offset = decoder.raw_decode(result, offset)
            if isinstance(data, dict) and "title" in data:
                records = [data]
            else:
                records = data.get("papers", data.get("results", [])) if isinstance(data, dict) else data
            if not isinstance(records, list):
                raise ValueError("学术检索没有返回结构化文献元数据")
            papers.extend(record for record in records if isinstance(record, dict))
        return papers

    def __call__(self, source):
        tools = self._tools()
        name, parameters = "mcp__fetch__fetch", {"url": source}
        if source.startswith("arxiv:"):
            name, parameters = "read_arxiv_paper", {"paper_id": source.removeprefix("arxiv:")}
        tool = tools.get(name)
        if tool is None:
            raise RuntimeError("引用原文读取工具不可用")
        found = []
        def record(item):
            found.append(item)
            if self.store and self.request.run_id:
                self.store.register_evidence(self.request.run_id, item.evidence_id)
        tool = EvidenceTool(tool, self.retrieval.evidence_store, record)
        self._invoke(tool, parameters, kind="source")
        return found or self.retrieval.evidence_store.find_source(source)
