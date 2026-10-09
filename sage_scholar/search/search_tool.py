import logging
from typing import Any
from sage_scholar.mcp.tool_adapter import MCPTool
from sage_scholar.tools import BaseTool, ToolParameter
from .adapter import BraveAdapter, TavilyAdapter
from sage_scholar.runtime.journal import WorkflowInterrupted

logger = logging.getLogger(__name__)


class SearchTool(BaseTool):
    source_access = "public"
    replay_policy = "safe_retry"
    def __init__(
        self,
        brave_tool: MCPTool | None = None,
        tavily_tool: MCPTool | None = None,
    ):
        if brave_tool is None and tavily_tool is None:
            raise ValueError("至少需要一个可用的搜索工具：Brave 或 Tavily。")
        super().__init__(
            name="search",
            description=(
                "Search the web for current information on any topic.\n"
                "\n"
                "When to use: need up-to-date information, fact verification, "
                "or gathering sources for a research topic.\n"
                "When NOT to use: the answer is already available in conversation "
                "context or previously retrieved documents.\n"
                "\n"
                "Returns: a list of search results, each containing title, URL, "
                "and a short content snippet (2-3 sentences). "
                "Snippets are previews only, not full article content."
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description=(
                        "The search query string. Use specific keywords for better results. "
                        'Example: "transformer self-attention mechanism 2017" '
                        'rather than "transformer".'
                    ),
                ),
                ToolParameter(
                    name="count",
                    type="integer",
                    description="Number of results to return. Range: 1-20. Default: 5.",
                    required=False,
                    default=5,
                ),
                ToolParameter(
                    name="time_filter",
                    type="string",
                    description=(
                        "Filter results by time. Either a relative keyword: "
                        '"day", "week", "month", "year", '
                        'or an absolute date range in "YYYY-MM-DDtoYYYY-MM-DD" format. '
                        'Example: "week" or "2025-01-01to2025-06-01".'
                    ),
                    required=False,
                ),
            ],
        )
        self.brave_adapter = BraveAdapter(brave_tool) if brave_tool is not None else None
        self.tavily_adapter = TavilyAdapter(tavily_tool) if tavily_tool is not None else None

    @staticmethod
    def _format_results(result_list: list) -> str:
        return "\n\n".join(f"[{i}] {result}" for i, result in enumerate(result_list, 1))

    def run_tool(self, parameters: dict[str, Any]) -> str:
        query = parameters.get("query")
        if not query:
            return "Error: missing required parameter 'query'. Provide a search query string."
        count = parameters.get("count")
        time_filter = parameters.get("time_filter")

        kwargs = {"query": query}
        if count is not None:
            kwargs["count"] = count
        if time_filter is not None:
            kwargs["time_filter"] = time_filter

        errors = 0
        adapters = [(name, adapter) for name, adapter in (
            ("Brave", self.brave_adapter), ("Tavily", self.tavily_adapter)
        ) if adapter is not None]
        for name, adapter in adapters:
            try:
                result_list = adapter.search(**kwargs)
                if result_list:
                    logger.info("[SearchTool] %s 返回 %d 条结果", name, len(result_list))
                    return self._format_results(result_list)
                logger.info("[SearchTool] %s 返回空结果", name)
            except WorkflowInterrupted:
                raise
            except Exception as e:
                errors += 1
                logger.warning("[SearchTool] %s 异常: %s", name, e)

        if errors == len(adapters):
            logger.error("[SearchTool] 所有搜索源不可用")
            return (
                "Error: all search services are currently unavailable. "
                "Continue working with the information you already have."
            )
        logger.warning("[SearchTool] 所有搜索源均无结果, query=%s", query)
        return (
            f"No results found for query '{query}'. "
            "Try rephrasing with different or broader keywords."
        )
