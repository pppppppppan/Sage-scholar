from .base_tool import BaseTool, ToolParameter


class RAGTool(BaseTool):
    source_access = "local"
    replay_policy = "safe_retry"

    def __init__(self, retrieval, policy, record=lambda evidence: None):
        super().__init__(
            name="rag_search",
            description="检索允许的本地文献，融合文字和 PDF 图像检索，返回 evidence_id/版本/位置。图像命中只是来源线索，视觉细节须用 read_pdf_visual 读取。",
            parameters=[ToolParameter(name="query", type="string", description="具体问题或关键词")],
        )
        self._retrieval, self._policy, self._record = retrieval, policy, record

    def run_tool(self, parameters):
        # Model arguments cannot override the run's document scope.
        warnings = []
        if hasattr(self._retrieval, "retrieve_with_warnings"):
            evidence, warnings = self._retrieval.retrieve_with_warnings(parameters["query"], self._policy)
        else:
            evidence = self._retrieval.retrieve_sync(parameters["query"], self._policy)
        for item in evidence:
            self._record(item)
        result = "\n\n".join(
            f"[{item.evidence_id}] document={item.document_id} section={item.section}\n{item.original_text}"
            for item in evidence
        ) or "本次允许的本地资料中未找到相关内容"
        return result + ("\n资料读取提示（非证据）：" + "；".join(warnings) if warnings else "")
