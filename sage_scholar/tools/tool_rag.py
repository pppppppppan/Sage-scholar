from .base_tool import BaseTool, ToolParameter


class RAGTool(BaseTool):
    source_access = "local"
    replay_policy = "safe_retry"

    def __init__(self, retrieval, policy, record=lambda evidence: None, *, visual_mode="auto"):
        super().__init__(
            name="rag_search",
            description="检索允许的本地文献，融合文字和 PDF 图像检索，返回 evidence_id/版本/位置。图像命中只是来源线索，视觉细节须用 read_pdf_visual 读取。",
            parameters=[ToolParameter(name="query", type="string", description="具体问题或关键词"),
                ToolParameter(name="visual_mode", type="string", required=False,
                    description="auto/off/on。文字不足且确需读原图时可用 on；充分的文字问答用 auto，不为普通 QA 强制检索图像。用户禁止读图时不能覆盖。")],
        )
        self._retrieval, self._policy, self._record = retrieval, policy, record
        self.visual_mode = visual_mode

    def run_tool(self, parameters):
        # Model arguments cannot override the run's document scope.
        warnings = []
        if hasattr(self._retrieval, "retrieve_with_warnings"):
            mode = "off" if self.visual_mode == "off" else parameters.get("visual_mode", self.visual_mode)
            evidence, warnings = self._retrieval.retrieve_with_warnings(parameters["query"], self._policy, visual_mode=mode)
        else:
            evidence = self._retrieval.retrieve_sync(parameters["query"], self._policy)
        for item in evidence:
            self._record(item)
        result = "\n\n".join(
            f"[{item.evidence_id}] document={item.document_id} section={item.section} page={item.page} location={item.location}\n{item.original_text}"
            for item in evidence
        ) or "本次允许的本地资料中未找到相关内容"
        return result + ("\n资料读取提示（非证据）：" + "；".join(warnings) if warnings else "")
