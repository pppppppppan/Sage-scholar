"""Expose existing section extraction as locators, not a second PDF pipeline."""
import json
from .base_tool import BaseTool, ToolParameter


class DocumentSectionsTool(BaseTool):
    source_access = "local"
    replay_policy = "safe_retry"

    def __init__(self, retrieval, policy, record):
        super().__init__("document_sections", "列出授权文档的章节和 evidence_id，不返回整篇原文。随后 read_context_artifact 定位精读。", [
            ToolParameter(name="offset", type="integer", description="章节目录分页起点", required=False),
        ])
        self.retrieval, self.policy, self.record = retrieval, policy, record

    def run_tool(self, parameters):
        sections = self.retrieval.read_documents_sync(self.policy)
        offset = max(0, int(parameters.get("offset", 0)))
        page = sections[offset:offset + 40]
        for item in page:
            self.record(item)
        return json.dumps({"sections": [{"evidence_id": e.evidence_id, "document_id": e.document_id,
            "section": e.section, "page": e.page, "location": e.location, "chars": len(e.original_text)} for e in page],
            "offset": offset, "total": len(sections), "next_offset": offset + 40 if offset + 40 < len(sections) else None}, ensure_ascii=False)
