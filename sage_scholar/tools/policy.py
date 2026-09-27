from __future__ import annotations

import hashlib
import json
import re

from sage_scholar.domain.evidence import EvidenceRef
from sage_scholar.domain.execution import SourcePolicy
from .base_tool import BaseTool, ToolCallError


class PolicyTool(BaseTool):
    """Defense at invocation as well as schema filtering."""

    def __init__(self, tool: BaseTool, policy: SourcePolicy):
        super().__init__(tool.name, tool.description, tool.parameters)
        self._tool, self._policy = tool, policy
        self.source_access = tool.source_access

    def to_schema(self):
        return self._tool.to_schema()

    def run_tool(self, parameters):
        if not self._policy.permits_tool(self.source_access):
            raise ToolCallError(f"Source policy denies tool: {self.name}", self.name)
        return self._tool.run_tool(parameters)


class EvidenceTool(BaseTool):
    """Discoveries/status/errors stay observations; only explicit source readers emit Evidence."""

    def __init__(self, tool, store, record, observe=lambda item: None):
        super().__init__(tool.name, tool.description, tool.parameters)
        self.source_access = tool.source_access
        self._tool, self._store, self._record, self._observe = tool, store, record, observe

    def to_schema(self):
        return self._tool.to_schema()

    def run_tool(self, parameters):
        try:
            result = self._tool.run_tool(parameters)
        except Exception as exc:
            self._observe({"tool": self.name, "parameters": parameters, "text": str(exc),
                           "evidence_ids": [], "kind": "error"})
            raise
        observation = {"tool": self.name, "parameters": parameters, "text": result, "evidence_ids": []}
        failed = bool(re.match(r"\s*(error|failed|exception|错误|失败|404|403)\b", result, re.I))
        status_only = False
        try:
            data = json.loads(result)
            if isinstance(data, dict):
                status_only = set(data) <= {"status", "message", "success", "path", "id"}
                failed = failed or bool(data.get("error") or data.get("isError")
                    or data.get("success") is False or data.get("status") in ("error", "failed"))
        except (ValueError, TypeError):
            pass
        source_id = parameters.get("url")
        if self.name == "read_arxiv_paper" and parameters.get("paper_id"):
            source_id = "arxiv:" + parameters["paper_id"]
        if (not failed and not status_only and result.strip() and self._tool.evidence_kind == "source_content"
                and isinstance(source_id, str) and source_id.startswith(("http://", "https://", "arxiv:"))):
            chunks = []
            version = hashlib.sha256(result.encode()).hexdigest()
            for start in range(0, len(result), 6000):
                content = result[start:start + 6000]
                evidence = EvidenceRef.create(
                    source_id=source_id, source_version=version,
                    parser_version="source-reader-v1", original_text=content,
                    location={"char_start": start, "char_end": start + len(content)},
                    metadata={"tool": self.name, "parameters": parameters, "kind": "source_content"},
                )
                self._store.save(evidence)
                self._record(evidence)
                observation["evidence_ids"].append(evidence.evidence_id)
                chunks.append(f"[{evidence.evidence_id}]\n{content}")
            self._observe(observation)
            return "\n\n".join(chunks)
        observation["kind"] = "error" if failed else "discovery_or_status"
        self._observe(observation)
        return "[Observation: 尚不能作为最终报告证据；请读取来源原文]\n" + result
