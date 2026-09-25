from __future__ import annotations

import hashlib
import json

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
    """Keep raw external observations. A tool response is not a parsed paper or a verified claim."""

    def __init__(self, tool, store, record):
        super().__init__(tool.name, tool.description, tool.parameters)
        self.source_access = tool.source_access
        self._tool, self._store, self._record = tool, store, record

    def to_schema(self):
        return self._tool.to_schema()

    def run_tool(self, parameters):
        result = self._tool.run_tool(parameters)
        params = json.dumps(parameters, sort_keys=True, ensure_ascii=False)
        evidence = EvidenceRef.create(
            source_id=f"tool:{self.name}:" + hashlib.sha256(params.encode()).hexdigest(),
            source_version=hashlib.sha256(result.encode()).hexdigest(),
            parser_version="raw-tool-response-v1", original_text=result,
            metadata={"tool": self.name, "parameters": parameters, "kind": "tool_observation"},
        )
        self._store.save(evidence)
        self._record(evidence)
        return f"[{evidence.evidence_id}]\n{result}"
