import json
from sage_scholar.tools.base_tool import BaseTool, ToolParameter


class ContextReadTool(BaseTool):
    source_access = "none"
    def __init__(self, manager, evidence, owner):
        super().__init__("read_context_artifact", "按 ID 和 offset 回读本次任务外置内容或已授权 Evidence；不联网。", [
            ToolParameter(name="identity", type="string", description="art_ 或 ev_ ID"),
            ToolParameter(name="offset", type="integer", description="字符起点", required=False),
            ToolParameter(name="limit", type="integer", description="回读字符数，默认 2000，最多 12000", required=False),
        ])
        self.manager, self.evidence, self.owner = manager, evidence, owner

    def run_tool(self, parameters):
        identity, offset = parameters["identity"], parameters.get("offset", 0)
        limit = parameters.get("limit", 2000)
        if identity.startswith("ev_"):
            result = self.manager.reread_evidence(self.evidence, identity, offset=offset, limit=limit)
        else:
            result = self.manager.artifacts.read(identity, self.owner, offset=offset, limit=limit)
        return json.dumps(result, ensure_ascii=False)
