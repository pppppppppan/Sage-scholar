import json
from sage_scholar.tools.base_tool import BaseTool, ToolParameter


class SkillsListTool(BaseTool):
    source_access = "none"
    replay_policy = "safe_retry"

    def __init__(self, service):
        super().__init__("skills_list", "列出可用 Skill 名称与简述，不读取正文；Skill 不扩展资料权限。", [])
        self.service = service

    def run_tool(self, parameters):
        return json.dumps({"skills": self.service.registry.catalog()}, ensure_ascii=False)


class SkillViewTool(BaseTool):
    source_access = "none"
    replay_policy = "safe_retry"

    def __init__(self, service, request):
        super().__init__("skill_view", "按需加载任务指南 SKILL.md 或 references 文件；同一 Run 固定版本。不是事实来源。", [
            ToolParameter(name="name", type="string", description="Skill Catalog 中的名称"),
            ToolParameter(name="file_path", type="string", description="可选：references/xxx.md；省略加载 SKILL.md", required=False),
        ])
        self.service, self.request = service, request

    def run_tool(self, parameters):
        return json.dumps(self.service.view(self.request, parameters["name"], parameters.get("file_path")), ensure_ascii=False)
