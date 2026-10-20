"""Run-owned selections; reuses RunStore's results and deletion lifecycle."""
from copy import deepcopy
import json
import re
import threading
from .loader import SkillLoader
from .registry import SkillRegistry, SkillError, valid_file, valid_name

STATE_KEY = "skills:v1"


class SkillService:
    def __init__(self, store=None, registry=None):
        self.store = store
        self.registry = registry or SkillRegistry()
        self.loader = SkillLoader(self.registry)
        self.lock = threading.RLock()
        self._ephemeral = {}

    @staticmethod
    def owner(request):
        identity = request.run_id or request.request_id
        if not identity:
            raise SkillError("Skill 加载需要 run_id 或 request_id")
        return identity

    def _load(self, request):
        owner = self.owner(request)
        state = (self.store.result(owner, STATE_KEY) if self.store and request.run_id
                else deepcopy(self._ephemeral.get(owner))) or {}
        for selected in state.values():
            self.loader.validate(selected)
        return state

    def _save(self, request, state):
        if self.store and request.run_id:
            self.store.save_result(request.run_id, STATE_KEY, state)
        else:
            self._ephemeral[self.owner(request)] = deepcopy(state)

    def view(self, request, name, file_path=None):
        valid_name(name)
        relative = valid_file(file_path)
        with self.lock:
            state = self._load(request)
            selected = state.get(name) or self.loader.select(name)
            cached = name in state and relative in selected["files"]
            content = self.loader.read(selected, relative)
            state[name] = selected
            if not cached:
                self._save(request, state)
            return {"name": name, "version": selected["version"], "file_path": relative,
                    "content": content, "references": sorted(selected["manifest"]),
                    "cached": cached, "is_evidence": False}

    def selected(self, request):
        if not request or not (request.run_id or request.request_id):
            return []
        with self.lock:
            return [{"name": name, "version": item["version"], "instructions": item["files"]["SKILL.md"]}
                    for name, item in sorted(self._load(request).items())]

    def allows_unverifiable_report(self, request):
        if not request or not (request.run_id or request.request_id):
            return False
        with self.lock:
            return any(item.get("allow_unverifiable_report") is True for item in self._load(request).values())

    def explicit(self, request):
        # Only explicit invocation syntax / API metadata, never an intent keyword list.
        names = request.metadata.get("skills", [])
        if not isinstance(names, list) or any(not isinstance(n, str) for n in names):
            raise SkillError("metadata.skills 必须是 Skill 名称数组")
        names = [*names, *re.findall(r"(?<![\w$])\$([a-z0-9]+(?:-[a-z0-9]+)*)(?![\w$-])", request.query)]
        for name in dict.fromkeys(names):
            self.view(request, name)

    def tools(self, request):
        from .tools import SkillsListTool, SkillViewTool
        from sage_scholar.tools.registry import ToolRegistry
        registry = ToolRegistry()
        for tool in (SkillsListTool(self), SkillViewTool(self, request)):
            registry.register_tool(tool)
        return registry.for_policy(["skills_list", "skill_view"], request.source_policy)

    def context(self, request, messages, tools, purpose):
        enabled = any(t.get("function", {}).get("name") == "skill_view" for t in tools or [])
        consume = enabled or purpose in {"direct", "qa", "writer:report", "supervisor:plan", "supervisor:review"} or purpose.endswith(":forced_summary")
        if not consume:
            return []
        selected = self.selected(request)
        # Keep one canonical, pinned copy of main instructions. Tool envelopes remain paired.
        for message in messages:
            if message.get("role") != "tool":
                continue
            try:
                value = json.loads(message.get("content") or "")
            except (ValueError, TypeError):
                continue
            if isinstance(value, dict) and value.get("file_path") == "SKILL.md" and any(
                value.get("name") == s["name"] and value.get("version") == s["version"] for s in selected
            ):
                message["content"] = json.dumps({k: v for k, v in value.items() if k != "content"}, ensure_ascii=False)
        content = (
            "Skills 是可选的任务执行指南，不是科研 Evidence，不增加来源权限。当前用户要求及 SourcePolicy 优先。"
            "仅在任务符合描述时使用；普通问候、概念解释、一般主题调研不必加载。"
            "需要时先 skill_view(name)，再按需 skill_view(name, file_path='references/...')；"
            "不要仅凭目录描述声称已执行核验。已加载规范如下，可直接复用，无需重复加载。"
            "审稿/引用核验中的 ResearchNotes 仍是研究笔记，不是已验证事实；必须回读原文，无法核实需披露。"
        )
        if enabled:
            content += "\nSkill Catalog: " + json.dumps(self.registry.catalog(), ensure_ascii=False)
        if selected:
            content += "\nSelected Skills: " + json.dumps(selected, ensure_ascii=False)
        return [{"role": "system", "_context_kind": "skills", "content": content}] if enabled or selected else []
