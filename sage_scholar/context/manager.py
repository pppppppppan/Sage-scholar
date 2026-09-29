"""One provider-independent request assembler. Protected constraints are never summarized away."""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
import json
import re

from .token_counter import TokenCounter
from .artifacts import ArtifactStore

CURRENT_REQUEST = ContextVar("sage_execution_request", default=None)


@contextmanager
def request_context(request):
    token = CURRENT_REQUEST.set(request)
    try:
        yield
    finally:
        CURRENT_REQUEST.reset(token)


class ContextBudgetExceeded(ValueError):
    pass


@dataclass
class PreparedContext:
    messages: list[dict]
    accounting: dict


class ContextManager:
    def __init__(self, data_dir, *, max_tokens=30000, reserve_ratio=0.15, memories=None):
        self.counter = TokenCounter()
        self.max_tokens, self.reserve_ratio = max_tokens, reserve_ratio
        self.artifacts = ArtifactStore(data_dir)
        self.memories = memories
        self.run_store = None

    def externalize(self, text, owner, kind="tool_result", preview_tokens=320):
        identity = self.artifacts.put(text, owner, kind)
        # Verbatim excerpt, explicitly not a verified or exhaustive summary.
        return json.dumps({"artifact_id": identity, "kind": kind, "is_evidence": False,
                           "evidence_ids": list(dict.fromkeys(re.findall(r"ev_[a-zA-Z0-9_-]+", text))),
                           "excerpt": self.counter.trim(text, preview_tokens), "truncated": True,
                           "reread": "read_context_artifact 可按 offset 回读原始内容"}, ensure_ascii=False)

    @staticmethod
    def protected_facts(messages):
        protected = []
        keys = {"constraints", "unknowns", "conflicts", "limitations", "missing", "remaining_gaps", "negative_evidence", "unresolved_gaps"}
        def walk(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in keys and item:
                        protected.append({key: item})
                    else:
                        walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)
        for msg in messages:
            try:
                walk(json.loads(msg.get("content") or ""))
            except (ValueError, TypeError):
                pass
        return protected

    @staticmethod
    def validate_pairs(messages):
        pending = set()
        for msg in messages:
            if msg.get("role") == "tool":
                identity = msg.get("tool_call_id")
                if identity not in pending:
                    raise ContextBudgetExceeded("工具结果缺少对应调用，不能发送不完整上下文")
                pending.remove(identity)
            else:
                if pending:
                    raise ContextBudgetExceeded("工具调用缺少结果，不能压缩成不完整轮次")
                for call in msg.get("tool_calls") or []:
                    pending.add(call["id"])
        if pending:
            raise ContextBudgetExceeded("工具调用缺少结果")

    def prepare(self, messages, *, tools=None, tool_choice=None, output_reserve=4096, request=None, include_memory=True):
        request = request or CURRENT_REQUEST.get()
        owner = (getattr(request, "run_id", None) or getattr(request, "request_id", None) or "standalone")
        result = deepcopy(messages)
        self.validate_pairs(result)
        protected = self.protected_facts(result)
        # Keep all actual user instructions, not a keyword-selected subset of prohibitions.
        user_history = list(dict.fromkeys(m["content"] for m in (request.history if request else []) if m.get("role") == "user"))
        pins = {"current_instruction": request.query, "prior_user_instructions": user_history,
                "source_policy": request.source_policy.model_dump(mode="json"),
                "constraints": request.metadata.get("constraints", []), "protected_findings": protected} if request else {"protected_findings": protected}
        if request or protected:
            result.insert(0, {"role": "system", "content": "当前用户指令优先于旧历史与 Memory；SourcePolicy 是硬限制。以下受保护信息不得在压缩后遗忘：\n" + json.dumps(pins, ensure_ascii=False)})
        memory_tokens = 0
        if include_memory and self.memories and request:
            records = self.memories.list(request.session_id)
            if records:
                text = "以下是用户明确保存的偏好/笔记，仅作背景，当前指令优先。Memory 不是 Evidence，不得用于科研主张引用：\n" + json.dumps([r.model_dump(mode="json") for r in records], ensure_ascii=False)
                memory_tokens = self.counter.count(text)
                result.insert(1, {"role": "system", "content": text})
        seen = set()
        deduplicated = []
        for message in result:
            content = message.get("content") or ""
            # Deduplicate text content, never remove tool roles or tool-call IDs.
            if message.get("role") == "tool" and len(content) > 6000:
                message["content"] = self.externalize(content, owner)
            elif message.get("role") == "assistant" and not message.get("tool_calls"):
                if content in seen and content:
                    continue
                seen.add(content)
            deduplicated.append(message)
        result = deduplicated
        reserve = max(output_reserve, int(self.max_tokens * self.reserve_ratio))
        def count():
            # Serialize the complete provider-facing payload, including schemas/choices and role overhead.
            return self.counter.count(json.dumps({"messages": result, "tools": tools, "tool_choice": tool_choice}, ensure_ascii=False)) + 64 + 8 * len(result)
        before = count()
        if before + reserve > self.max_tokens:
            # Compress old observations/assistant explanations into reloadable extracts.
            # User/system text and every tool call/result envelope remain intact.
            for message in result[:-1]:
                if message.get("role") in {"tool", "assistant"} and not message.get("tool_calls") and len(message.get("content") or "") > 600:
                    message["content"] = self.externalize(message["content"], owner, "history", preview_tokens=100)
                    if count() + reserve <= self.max_tokens:
                        break
        used = count()
        self.validate_pairs(result)
        if used + reserve > self.max_tokens:
            raise ContextBudgetExceeded(f"完整请求超出上下文预算：输入 {used} + 输出预留 {reserve} > {self.max_tokens}；请缩小资料范围，未丢弃用户限制或工具配对。")
        return PreparedContext(result, {"input_tokens": used, "before_compression": before,
            "tools_tokens": self.counter.count(json.dumps(tools or [], ensure_ascii=False)),
            "memory_tokens": memory_tokens, "output_reserve": reserve, "total_tokens": used + reserve,
            "max_tokens": self.max_tokens, "tokenizer": "o200k_base estimate + framing margin"})

    def read_evidence(self, context, identities, *, budget_tokens=6000):
        """Select bounded original-text windows, keeping all IDs and disclosing omissions.

        Callers supply task-relevant IDs from retrieval/notes. Allocation does not score truth.
        The final prepare() gate also counts the prompt, constraints, memory and schemas.
        """
        identities = list(dict.fromkeys(identities))
        per_source = max(0, budget_tokens // max(1, len(identities)))
        records = []
        for identity in identities:
            item = context.get(identity)
            selected = self.counter.trim(item.original_text, per_source)
            records.append({"evidence_id": identity, "source_id": item.source_id,
                "document_id": item.document_id, "source_version": item.source_version,
                "parser_version": item.parser_version, "section": item.section,
                "page": item.page, "location": item.location, "original_text": selected,
                "truncated": selected != item.original_text})
        return records

    def reread_evidence(self, context, identity, *, offset=0, limit=6000):
        item = context.get(identity)  # Existing source policy/version authorization remains authoritative.
        start, size = max(0, offset), min(12000, max(1, limit))
        return {"evidence_id": identity, "source_id": item.source_id, "source_version": item.source_version,
                "location": item.location, "offset": start, "original_text": item.original_text[start:start + size],
                "total_chars": len(item.original_text)}
