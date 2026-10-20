"""Read the frozen request history and its original report maps, never another conversation."""
import json
from .base_tool import BaseTool, ToolParameter


class ConversationReadTool(BaseTool):
    source_access = "none"
    replay_policy = "safe_retry"

    def __init__(self, request, store=None, evidence_context=None, record=lambda item: None):
        super().__init__("read_conversation", "回读本次请求之前的原始对话/报告及引用映射。无 index 时列目录；历史回答不是 Evidence，核验必须读取原文。", [
            ToolParameter(name="index", type="integer", description="目录中的消息索引；省略列出目录", required=False),
            ToolParameter(name="offset", type="integer", description="正文字符起点", required=False),
            ToolParameter(name="limit", type="integer", description="正文字符数，默认 4000，最多 12000", required=False),
        ])
        self.request, self.store, self.context, self.record = request, store, evidence_context, record

    def run_tool(self, parameters):
        history = self.request.history
        index = parameters.get("index")
        if index is None:
            return json.dumps({"messages": [{"index": i, "message_id": m.get("id"), "role": m["role"],
                "preview": m.get("content", "")[:180], "chars": len(m.get("content", ""))}
                for i, m in enumerate(history)], "is_evidence": False}, ensure_ascii=False)
        if not isinstance(index, int) or not 0 <= index < len(history):
            raise ValueError("对话消息索引不存在")
        message = history[index]
        start = max(0, int(parameters.get("offset", 0)))
        size = max(1, min(12000, int(parameters.get("limit", 4000))))
        text = message.get("content", "")
        result = {"index": index, "message_id": message.get("id"), "role": message["role"],
                  "content": text[start:start + size], "offset": start, "total_chars": len(text), "is_evidence": False}
        if self.store and self.request.conversation_id:
            conversation = self.store.conversations.get(self.request.conversation_id)
            if not conversation or conversation.workspace_id != self.request.workspace_id:
                raise ValueError("对话不存在或不在当前工作区")
            saved = next((m for m in self.store.conversations.messages(conversation.conversation_id)
                          if m.message_id == message.get("id") and m.role == "assistant"), None)
            run = self.store.get(saved.run_id) if saved else None
            if run and run.execution.conversation_id == conversation.conversation_id and run.execution.workspace_id == self.request.workspace_id:
                data = run.result or {}
                result["citations"] = data.get("citations", {})
                result["citation_evidence"] = data.get("citation_evidence", {})
                # Import only authorized historical source records, not the report itself.
                identities = set(data.get("evidence_ids", []))
                for value in result["citations"].values():
                    if isinstance(value, str):
                        identities.add(value)
                for values in result["citation_evidence"].values():
                    if isinstance(values, list):
                        identities.update(v for v in values if isinstance(v, str))
                readable = []
                if self.context:
                    for identity in sorted(identities):
                        item = self.context.store.get(identity)
                        if item:
                            self.context.register(item)
                            try:
                                self.context.get(identity)  # current SourcePolicy + version validation
                            except ValueError:
                                continue
                            self.record(item)
                            readable.append(identity)
                result["readable_evidence_ids"] = readable
                result["notice"] = "引用映射只说明报告原来引用了什么；可读原文须用 read_context_artifact；来源受当前 SourcePolicy 约束。"
        return json.dumps(result, ensure_ascii=False)
