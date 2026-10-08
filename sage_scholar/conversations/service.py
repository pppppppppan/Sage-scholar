"""Conversation ownership, submission deduplication and safe deletion."""
import asyncio
import hashlib
import json
from dataclasses import asdict
from sage_scholar.application.query_service import QueryResult, QueryStatus
from sage_scholar.routing import RouteDecision
from .store import ACTIVE


class ConversationService:
    def __init__(self, store, runs, query, memory_background):
        self.store, self.runs, self.query, self.memory = store, runs, query, memory_background
        self._locks = {}

    def lock(self, identity):
        return self._locks.setdefault(identity, asyncio.Lock())

    async def submit(self, command):
        if not command.conversation_id:
            command.conversation_id = self.store.create(command.workspace_id).conversation_id
        identity = command.conversation_id
        async with self.lock(identity):
            conversation = self.store.assert_open(identity)
            if conversation.workspace_id != command.workspace_id:
                raise ValueError("对话不属于当前工作区")
            fingerprint = hashlib.sha256(json.dumps({k:v for k,v in asdict(command).items()
                if k not in {"history", "session_id", "request_id"}}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            previous = self.store.request(identity, command.request_id)
            if previous:
                if previous[1] != fingerprint:
                    raise ValueError("同一 request_id 不可提交不同请求")
                run = self.runs.get(previous[0])
                return QueryResult(request_id=command.request_id, status=QueryStatus.ACCEPTED, run_id=run.run_id,
                    route=RouteDecision(reason_code="idempotent_replay", **run.execution.model_dump(include={"workflow","source_scope","document_ids","allow_web","execution_mode"})))
            if any(str(r.status) in ACTIVE for r in self.store.runs(identity)):
                raise ValueError("当前对话仍有任务运行或等待回答，请先停止或回答该任务")
            command.history = self.store.history(identity)
            command.session_id = identity
            command.metadata = {**{k:v for k,v in command.metadata.items() if not k.startswith("_")}, "_submission_fingerprint": fingerprint}
            return await self.query.execute(command)

    async def delete(self, identity):
        async with self.lock(identity):
            self.store.mark_deleted(identity)  # durable barrier before awaiting cancellation
            if self.memory:
                await self.memory.cancel_conversation(identity)
            for run in self.store.runs(identity):
                await self.runs.remove(run.run_id)
            self.store.purge_messages(identity)

    def detail(self, identity):
        conversation = self.store.get(identity)
        if not conversation:
            raise KeyError(identity)
        return {"conversation": conversation, "messages": self.store.messages(identity),
                "runs": [self.runs.get(run.run_id) or run for run in self.store.runs(identity)]}
