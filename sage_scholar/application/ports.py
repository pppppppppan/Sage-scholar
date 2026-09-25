from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from sage_scholar.domain import EvidenceRef
from sage_scholar.domain.execution import ExecutionRequest, SourcePolicy, RunOutcome


class ModelGateway(Protocol):
    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0,
        tag: str = "",
    ) -> str: ...


class RetrievalService(Protocol):
    async def retrieve(
        self,
        query: str,
        *,
        policy: SourcePolicy,
        top_k: int = 5,
    ) -> list[EvidenceRef]: ...

    async def read_documents(self, *, policy: SourcePolicy) -> list[EvidenceRef]: ...


EventEmitter = Callable[[str, dict], Awaitable[None]]


class LongTaskRunner(Protocol):
    async def run(self, request: ExecutionRequest, emit: EventEmitter, cancelled: Callable[[], bool]) -> RunOutcome: ...
