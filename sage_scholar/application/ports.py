from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from sage_scholar.domain import EvidenceRef


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
        document_ids: list[str],
        top_k: int = 5,
    ) -> list[EvidenceRef]: ...


EventEmitter = Callable[[str, dict], Awaitable[None]]


class LongTaskRunner(Protocol):
    async def run(self, query: str, emit: EventEmitter, cancelled: Callable[[], bool]) -> dict: ...
