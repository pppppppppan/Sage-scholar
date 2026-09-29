from __future__ import annotations

import asyncio
import threading

from sage_scholar.base import LLMClient
from sage_scholar.config import Config


class LegacyModelGateway:
    """Lazy adapter around the proven SAGE Research model client."""

    def __init__(self, config: Config, context_manager=None):
        self._config = config
        self.context_manager = context_manager
        self._client: LLMClient | None = None
        self._lock = threading.Lock()

    @property
    def client(self) -> LLMClient:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = LLMClient(
                        model=self._config.llm.model,
                        timeout=self._config.llm.timeout,
                        context_manager=self.context_manager,
                    )
        return self._client

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0,
        tag: str = "",
    ) -> str:
        response = await asyncio.to_thread(
            self.client.invoke,
            messages,
            temperature,
            None,
            None,
            4096,
            tag,
        )
        return response.content or ""
