from __future__ import annotations

from sage_scholar.runtime.control import cooperative_thread
from collections.abc import Callable
from datetime import datetime
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .ports import ModelGateway
from sage_scholar.domain.execution import ExecutionRequest


class DirectService:
    def __init__(self, model: ModelGateway, clock: Callable[[ZoneInfo], datetime] | None = None, *, context_manager=None):
        self._model = model
        self._context_manager = context_manager
        self._clock = clock or (lambda timezone: datetime.now(timezone))

    async def answer(
        self,
        request: ExecutionRequest,
    ) -> str:
        query, timezone, history = request.query, request.timezone, request.history
        if re.search(r"今天.{0,3}(星期几|周几)|今天是星期|今天是周", query):
            try:
                zone = ZoneInfo(timezone)
            except ZoneInfoNotFoundError:
                zone = ZoneInfo("Asia/Shanghai")
            now = self._clock(zone)
            weekdays = "一二三四五六日"
            return f"今天是{now:%Y年%m月%d日}，星期{weekdays[now.weekday()]}。"

        messages = [
            {
                "role": "system",
                "content": (
                    "你是 SAGE Scholar 的简洁问答助手。直接回答当前问题。"
                    "不要假装做过检索；需要最新外部资料时明确建议用户使用搜索或深度调研。"
                ),
            }
        ]
        if self._context_manager:
            bundle = await cooperative_thread(self._context_manager.conversation, request,
                summarize=getattr(self._model, "summarize_history", None))
            messages.extend(bundle.messages())
        else:
            # Compatibility with independently constructed gateways; provider boundary still budgets.
            for message in (history or []):
                role = message.get("role")
                content = message.get("content", "").strip()
                if role in {"user", "assistant"} and content:
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": query})

        return await self._model.complete(
            messages,
            tag="direct",
        )
