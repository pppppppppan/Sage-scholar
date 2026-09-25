from __future__ import annotations

from dataclasses import dataclass

from sage_scholar.domain import EvidenceRef

from .ports import ModelGateway, RetrievalService


@dataclass(slots=True)
class QAResult:
    answer: str
    evidence: list[EvidenceRef]


class QAService:
    def __init__(self, retrieval: RetrievalService, model: ModelGateway):
        self._retrieval = retrieval
        self._model = model

    async def answer(self, query: str, document_ids: list[str]) -> QAResult:
        evidence = await self._retrieval.retrieve(query, document_ids=document_ids)
        if not evidence:
            return QAResult(
                answer="当前文献库中没有找到足以回答该问题的证据。请上传相关论文，或调整资料范围。",
                evidence=[],
            )

        context = "\n\n".join(
            f"[{index}] source={item.source_id}\n{item.text}"
            for index, item in enumerate(evidence, 1)
        )
        answer = await self._model.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "只依据给定论文证据回答。使用 [1] 形式引用证据。"
                        "证据不足时明确说不知道，不得用外部记忆补全。"
                    ),
                },
                {"role": "user", "content": f"问题：{query}\n\n证据：\n{context}"},
            ],
            tag="qa",
        )
        return QAResult(answer=answer, evidence=evidence)
