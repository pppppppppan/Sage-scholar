"""Short workflows use the same cancellable lifecycle as research."""
import re
from sage_scholar.domain.execution import RunOutcome
from sage_scholar.routing import Workflow
from sage_scholar.runtime.control import cooperative_thread, check_control
from sage_scholar.memory import MemoryInput


class InlineRunner:
    def __init__(self, direct, qa, memory=None):
        self.direct, self.qa, self.memory = direct, qa, memory

    async def run(self, request, emit, cancelled):
        check_control()
        explicit = re.fullmatch(r"(?:请)?记住[：:]\s*(.+)", request.query.strip(), re.S)
        if request.metadata.get("clarification_message"):
            result = {"answer": request.metadata["clarification_message"]}
        elif explicit and self.memory:
            if hasattr(self.memory, "remember"):
                memory = await cooperative_thread(self.memory.remember, explicit[1],
                    workspace_id=request.workspace_id, session_id=request.session_id or request.conversation_id)
            else:
                memory = await cooperative_thread(self.memory.add, MemoryInput(content=explicit[1]))
            result = {"answer": f"已记住：{memory.content}。可在记忆管理中修改或删除。"}
        elif request.workflow == Workflow.QA:
            answer = await self.qa.answer(request)
            result = {"answer": answer.answer, "evidence": [e.model_dump(mode="json") for e in answer.evidence],
                      "citations": answer.citations, "citation_issues": answer.citation_issues,
                      "reading_strategy": answer.reading_strategy}
        else:
            result = {"answer": await self.direct.answer(request)}
        check_control()
        await emit("answer", result)
        return RunOutcome(result=result)
