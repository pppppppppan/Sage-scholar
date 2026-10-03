"""Best-effort post-response extraction; no workflow state or distributed worker."""
import asyncio
import contextvars
import logging
from dataclasses import dataclass
from sage_scholar.domain.documents import utc_now
from sage_scholar.research.evidence_context import ResearchEvidenceContext

logger = logging.getLogger(__name__)


def research_messages(run, events, retrieval):
    """Only server-owned findings/claims with readable Evidence can ground a research lead."""
    messages = [{"role": "user", "content": run.execution.query}]
    context = ResearchEvidenceContext(retrieval.evidence_store, run.execution.source_policy, retrieval.documents)
    findings = []
    latest = {}
    for event in events:
        if event.type == "research_task_completed":
            note = event.data.get("note", {})
            latest[note.get("task_id", note.get("note_id"))] = note
    for note in latest.values():
        findings.extend(note.get("findings", []))
    findings.extend((run.result or {}).get("claims", []))
    seen, size = set(), len(run.execution.query)
    for finding in findings:
        text = finding.get("text", "")
        if not text or text in seen or size + len(text) > 80000:
            continue
        ids = []
        for identity in finding.get("evidence_ids", []):
            try:
                item = retrieval.evidence_store.get(identity)
                if item is None:
                    continue
                context.register(item)
                context.get(identity)
                ids.append(identity)
            except (ValueError, KeyError):
                pass
        if not ids:
            continue
        seen.add(text)
        size += len(text)
        messages.append({"role": "assistant", "content": text, "evidence_ids": list(dict.fromkeys(ids))})
        if len(messages) >= 100:
            break
    return messages


@dataclass
class MemoryJob:
    request: object
    messages: list | None
    started_at: object
    run: object = None
    events: tuple = ()


class MemoryBackground:
    def __init__(self, service, retrieval):
        self.service, self.retrieval = service, retrieval
        self.queue = asyncio.Queue(maxsize=32)
        self.worker = None
        self.closing = False

    def _enqueue(self, job):
        if self.closing:
            return
        try:
            self.queue.put_nowait(job)
            if self.worker is None or self.worker.done():
                # Extraction must not inherit the finished Run's budget/context or recurse into memory recall.
                self.worker = asyncio.create_task(self._work(), context=contextvars.Context(), name="memory-extraction")
        except (asyncio.QueueFull, RuntimeError):
            logger.warning("Memory extraction queue unavailable; main response is unaffected")

    def conversation(self, request, answer, *, started_at=None):
        # Process the new turn only: replaying old history could undo a newer long-term decision.
        self._enqueue(MemoryJob(request, [{"role": "user", "content": request.query[:16000]},
            {"role": "assistant", "content": (answer or "")[:16000]}], started_at or utc_now()))

    def completed(self, run, events):
        if str(run.workflow) == "deep_research" and str(run.status) in {"completed", "partial"}:
            self._enqueue(MemoryJob(run.execution, None, run.created_at, run, tuple(events)))

    def _extract(self, job):
        messages = job.messages if job.messages is not None else research_messages(job.run, job.events, self.retrieval)
        request = job.request
        self.service.extract(messages, workspace_id=request.workspace_id,
            session_id=request.session_id or request.conversation_id,
            reference=request.run_id or request.request_id, research=job.run is not None,
            started_at=job.started_at, cancelled=lambda: self.closing)

    async def _work(self):
        while not self.queue.empty() and not self.closing:
            job = self.queue.get_nowait()
            try:
                await asyncio.to_thread(self._extract, job)
            except Exception:
                logger.exception("Automatic memory extraction failed; main response is unaffected")
            finally:
                self.queue.task_done()

    async def close(self):
        self.closing = True
        while not self.queue.empty():
            self.queue.get_nowait()
            self.queue.task_done()
        if self.worker and not self.worker.done():
            # SDK requests have their own timeout. A late thread cannot commit after closing.
            _, pending = await asyncio.wait([self.worker], timeout=2)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
