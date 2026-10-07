"""Durable extraction scheduler within the application process; not a Research workflow."""
import asyncio
import contextvars
import hashlib
import json
import logging
import threading
from datetime import datetime
from sage_scholar.domain.documents import utc_now
from .models import MemoryCandidate
from .research_source import ResearchMemorySource

logger = logging.getLogger(__name__)


class MemoryBackground:
    def __init__(self, service, retrieval, jobs, checkpoints=None):
        self.service, self.jobs = service, jobs
        self.source = ResearchMemorySource(jobs.db, retrieval, checkpoints)
        self.worker = None
        self.closing = threading.Event()
        self.wakeup = asyncio.Event()

    def _wake(self):
        if self.closing.is_set():
            return
        self.wakeup.set()
        if self.worker is None or self.worker.done():
            self.worker = asyncio.create_task(self._work(), context=contextvars.Context(), name='memory-extraction')

    async def recover(self):
        self.jobs.recover()
        # Covers the crash gap after persisted run_finished but before its completion callback.
        for run in self.jobs.db.finished_research():
            self.enqueue_research(run.run_id, wake=False)
        self._wake()

    def enqueue_research(self, run_id, *, retry=False, wake=True):
        run = self.jobs.db.get(run_id)
        if not run or str(run.workflow) != 'deep_research' or str(run.status) not in {'completed','partial'}:
            raise ValueError('请选择已完成的持久化研究任务')
        job = self.jobs.enqueue('research', run_id, run.execution.workspace_id,
            run.execution.session_id or run.execution.conversation_id, {'run_id': run_id}, run.created_at, retry=retry)
        if wake:
            self._wake()
        return job

    def enqueue_conversation(self, messages, *, workspace_id='default', session_id=None, source_id=None,
                             started_at=None, retry=False):
        source_id = source_id or 'conversation:' + hashlib.sha256(
            json.dumps([workspace_id,session_id,messages],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        job = self.jobs.enqueue('conversation', source_id, workspace_id, session_id,
            {'messages': messages}, started_at or utc_now(), retry=retry)
        self._wake()
        return job

    def conversation(self, request, answer, *, started_at=None):
        try:
            # Small durable write completes before the HTTP response; inference stays background-only.
            return self.enqueue_conversation([{'role':'user','content':request.query},
                {'role':'assistant','content':answer or ''}], workspace_id=request.workspace_id,
                session_id=request.session_id or request.conversation_id, source_id=request.request_id,
                started_at=started_at)
        except Exception:
            logger.exception('Could not persist memory extraction job; main response retained')

    def completed(self, run, events):
        if str(run.workflow) == 'deep_research' and str(run.status) in {'completed','partial'}:
            try:
                self.enqueue_research(run.run_id)
            except Exception:
                logger.exception('Memory enqueue failed; durable completed Run will be reconciled on restart')

    def _extract(self, job):
        if self.closing.is_set():
            raise InterruptedError('Memory worker stopping')
        prepared = job['prepared']
        if prepared is None:
            if job['source_type'] == 'research':
                _, messages, background = self.source.load(job['source_id'])
            else:
                messages, background = job['payload']['messages'], {}
            candidates = self.service.prepare_extraction(messages, workspace_id=job['workspace_id'],
                session_id=job['session_id'], reference=job['source_id'],
                research=job['source_type']=='research', started_at=datetime.fromisoformat(job['source_time']),
                background=background)
            if self.closing.is_set():
                raise InterruptedError('Memory worker stopping')
            prepared = [c.model_dump(mode='json') for c in candidates]
            self.jobs.prepare(job['job_id'], prepared)
        candidates = [MemoryCandidate.model_validate(c) for c in prepared]
        self.service.commit_candidates(candidates, batch_id=job['job_id'], cancelled=self.closing.is_set)

    async def _work(self):
        while not self.closing.is_set():
            job = self.jobs.claim()
            if job:
                try:
                    # Shield lets shutdown know whether the thread really finished before closing SQLite.
                    execution = asyncio.create_task(asyncio.to_thread(self._extract, job))
                    await asyncio.shield(execution)
                    self.jobs.succeed(job['job_id'])
                except InterruptedError:
                    self.jobs.release(job['job_id'])
                except asyncio.CancelledError:
                    self.closing.set()
                    try:
                        await asyncio.shield(execution)
                    except InterruptedError:
                        pass
                    finally:
                        self.jobs.release(job['job_id'])
                    raise
                except Exception as exc:
                    logger.exception('Memory extraction failed; persisted job may retry')
                    self.jobs.fail(job['job_id'], f'{type(exc).__name__}: {exc}')
                continue
            self.wakeup.clear()
            delay = self.jobs.next_delay()
            if delay is None:
                return
            try:
                await asyncio.wait_for(self.wakeup.wait(), timeout=max(.05,delay))
            except TimeoutError:
                pass

    async def drain(self):
        """Wait for scheduled work (including bounded retries); useful to management/tests."""
        if self.worker:
            await asyncio.shield(self.worker)

    async def close(self):
        self.closing.set()
        self.wakeup.set()
        if self.worker and not self.worker.done():
            # Never close shared SQLite underneath an active thread. Gateway enforces provider timeout.
            await asyncio.shield(self.worker)
