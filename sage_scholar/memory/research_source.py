"""One durable research input reader for automatic extraction and manual re-extraction."""
from pydantic import BaseModel
from .conversation_source import conversation_messages
from sage_scholar.domain.research import ResearchContract
from sage_scholar.research.evidence_context import ResearchEvidenceContext


def plain(value):
    if isinstance(value, BaseModel):
        return value.model_dump(mode='json')
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


class ResearchMemorySource:
    def __init__(self, run_store, retrieval, checkpoints=None):
        self.store, self.retrieval, self.checkpoints = run_store, retrieval, checkpoints

    def load(self, run_id):
        run = self.store.get(run_id)
        if not run or str(run.workflow) != 'deep_research' or str(run.status) not in {'completed', 'partial'}:
            raise ValueError('请选择已完成或 PARTIAL 的持久化研究任务')
        values = {}
        if self.checkpoints and self.checkpoints.compatible(run):
            checkpoint = self.checkpoints.saver.get_tuple(self.checkpoints.config(run_id))
            values = plain(checkpoint.checkpoint.get('channel_values', {})) if checkpoint else {}
        # Events are read from SQLite, never from the live RunManager's in-memory list.
        notes, review = {}, {}
        for event in self.store.events_for(run_id):
            if event.type == 'research_task_completed':
                note = event.data.get('note', {})
                notes[note.get('task_id', note.get('note_id'))] = note
            elif event.type == 'gap_reviewed':
                review = event.data
        if 'research_notes' in values:
            notes = {key: note for key, note in values['research_notes'].items() if note is not None}
        review = values.get('gap_review') or review
        contract = values.get('research_contract') or plain(ResearchContract.from_request(run.execution))
        if isinstance(contract.get('execution'), dict):
            contract['execution'].pop('history', None)
        # The source request is authority for preferences; research material is only a lead.
        messages = conversation_messages(run.execution)
        access = ResearchEvidenceContext(self.retrieval.evidence_store, run.execution.source_policy, self.retrieval.documents)
        findings = []
        for note in notes.values():
            for finding in note.get('findings', []):
                findings.append((finding, {'research_note_id': note.get('note_id'), 'task_id': note.get('task_id')}))
        for claim in (run.result or {}).get('claims', []):
            findings.append((claim, {'claim_id': claim.get('claim_id')}))
        seen = set()
        for finding, provenance in findings:
            text = finding.get('text', '')
            identity = (text, tuple(provenance.items()))
            if not text or identity in seen:
                continue
            ids = []
            for eid in finding.get('evidence_ids', []):
                try:
                    item = self.retrieval.evidence_store.get(eid)
                    if item is None:
                        continue
                    access.register(item)
                    access.get(eid)
                    ids.append(eid)
                except (ValueError, KeyError):
                    continue
            if ids:
                messages.append({'role': 'assistant', 'content': text, 'evidence_ids': list(dict.fromkeys(ids)), **provenance})
                seen.add(identity)
        background = {'contract': contract, 'gap_review': review, 'remaining_gaps': run.remaining_gaps,
            'research_notes': [{k: n.get(k) for k in ('note_id','task_id','question','unknowns','conflicts','limitations')}
                               for n in notes.values()]}
        return run, messages, background
