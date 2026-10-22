"""Read an existing Run through read-only SQLite; never invoke/resume its graph."""
import json
from pathlib import Path
import sqlite3


def read_run(data_dir, run_id):
    path = Path(data_dir) / 'runtime.sqlite3'
    db = sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True)
    try:
        row = db.execute('SELECT snapshot FROM runs WHERE id=?', (run_id,)).fetchone()
        if row is None:
            raise ValueError(f'Run not found: {run_id}')
        snapshot = json.loads(row[0])
        events = [json.loads(row[0]) for row in db.execute('SELECT payload FROM events WHERE run_id=? ORDER BY sequence', (run_id,))]
        tasks, notes = {}, {}
        for event in events:
            data = event['data']
            if event['type'] == 'research_planned':
                tasks.update({t['task_id']: t for t in data.get('tasks', [])})
            if event['type'] == 'research_task_completed':
                n = data.get('note', {})
                if n.get('task_id'):
                    notes[n['task_id']] = n
        # Notes committed before SSE emission are still valid offline evidence.
        results = {key: json.loads(payload) for key, payload in db.execute('SELECT result_key,payload FROM results WHERE run_id=?', (run_id,))}
        for key, note in results.items():
            if key.startswith('note:markdown:') and note.get('task_id'):
                notes[note['task_id']] = note
        tool_calls = [call for key, value in results.items() if key.startswith('react:')
                      for message in value.get('history', []) for call in message.get('tool_calls') or []]
        source_tools = sorted({call['function']['name'] for call in tool_calls
                               if call.get('function', {}).get('name') not in {'skill_view','skills_list'}})
        skills_row = db.execute("SELECT payload FROM results WHERE run_id=? AND result_key='skills:v1'", (run_id,)).fetchone()
        skills = json.loads(skills_row[0]) if skills_row else {}
        calls = [dict(zip(('key','status','payload'), row)) for row in db.execute('SELECT call_key,status,payload FROM calls WHERE run_id=?', (run_id,))]
        evidence_ids = [r[0] for r in db.execute('SELECT evidence_id FROM run_evidence WHERE run_id=?', (run_id,))]
        # EvidenceStore uses content-addressed JSON; direct file reads avoid any writes.
        from sage_scholar.infrastructure.evidence_store import JsonEvidenceStore
        store = JsonEvidenceStore(data_dir)
        evidence = [e.model_dump(mode='json') for identity in evidence_ids if (e := store.get(identity))]
        budget_row = db.execute('SELECT * FROM budget WHERE run_id=?', (run_id,))
        values = budget_row.fetchone()
        budget = dict(zip([c[0] for c in budget_row.description],values)) if values else {}
        stages_cursor = db.execute('SELECT * FROM stage_usage WHERE run_id=?', (run_id,))
        stages = {row['stage']: row for values in stages_cursor.fetchall()
                  if (row := dict(zip([c[0] for c in stages_cursor.description], values)))}
        usage = {name: budget[column] for name, column in (
            ('prompt_tokens','prompt_tokens'), ('completion_tokens','completion_tokens'),
            ('tool_calls','used'), ('active_seconds','active_seconds')) if column in budget}
        return {**usage, 'tasks': list(tasks.values()), 'notes': list(notes.values()), 'evidence': evidence,
                'selected_skills': list(skills), 'skill_versions': {n:s.get('version') for n,s in skills.items()},
                'tool_trace': calls, 'source_tools': source_tools, 'source_tool_trace_complete': bool(tool_calls), 'events': events, 'result': snapshot.get('result'),
                'answer': (snapshot.get('result') or {}).get('report', (snapshot.get('result') or {}).get('answer','')),
                'workflow': snapshot['workflow'], 'status': snapshot['status'],
                'stage_usage': stages, 'source_versions': sorted({(e['document_id'], e['source_version']) for e in evidence if e.get('document_id')})}
    finally:
        db.close()
