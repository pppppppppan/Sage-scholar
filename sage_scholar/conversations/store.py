"""Conversation and message records share RunStore's SQLite transaction boundary."""
import json
from uuid import uuid4
from sage_scholar.runtime.models import utc_now
from .models import Conversation, ConversationMessage

ACTIVE = {'pending', 'running', 'cancel_requested', 'waiting_for_user'}


class ConversationStore:
    def __init__(self, store):
        self.store = store
        with store.transaction() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL,
                    title TEXT NOT NULL, archived INTEGER NOT NULL DEFAULT 0, deleted_at TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS conversation_messages (id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL,
                    run_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, kind TEXT NOT NULL,
                    created_at TEXT NOT NULL, data TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS conversation_messages_owner ON conversation_messages(conversation_id,created_at);
                CREATE TABLE IF NOT EXISTS conversation_requests (conversation_id TEXT, request_id TEXT,
                    run_id TEXT NOT NULL, fingerprint TEXT NOT NULL, PRIMARY KEY(conversation_id,request_id));
            ''')

    def create(self, workspace_id='default', title='新对话', identity=None):
        identity, now = identity or uuid4().hex, utc_now().isoformat()
        with self.store.transaction() as db:
            db.execute('INSERT INTO conversations VALUES (?,?,?,0,NULL,?,?)',
                       (identity, workspace_id, title.strip() or '新对话', now, now))
        return self.get(identity)

    def get(self, identity, *, include_deleted=False):
        with self.store.lock:
            row = self.store.conn.execute('SELECT * FROM conversations WHERE id=?', (identity,)).fetchone()
            if not row or (row[4] and not include_deleted):
                return None
            values = dict(zip(('conversation_id','workspace_id','title','archived','deleted_at','created_at','updated_at'), row))
            runs = self.runs(identity)
            active = next((r for r in reversed(runs) if str(r.status) in ACTIVE), None)
            values['status'] = str(active.status if active else runs[-1].status if runs else 'completed')
            return Conversation.model_validate(values)

    def list(self, workspace_id='default', search='', archived=False):
        with self.store.lock:
            ids = [r[0] for r in self.store.conn.execute('''SELECT id FROM conversations
                WHERE workspace_id=? AND archived=? AND deleted_at IS NULL ORDER BY updated_at DESC''', (workspace_id, archived))]
            result = [self.get(identity) for identity in ids]
            text = search.strip().casefold()
            return [c for c in result if not text or text in c.title.casefold()
                    or any(text in m.content.casefold() for m in self.messages(c.conversation_id))]

    def update(self, identity, *, title=None, archived=None):
        with self.store.transaction() as db:
            if not self.get(identity):
                raise KeyError(identity)
            if title is not None:
                if not title.strip():
                    raise ValueError('标题不能为空')
                db.execute('UPDATE conversations SET title=? WHERE id=?', (title.strip()[:200], identity))
            if archived is not None:
                db.execute('UPDATE conversations SET archived=? WHERE id=?', (archived, identity))
            db.execute('UPDATE conversations SET updated_at=? WHERE id=?', (utc_now().isoformat(), identity))
        return self.get(identity)

    def runs(self, identity):
        from sage_scholar.runtime.models import RunSnapshot
        with self.store.lock:
            return [run for row in self.store.conn.execute('SELECT snapshot FROM runs ORDER BY rowid')
                    if (run := RunSnapshot.model_validate_json(row[0])).execution.conversation_id == identity]

    def messages(self, identity):
        with self.store.lock:
            return [ConversationMessage.model_validate(dict(zip(
                ('message_id','conversation_id','run_id','role','content','kind','created_at','data'),
                (*row[:-1], json.loads(row[-1]))))) for row in self.store.conn.execute(
                    'SELECT * FROM conversation_messages WHERE conversation_id=? ORDER BY created_at,rowid', (identity,))]

    def history(self, identity):
        return [{'role': m.role, 'content': m.content, 'id': m.message_id}
                for m in self.messages(identity) if m.content and m.kind != 'status']

    def assert_open(self, identity):
        conversation = self.get(identity)
        if not conversation:
            raise KeyError(identity)
        if conversation.archived:
            raise ValueError('请先取消归档再继续对话')
        return conversation

    def mark_deleted(self, identity):
        with self.store.transaction() as db:
            if not self.get(identity, include_deleted=True):
                raise KeyError(identity)
            db.execute('UPDATE conversations SET deleted_at=COALESCE(deleted_at,?),title=? WHERE id=?',
                       (utc_now().isoformat(), '已删除对话', identity))

    def source_deleted(self, session_id):
        with self.store.lock:
            return bool(session_id and self.store.conn.execute(
                'SELECT 1 FROM conversations WHERE id=? AND deleted_at IS NOT NULL', (session_id,)).fetchone())

    def purge_messages(self, identity):
        with self.store.transaction() as db:
            db.execute('DELETE FROM conversation_messages WHERE conversation_id=?', (identity,))
            db.execute('DELETE FROM conversation_requests WHERE conversation_id=?', (identity,))

    def request(self, identity, request_id):
        with self.store.lock:
            return self.store.conn.execute('SELECT run_id,fingerprint FROM conversation_requests WHERE conversation_id=? AND request_id=?', (identity, request_id)).fetchone() if request_id else None

    def record_run(self, db, run, event):
        identity = run.execution.conversation_id
        if not identity or not self.get(identity):
            return
        now = event.created_at.isoformat()
        if event.type == 'run_created':
            if run.execution.request_id:
                db.execute('INSERT OR IGNORE INTO conversation_requests VALUES (?,?,?,?)',
                    (identity, run.execution.request_id, run.run_id, run.execution.metadata.get('_submission_fingerprint', '')))
            db.execute('UPDATE conversations SET title=CASE WHEN title=? THEN ? ELSE title END, updated_at=? WHERE id=?',
                       ('新对话', run.query.strip().replace('\n', ' ')[:60] or '新对话', now, identity))
            for role, content in [('user', run.query), ('assistant', '')]:
                db.execute('INSERT OR IGNORE INTO conversation_messages VALUES (?,?,?,?,?,?,?,?)',
                    (run.run_id + ':' + role, identity, run.run_id, role, content, 'message', now, '{}'))
        if event.type == 'human_responded':
            data = event.data
            db.execute('INSERT OR IGNORE INTO conversation_messages VALUES (?,?,?,?,?,?,?,?)',
                (run.run_id + ':' + data['interrupt_id'], identity, run.run_id, 'user',
                 data.get('answer') or data.get('choice') or '跳过此问题', 'interaction', now, json.dumps(data, ensure_ascii=False)))
        result = run.result or {}
        content = result.get('answer') or result.get('report') or ''
        db.execute('UPDATE conversation_messages SET content=?,data=? WHERE id=?',
                   (content, json.dumps({'status': str(run.status), 'evidence': result.get('evidence', []),
                    'reason': run.reason, 'error': run.error}, ensure_ascii=False), run.run_id + ':assistant'))
        db.execute('UPDATE conversations SET updated_at=? WHERE id=?', (now, identity))

    def adopt_existing_runs(self):
        """Idempotently project legacy durable Runs into the conversation history."""
        from sage_scholar.runtime.models import RunSnapshot, RunEvent
        with self.store.transaction() as db:
            rows = db.execute('SELECT snapshot FROM runs ORDER BY rowid').fetchall()
            for row in rows:
                run = RunSnapshot.model_validate_json(row[0])
                owner = run.execution.conversation_id or run.execution.session_id or 'legacy-' + run.run_id
                if self.source_deleted(owner):
                    continue
                current = self.get(owner)
                if current and current.workspace_id != run.execution.workspace_id:
                    owner = 'legacy-' + run.run_id
                    current = self.get(owner)
                if not current:
                    self.create(run.execution.workspace_id, identity=owner)
                if run.execution.conversation_id != owner:
                    run.execution = run.execution.model_copy(update={'conversation_id': owner})
                    db.execute('UPDATE runs SET snapshot=? WHERE id=?', (run.model_dump_json(), run.run_id))
                if not db.execute('SELECT 1 FROM conversation_messages WHERE id=?', (run.run_id + ':user',)).fetchone():
                    self.record_run(db, run, RunEvent(sequence=0, type='run_created', created_at=run.created_at))
                    self.record_run(db, run, RunEvent(sequence=0, type='migration', created_at=run.updated_at))
