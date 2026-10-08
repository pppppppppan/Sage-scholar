"""Single-process SQLite extraction journal, using the existing RunStore transaction boundary."""
import hashlib
import json
import time
from sage_scholar.domain.documents import utc_now


class MemoryJobStore:
    MAX_ATTEMPTS = 3

    def __init__(self, run_store):
        self.db = run_store
        with self.db.transaction() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS memory_jobs (
                job_id TEXT PRIMARY KEY, source_key TEXT UNIQUE NOT NULL,
                source_type TEXT NOT NULL, source_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL, session_id TEXT,
                status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL, source_time TEXT NOT NULL,
                payload TEXT NOT NULL, prepared TEXT, error TEXT, available_at REAL NOT NULL DEFAULT 0)''')

    @staticmethod
    def _decode(row):
        if row is None:
            return None
        fields = ('job_id source_key source_type source_id workspace_id session_id status attempts '
                  'created_at updated_at source_time payload prepared error available_at').split()
        data = dict(zip(fields, row))
        data['payload'] = json.loads(data['payload'])
        data['prepared'] = json.loads(data['prepared']) if data['prepared'] is not None else None
        return data

    def enqueue(self, source_type, source_id, workspace_id, session_id, payload, source_time, *, retry=False):
        if self.db.conversations.source_deleted(session_id):
            raise ValueError("已删除对话不能重新抽取记忆")
        key = json.dumps([source_type, source_id, workspace_id, session_id], ensure_ascii=False)
        job_id = 'mj_' + hashlib.sha256(key.encode()).hexdigest()
        now = utc_now().isoformat()
        with self.db.transaction() as db:
            db.execute('''INSERT OR IGNORE INTO memory_jobs
                (job_id,source_key,source_type,source_id,workspace_id,session_id,status,
                 created_at,updated_at,source_time,payload) VALUES (?,?,?,?,?,?,'pending',?,?,?,?)''',
                (job_id,key,source_type,source_id,workspace_id,session_id,now,now,
                 source_time.isoformat(),json.dumps(payload,ensure_ascii=False)))
            if retry:
                # A manual repeat uses the same source/job. Commit receipts still prevent resurrection.
                db.execute("UPDATE memory_jobs SET status='pending', attempts=0, available_at=0, error=NULL, prepared=NULL, updated_at=? WHERE job_id=? AND status IN ('failed','succeeded')", (now,job_id))
            return self._decode(db.execute('SELECT * FROM memory_jobs WHERE job_id=?', (job_id,)).fetchone())

    def get(self, job_id):
        with self.db.lock:
            return self._decode(self.db.conn.execute('SELECT * FROM memory_jobs WHERE job_id=?', (job_id,)).fetchone())

    def list(self, workspace_id='default', session_id=None, limit=30):
        with self.db.lock:
            rows = self.db.conn.execute('''SELECT * FROM memory_jobs WHERE workspace_id=?
                AND (? IS NULL OR session_id=?) ORDER BY created_at DESC LIMIT ?''',
                (workspace_id,session_id,session_id,limit)).fetchall()
            return [self._decode(row) for row in rows]

    def cancel_conversation(self, identity):
        with self.db.transaction() as db:
            db.execute("UPDATE memory_jobs SET status='cancelled',payload='{}',prepared=NULL,error=NULL WHERE session_id=?", (identity,))

    def recover(self):
        with self.db.transaction() as db:
            db.execute("UPDATE memory_jobs SET status='cancelled',payload='{}',prepared=NULL WHERE session_id IN (SELECT id FROM conversations WHERE deleted_at IS NOT NULL)")
            db.execute("UPDATE memory_jobs SET status='pending', available_at=0, updated_at=? WHERE status='running'", (utc_now().isoformat(),))

    def claim(self):
        with self.db.transaction() as db:
            row = db.execute("SELECT * FROM memory_jobs WHERE status IN ('pending','retryable') AND available_at<=? ORDER BY created_at LIMIT 1", (time.time(),)).fetchone()
            if not row:
                return None
            db.execute("UPDATE memory_jobs SET status='running', attempts=attempts+1, updated_at=? WHERE job_id=?", (utc_now().isoformat(),row[0]))
            return self._decode(db.execute('SELECT * FROM memory_jobs WHERE job_id=?', (row[0],)).fetchone())

    def next_delay(self):
        with self.db.lock:
            value = self.db.conn.execute("SELECT MIN(available_at) FROM memory_jobs WHERE status IN ('pending','retryable')").fetchone()[0]
            return None if value is None else max(0, value - time.time())

    def prepare(self, identity, candidates):
        with self.db.transaction() as db:
            db.execute("UPDATE memory_jobs SET prepared=?,updated_at=? WHERE job_id=? AND prepared IS NULL AND status='running'",
                (json.dumps(candidates,ensure_ascii=False),utc_now().isoformat(),identity))

    def succeed(self, identity):
        with self.db.transaction() as db:
            db.execute("UPDATE memory_jobs SET status='succeeded',error=NULL,updated_at=? WHERE job_id=? AND status='running'", (utc_now().isoformat(),identity))

    def fail(self, identity, error):
        with self.db.transaction() as db:
            if self.get(identity)['status'] == 'cancelled':
                return
            attempts = db.execute('SELECT attempts FROM memory_jobs WHERE job_id=?', (identity,)).fetchone()[0]
            status = 'failed' if attempts >= self.MAX_ATTEMPTS else 'retryable'
            db.execute('UPDATE memory_jobs SET status=?,error=?,available_at=?,updated_at=? WHERE job_id=?',
                (status,error[:1000],time.time()+min(30,2**attempts),utc_now().isoformat(),identity))

    def release(self, identity):
        with self.db.transaction() as db:
            db.execute("UPDATE memory_jobs SET status='pending',available_at=0,updated_at=? WHERE job_id=? AND status='running'", (utc_now().isoformat(),identity))
