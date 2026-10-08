"""SQLite lifecycle/event storage and durable execution boundaries, not workflow state."""
import json
import sqlite3
import threading
from pathlib import Path
from contextlib import contextmanager
from .models import RunSnapshot, RunEvent


class RunStore:
    def __init__(self, data_dir):
        path = Path(data_dir) / "runtime.sqlite3"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.lock = threading.RLock()
        self.conn.executescript('''
        CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, snapshot TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events (run_id TEXT, sequence INTEGER, payload TEXT NOT NULL, PRIMARY KEY(run_id,sequence));
        CREATE TABLE IF NOT EXISTS budget (run_id TEXT PRIMARY KEY, used INTEGER DEFAULT 0, denied INTEGER DEFAULT 0, blocked INTEGER DEFAULT 0, prompt_tokens INTEGER DEFAULT 0, completion_tokens INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS calls (run_id TEXT, call_key TEXT, status TEXT NOT NULL, payload TEXT, PRIMARY KEY(run_id,call_key));
        CREATE TABLE IF NOT EXISTS results (run_id TEXT, result_key TEXT, payload TEXT NOT NULL, PRIMARY KEY(run_id,result_key));
        CREATE TABLE IF NOT EXISTS run_evidence (run_id TEXT, evidence_id TEXT, PRIMARY KEY(run_id,evidence_id));
        ''')
        self.conn.commit()
        columns = {r[1] for r in self.conn.execute('PRAGMA table_info(budget)')}
        if 'active_seconds' not in columns:
            self.conn.execute('ALTER TABLE budget ADD COLUMN active_seconds REAL NOT NULL DEFAULT 0')
        for column in ('cached_input_tokens', 'cache_observed_input_tokens', 'cache_observed_calls'):
            if column not in columns:
                self.conn.execute(f'ALTER TABLE budget ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0')
        self.conn.execute('CREATE TABLE IF NOT EXISTS model_reservations (run_id TEXT, lease TEXT PRIMARY KEY, tokens INTEGER)')
        self.conn.commit()
        from sage_scholar.conversations.store import ConversationStore
        self.conversations = ConversationStore(self)

    @contextmanager
    def transaction(self):
        with self.lock, self.conn:
            yield self.conn

    def load(self):
        with self.lock:
            return [(RunSnapshot.model_validate_json(row[0]), [RunEvent.model_validate_json(e[0]) for e in self.conn.execute(
                "SELECT payload FROM events WHERE run_id=? ORDER BY sequence", (row[1],))])
                for row in self.conn.execute("SELECT snapshot,id FROM runs").fetchall()]

    def get(self, run_id):
        with self.lock:
            row = self.conn.execute("SELECT snapshot FROM runs WHERE id=?", (run_id,)).fetchone()
            return RunSnapshot.model_validate_json(row[0]) if row else None

    def events_for(self, run_id):
        with self.lock:
            return [RunEvent.model_validate_json(row[0]) for row in self.conn.execute(
                "SELECT payload FROM events WHERE run_id=? ORDER BY sequence", (run_id,))]

    def finished_research(self):
        with self.lock:
            return [run for row in self.conn.execute("SELECT snapshot FROM runs")
                    if (run := RunSnapshot.model_validate_json(row[0])).workflow == "deep_research"
                    and run.status in {"completed", "partial"}]

    def append(self, snapshot, event):
        # Event and lifecycle snapshot become visible atomically before SSE notification.
        with self.transaction() as db:
            db.execute("INSERT OR REPLACE INTO runs VALUES (?,?)", (snapshot.run_id, snapshot.model_dump_json()))
            db.execute("INSERT INTO events VALUES (?,?,?)", (snapshot.run_id, event.sequence, event.model_dump_json()))
            self.conversations.record_run(db, snapshot, event)

    def budget(self, run_id):
        with self.lock:
            row = self.conn.execute("SELECT used,denied,blocked,prompt_tokens,completion_tokens,active_seconds,cached_input_tokens,cache_observed_input_tokens,cache_observed_calls FROM budget WHERE run_id=?", (run_id,)).fetchone() or (0,0,0,0,0,0,0,0,0)
        return dict(zip(("tool_calls", "denied_calls", "blocked_by_budget", "prompt_tokens", "completion_tokens", "active_seconds", "cached_input_tokens", "cache_observed_input_tokens", "cache_observed_calls"), row))

    def reserve(self, run_id, limit):
        with self.transaction() as db:
            db.execute("INSERT OR IGNORE INTO budget(run_id) VALUES (?)", (run_id,))
            used = db.execute("SELECT used FROM budget WHERE run_id=?", (run_id,)).fetchone()[0]
            if used >= limit:
                db.execute("UPDATE budget SET denied=denied+1,blocked=1 WHERE run_id=?", (run_id,))
                allowed = False
            else:
                db.execute("UPDATE budget SET used=used+1 WHERE run_id=?", (run_id,))
                allowed = True
        return allowed

    def mark_blocked(self, run_id):
        with self.transaction() as db:
            db.execute("INSERT OR IGNORE INTO budget(run_id) VALUES (?)", (run_id,))
            db.execute("UPDATE budget SET blocked=1 WHERE run_id=?", (run_id,))

    def record_usage(self, run_id, prompt, completion, cached_tokens=None):
        with self.transaction() as db:
            db.execute("INSERT OR IGNORE INTO budget(run_id) VALUES (?)", (run_id,))
            db.execute("UPDATE budget SET prompt_tokens=prompt_tokens+?,completion_tokens=completion_tokens+? WHERE run_id=?", (prompt,completion,run_id))

            if cached_tokens is not None:
                db.execute("UPDATE budget SET cached_input_tokens=cached_input_tokens+?,cache_observed_input_tokens=cache_observed_input_tokens+?,cache_observed_calls=cache_observed_calls+1 WHERE run_id=?",
                           (min(prompt, max(0, cached_tokens)), prompt, run_id))

    def model_load(self, run_id):
        with self.lock:
            used = self.budget(run_id)
            reserved = self.conn.execute('SELECT COALESCE(SUM(tokens),0) FROM model_reservations WHERE run_id=?', (run_id,)).fetchone()[0]
            return used['prompt_tokens'] + used['completion_tokens'] + reserved

    def result(self, run_id, key):
        with self.lock:
            row = self.conn.execute("SELECT payload FROM results WHERE run_id=? AND result_key=?", (run_id,key)).fetchone()
        return json.loads(row[0]) if row else None

    def save_result(self, run_id, key, value):
        with self.transaction() as db:
            db.execute("INSERT OR REPLACE INTO results VALUES (?,?,?)", (run_id,key,json.dumps(value,ensure_ascii=False)))

    def register_evidence(self, run_id, identity):
        with self.transaction() as db:
            db.execute("INSERT OR IGNORE INTO run_evidence VALUES (?,?)", (run_id,identity))

    def evidence_ids(self, run_id):
        with self.lock:
            return [r[0] for r in self.conn.execute("SELECT evidence_id FROM run_evidence WHERE run_id=?", (run_id,))]

    def call(self, run_id, key):
        with self.lock:
            row = self.conn.execute("SELECT status,payload FROM calls WHERE run_id=? AND call_key=?", (run_id,key)).fetchone()
        return (row[0], json.loads(row[1]) if row[1] else None) if row else None

    def call_started(self, run_id, key, intent=None):
        with self.transaction() as db:
            db.execute("INSERT INTO calls VALUES (?,?,?,?)", (run_id,key,"started",json.dumps(intent, ensure_ascii=False)))

    def call_finished(self, run_id, key, payload):
        with self.transaction() as db:
            db.execute("UPDATE calls SET status='completed',payload=? WHERE run_id=? AND call_key=?", (json.dumps(payload,ensure_ascii=False),run_id,key))

    def unknown_calls(self, run_id, *, mark=False):
        with self.transaction() as db:
            if mark:
                db.execute("UPDATE calls SET status='unknown' WHERE run_id=? AND status='started'", (run_id,))
            return [r[0] for r in db.execute("SELECT call_key FROM calls WHERE run_id=? AND status IN ('started','unknown')", (run_id,))]

    def add_seconds(self, run_id, seconds):
        with self.transaction() as db:
            db.execute('INSERT OR IGNORE INTO budget(run_id) VALUES (?)', (run_id,))
            db.execute('UPDATE budget SET active_seconds=active_seconds+? WHERE run_id=?', (seconds, run_id))

    def reserve_model(self, run_id, tokens, limit):
        from uuid import uuid4
        with self.transaction() as db:
            used = self.budget(run_id)
            reserved = db.execute('SELECT COALESCE(SUM(tokens),0) FROM model_reservations WHERE run_id=?', (run_id,)).fetchone()[0]
            if used['prompt_tokens'] + used['completion_tokens'] + reserved + tokens > limit:
                return None
            lease = uuid4().hex
            db.execute('INSERT INTO model_reservations VALUES (?,?,?)', (run_id, lease, tokens))
            return lease

    def release_model(self, run_id, lease):
        with self.transaction() as db:
            db.execute('DELETE FROM model_reservations WHERE run_id=? AND lease=?', (run_id, lease))

    def clear_reservations(self, run_id):
        with self.transaction() as db:
            reserved = db.execute('SELECT COALESCE(SUM(tokens),0) FROM model_reservations WHERE run_id=?', (run_id,)).fetchone()[0]
            if reserved:
                self.record_usage(run_id, reserved, 0)
            db.execute('DELETE FROM model_reservations WHERE run_id=?', (run_id,))

    def reset_blocked(self, run_id):
        with self.transaction() as db:
            db.execute('UPDATE budget SET blocked=0 WHERE run_id=?', (run_id,))

    def purge_run(self, run_id):
        with self.transaction() as db:
            for table in ('events','budget','calls','results','run_evidence','model_reservations'):
                db.execute(f'DELETE FROM {table} WHERE run_id=?', (run_id,))
            db.execute('DELETE FROM runs WHERE id=?', (run_id,))

    def close(self):
        self.conn.close()
