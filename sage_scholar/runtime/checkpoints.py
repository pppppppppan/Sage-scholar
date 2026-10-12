"""Official LangGraph SQLite saver owns graph channels, tasks and pending writes."""
import sqlite3
from pathlib import Path
from pydantic import BaseModel
from enum import Enum
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from sage_scholar.domain import research, execution
from sage_scholar.routing import models

SCHEMAS = {"deep_research": "research-supervisor-markdown-v2"}
ENTRY_SCHEMA = "semantic-query-v1"


class Checkpoints:
    def __init__(self, data_dir):
        path = Path(data_dir) / "checkpoints.sqlite3"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA synchronous=FULL")
        allowed = [(module.__name__, name) for module in (research, execution, models)
                   for name, value in vars(module).items() if isinstance(value, type) and issubclass(value, (BaseModel, Enum)) and value.__module__ == module.__name__]
        self.saver = SqliteSaver(self.conn, serde=JsonPlusSerializer(allowed_msgpack_modules=allowed, pickle_fallback=False))
        self.saver.setup()

    @staticmethod
    def config(run_id):
        return {"configurable": {"thread_id": run_id, "checkpoint_ns": ""}, "recursion_limit": 2000}

    @classmethod
    def research_config(cls, request):
        identity = request.run_id + ":research" if request.metadata.get("_entry_graph") else request.run_id
        return cls.config(identity)

    def reference(self, run_id, workflow):
        checkpoint = self.saver.get_tuple(self.config(run_id))
        values = checkpoint.checkpoint.get("channel_values", {}) if checkpoint else {}
        schema = ENTRY_SCHEMA if values.get("entry_version") == ENTRY_SCHEMA else SCHEMAS.get(str(workflow))
        return {"thread_id": run_id, "schema": schema,
                "checkpoint_id": checkpoint.config["configurable"]["checkpoint_id"] if checkpoint else None}

    def compatible(self, snapshot):
        expected = ENTRY_SCHEMA if snapshot.execution.metadata.get("_entry_graph") == ENTRY_SCHEMA else SCHEMAS.get(str(snapshot.workflow))
        if expected is None or snapshot.schema_version != 4 or snapshot.checkpoint.get("schema") != expected:
            return False
        try:
            return self.saver.get_tuple(self.config(snapshot.run_id)) is not None
        except Exception:
            return False

    def pending_interrupts(self, run_id):
        item = self.saver.get_tuple(self.config(run_id))
        return [value for _, channel, values in (item.pending_writes or []) if channel == "__interrupt__" for value in values] if item else []

    def delete(self, run_id):
        self.saver.delete_thread(run_id)
        self.saver.delete_thread(run_id + ":research")

    def close(self):
        self.conn.close()
