"""Official LangGraph SQLite saver owns graph channels, tasks and pending writes."""
import sqlite3
from pathlib import Path
from pydantic import BaseModel
from enum import Enum
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from sage_scholar.domain import research, execution
from sage_scholar.routing import models
from sage_scholar.audit import models as audit_models

SCHEMAS = {"deep_research": "research-v4-1", "citation_audit": "audit-v4.1-1"}


class Checkpoints:
    def __init__(self, data_dir):
        path = Path(data_dir) / "checkpoints.sqlite3"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA synchronous=FULL")
        allowed = [(module.__name__, name) for module in (research, execution, models, audit_models)
                   for name, value in vars(module).items() if isinstance(value, type) and issubclass(value, (BaseModel, Enum)) and value.__module__ == module.__name__]
        self.saver = SqliteSaver(self.conn, serde=JsonPlusSerializer(allowed_msgpack_modules=allowed, pickle_fallback=False))
        self.saver.setup()

    @staticmethod
    def config(run_id):
        return {"configurable": {"thread_id": run_id}, "recursion_limit": 2000}

    def reference(self, run_id, workflow):
        checkpoint = self.saver.get_tuple(self.config(run_id))
        return {"thread_id": run_id, "schema": SCHEMAS.get(str(workflow)),
                "checkpoint_id": checkpoint.config["configurable"]["checkpoint_id"] if checkpoint else None}

    def compatible(self, snapshot):
        if snapshot.schema_version != 4 or snapshot.checkpoint.get("schema") != SCHEMAS.get(str(snapshot.workflow)):
            return False
        try:
            return self.saver.get_tuple(self.config(snapshot.run_id)) is not None
        except Exception:
            return False

    def close(self):
        self.conn.close()
