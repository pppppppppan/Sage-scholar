"""Small single-process JSON stores: shared locks and atomic file replacement."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading

_locks: dict[str, threading.RLock] = {}
_guard = threading.Lock()


def file_lock(path: Path):
    with _guard:
        return _locks.setdefault(str(path.resolve()), threading.RLock())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
