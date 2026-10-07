"""Atomic temporal memories and extraction receipts; indexes remain disposable."""
import json
import os
import tempfile
from pathlib import Path
from sage_scholar.domain.documents import utc_now
from sage_scholar.infrastructure.json_store import file_lock, write_json
from .models import MemoryInput, MemoryRecord, MemorySource, visible, same_scope, scope_fingerprint
from .recall import normalized


class MemoryConflict(ValueError):
    pass


class MemoryStore:
    def __init__(self, data_dir):
        self.path = Path(data_dir) / "memory.json"
        self.lock = file_lock(self.path)

    def _read(self):
        raw = json.loads(self.path.read_text()) if self.path.exists() else []
        if isinstance(raw, list):
            raw = {"revision": 0, "memories": [dict(r, source=r.get("source", {"kind": "legacy"})) for r in raw]}
        return raw["revision"], [MemoryRecord.model_validate(r) for r in raw["memories"]], raw.get("receipts", {})

    def _write(self, revision, items, receipts, *, changed=True):
        # Preserve the prior schema, including obsolete unapproved candidates, once on migration.
        backup = self.path.with_name("memory.pre-finalization.json")
        if self.path.exists() and not backup.exists():
            old = json.loads(self.path.read_text())
            if isinstance(old, list) or old.get("schema", 0) < 3:
                write_json(backup, old)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=self.path.parent, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as file:
                json.dump({"schema": 3, "revision": revision + int(changed),
                    "memories": [r.model_dump(mode="json") for r in items], "receipts": receipts}, file, ensure_ascii=False)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        # Receipt and memory become durable together, before the SQLite job can acknowledge success.
        fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def snapshot(self):
        with self.lock:
            revision, items, _ = self._read()
            return revision, items

    def list(self, session_id=None, *, workspace_id="default", all_scopes=False, include_inactive=False):
        _, items = self.snapshot()
        return [r for r in items if (all_scopes or visible(r, workspace_id, session_id))
                and r.status != "deleted" and (include_inactive or r.status == "active")]

    def processed(self, key):
        with self.lock:
            return bool(key and key in self._read()[2])

    def operation_results(self, batch_id):
        with self.lock:
            return {value for key, value in self._read()[2].items() if key.startswith(batch_id + ":") and value}

    @staticmethod
    def _combined(targets, candidate):
        # Lossless merge: copy complete source-backed statements, never ask a model to rewrite them.
        parts, seen = [], set()
        for text in [*(r.content for r in targets), candidate.content]:
            if normalized(text) not in seen:
                parts.append(text)
                seen.add(normalized(text))
        return "\n\n".join(parts)

    def commit(self, candidate, *, expected_revision=None, manual=False):
        with self.lock:
            revision, items, receipts = self._read()
            key = candidate.operation_key
            if key and key in receipts:
                # Even if the original result was subsequently edited/deleted, never resurrect it.
                return next((r for r in items if r.memory_id == receipts[key] and r.status == "active"), None)
            if expected_revision is not None and revision != expected_revision:
                raise MemoryConflict("记忆已被修改，请重新整理")
            if candidate.scope_fingerprint != scope_fingerprint(items, candidate):
                raise MemoryConflict("该范围的记忆已变化，请重新整理")
            targets = [r for r in items if r.memory_id in candidate.proposal.targets]
            if len(targets) != len(candidate.proposal.targets):
                raise MemoryConflict("目标记忆已不存在")
            action = candidate.proposal.action
            for target in targets:
                if not same_scope(target, candidate) or target.version != candidate.target_versions.get(target.memory_id):
                    raise MemoryConflict("记忆范围或版本已变化")
            suppressed = not manual and (
                any(r.status == "deleted" and same_scope(r, candidate)
                    and normalized(r.content) == normalized(candidate.content) for r in items)
                or any(r.status != "active" for r in targets)
                or (action != "ignore" and any(r.manual_revision for r in targets))
                or any(r.updated_at > candidate.created_at and same_scope(r, candidate)
                       and (r.manual_revision or r.status == "deleted") for r in items))
            if suppressed:
                if key:
                    receipts[key] = None
                    self._write(revision, items, receipts, changed=False)
                return None
            if any(r.status != "active" for r in targets):
                raise MemoryConflict("目标记忆不再有效")
            if action == "ignore":
                result = targets[0] if targets else None
                if key:
                    receipts[key] = result.memory_id if result else None
                    self._write(revision, items, receipts, changed=False)
                return result  # No original content, timestamps, or source metadata mutation.
            content = self._combined(targets, candidate) if action == "merge" else candidate.content
            if len(content) > 8000:
                # A safe merge cannot fit the record contract: retain both active records.
                content, targets = candidate.content, []
            sources = [candidate.source]
            for target in targets:
                for source in target.sources or [target.source]:
                    if source not in sources:
                        sources.append(source)
            result = MemoryRecord(content=content, memory_type=candidate.memory_type, scope=candidate.scope,
                workspace_id=candidate.workspace_id, session_id=candidate.session_id,
                importance=candidate.importance, confidence=candidate.confidence, source=candidate.source,
                sources=sources, supersedes=[r.memory_id for r in targets],
                version=max([r.version for r in targets], default=0) + 1, manual_revision=manual)
            for target in targets:
                target.status, target.superseded_by, target.updated_at = "superseded", result.memory_id, utc_now()
            items.append(result)
            if key:
                receipts[key] = result.memory_id
            self._write(revision, items, receipts)
            return result

    def add(self, value: MemoryInput):
        with self.lock:
            revision, items, receipts = self._read()
            duplicate = next((r for r in items if r.status == "active" and same_scope(r, value) and r.content == value.content), None)
            if duplicate:
                return duplicate
            item = MemoryRecord(**value.model_dump())
            items.append(item)
            self._write(revision, items, receipts)
            return item

    def update(self, identity, value: MemoryInput, *, expected_revision=None, supersede=()):
        with self.lock:
            revision, items, receipts = self._read()
            old = next((r for r in items if r.memory_id == identity and r.status == "active"), None)
            if not old:
                raise KeyError(identity)
            if expected_revision is not None and revision != expected_revision:
                raise MemoryConflict("记忆已变更，请重新修改")
            targets = [r for r in items if r.memory_id in supersede and r.memory_id != identity]
            if any(r.status != "active" or not same_scope(r, value) for r in targets):
                raise MemoryConflict("不能覆盖其他范围或失效记忆")
            archived = old.model_copy(deep=True, update={"memory_id": f"{identity}_v{old.version}",
                "status": "superseded", "superseded_by": identity})
            replacement = MemoryRecord(**value.model_dump(), memory_id=identity, created_at=old.created_at,
                source=MemorySource(kind="manual", quote=value.content),
                sources=old.sources or [old.source], version=old.version + 1,
                supersedes=[archived.memory_id], manual_revision=True)
            for previous in items:
                if previous.superseded_by == identity:
                    previous.superseded_by = archived.memory_id
            for target in targets:
                target.status, target.superseded_by, target.updated_at = "superseded", identity, utc_now()
                replacement.supersedes.append(target.memory_id)
            items = [replacement if r.memory_id == identity else r for r in items] + [archived]
            self._write(revision, items, receipts)
            return replacement

    def delete(self, identity):
        with self.lock:
            revision, items, receipts = self._read()
            item = next((r for r in items if r.memory_id == identity and r.status != "deleted"), None)
            if not item:
                raise KeyError(identity)
            item.status, item.manual_revision, item.updated_at = "deleted", True, utc_now()
            self._write(revision, items, receipts)

    def recall(self, query, *, workspace_id="default", session_id=None, top_k=5, **kwargs):
        from .recall import lexical_rank
        return [r for r, _ in lexical_rank(query, self.list(session_id, workspace_id=workspace_id))[:max(0, min(5, top_k))]]
