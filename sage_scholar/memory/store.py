"""Atomic memory/candidate lifecycle; vector files are disposable projections."""
import json
from pathlib import Path
from sage_scholar.domain.documents import utc_now
from sage_scholar.infrastructure.json_store import file_lock, write_json
from .models import MemoryInput, MemoryRecord, MemoryCandidate, visible, same_scope, scope_fingerprint


class MemoryConflict(ValueError):
    pass


class MemoryStore:
    def __init__(self, data_dir):
        self.path = Path(data_dir) / "memory.json"
        self.lock = file_lock(self.path)

    def _read(self):
        raw = json.loads(self.path.read_text()) if self.path.exists() else []
        if isinstance(raw, list):  # Lossless migration of V4 preferences; old workspace means default.
            raw = {"revision": 0, "memories": [dict(r, source=r.get("source", {"kind": "legacy"})) for r in raw], "candidates": []}
        return raw["revision"], [MemoryRecord.model_validate(r) for r in raw["memories"]], [MemoryCandidate.model_validate(c) for c in raw["candidates"]]

    def _write(self, revision, items, candidates):
        write_json(self.path, {"schema": 2, "revision": revision + 1,
            "memories": [r.model_dump(mode="json") for r in items],
            "candidates": [c.model_dump(mode="json") for c in candidates]})

    def snapshot(self):
        with self.lock:
            return self._read()

    def list(self, session_id=None, *, workspace_id="default", all_scopes=False, include_inactive=False):
        _, items, _ = self.snapshot()
        return [r for r in items if (all_scopes or visible(r, workspace_id, session_id))
                and r.status != "deleted" and (include_inactive or r.status == "active")]

    def candidates(self, *, workspace_id="default", session_id=None):
        return [c for c in self.snapshot()[2] if c.status == "pending" and visible(c, workspace_id, session_id)]

    def save_candidate(self, value, expected_revision):
        with self.lock:
            revision, items, candidates = self._read()
            if revision != expected_revision:
                raise MemoryConflict("记忆已被修改，请重新检查候选冲突")
            for previous in candidates:
                if same_scope(previous, value) and previous.content == value.content and previous.source == value.source:
                    if previous.status != "pending":
                        return previous  # Accepted/rejected user decisions survive extraction retries.
                    value.candidate_id = previous.candidate_id
                    value.revision = previous.revision + 1
                    candidates = [value if c.candidate_id == previous.candidate_id else c for c in candidates]
                    self._write(revision, items, candidates)
                    return value
            if len([c for c in candidates if c.status == "pending" and same_scope(c, value)]) >= 100:
                raise ValueError("候选记忆已达 100 条，请先确认或丢弃已有候选")
            candidates.append(value)
            self._write(revision, items, candidates)
            return value

    def reject(self, identity):
        with self.lock:
            revision, items, candidates = self._read()
            candidate = next((c for c in candidates if c.candidate_id == identity), None)
            if not candidate:
                raise KeyError(identity)
            if candidate.status != "pending":
                raise MemoryConflict("候选已处理")
            candidate.status = "rejected"
            self._write(revision, items, candidates)

    def commit(self, candidate, *, expected_revision=None, manual=False):
        """One write commits candidate status and all temporal relations, with stale-plan checks."""
        with self.lock:
            revision, items, candidates = self._read()
            stored = next((c for c in candidates if c.candidate_id == candidate.candidate_id), None)
            if stored and stored.status == "accepted":
                result = next((r for r in items if r.memory_id == stored.result_id and r.status == "active"), None)
                if result:
                    return result
                raise MemoryConflict("候选的结果已变更，不能重新激活旧版本")
            if stored and stored.status != "pending":
                raise MemoryConflict("候选已被拒绝")
            if expected_revision is not None and revision != expected_revision:
                raise MemoryConflict("记忆已被修改，请重试")
            if stored and stored.revision != candidate.revision:
                raise MemoryConflict("候选方案已更新，请刷新后重新确认")
            if candidate.scope_fingerprint != scope_fingerprint(items, candidate):
                raise MemoryConflict("该范围的记忆已变化，请重新提取并确认合并方案")
            targets = [r for r in items if r.memory_id in candidate.proposal.targets]
            if len(targets) != len(candidate.proposal.targets):
                raise MemoryConflict("目标记忆已不存在")
            for target in targets:
                if not same_scope(target, candidate) or target.status != "active" or target.version != candidate.target_versions.get(target.memory_id):
                    raise MemoryConflict("候选关联的记忆已被修改/删除，请重新提取候选")
            # A manual correction/deletion creates a barrier against pending automatic reintroductions.
            if not manual and any(r.updated_at > candidate.created_at and same_scope(r, candidate)
                                  and (r.manual_revision or r.status == "deleted") for r in items):
                raise MemoryConflict("候选产生后已有手动修改，请重新提取确认")
            action = candidate.proposal.action
            if action == "ignore" and targets:
                result = targets[0]
            else:
                result = MemoryRecord(
                    content=candidate.content, memory_type=candidate.memory_type, scope=candidate.scope,
                    workspace_id=candidate.workspace_id, session_id=candidate.session_id, confirmed=True,
                    importance=candidate.importance, confidence=candidate.confidence, source=candidate.source,
                    sources=[candidate.source], supersedes=[r.memory_id for r in targets],
                    version=max([r.version for r in targets], default=0) + 1, manual_revision=manual,
                )
                for target in targets:
                    target.status, target.superseded_by, target.updated_at = "superseded", result.memory_id, utc_now()
                    result.sources.extend(target.sources or [target.source])
                items.append(result)
            if stored:
                stored.status, stored.result_id = "accepted", result.memory_id
            self._write(revision, items, candidates)
            return result

    def add(self, value: MemoryInput):
        if not value.confirmed:
            raise ValueError("只保存用户明确要求或确认的记忆")
        # Service adds semantic consolidation. The storage primitive remains deterministic.
        with self.lock:
            revision, items, candidates = self._read()
            duplicate = next((r for r in items if r.status == "active" and same_scope(r, value) and r.content == value.content), None)
            if duplicate:
                return duplicate
            item = MemoryRecord(**value.model_dump())
            items.append(item)
            self._write(revision, items, candidates)
            return item

    def update(self, identity, value: MemoryInput, *, expected_revision=None, supersede=()):
        if not value.confirmed:
            raise ValueError("修改记忆需要用户明确确认")
        with self.lock:
            revision, items, candidates = self._read()
            old = next((r for r in items if r.memory_id == identity and r.status == "active"), None)
            if not old:
                raise KeyError(identity)
            if expected_revision is not None and revision != expected_revision:
                raise MemoryConflict("记忆已变更，请重新修改")
            other_targets = [r for r in items if r.memory_id in supersede and r.memory_id != identity]
            if any(r.status != "active" or not same_scope(r, value) for r in other_targets):
                raise MemoryConflict("不能覆盖其他范围或失效的记忆")
            # Preserve API ID for manual edits; retain the prior version as an immutable predecessor.
            archived = old.model_copy(deep=True, update={"memory_id": f"{identity}_v{old.version}",
                "status": "superseded", "superseded_by": identity})
            replacement = MemoryRecord(**value.model_dump(), memory_id=identity, created_at=old.created_at,
                version=old.version + 1, supersedes=[archived.memory_id], manual_revision=True)
            for previous in items:
                if previous.superseded_by == identity:
                    previous.superseded_by = archived.memory_id
            for target in other_targets:
                target.status, target.superseded_by, target.updated_at = "superseded", identity, utc_now()
                replacement.supersedes.append(target.memory_id)
            items = [replacement if r.memory_id == identity else r for r in items] + [archived]
            self._write(revision, items, candidates)
            return replacement

    def delete(self, identity):
        with self.lock:
            revision, items, candidates = self._read()
            item = next((r for r in items if r.memory_id == identity and r.status != "deleted"), None)
            if not item:
                raise KeyError(identity)
            item.status, item.manual_revision, item.updated_at = "deleted", True, utc_now()
            self._write(revision, items, candidates)

    def recall(self, query, *, workspace_id="default", session_id=None, top_k=5, **kwargs):
        # Offline/standalone store users get bounded lexical recall, never list()->prompt.
        from .recall import lexical_rank
        return [r for r, _ in lexical_rank(query, self.list(session_id, workspace_id=workspace_id))[:max(0, min(5, top_k))]]
