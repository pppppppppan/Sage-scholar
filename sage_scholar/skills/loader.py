"""Lazy, bounded reads with a pinned reference manifest for resumable Runs."""
import hashlib
import json
from .registry import SkillError, valid_file, valid_name

MAX_FILE_BYTES = 32_000
MAX_FILES = 16


def signature(path):
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns, stat.st_ino]


def read_bounded(path, expected=None):
    before = signature(path)
    if expected is not None and before != expected:
        raise SkillError("Skill 参考文件已改变；原 Run 不混用新版，请新建任务使用新版 Skill")
    if before[0] > MAX_FILE_BYTES:
        raise SkillError("Skill 文件过大，请拆分参考资料")
    with path.open("rb") as stream:
        data = stream.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES or signature(path) != before:
        raise SkillError("Skill 文件过大或读取期间被修改")
    return data.decode("utf-8")


class SkillLoader:
    def __init__(self, registry):
        self.registry = registry

    def select(self, name):
        valid_name(name)
        if name not in self.registry.entries:
            raise SkillError(f"不存在 Skill: {name}")
        main = self.registry.path(name, "SKILL.md")
        content = read_bounded(main)
        manifest = {}
        for path in sorted((main.parent / "references").rglob("*.md")):
            relative = path.relative_to(main.parent).as_posix()
            manifest[relative] = signature(self.registry.path(name, relative))
            if len(manifest) > MAX_FILES:
                raise SkillError("Skill 参考文件数量超限")
        # Reference bytes stay lazy. Already-read files live in SQLite; changed unread
        # files are refused on resume rather than silently mixing revisions.
        version = hashlib.sha256((content + json.dumps(manifest, sort_keys=True)).encode()).hexdigest()
        return {"name": name, "version": version, "manifest": manifest,
                "allow_unverifiable_report": self.registry.entries[name].allow_unverifiable_report,
                "files": {"SKILL.md": content}}

    def read(self, selected, file_path=None):
        relative = valid_file(file_path)
        if relative in selected["files"]:
            return selected["files"][relative]
        if relative not in selected["manifest"]:
            raise SkillError("参考文件不在本次 Run 固定的 Skill 版本中")
        text = read_bounded(self.registry.path(selected["name"], relative), selected["manifest"][relative])
        selected["files"][relative] = text
        return text
