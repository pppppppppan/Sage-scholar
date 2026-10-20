"""Bounded content-addressed snapshots, with reference bodies disclosed only on demand."""
import hashlib
import json
from .registry import SkillError, valid_file, valid_name

MAX_FILE_BYTES = 32_000
MAX_FILES = 16
SNAPSHOT_SCHEMA = 2


def digest(content):
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def read_bounded(path, expected=None):
    if path.stat().st_size > MAX_FILE_BYTES:
        raise SkillError("Skill 文件过大，请拆分参考资料")
    with path.open('rb') as stream:
        data = stream.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise SkillError("Skill 文件过大，请拆分参考资料")
    if expected is not None and hashlib.sha256(data).hexdigest() != expected:
        raise SkillError("Skill 参考文件已改变；原 Run 不混用新版，请新建任务使用新版 Skill")
    return data.decode('utf-8')


class SkillLoader:
    def __init__(self, registry):
        self.registry = registry

    def select(self, name):
        valid_name(name)
        if name not in self.registry.entries:
            raise SkillError(f"不存在 Skill: {name}")
        main = self.registry.path(name, 'SKILL.md')
        content = read_bounded(main)
        manifest = {}
        for path in sorted((main.parent / 'references').rglob('*.md')):
            if len(manifest) >= MAX_FILES:
                raise SkillError("Skill 参考文件数量超限")
            relative = path.relative_to(main.parent).as_posix()
            manifest[relative] = digest(read_bounded(self.registry.path(name, relative)))
        hashes = {'SKILL.md': digest(content), **manifest}
        version = digest(json.dumps(hashes, sort_keys=True))
        return {'schema_version': SNAPSHOT_SCHEMA, 'name': name, 'version': version,
                'main_sha256': hashes['SKILL.md'], 'manifest': manifest,
                'allow_unverifiable_report': self.registry.entries[name].allow_unverifiable_report,
                'files': {'SKILL.md': content}}

    def validate(self, selected):
        # Legacy stat manifests cannot prove the identity of unread references. Do not
        # silently upgrade their identity using today's bytes (including on resume).
        if selected.get('schema_version') != SNAPSHOT_SCHEMA:
            raise SkillError("旧 Skill 快照缺少内容哈希，无法安全恢复；请重新开始任务")
        hashes = {'SKILL.md': selected['main_sha256'], **selected['manifest']}
        if digest(json.dumps(hashes, sort_keys=True)) != selected['version']:
            raise SkillError("Skill 快照版本校验失败")
        for name, content in selected['files'].items():
            valid_file(name)
            if digest(content) != hashes.get(name):
                raise SkillError("Skill 已保存内容校验失败")

    def read(self, selected, file_path=None):
        relative = valid_file(file_path)
        self.validate(selected)
        # The immutable persisted body is the authority, not mutable working files.
        if relative in selected['files']:
            return selected['files'][relative]
        if relative not in selected['manifest']:
            raise SkillError("参考文件不在本次 Run 固定的 Skill 版本中")
        text = read_bounded(self.registry.path(selected['name'], relative), selected['manifest'][relative])
        selected['files'][relative] = text
        return text
