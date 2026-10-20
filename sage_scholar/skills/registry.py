"""Metadata-only discovery. Skill bodies and references are not read at startup."""
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import json


class SkillError(ValueError):
    pass


def valid_name(name):
    if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) or len(name) > 64:
        raise SkillError("非法 Skill 名称")
    return name


def valid_file(file_path):
    value = "SKILL.md" if file_path is None else file_path
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise SkillError("非法 Skill 文件路径")
    path = PurePosixPath(value)
    if path.is_absolute() or PureWindowsPath(value).drive or ".." in path.parts:
        raise SkillError("Skill 文件不能越过自身目录")
    if value != "SKILL.md" and (not value.startswith("references/") or path.suffix != ".md"):
        raise SkillError("只允许 SKILL.md 和 references/*.md")
    return path.as_posix()


@dataclass(frozen=True)
class SkillMetadata:
    name: str
    description: str
    allow_unverifiable_report: bool = False


class SkillRegistry:
    def __init__(self, root=None):
        self.root = Path(root or Path(__file__).resolve().parents[2] / "skills").resolve()
        self.entries = {}
        if not self.root.is_dir():
            raise SkillError(f"Skill 目录不存在: {self.root}")
        for folder in sorted(self.root.iterdir()):
            if not folder.is_dir() or not (folder / "SKILL.md").exists():
                continue
            name = valid_name(folder.name)
            path = self.path(name, "SKILL.md")
            # The built-in manifest uses single-line YAML scalars; reject unsupported syntax
            # instead of accidentally reading body text as metadata. No extra YAML dependency.
            fields = {}
            with path.open(encoding="utf-8") as stream:
                if stream.readline().strip() != "---":
                    raise SkillError(f"{name} 缺少 frontmatter")
                for _ in range(20):
                    line = stream.readline(2048)
                    if line.strip() == "---":
                        break
                    key, separator, value = line.partition(":")
                    if separator:
                        fields[key.strip()] = value.strip().strip('\"\'')
                else:
                    raise SkillError(f"{name} frontmatter 过长")
            if fields.get("name") != name or not fields.get("description") or len(fields["description"]) > 800:
                raise SkillError(f"{name} 元数据不合法")
            metadata = json.loads(fields.get("metadata", "{}"))
            if not isinstance(metadata, dict) or not isinstance(metadata.get("allow_unverifiable_report", False), bool):
                raise SkillError(f"{name} metadata 必须是含布尔能力标识的内联 JSON 对象")
            self.entries[name] = SkillMetadata(name, fields["description"], metadata.get("allow_unverifiable_report", False))

    def path(self, name, file_path):
        valid_name(name)
        relative = valid_file(file_path)
        folder = (self.root / name).resolve()
        path = (folder / relative).resolve()
        if folder.parent != self.root or not path.is_relative_to(folder) or not path.is_file():
            raise SkillError("Skill 文件不存在或越过允许目录")
        return path

    def catalog(self):
        return [{"name": item.name, "description": item.description} for item in self.entries.values()]
