"""Resolve explicit document references against the catalog before workflow routing."""
import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DocumentResolution:
    document_ids: list[str] = field(default_factory=list)
    message: str | None = None


class DocumentResolver:
    def __init__(self, documents):
        self._documents = documents

    async def resolve(self, query: str, selected_ids: list[str]) -> DocumentResolution:
        documents = await asyncio.to_thread(self._documents.list)
        if selected_ids:
            by_id = {doc.document_id: doc for doc in documents}
            for identity in selected_ids:
                doc = by_id.get(identity)
                if doc is None:
                    return DocumentResolution(message="所选论文已不存在，请在文献库重新选择。")
                if not doc.active_version or not doc.converted_path:
                    return DocumentResolution(message=f"《{doc.title}》已上传，但尚未完成解析和索引，请在文献库查看处理状态。")
            return DocumentResolution(document_ids=list(dict.fromkeys(selected_ids)))

        text = query.casefold()
        matches = {}
        for doc in documents:
            title = doc.title.strip()
            alias = Path(title).stem if Path(title).suffix.lower() in {".pdf", ".md", ".txt", ".markdown"} else title
            alias = alias.casefold()
            # ASCII boundaries distinguish e.g. 34552 from 134552/345520, while allowing Chinese suffixes.
            if len(alias) >= 2 and re.search(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", text):
                matches.setdefault(alias, []).append(doc)
        if any(len(group) > 1 for group in matches.values()):
            return DocumentResolution(message="文献库中有多篇同名论文，请勾选本次要使用的论文后再提问。")
        found = [group[0] for group in matches.values()]
        for doc in found:
            if not doc.active_version or not doc.converted_path:
                return DocumentResolution(message=f"《{doc.title}》已上传，但尚未完成解析和索引，请在文献库查看处理状态。")
        return DocumentResolution(document_ids=[doc.document_id for doc in found])
