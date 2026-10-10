"""Deterministic reference validation; never judges scientific truth."""
import re
from urllib.parse import urlparse

_CITE = re.compile(r"(?<!!)\[(\d+)\](?!\()")
_LITERAL = re.compile(r"(```[\s\S]*?```|`[^`]*`|\$\$[\s\S]*?\$\$|(?<!\\)\$[^\n$]+?\$)")
_INTERNAL = re.compile(r"\bev_[a-zA-Z0-9_]{6,}|prior_findings|ResearchNote|Traceback|ContextBudgetExceeded|tool_errors|完整请求超出上下文预算")


def citation_numbers(markdown):
    prose = _LITERAL.sub("", markdown)
    return list(dict.fromkeys(int(m.group(1)) for m in _CITE.finditer(prose)))


def reference_catalog(context, identities):
    result = {}
    for identity in dict.fromkeys(identities):
        context.get(identity)  # Existing ownership/version/SourcePolicy checks.
        result[len(result) + 1] = identity
    return result


def source_label(item):
    title = item.metadata.get("title") or item.metadata.get("filename") or item.document_id or item.source_id
    return re.sub(r"[\[\]\n\r<>]", " ", str(title))[:220]


def writer_material(notes, mapping, context):
    from .notes import SOURCE_ID
    reverse = {identity: number for number, identity in mapping.items()}
    materials = []
    for note in notes:
        text = re.sub(r"\[(ev_[a-zA-Z0-9_-]+)\]", r"\1", note.content)
        text = SOURCE_ID.sub(lambda m: f"[{reverse[m.group()]}]" if m.group() in reverse else "[来源未确认]", text)
        materials.append({"question": note.question, "content": text, "status": note.status})
    # Identity/locator only: the Markdown notes carry the findings, originals stay reloadable.
    sources = []
    for number, identity in mapping.items():
        item = context.get(identity)
        sources.append({"number": number, "title": source_label(item), "source": item.source_id,
                        "section": item.section, "page": item.page})
    return materials, sources


def finalize_report(text, mapping, context, gaps=()):
    from sage_scholar.domain.research import ResearchReport
    # The bibliography is generated below from authoritative source identities.
    text = re.split(r"^#{1,3}\s*(?:参考文献|References|Sources)\s*:?[ \t]*$", text, flags=re.M | re.I)[0].strip()
    remaining, used = list(gaps), {}
    for number in citation_numbers(text):
        identity = mapping.get(number)
        try:
            if identity is None:
                raise ValueError("unknown citation")
            context.get(identity)
            used[number] = identity
        except ValueError:
            remaining.append("部分引用未能定位到本次研究来源，已标明待确认")
    if not used:
        remaining.append("报告未提供可追溯的正文引用")
    # Bibliography is generated from validated identities, never model-created URLs.
    renumber, source_numbers, citation_evidence = {}, {}, {}
    for old, identity in used.items():
        item = context.get(identity)
        key = (item.source_id, item.source_version, item.document_id)
        number = source_numbers.setdefault(key, len(source_numbers) + 1)
        renumber[old] = number
        citation_evidence.setdefault(number, []).append(identity)
    pieces = _LITERAL.split(text)
    for index in range(0, len(pieces), 2):
        pieces[index] = _CITE.sub(lambda m: f"[{renumber[int(m.group(1))]}]" if int(m.group(1)) in renumber else "[来源未确认]", pieces[index])
        pieces[index] = re.sub(r"(\[\d+\])(?:\1)+", r"\1", pieces[index])
    text = "".join(pieces)
    if _INTERNAL.search(text):
        text = _INTERNAL.sub("（内部标识已省略）", text)
        remaining.append("正文出现内部标识，已移除，需复核相关表述")
    refs = []
    listed = set()
    for old, identity in used.items():
        if renumber[old] in listed:
            continue
        listed.add(renumber[old])
        item = context.get(identity)
        source = item.source_id
        if source.startswith("arxiv:"):
            source = "https://arxiv.org/abs/" + source[6:]
        if urlparse(source).scheme not in {"http", "https"}:
            source = f"/api/evidence/{identity}"
        source = source.replace("(", "%28").replace(")", "%29").replace(" ", "%20")
        location = re.sub(r"[\n\r<>\[\]]", " ", item.section or (f"第 {item.page} 页" if item.page else ""))
        refs.append(f"[{renumber[old]}] [{source_label(item)}]({source})" + (f" · {location}" if location else ""))
    remaining = list(dict.fromkeys(remaining))
    notice = "\n\n> 阶段性报告：仍有未覆盖或待确认内容，具体缺口见任务状态。" if remaining else ""
    report = text + notice
    if refs:
        report += "\n\n## 参考文献\n\n" + "\n".join(refs)
    return ResearchReport(report=report, citations={renumber[n]: i for n, i in used.items()},
                          citation_evidence=citation_evidence, remaining_gaps=remaining, partial=bool(remaining))
