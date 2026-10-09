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


def valid_heading(value):
    return bool(value.strip()) and not (_INTERNAL.search(value) or re.search(r"[\n\r<>\[\]]", value))


def source_label(item):
    title = item.metadata.get("title") or item.metadata.get("filename") or item.document_id or item.source_id
    return re.sub(r"[\[\]\n\r<>]", " ", str(title))[:220]


def reference_material(context, mapping, identities, *, tokens=4500):
    wanted = set(identities)
    subset = {number: identity for number, identity in mapping.items() if identity in wanted}
    records = context.read(list(subset.values()), max_chars=tokens * 2)
    by_id = {r["evidence_id"]: r for r in records}
    return [{"number": n, "title": source_label(context.get(i)), **by_id[i]}
            for n, i in subset.items()]


def check_section(section, mapping, context, allowed_ids=None):
    issues = []
    if not section.markdown.strip():
        return ["章节没有正文"]
    if not valid_heading(section.heading):
        issues.append("章节标题含有内部标识或非法结构")
    if _INTERNAL.search(section.markdown):
        issues.append("正文含有内部标识或运行诊断，必须改写为面向读者的内容")
    if re.search(r"^#{1,3}\s*(参考文献|References|Sources)\s*$", section.markdown, re.M | re.I):
        issues.append("章节不能自行生成参考文献表")
    numbers = citation_numbers(section.markdown)
    if not numbers:
        issues.append("本节没有可追溯引用；若确无证据，应明确说明而非陈述科研结论")
    for number in numbers:
        identity = mapping.get(number)
        if not identity or (allowed_ids is not None and identity not in allowed_ids):
            issues.append(f"引用 [{number}] 不属于该章节可用证据")
            continue
        try:
            item = context.get(identity)
            source = " ".join(item.original_text.split())
            body = " ".join(section.markdown.split())
            if len(source) > 600 and source[:600] in body:
                issues.append("正文复制了大段来源原文，应综合改写并保留引用")
        except ValueError:
            issues.append(f"引用 [{number}] 的来源版本或访问范围无效")
    if section.markdown.count("$$") % 2:
        issues.append("块级公式分隔符不成对")
    return list(dict.fromkeys(issues))


def finalize_report(outline, sections, mapping, context, reviews, gaps):
    from sage_scholar.domain.research import ResearchReport
    if not valid_heading(outline.title):
        raise ValueError("报告标题格式不合法，未发布报告")
    body, used, remaining = [], {}, list(gaps)
    for plan in outline.sections:
        section = sections.get(plan.section_id)
        if section is None:
            remaining.append(f"尚未完成：{plan.heading}")
            continue
        identity_issues = check_section(section, mapping, context)
        if identity_issues:
            remaining.append(f"{plan.heading}未通过格式或来源检查，本节暂未发布")
            continue
        review = reviews.get(plan.section_id)
        if review is None or not review.acceptable:
            # A failed review never silently becomes a finished scientific section.
            remaining.append(f"{plan.heading}仍需修订，暂未纳入报告")
            continue
        for number in citation_numbers(section.markdown):
            used[number] = mapping[number]
        body.append(f"## {plan.heading}\n\n{section.markdown.strip()}")
    if not body:
        raise ValueError("尚未形成可发布的研究报告，已收集的证据和综合结果已保存")
    # Bibliography is generated from validated identities, never model-created URLs.
    renumber, source_numbers, citation_evidence = {}, {}, {}
    for old, identity in used.items():
        item = context.get(identity)
        key = (item.source_id, item.source_version, item.document_id)
        number = source_numbers.setdefault(key, len(source_numbers) + 1)
        renumber[old] = number
        citation_evidence.setdefault(number, []).append(identity)
    text = "\n\n".join(body)
    pieces = _LITERAL.split(text)
    for index in range(0, len(pieces), 2):
        pieces[index] = _CITE.sub(lambda m: f"[{renumber[int(m.group(1))]}]" if int(m.group(1)) in renumber else m.group(), pieces[index])
        pieces[index] = re.sub(r"(\[\d+\])(?:\1)+", r"\1", pieces[index])
    text = "".join(pieces)
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
    notice = "\n\n> 阶段性报告：部分研究内容尚未完成或通过审查，以下仅保留已完成部分。" if remaining else ""
    report = f"# {outline.title}{notice}\n\n{text}\n\n## 参考文献\n\n" + "\n".join(refs)
    return ResearchReport(report=report, citations={renumber[n]: i for n, i in used.items()},
                          citation_evidence=citation_evidence, remaining_gaps=remaining,
                          section_reviews=reviews, partial=bool(remaining))
