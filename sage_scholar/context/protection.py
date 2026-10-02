"""Extract restrictions, not every question. No keyword-based factual scoring."""
import json
import re

PROTECTED_KEYS = frozenset({
    "constraints", "unknowns", "conflicts", "limitations", "missing", "remaining_gaps",
    "negative_evidence", "unresolved_gaps", "requested_dimensions", "comparison_dimensions",
})
# Only directive clauses from actual user messages are eligible. Source/tool text is not authority.
_DIRECTIVE = re.compile(
    r"(?:不要|不得|禁止|不能|不允许|不考虑|不使用|只(?:比较|使用|考虑|研究|分析|读取|允许)|"
    r"仅(?:比较|使用|考虑|研究|分析|读取)|必须|务必|请(?:从|按|以)|比较维度|对比维度|"
    r"(?:从|按照|按).{1,64}(?:维度|方面|指标)|重点比较|优先考虑|暂不|排除|"
    r"\b(?:do not|don't|never|must|only (?:use|compare|consider)|exclude)\b)", re.I,
)


def unique(values):
    seen, result = set(), []
    for value in values:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def user_constraints(history):
    clauses = []
    for message in history:
        if message.get("role") != "user":
            continue
        for clause in re.findall(r"[^。！？!?；;\n]+[。！？!?；;]?", message.get("content", "")):
            clause = clause.strip()
            if clause.endswith(("?", "？")) and not re.match(r"(?:请|不要|不得|禁止|必须|只|仅)", clause):
                continue
            if clause and _DIRECTIVE.search(clause):
                clauses.append(clause)
    return unique(clauses)


def protected_facts(messages):
    """Only structured findings are promoted as *data*, never as system instructions."""
    facts = []

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in PROTECTED_KEYS and item:
                    facts.append({key: item})
                else:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for message in messages:
        try:
            walk(json.loads(message.get("content") or ""))
        except (ValueError, TypeError):
            pass
    return unique(facts)


def prompt_contract(contract):
    """Persist the full ExecutionRequest; serialize its history only through ContextManager."""
    return contract.model_dump(mode="json", exclude={"execution": {"history"}})
