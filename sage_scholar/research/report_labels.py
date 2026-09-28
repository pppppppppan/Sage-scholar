"""Conservative navigation labels; no model calls or factual title rewriting.

Unknown prose falls back to a fixed label. Only known categories and a narrow
identifier-based comparison grammar pass, rather than a growing assertion blacklist.
"""
import re
import unicodedata
from difflib import SequenceMatcher


_CATEGORIES = frozenset(label.casefold() for label in (
    "研究报告", "最终报告", "研究综述", "方法比较", "主要方法", "研究方法", "数据集",
    "评测指标", "数据集与评测指标", "实验结果比较", "局限与未解决问题", "研究发现",
    "其他发现", "执行摘要", "结论", "未解决事项", "参考资料",
    "Report", "Research Report", "Comparison", "Results", "Findings", "Methods",
    "Datasets", "Evaluation Metrics", "Limitations", "Executive Summary", "Conclusion",
))
_IDENTIFIER = r"[A-Za-z][A-Za-z0-9_.+-]{0,63}"
_COMPARISON = re.compile(
    rf"(?:方法\s*{_IDENTIFIER}\s*与\s*方法\s*{_IDENTIFIER}\s*的(?:性能|结果|方法)?比较"
    rf"|{_IDENTIFIER}\s+(?:方法比较|研究综述)"
    rf"|Method\s+{_IDENTIFIER}\s+and\s+Method\s+{_IDENTIFIER}\s+(?:performance|results)\s+comparison)",
    re.IGNORECASE,
)


def _compact(text):
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text).casefold())


def neutral_label(text, *, fallback, rejected_claim_texts=()):
    label = unicodedata.normalize("NFKC", text).strip()
    # Reject Markdown/HTML, multiline text, all citation brackets and raw evidence IDs.
    if (not label or len(label) > 120 or re.search(r"[\r\n\[\]<>#`]|ev_", label, re.I)):
        return fallback
    compact = _compact(label)
    for claim in rejected_claim_texts:
        rejected = _compact(claim)
        if rejected and (rejected in compact or compact in rejected
                         or SequenceMatcher(None, compact, rejected).ratio() >= 0.8):
            return fallback
    if label.casefold() in _CATEGORIES or _COMPARISON.fullmatch(label):
        return label
    return fallback
