"""Only tool envelopes determine failure, never words inside a paper body."""
import json
import re


def is_error_result(result):
    if re.match(r"\s*(?:Error:|Failed:|<error>)", result, re.I):
        return True
    try:
        value = json.loads(result)
    except (ValueError, TypeError):
        return False
    return isinstance(value, dict) and bool(value.get("error") or value.get("isError")
        or value.get("success") is False or value.get("status") in ("error", "failed"))
