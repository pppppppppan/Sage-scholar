"""Normalize reported cache usage. Missing metrics are unknown, never zero hits."""

def field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def cached_input(usage):
    value = field(field(usage, "prompt_tokens_details"), "cached_tokens")
    if value is None:
        value = field(usage, "prompt_cache_hit_tokens")  # DeepSeek-compatible API
    return max(0, int(value)) if isinstance(value, (int, float)) else None


def anthropic_input(usage):
    read = field(usage, "cache_read_input_tokens")
    write = field(usage, "cache_creation_input_tokens", 0) or 0
    return (field(usage, "input_tokens", 0) + (read or 0) + write, read)
