"""Incremental text delivery without changing the shared model client lifetime."""
import time
from openai.types.chat import ChatCompletionMessage


def openai_text(client, kwargs, control, track, tag):
    start = time.monotonic()
    text, usage = [], None
    stream = client.chat.completions.create(**kwargs, stream=True, stream_options={"include_usage": True})
    try:
        for chunk in stream:
            control.check()
            if chunk.usage:
                usage = chunk.usage
            if chunk.choices:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    text.append(delta)
                    control.emit("answer_delta", {"text": delta})
    finally:
        stream.close()
        msg = ChatCompletionMessage(role="assistant", content="".join(text))
        # Interrupted streams often omit final usage. Conservatively account locally.
        from sage_scholar.context.token_counter import TokenCounter
        count_tokens = TokenCounter().count
        prompt = usage.prompt_tokens if usage else count_tokens(str(kwargs["messages"]))
        completion = usage.completion_tokens if usage else count_tokens(msg.content)
        track(prompt, completion, time.monotonic() - start, tag, msg)
    return msg
