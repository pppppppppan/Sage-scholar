"""Bounded, typed model responses using the existing provider client."""
import json_repair


def structured_response(llm, model, messages, *, tag, temperature=0, max_tokens=3000, tools=None):
    schema = {
        "type": "function",
        "function": {"name": "submit_result", "description": "Submit the structured result.",
                     "parameters": model.model_json_schema()},
    }
    for attempt in range(2):
        if tools:
            from sage_scholar.tools.calling import invoke_tools
            response = invoke_tools(llm, messages, tools, final_schema=schema, tag=tag,
                                    temperature=temperature, max_tokens=max_tokens)
        else:
            response = llm.invoke(messages=messages, tools=[schema], tool_choice="required",
                                  temperature=temperature, tag=tag, max_tokens=max_tokens)
        try:
            raw = response.tool_calls[0].function.arguments if response.tool_calls else response.content
            return model.model_validate(json_repair.loads(raw or ""))
        except (ValueError, TypeError, IndexError):
            if attempt:
                raise
    raise AssertionError("unreachable")
