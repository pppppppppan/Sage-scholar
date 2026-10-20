"""Small bounded tool loop for existing services/structured calls, not a new Agent."""
from copy import deepcopy
import json
from sage_scholar.runtime.control import check_control, ResearchBudgetReached, CURRENT_CONTROL
from sage_scholar.runtime.journal import WorkflowInterrupted
from sage_scholar.tools.base_tool import ToolCallError


def invoke_tools(llm, messages, tools, *, tag, max_tokens=4096, temperature=0,
                 final_schema=None, max_rounds=4, max_calls=12, images=None):
    history = deepcopy(messages)
    mapping = {t.name: t for t in tools}
    schemas = [t.to_schema() for t in tools]
    terminal = final_schema["function"]["name"] if final_schema else None
    if final_schema:
        schemas.append(final_schema)
    used = 0
    for step in range(max_rounds + 1):
        check_control()
        final = step == max_rounds or used >= max_calls
        available = ([final_schema] if final_schema else None) if final else schemas
        choice = {"type": "function", "function": {"name": terminal}} if final_schema and final else ("none" if final else "auto")
        response = llm.invoke(messages=history, tools=available, tool_choice=choice,
            max_tokens=max_tokens, temperature=temperature, tag=tag, **({"images": images} if images else {}))
        check_control()
        calls = list(response.tool_calls or [])
        if not calls:
            if not (response.content or "").strip():
                raise ValueError("模型返回空回答，未将任务标记为完成")
            return response
        # A structured submission cannot precede pending tool observations in the same batch.
        if terminal and len(calls) == 1 and calls[0].function.name == terminal:
            return response
        if final:
            raise ToolCallError("模型在工具预算结束后仍要求调用工具，未伪装任务完成")
        history.append({"role": "assistant", "content": response.content,
                        "tool_calls": [c.model_dump() for c in calls]})
        for call in calls:
            check_control()
            try:
                if call.function.name == terminal:
                    raise ToolCallError("请先读取本批工具结果，再单独 submit_result")
                if used >= max_calls:
                    raise ToolCallError("本轮工具次数已用完，请依据已有结果总结并说明局限")
                used += 1
                tool = mapping.get(call.function.name)
                if tool is None:
                    raise ToolCallError("本次请求没有此工具权限")
                parameters = json.loads(call.function.arguments or "{}")
                if not isinstance(parameters, dict):
                    raise ToolCallError("工具参数必须是对象")
                control = CURRENT_CONTROL.get()
                if control and control.store and not control.store.reserve(control.request.run_id, control.request.budget.get("max_tool_calls", 0)):
                    used = max_calls
                    raise ToolCallError("Run 工具预算已耗尽；不能继续核验的部分请明确说明")
                content = tool.run_tool(parameters)
            except (WorkflowInterrupted, ResearchBudgetReached):
                raise
            except Exception as exc:
                content = json.dumps({"error": str(exc)}, ensure_ascii=False)
            history.append({"role": "tool", "tool_call_id": call.id, "content": content})
        if used >= max_calls or step == max_rounds - 1:
            history.append({"role": "user", "content": "工具读取已结束。用现有资料完成回答；未核实项明确标为未知，不声称核验成功。"})
    raise AssertionError("unreachable")
