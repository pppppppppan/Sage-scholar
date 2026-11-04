"""Small bounded tool loop for existing services/structured calls, not a new Agent."""
from copy import deepcopy
import json
from sage_scholar.runtime.control import check_control, ResearchBudgetReached, CURRENT_CONTROL
from sage_scholar.runtime.journal import WorkflowInterrupted
from sage_scholar.tools.base_tool import ToolCallError
from sage_scholar.observability import execute_tool
from sage_scholar.tools.results import is_error_result


def invoke_tools(llm, messages, tools, *, tag, max_tokens=4096, temperature=0,
                 final_schema=None, max_rounds=4, max_calls=12, images=None, finish_when=None,
                 vision_llm=None, defer_images=True, visual_state=None):
    history = deepcopy(messages)
    mapping = {t.name: t for t in tools}
    schemas = [t.to_schema() for t in tools]
    terminal = final_schema["function"]["name"] if final_schema else None
    if final_schema:
        schemas.append(final_schema)
    if images and defer_images and tools:
        history.append({"role": "user", "content":
            "原图已选定，将在工具收集完成后的最终回答阶段提供。当前先加载需要的 Skill、"
            "检索资料和定位证据；不要从图注猜测视觉细节。资料就绪时简短表示可以开始读图。"})
    used = 0
    failures = {}
    visual_state = visual_state if visual_state is not None else {}

    def visual_answer():
        check_control()
        try:
            response = (vision_llm or llm).invoke(messages=[*history, {"role": "user", "content":
                "工具资料已收集完成。现在结合所附原图与工具结果回答原问题，遵守已加载的 Skill；"
                "只对实际看到的原图描述视觉细节，不能把图注或工具摘要当作原图验证。"}],
                tools=[final_schema] if final_schema else None,
                tool_choice={"type": "function", "function": {"name": terminal}} if terminal else "none",
                max_tokens=max_tokens, temperature=temperature, tag=tag, images=images)
            check_control()
            if not terminal and (response.tool_calls or not (response.content or "").strip()):
                raise ToolCallError("最终视觉请求未返回有效答案")
            return response
        except (WorkflowInterrupted, ResearchBudgetReached):
            raise
        except Exception:
            # Keep already acquired tool observations/Skill instructions. Do not
            # restart the tool loop or pay for the same retrieval a second time.
            visual_state["failed"] = True
            response = llm.invoke(messages=[*history, {"role": "user", "content":
                "视觉模型不可用。本次仅依据已获取的文字证据回答；图像内容尚未核验，"
                "不得声称已看过原图或由图注猜测视觉细节。没有文字证据则明确资料不足。"}],
                tools=[final_schema] if final_schema else None,
                tool_choice={"type": "function", "function": {"name": terminal}} if terminal else "none",
                max_tokens=max_tokens, temperature=temperature, tag=tag)
            check_control()
            if not terminal and (response.tool_calls or not (response.content or "").strip()):
                raise ToolCallError("视觉降级后未返回有效文字答案")
            return response

    for step in range(max_rounds + 1):
        check_control()
        final = step == max_rounds or used >= max_calls or bool(finish_when and finish_when())
        available = ([final_schema] if final_schema else None) if final else schemas
        choice = {"type": "function", "function": {"name": terminal}} if final_schema and final else ("none" if final else "auto")
        if images and final:
            return visual_answer()
        # Ordinary visual QA can finish in one request. Only that initial
        # capability-discovery call may carry pixels; once tools are requested,
        # all intermediate turns use text, followed by a real visual answer.
        initial_visual = bool(images and step == 0 and (not defer_images or not tools))
        if initial_visual:
            try:
                response = (vision_llm or llm).invoke(messages=history, tools=available, tool_choice=choice,
                    max_tokens=max_tokens, temperature=temperature, tag=tag, images=images)
            except (WorkflowInterrupted, ResearchBudgetReached):
                raise
            except Exception:
                visual_state["failed"] = True
                history.append({"role": "user", "content":
                    "视觉模型不可用，仅使用文字证据；图像内容尚未核验，不得依据图注猜测视觉细节。"})
                images = None
                response = llm.invoke(messages=history, tools=available, tool_choice=choice,
                    max_tokens=max_tokens, temperature=temperature, tag=tag)
        else:
            response = llm.invoke(messages=history, tools=available, tool_choice=choice,
                max_tokens=max_tokens, temperature=temperature, tag=tag)
        check_control()
        calls = list(response.tool_calls or [])
        if not calls:
            if not (response.content or "").strip():
                raise ValueError("模型返回空回答，未将任务标记为完成")
            return visual_answer() if images and not initial_visual else response
        # A structured submission cannot precede pending tool observations in the same batch.
        if terminal and len(calls) == 1 and calls[0].function.name == terminal:
            return visual_answer() if images and not initial_visual else response
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
                content = execute_tool(tool, parameters, retry_count=failures.get(tool.name, 0))
                failures[tool.name] = failures.get(tool.name, 0) + 1 if is_error_result(content) else 0
            except (WorkflowInterrupted, ResearchBudgetReached):
                raise
            except Exception as exc:
                failures[call.function.name] = failures.get(call.function.name, 0) + 1
                content = json.dumps({"error": str(exc)}, ensure_ascii=False)
            history.append({"role": "tool", "tool_call_id": call.id, "content": content})
        if used >= max_calls or step == max_rounds - 1:
            history.append({"role": "user", "content": "工具读取已结束。用现有资料完成回答；未核实项明确标为未知，不声称核验成功。"})
    raise AssertionError("unreachable")
