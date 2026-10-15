"""Provider ceilings, independent from per-Agent working budgets.

DeepSeek official Chat API, checked 2026-10-15:
https://api-docs.deepseek.com/quick_start/pricing/
https://api-docs.deepseek.com/api/create-chat-completion/
Custom gateways may impose lower ceilings: configure the environment overrides.
"""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelLimits:
    context: int | None
    output: int | None

    def validate_output(self, tokens):
        if tokens < 1:
            raise ValueError("输出上限必须大于零")
        if self.output is None and tokens > 8192:
            raise ValueError("该模型的长输出能力未登记，请核对提供商后设置 LLM_MAX_OUTPUT_TOKENS 和 LLM_CONTEXT_WINDOW")
        if self.output is not None and tokens > self.output:
            raise ValueError(f"请求输出 {tokens} 超过模型/网关上限 {self.output}；请调整 Agent 配置或模型能力配置")


def limits_for(model):
    known = model in {"deepseek-flash", "deepseek-v4-flash", "deepseek-v4-pro", "deepseek-v4-pro-0813"}
    context, output = (1048576, 393216) if known else (None, None)
    return ModelLimits(
        int(os.environ["LLM_CONTEXT_WINDOW"]) if os.getenv("LLM_CONTEXT_WINDOW") else context,
        int(os.environ["LLM_MAX_OUTPUT_TOKENS"]) if os.getenv("LLM_MAX_OUTPUT_TOKENS") else output)
