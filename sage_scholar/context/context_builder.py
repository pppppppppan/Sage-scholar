"""Compatibility adapter for the migrated agents; final budgeting happens at LLMClient.invoke."""
from .history_compactor import HistoryCompactor
from .token_counter import TokenCounter


class ContextBuilder:
    def __init__(self, history_compactor: HistoryCompactor, token_counter: TokenCounter,
                 reserve_ratio=0.15, max_tokens=30000, memory_manager=None, manager=None):
        self.history_compactor, self.token_counter = history_compactor, token_counter
        self.reserve_ratio, self.max_tokens = reserve_ratio, max_tokens
        self.manager = manager

    def build_context(self, system_prompt, history):
        return [{"role": "system", "content": system_prompt}, *[msg.to_dict() for msg in history]]

    def compress_old_rounds(self, history, sub_question, system_prompt, keep_recent=1):
        # The unified manager sees tools, reserve and the full request at the invocation boundary.
        # Do not run the legacy destructive summarizer before that accounting pass.
        return
