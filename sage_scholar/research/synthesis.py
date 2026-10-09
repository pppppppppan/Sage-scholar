"""Cross-paper synthesis, hierarchically reduced at checkpoint boundaries."""
from sage_scholar.domain.research import Synthesis
from sage_scholar.context.protection import prompt_contract
from .materials import encoded, source_read
from .structured import structured_response


class ResearchSynthesis:
    def __init__(self, llm, context):
        self.llm, self.context = llm, context

    def run(self, contract, materials, *, allowed_ids, max_blocks=4):
        result = structured_response(self.llm, Synthesis, [
            {"role": "system", "content": (
                f"综合科研材料，输出最多 {max_blocks} 个分析主题，每个 analysis 约 300-600 字。"
                "任务是跨论文推理与归纳，不是换格式抄写 findings：归纳技术路线，解释共同机制与差异，"
                "比较目标函数、假设、数据、实验条件、成本和适用范围；条件不可比时禁止数字排名。"
                "明确区分已获支持的事实、跨论文归纳、推论与未知；说明冲突可能来自的实验条件。"
                "每个主题列出支持分析的 evidence_ids，原文窗口不足时保留不确定性；笔记和记忆不是原始证据。"
                "合并已有综合主题时必须保留反证、限定条件和未解决问题。不要产生过程记录或工具错误。"
            )},
            {"role": "user", "content": encoded({"contract": prompt_contract(contract),
                "materials": materials, "evidence": source_read(self.context, allowed_ids, 3000)})},
        ], tag="synthesis:combine")
        allowed = set(allowed_ids)
        if not result.blocks:
            raise ValueError("综合未返回分析主题")
        if len(result.blocks) > max_blocks:
            raise ValueError("综合主题超过本次阶段限额")
        for block in result.blocks:
            if set(block.evidence_ids) - allowed:
                raise ValueError("综合引用了输入材料之外的证据")
            if not block.evidence_ids:
                block.limitations.append("此项分析尚缺少可追溯的原文支持")
        return result
