"""Review argument and important evidence together, once per section, no Claim agent."""
from sage_scholar.domain.research import ReportReview
from sage_scholar.context.protection import prompt_contract
from .materials import encoded
from .references import check_section, citation_numbers, reference_material
from .structured import structured_response


class ReportReviewer:
    def __init__(self, llm, context):
        self.llm, self.context = llm, context

    def review(self, contract, outline, section, references, siblings):
        deterministic = check_section(section, references, self.context)
        if deterministic:
            return ReportReview(acceptable=False, issues=deterministic,
                                revision_instructions="；".join(deterministic))
        ids = [references[n] for n in citation_numbers(section.markdown)]
        result = structured_response(self.llm, ReportReview, [
            {"role": "system", "content": (
                "审查完整章节的论证质量及重要结论与引用的支持关系，不逐句生成 Claim 列表。"
                "判断是否回答章节目标、与全篇结构衔接、存在跨论文综合而非流水账、重复空洞或内部信息泄漏。"
                "优先检查关键公式、实验数字、因果和优劣判断，核对来源与限定条件；不能用二手摘要冒充完整原文。"
                "引用窗口截断时不能假定未展示内容支持结论；要求收窄或明确局限。"
                "对重大错误返回具体句子/表格及修订建议；一般不确定性如已明确披露可以接受。"
                "acceptability 不取决于篇幅与引用数量。acceptable=true 时 issues 为空；"
                "修订仅限当前章节，不要求重做全文或无限检索。不要输出大段原文或技术异常。"
                "若核心结论缺少可取得的原文细节，可在 source_request 提出一个具体补读目标；"
                "普通局限或措辞问题留空，不要求无目标搜索。"
            )},
            {"role": "user", "content": encoded({"contract": prompt_contract(contract),
                "outline": outline.model_dump(), "section": section.model_dump(),
                "other_section_openings": {key: value.markdown[:300] for key, value in siblings.items()
                                           if key != section.section_id},
                "sources": reference_material(self.context, references, ids, tokens=6500)})},
        ], tag="report_review:section")
        if result.issues or result.source_request:
            result.acceptable = False
        return result
