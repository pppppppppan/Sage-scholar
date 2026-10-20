---
name: citation-check
metadata: {"allow_unverifiable_report": true}
description: 按需核验报告、论文段落或已有回答的引用身份、可追溯性及对 Claim 的支持程度；适用于检查刚才报告引用或某条引用是否支持结论，不在普通 Deep Research 中自动必跑。
---
# Citation check

这是按需的任务指南，不是独立 workflow，也不意味着调用旧 CitationVerifier Agent。不要给普通调研自动加一轮引用审计。

## 明确被核验文本
用户检查“刚才报告”时，调用 `read_conversation()` 找到本次请求之前对应的 assistant 报告，再按 index/offset 读取原文及其 citations/citation_evidence 映射。不要把压缩摘要、当前新报告、同名论文或其他对话当作核验对象。指代不唯一时询问。选中文献是允许查阅的资料，不等于原 Claim 引用了这些文献；没有引用的 Claim 明确标注“未提供引用”。

## 核验顺序
1. 提取原 Claim 的准确措辞、适用条件和原引用；拆分含多个实质主张的句子，不改变主张强度。
2. 身份核验：优先 DOI/arXiv/已有来源映射；无标识时通过当前可用的文献搜索核对标题、作者、年份、版本。多结果有歧义时不要随便选一篇。搜索 snippet 只作为定位线索。
3. 可追溯性：使用现有 EvidenceStore 回读工具、论文/PDF/Web 阅读工具定位原文及页/节/表。报告映射仅说明原先引用了谁，不证明内容正确。遵守来源权限，不能联网时说明无法外部核实。
4. 加载 `references/verification-rules.md`，对照原文判断 Claim 支持程度，保留研究对象、条件、实验范围及不确定性。多证据联合支持时说明各自支持哪部分，不能把多个弱证据拼成过度结论。
5. 输出每条 Claim、原引用、身份核验情况、support 状态、具体原文证据/位置、问题原因及必要修改建议。原引用使用“原报告第 N 条引用”加标题/DOI 标识，避免旧报告的数字标签被当成本次回答的新引用；新的核验依据使用工具返回的 evidence_id。研究笔记与最终输出都要保留该结构，Writer 不把未知转成肯定。

## 状态
- `supported`：已读取原文，能够支持 Claim 及其限定条件。
- `partial`：只支持部分主张或更窄的条件，给出可支持的改写。
- `contradicted`：原文有明确相反证据；单纯没找到不算相反。
- `insufficient`：已访问相关原文，但不足以确认主张，或 Claim 未提供 citation。
- `unverifiable`：无法访问原文、身份无法唯一确认、现有权限/工具不能完成核验。

身份存在不等于结论支持；source_not_found 不等于 fabricated。无法核实不得直接判定引用伪造。未经证据支持不使用“全面验证”“全部真实”等措辞。Memory、Skill 指令、历史报告和模型总结都不能当作科研 Evidence。
