# Sage Scholar 离线评测与持久化检索

Eval 不在请求链路中运行，不新增 Agent，不会自动重放 Research。默认不连接 LLM、MCP 或下载模型。

## 快速运行

在仓库根目录、项目 Python 环境中运行：

```bash
python -m pip install -r requirements.txt
python -m evals run --suite all
python -m evals run --suite rag
python -m evals run --suite router
python -m evals run --suite deep_research
python -m evals run --suite citations
python -m evals run --suite skills
python -m evals run --suite rag --baseline eval-results/上次报告.json --k 5 --seed 42
```

每次生成带时间戳的 JSON 明细与 Markdown 汇总，默认 `eval-results/`（已忽略 Git）。JSON 包含样本、原始观测、数据 SHA256、seed、Prompt 文件 SHA256、运行配置、错误与已标注样本数。存在执行异常时 CLI 返回非零；质量误判仍产出报告并显示错误样本，不把它当程序崩溃。

内置 `synthetic-v1` 数据是明确标记的合成 Fixture：

- RAG：实际执行 SQLite/FAISS/RRF，覆盖正文、公式、表格、章节、跨章节、Figure 文字定位。使用确定性 hash 向量，无模型/重排，**不是多语言模型或图片识别准确率**。
- Router：实际执行 LocalClassifier 的解析和 QueryRouter 的策略，但模型 HTTP 返回值是脚本替身。它检查路由契约；需要 `--live` 才能评估 Qwen。
- Deep Research、Citation、Skills：合成观测用于检验指标和报告，不执行这些 Agent，不代表当前 Agent 已答对。没有伪造 Token/延迟消耗，也不填充没有人工依据的语义分数。

报告的 `mode` 区分 `synthetic_index_execution`、`router_contract_fixture`、`synthetic_record`、`recorded_prediction`、`recorded_run` 和 `live`。不要把不同模式的分数合称“模型准确率”。Baseline 仅在数据 hash、K、live/answers 和 Judge 模型一致时给数值差值；模型/Prompt 改动本来就是可比较的自变量，已单独记录。

## 指标定义

| 模块 | 自动计算 | 需要人工标注 / 可选 Judge |
|---|---|---|
| RAG | Recall@K、Precision@K、MRR、nDCG、检索延迟、答案精确匹配、引用 ID precision | Answer Correctness、Faithfulness、Citation Accuracy |
| Router | workflow / source_scope / allow_web / document_ids / referent 精确匹配，混淆矩阵，各类 P/R/F1，模型不可用行为，实测延迟/Token | 无隐含语义评分 |
| Deep Research | 任务数、gold 任务数上限、完全相同 objective 的重复率、笔记来源 ID 可解析率、实际 Token/工具数/活动时长/阶段用量 | Task Coverage、语义 Task Redundancy、Evidence/Note/Report Quality |
| Citations | supported / partial / contradicted / insufficient / unverifiable 分类 P/R/F1、混淆矩阵、误判 | gold 来自人工核对原文；不由“有 citation”推导支持性 |
| Skills | 已加载 Skill 集合、必需来源工具覆盖率、已记录加载开销、同输入人工 A/B 质量差 | Skill 的答案质量增益，来源内容是否真正支持结论 |

排序指标先去重，再取前 K：Recall=命中数/gold 正例数；Precision=命中数/K（不足 K 仍以 K 为分母）；MRR=首个正例倒数排名；nDCG 使用 `2^relevance-1` 与 `log2(rank+1)`。没有 gold 或没有 gold 正例时不计算这些指标。未标注文档按不相关处理，故真实实验需尽量完整标注。没有语义标注时相关指标缺省，绝不以关键词/格式检查冒充事实正确率。

`exact_duplicate_task_rate` 只测完全重复，不能衡量语义重复；`note_source_resolvable_rate` 只测 ID 能回到原文，不能证明支持。任务数量合规也不证明覆盖充分。`task_redundancy` 越低越好，其他质量分通常越高越好；以标注 rationale 为准。

成本支持样本 `pricing_per_million: {"input": 数值, "output": 数值}`，输出 `estimated_cost`；同一货币单位由标注者规定，缓存价格不同需提供真实 `observed.cost`，本工具不推测供应商账单。缺失用量不当作零。Skills 的精确 Token/耗时只有在观测包含 `skill_overhead_tokens/skill_overhead_ms` 时统计，已有 Run 未单独记录的部分保留未测，不用整轮 Token 假冒 Skill 开销。

## 用自己的 PDF 评估文本 / 图片 RAG

PDF 仍从文献库正常导入，不由 Eval 再解析。使用真实 `document_id` 和对应版本的 Evidence ID 创建 JSONL（每行一个对象）：

```json
{"id":"my-paper-figure-2","suite":"rag","dataset_version":"paper-v1-labels-1","query":"图2的曲线说明什么？","provenance":{"kind":"human","basis":"人工核对PDF第4页Figure 2"},"source_versions":{"doc_实际ID":"v1"},"required_source_versions":[["doc_实际ID","v1"]],"policy":{"source_scope":"selected_documents","document_ids":["doc_实际ID"],"allow_web":false},"visual_mode":"auto","gold_evidence":{"ev_实际ID":2},"expected":{"citations":["ev_实际ID"]}}
```

```bash
# 显式使用当前 Embedding、Reranker 和已有视觉索引；不会调用答案生成模型
python -m evals run --suite rag --live --dataset /path/my-rag.jsonl --data-dir ./data
# 如需测答案与原图理解，明确允许真实 QA/视觉模型调用（可能收费）
python -m evals run --suite rag --live --answers --dataset /path/my-rag.jsonl --data-dir ./data
```

首次可省略 gold，查看报告中的实际 `evidence`，再人工标注。图片样本 gold 应指向真实视觉 Evidence，不能只标 Caption 的文本 ID。`auto` 沿用应用的文本/视觉融合，不改图像预算和 SourcePolicy；`--answers` 沿用现有视觉读图与失败降级，会记录 warnings / 实际模型。默认模型可能需要本地缓存；远程 Embedding 或 `--answers` 可能产生 API 费用，均需显式 `--live`。本轮性能 benchmark 使用文本索引，**没有声称测过真实 PDF 像素问答质量**。

`source_versions` 是数据标注信息；可用 `required_source_versions` 指定此次必须返回的 document/version 对，缺失或不一致会报告执行错误并不计算分数，防止把 v2 结果套用 v1 gold。若要评价“是否漏召回该文献”，省略硬版本要求，用对应版本 gold ID 的 Recall 计分。

## 评估真实本地 Router

```bash
python -m evals run --suite router --live --dataset /path/router.jsonl
```

沿用 `.env` 的 `ROUTER_MODEL/ROUTER_BASE_URL/ROUTER_TIMEOUT`，记录实际模型与阈值；不回退到云端。`route_input` 支持当前 QueryRouter 的 document_ids、requested_scope、allow_web、history 等真实参数；多轮指代用 `history` 和期望 document_ids 验收。不可用场景可在 Ollama 关闭时使用单独不可用样本，不能指望一次在线运行同时满足在线/离线场景。内置 synthetic unavailable 样本仅在离线模式模拟连接故障。

## 评估已有 Research / Skills / Citation Run

```json
{"id":"opd-run","suite":"deep_research","dataset_version":"opd-labels-1","query":"调研OPD核心原理","provenance":{"kind":"human","basis":"对照用户需求及原论文逐项审阅"},"run_id":"真实run_id","expected":{"max_tasks":1},"annotations":{"task_coverage":[{"method":"human","annotator":"reviewer-1","unit":"core-principles","score":0.75,"rationale":"已覆盖采样及KL目标，缺少训练步骤解释"}]}}
```

```bash
python -m evals run --suite deep_research --data-dir ./data --dataset /path/research.jsonl
python -m evals run --suite skills --data-dir ./data --dataset /path/skills.jsonl
python -m evals run --suite citations --data-dir ./data --dataset /path/citations.jsonl
```

Run reader 用只读 SQLite 读取 Run snapshot、SSE、预算/阶段用量、持久化 ResearchNote、Researcher ReAct history、Skills 状态及 EvidenceStore。不会恢复/重跑 Run，不会再调用搜索。报告保留完整观测，可人工复核。工具 trace 不全时不计算必需工具覆盖率；History 中的 tool call 只证明曾请求该工具，工具成功与内容质量仍需检查记录/原文。

Citation 的一个样本对应一个 Claim。因为 Skill 输出是自然语言，不硬写关键词解析器伪装语义理解：真实 Run 样本需要 `prediction: {"status":"partial","quote":"从该Run最终回答复制的核验原话"}`，其 quote 必须真实出现在报告中；`expected.status` 由人工原文核对标注。Run 的 `completed/partial` 生命周期与引用的 `supported/partial` **不同**，不会混用。也可直接给 `observed: {"status":"...", "claim":"...", "original_text":"...", "source_id":"..."}` 导入外部整理的核验结果。

Skills 样本使用 `expected.selected_skills` 和 `expected.required_source_tools`（实际工具名）。比较使用前后质量时，可传入：

```json
{"paired_quality":{"method":"human","same_input":true,"rationale":"同题同资料，仅是否加载Skill不同，逐条人工比较","with_skill":0.8,"without_skill":0.6}}
```

该结构放在 `observed` 中，两边输出和 Run ID 应同时保存以便复核；不自动重跑收费 A/B。任何 0..1 语义 annotation 必须包含 human、annotator、rationale；可添加 unit、Evidence ID、quote，保留在 JSON。没有真实测量就不要填写数值。

## 可选 LLM Judge

```bash
export EVAL_JUDGE_API_KEY='你的专用密钥'
python -m evals run --suite deep_research --dataset /path/research.jsonl --data-dir ./data \
  --judge-model 模型ID --judge-base-url https://你的服务/v1
```

只有指定 `--judge-model` 才调用，显式记录 Judge 模型、完整 Prompt、输入、输出、usage，分数前缀 `llm_judge.` 与人工 gold 分开。材料不足可返回 null；超出输入限额直接报错，避免静默截断原文。Judge 是可审阅的模型意见，不是事实真值。报告包含论文/对话内容，分享前自行确认范围。

## 新旧性能 Benchmark

```bash
# 快速、完全离线、合成向量及合成重排；仅检查性能脚本
python -m evals benchmark --models fixture --sizes 128,1024,4096 --queries 12
# 真实本地已缓存模型，完整 Embedding → Hybrid → RRF → 现有 CrossEncoder
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
 python -m evals benchmark --models local --sizes 128,1024,4096 --queries 6 --output evals/reports
```

旧版调用原 VectorStore，正式服务不会使用它；新版调用 PersistentStore。两边同一语料/预计算 E5 文档向量/查询集/重排器，按 backend/size 隔离进程；查询包括 query embedding 和 rerank。构建计时不含文档 embedding（JSON 记录身份），增删替换包含持久化，版本切换测 Catalog 激活及限定版本 Hybrid 检索，旧版本保留。冷加载为新建索引对象，未清 OS 页缓存；峰值 RSS 包含模型和进程，不能解读成索引独占内存。每组仅 6 查询，P95 不是生产 SLA。

实测报告在 `reports/benchmark-local-*.{json,md}`，完整原始延迟也在 JSON。查询是中英文混合、语料为已标注主题的合成英文段落，不是真实论文金标集。gold 是该主题所有段落，因此大库 Recall@5 数值自然低；MRR/nDCG 衡量前列排序。FAISS/FTS 的词项统计和旧 BM25 的零分候选处理不同，会改变质量，所有增减均报告。

当前 FTS/metadata 按受影响 Chunk 增量更新；FAISS 变更重建快照，以可靠原子发布换取写入成本。读请求不重建。小库 Flat 精确、大库 HNSW 近似，后者可能有召回损失；未来是否优化批量写入由真实文献规模 benchmark 决定。
