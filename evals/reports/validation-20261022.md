# 2026-10-22 验证记录

## 实际运行

- 后端完整 `python -m pytest -q`：372 passed，6 个依赖弃用警告，无失败/跳过。
- `python -m compileall -q sage_scholar tests evals`：通过。
- 前端 `npm --prefix web run test`：11 passed；lint、build：通过。未修改前端源码。
- `python -m evals run --suite all`：29 个离线样本，五个模块，0 执行错误。
- `git diff --check`：通过。
- 全量后端测试含真实 SQLite/FAISS、PDF 结构、导入取消/失败/进程退出恢复、来源过滤、旧版本、Evidence、QA/Research/Skills/Context/Run 回归；没有调用收费 LLM 或 MCP 服务。
- 第一次数值模型 benchmark 曾出现进程崩溃；明确固定 OMP/OpenBLAS/FAISS 单线程后完成。沙箱限制导致的 ps、前端写缓存/构建失败已在获准权限后重跑通过，未删测试或跳过断言。

## 真实本地索引迁移

`python -m sage_scholar.rag.migrate --data-dir ./data` 已运行：145 Chunk，schema 3 / revision 1，0 orphan path。

迁移前后 vector_store.json SHA256 均为：
`d5377f2dbcf9a2feceb596475cb90d3f69f9f92ed9d3907107c39ad2074f4722`。

全部 Chunk metadata 逐项相等；向量与旧值在 float32 精度内一致；SQLite integrity_check 为 ok。没有重解析 PDF、调用 Embedding 或删除旧 JSON。正式服务下次启动直接加载新索引。

## 实测性能

[最终 benchmark](benchmark-local-20261022-203506.md) / [原始 JSON](benchmark-local-20261022-203506.json)。JSON 包含实际索引/管线/重排源码 SHA256、模型配置和全部查询延迟。

同一多语言 E5 / CrossEncoder，128/1024/4096 Chunk 合成英文主题语料、6 条中英文查询。4096 Chunk 平均端到端查询 141.85 → 40.49 ms（3.50x），P95 149.34 → 50.43 ms；冷索引加载 0.236 → 0.006 s；文件总大小 50.63 → 17.44 MB。

没有隐藏负面结果：128 Chunk 的包含模型进程峰值 RSS 从约1093升至1248MB；4096 Chunk 版本切换从1.45升至1.55ms。小样本硬件波动可能显著，不能声称内存或全部操作均改善。没有真实论文金标质量、Qwen Router、视觉问答、付费 LLM Judge 或完整线上 Research 的效果承诺。离线 Fixture 分数只验证执行/指标契约。

## 修改文件

新增：

- sage_scholar/rag/persistent_store.py：SQLite/FTS5/FAISS、稳定ID、事务快照、恢复、JSON一次迁移。
- sage_scholar/rag/migrate.py：不加载模型的离线迁移/恢复CLI。
- evals/__main__.py、runners/{runner,trace,fixture,benchmark}.py、metrics/__init__.py、judges/llm.py：统一评测、只读Run观测、计算指标和可选Judge。
- evals/datasets/{rag,router,deep_research,citations,skills}/smoke.jsonl：明确标记合成Fixture。
- evals/README.md、reports/：数据格式、指标口径、运行方法和实测报告。
- tests/test_persistent_retrieval.py、tests/test_evals.py：新增回归。

修改：

- sage_scholar/rag/pipeline.py：唯一生产检索底座切换，复用现有RRF/Reranker。
- sage_scholar/rag/vector_store.py：增加路径集合接口用于恢复，原实现仅迁移/benchmark/历史测试。
- sage_scholar/library/library_manager.py：Catalog提交后同步文档版本元数据。
- tests/test_ingestion_rollback.py、tests/test_seven_structured_rag.py：在SQLite持久化上保留原回滚/重启断言。
- requirements.txt、.env.example、README.md：依赖、配置、迁移和使用说明。

## 明确边界

FAISS较大库为近似HNSW，需在真实金标论文集持续测Recall；新旧BM25词项统计不同，不能假设排序完全相同。FTS/metadata增量写入，FAISS变更仍重建快照（无重复Embedding），大规模频繁写入成本仍需关注。沿用单进程共享实例与ingestion锁，不支持多个独立后端同时写同一库。旧Run没有逐Skill开销观测时该指标缺省，不用总Token替代；语义质量没有人工gold/Judge时不伪造分数。
