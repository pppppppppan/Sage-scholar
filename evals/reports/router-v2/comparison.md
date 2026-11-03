# Router V2 实测对比（2026-11-03）

## 实验口径

仅调用本机 Ollama qwen3.5:2b，未修改模型 tag、未调用付费 API；Ollama 0.40.3，digest `dfaf2605162e689c8cac330dc4a190c4c50ad30aa47814a8fe9b79fa5127e97e`。temperature=0、think=false、num_ctx=16384、num_predict=768。
原始 144 题及 42 题开发集标签未修改，均为产品候选行为标注，未声称经过独立人工审核。full 包含开发集；holdout 是 baseline 前冻结的 42 个非 smoke 样本。只根据 smoke 调整两版提示词，最终提示词冻结后运行 full。
最初的无示例草稿在 smoke 降至 45.24%，已废弃，保留 after-smoke 记录；带通用对照示例的最终提示词独立 smoke 为 88.10%。下表 smoke 是最终 full 中的对应子集，不隐藏两次调用的差异。
baseline 在行为改造之前运行，原始代码见 baseline/source-snapshot.json。代码快照不包含 .env。逐题 JSON 包含原始输出、后处理诊断、期望标签和错误；Markdown 给出汇总。

## 结果

| 数据范围 | 指标 | 改前 | 改后 |
|---|---|---:|---:|
| full | workflow accuracy | 0.7639 | 0.8958 |
| full | macro-F1 | 0.7416 | 0.8621 |
| full | clarify precision | 0.4167 | 0.7222 |
| full | clarify recall | 0.8824 | 0.7647 |
| full | 误澄清率（非 clarify 标签） | 0.1654 | 0.0394 |
| full | source_scope_accuracy (N=127) | 0.8425 | 0.9685 |
| full | allow_web_accuracy (N=127) | 0.8898 | 0.9921 |
| full | document_ids_accuracy (N=105) | 0.8571 | 0.9810 |
| full | source_policy_accuracy (N=127) | 0.8189 | 0.9685 |
| full | 明确权限违例数 | 0.0000 | 0.0000 |
| full | p50 ms | 1132.5403 | 1894.4200 |
| full | p95 ms | 1679.9280 | 2595.0976 |
| holdout | workflow accuracy | 0.7857 | 0.8095 |
| holdout | macro-F1 | 0.7639 | 0.7734 |
| holdout | clarify precision | 0.4444 | 0.4286 |
| holdout | clarify recall | 1.0000 | 0.7500 |
| holdout | 误澄清率（非 clarify 标签） | 0.1316 | 0.1053 |
| holdout | source_scope_accuracy (N=38) | 0.8421 | 0.9474 |
| holdout | allow_web_accuracy (N=38) | 0.8947 | 0.9737 |
| holdout | document_ids_accuracy (N=34) | 0.8529 | 0.9706 |
| holdout | source_policy_accuracy (N=38) | 0.8158 | 0.9474 |
| holdout | 明确权限违例数 | 0.0000 | 0.0000 |
| holdout | p50 ms | 1062.5769 | 1839.4568 |
| holdout | p95 ms | 1672.4686 | 2543.4266 |
| smoke | workflow accuracy | 0.7381 | 0.9048 |
| smoke | macro-F1 | 0.7296 | 0.8540 |
| smoke | clarify precision | 0.4286 | 0.8000 |
| smoke | clarify recall | 1.0000 | 0.6667 |
| smoke | 误澄清率（非 clarify 标签） | 0.2222 | 0.0278 |
| smoke | source_scope_accuracy (N=36) | 0.7778 | 0.9722 |
| smoke | allow_web_accuracy (N=36) | 0.8333 | 1.0000 |
| smoke | document_ids_accuracy (N=29) | 0.7931 | 0.9655 |
| smoke | source_policy_accuracy (N=36) | 0.7500 | 0.9722 |
| smoke | 明确权限违例数 | 0.0000 | 0.0000 |
| smoke | p50 ms | 1138.9533 | 1934.7754 |
| smoke | p95 ms | 1645.9640 | 2594.3406 |

质量门禁：**passed**。要求 live workflow accuracy 和 macro-F1 不低于同数据 baseline、明确权限违例为零；至少 40 个标注样本。

## 错误与限制

以下列出最终全量集所有与候选标签不一致的样本。分组为诊断启发式；模型误判和标签争议需要人工区分，未擅自更改 Gold。输入中自然语言的权限意图依赖模型识别，不能将有限样本的零违例推广为对任意指令注入免疫。

| ID | 分组 | 不一致字段 | 期望 workflow | 实际 workflow |
|---|---|---|---|---|
| sage-router-v2-003 | 模型误判 | workflow | direct | clarify |
| sage-router-v2-005 | 模型误判 | workflow | direct | clarify |
| sage-router-v2-009 | 模型误判 | workflow | direct | clarify |
| sage-router-v2-016 | 模型误判 | workflow | direct | clarify |
| sage-router-v2-017 | 模型误判 | workflow, source_scope, allow_web | direct | deep_research |
| sage-router-v2-018 | 模型误判 | workflow | direct | clarify |
| sage-router-v2-060 | 模型误判 | workflow | qa | deep_research |
| sage-router-v2-064 | 模型误判 | workflow, source_scope | qa | direct |
| sage-router-v2-095 | 模型误判 | workflow | deep_research | qa |
| sage-router-v2-121 | 模型误判 | workflow | clarify | qa |
| sage-router-v2-122 | 模型误判 | workflow | clarify | qa |
| sage-router-v2-123 | 模型误判 | workflow | clarify | qa |
| sage-router-v2-125 | 模型误判 | workflow | clarify | deep_research |
| sage-router-v2-022 | 文档指代 | workflow, source_scope, document_ids | direct | qa |
| sage-router-v2-028 | 文档指代 | workflow, source_scope, document_ids | direct | qa |

## 置信度 A/B

Deterministic replay of identical final model predictions, confidence threshold=0.65; not a second live model test.

| 门禁 | accuracy | macro-F1 | 误澄清率 |
|---|---:|---:|---:|
| False | 0.8958 | 0.8621 | 0.0394 |
| True | 0.8958 | 0.8621 | 0.0394 |

小模型仍会自信地误判；关闭自报 confidence 门禁不等同于修复语义分类。延迟是单机串行请求观测，包含本地运行负载影响；不是受控性能基准。

## 验证与复现

全量 pytest：401 passed、5 failed；失败全部位于 MCP 生命周期测试，沙箱拒绝执行 `ps`（PermissionError），未跳过或修改其断言。最终 Router / SourcePolicy / QueryGraph / Eval 定向测试 64 passed；compileall 成功；离线 Eval quality gate passed。前端与 Research 核心算法未修改。

在项目根目录、deep-research 环境执行：
```bash
python -m pytest -q
python -m compileall -q sage_scholar tests evals
python -m evals validate --dataset evals/datasets/gold/sage_router_v2_full.jsonl
python -m evals run --suite all --gate evals/gate_configs/offline.json
python -m evals run --suite router --live --dataset /Users/pan/Desktop/1/sage-scholar/evals/datasets/gold/sage_router_v2_full.jsonl --output eval-results/router-v2/after-full --baseline evals/reports/router-v2/baseline/20261103T100302441930-router.json --gate evals/gate_configs/router-v2.json
```

本机 pytest / offline Eval 需要 `DYLD_LIBRARY_PATH=/opt/miniconda3/envs/deep-research/lib/python3.14/site-packages/torch/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false`，避免现有 FAISS/Torch OpenMP 冲突；未改安装依赖。Baseline 比较会核对数据路径及内容哈希，迁移目录后应重新建立 baseline。

## 后续建议

冻结当前提示词；优先人工审核剩余语义边界与标签争议，再用全新场景测试泛化。不要对这里已经公开的 holdout 反复调参。若进一步比较 4B/其他模型，使用相同服务参数重新成对评测准确率、内存与延迟，不假定换大模型必然提高。
