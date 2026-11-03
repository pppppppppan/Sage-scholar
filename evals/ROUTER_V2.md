# Router V2：语义理解、资料解析与权限校验

## 实验范围

输入来自用户提供的 Router Improvement Kit，原始 smoke 42 / full 144 JSONL 原样保留。它们是候选行为标注，不是经项目负责人逐条审核的线上 Gold。原有 `gold/router.jsonl` 不修改。

本机 RouterConfig 与 Ollama 实际报告的 tag 为 `qwen3.5:2b`（并非附件文字描述的 4B）；不更换模型。服务版本、模型 digest、参数、Prompt hash 与逐题原始输出随评测记录。仅调用本机分类服务，不调用付费模型或 Research。

开发只使用 smoke；`sage_router_v2_split.json` 在 baseline 前固定了 42 个非 smoke holdout ID。full 包含开发集，不能把全量分数称为完全独立的泛化成绩。最终从 full 结果按固定 ID 分出 holdout，不反复对 holdout 调 Prompt。

## 契约与设计

旧链路：模型同时预测 workflow / scope / IDs / allow_web，再由混合后处理覆盖；UI 选中项会覆盖 public，confidence 阈值可能制造澄清。

新链路：IntentClassifier 单次结构化语义理解 → SourceResolver 解析/检查文献身份 → PolicyValidator 执行显式权限 → 保持原 RouteDecision。没有新增 Agent、关键词主路由、付费 fallback 或 Research 图重构。

- UI 选择是潜在上下文；仅 requested_scope=selected_documents 或当前用户明确“只使用这些论文”是限制。Direct 清空文献/联网；public 不携带选中项。
- 文档来源仅来自当前目录/选择；有目录时验证存在与 ready；没有目录的轻量调用仍可接受调用方已校验的选择。应用真实请求始终传 Catalog，执行前再次 resolve。
- requested_scope 和 allow_web=False 是硬权限；True 只表示允许，不强制 Direct/本地 QA 联网。文字限制由语义模型识别，无法声称对任意 Prompt injection 完全免疫；服务器硬权限不依赖该判断。
- 缺失对象、不可用资料或互相矛盾的资料限制进入具体澄清。陌生术语不是澄清理由。
- confidence 是诊断值；默认不触发澄清，保留可配置 A/B 门禁。
- 公共参数、RouteDecision、SSE、HITL 与 checkpoint schema 不变。已经选择分支的恢复不重新路由；旧已等待澄清任务仍保留原问题，用户补充后用新 Router。

## 诊断与指标

评测通过 ContextVar 启用独立 trace，不存储在常规 API/Run 结果中，不共享并发请求的 trace。保存 raw_intent、parsed_intent、最终决策、资料解析、覆盖字段、clarify_trigger、模型配置、耗时与异常类型；trace 中对常见 key/token/email 脱敏，不声称通用 PII 识别。普通 Eval 样本本身仍含问题与历史，仅应使用允许共享的数据。

输出 workflow accuracy、macro-F1/混淆矩阵、澄清 precision/recall、非澄清 Gold 上的误澄清率、各 source 字段准确率、明确权限违例数及 p50/p95 延迟。未标注字段不评分。错误分组是帮助定位的启发式，不是根因已被证明；有争议的预期需人工裁定，不改标签凑分。

## 命令

```bash
python -m evals validate --dataset evals/datasets/gold/sage_router_v2_smoke_42.jsonl
python -m evals run --suite router --live --dataset evals/datasets/gold/sage_router_v2_smoke_42.jsonl
python -m evals run --suite router --live --dataset evals/datasets/gold/sage_router_v2_full.jsonl
ROUTER_CONFIDENCE_GATE=true python -m evals run --suite router --live --dataset evals/datasets/gold/sage_router_v2_smoke_42.jsonl
```

## 最终实测结果

详见 [完整比较报告](reports/router-v2/comparison.md)，同目录保留 baseline、两版开发实验、最终 full 的逐题 JSON/Markdown 与源码快照。

- 全量候选集 144 题：workflow 76.39% → 89.58%，macro-F1 0.7416 → 0.8621。
- 冻结 holdout 42 题：workflow 78.57% → 80.95%，macro-F1 0.7639 → 0.7734。不能将开发集的大幅提升推广到所有未见问题。
- 全量 p50：1133 → 1894 ms；p95：1680 → 2595 ms。提示词增加了延迟，保留此取舍。
- 明确权限违例：0；这不证明任意自然语言权限或指令注入都能被模型识别。
- baseline 比较门禁通过；本机实际模型为 2B，不是附件提到的 4B。
- 全量测试 401 passed / 5 failed（MCP 测试的 `ps` 被沙箱拒绝）；最终定向回归 64 passed；compileall 与默认离线 quality gate 通过。未改变失败断言，没有宣称全量全绿。

启用基线门禁（路径保持与 baseline 相同）：

```bash
python -m evals run --suite router --live \
  --dataset /Users/pan/Desktop/1/sage-scholar/evals/datasets/gold/sage_router_v2_full.jsonl \
  --baseline evals/reports/router-v2/baseline/20261103T100302441930-router.json \
  --gate evals/gate_configs/router-v2.json
```

## 本轮变更清单

- `sage_scholar/routing/classifier.py`：精简模型输出、通用对照示例、结构化解析与评测 trace。
- `sage_scholar/routing/router.py`、新增 `sources.py` / `policy.py`：单模型调用、身份解析、硬权限验证。
- 新增 `routing/diagnostics.py`：按调用启用且隔离的脱敏诊断。
- `sage_scholar/config.py`、`.env.example`：默认关闭 confidence 门禁，可选 A/B。
- `sage_scholar/graph/query_graph.py`：仅校验实际使用的文档，避免无关 UI 选择阻断 public 任务；checkpoint/HITL schema 不变。
- `evals/dataset.py`、`metrics/__init__.py`、`gates.py`、`runners/runner.py` / `fixture.py`、新增 `runners/router_diagnostics.py`：候选标签检查、真实诊断、误澄清/权限/延迟指标和基线门禁。
- `evals/datasets/gold/sage_router_v2_*`、`gate_configs/router-v2.json`、`reports/router-v2/`：冻结数据、门禁及实测记录。
- `tests/test_router_policy_v2.py`、`tests/test_semantic_query_graph.py`、`.github/workflows/quality.yml`：权限/解析/图恢复测试和离线 CI。
- `README.md`、`evals/README.md`、本说明：实际语义、配置、限制和运行方法。

此前尚未提交的 BGE 默认模型修改保持原样，不属于本轮 Router 改造。未修改 Research、RAG、Memory、RunManager 核心算法，也未修改原有 gold/router.jsonl。
