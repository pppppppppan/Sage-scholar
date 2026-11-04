# Router Clarify 局部修复（2026-11-04）

## 恢复流程

仍由单次本地分类 → SourceResolver → PolicyValidator → LangGraph 执行；不新增 Agent，不改变 Research 主链。
- 普通信息补充：将澄清问题和回答成对加入 history，并保存 clarification_turns、更新后的 query；清除已消费的 human_question，恢复 Router 阶段预算后重新路由。三次澄清上限不变，不强行将尚缺信息的任务判为可执行。
- 权限冲突：PolicyValidator 区分 scope_conflict、web_forbidden 和 selected_boundary。Router 发出带当前配置的 RouterQuestion。用户在交互卡片修改资料范围、联网权限或选中文献，并勾选确认后，通过现有 respond API 的 routing_update 提交。纯文字不解除硬限制。
- RunManager 校验 conversation_id / run_id / interrupt_id / WAITING 状态及允许的更新字段，拒绝过期、重复、空改动与不属于 Router 的配置变更。持久化使用 exclude_unset=True，避免省略的 allow_web 被序列化成 null 而重置旧禁止。
- LangGraph 消费绑定回答后将 _routing 与问答一起提交到 checkpoint；下一次分类读取更新后的配置，文档身份仍须经过 SourceResolver / Catalog 检查。此前选中的分支、Research 图和 checkpoint 版本不变。
- RouterQuestion 的资料设置不出现在 Research LLM 使用的 HumanQuestion schema 中；模型不能生成可修改权限的交互协议。全局 .env 不变，更新仅对本次 Run 生效。

## 误判分析与提示词

原 V2 有四个漏澄清：无明确指代却从目录中挑两篇；把“它”的实验解释为整个库；将尚不支持的代码复现当作普通问答；综述缺少主题却开始研究。还有普通概念被判 QA 后因缺少文档触发误澄清。
最终仅修改原提示词的五处说明：必要主题/对象的前置检查、目录不是选择、通用概念与文献问答边界、未支持执行操作的替代方案、恢复时消费问答对；移除配置优先级裁决职责。保留原有示例数量，不提高 confidence 门禁、不增加 LLM 调用。
前两版尝试已保留：第一版提高 Clarify，但 QA/Research 回归；第二版 smoke 未显示足够收益。最终回到原 V2 框架做较小修改。未改标签、未硬编码具体 Query。当前全量集已用于错误分析，属于回归集，不再作为独立 holdout 泛化证明。

## 真实模型回归

基线：`../router-v2/after-full/20261103T124539232837-router.json`。同一候选行为集 144 题、同一本机 Qwen3.5:2b，模型摘要和运行配置见逐题 JSON。未调用云模型。

| 指标 | V2 | 本次 |
|---|---:|---:|
| 准确率 | 0.8958 | 0.9028 |
| 宏平均 F1 | 0.8621 | 0.8834 |
| Clarify Precision | 0.7222 | 0.7368 |
| Clarify Recall | 0.7647 | 0.8235 |
| Clarify F1 | 0.7429 | 0.7778 |
| 误澄清率 | 0.0394 | 0.0394 |
| p50 ms | 1894.4200 | 2017.5569 |
| p95 ms | 2595.0976 | 2904.7391 |

按 workflow 单独列出召回与 F1，避免总体成绩掩盖回归：

| 类别（Gold N） | V2 Recall | 本次 Recall | V2 F1 | 本次 F1 |
|---|---:|---:|---:|---:|
| clarify (17) | 0.7647 | 0.8235 | 0.7429 | 0.7778 |
| deep_research (58) | 0.9828 | 0.9310 | 0.9661 | 0.9310 |
| direct (32) | 0.7500 | 0.8750 | 0.8421 | 0.9180 |
| qa (37) | 0.9459 | 0.9189 | 0.8974 | 0.9067 |

不能据此宣称“其他类别完全无回归”：QA 比基线少判对 1 题，Deep Research 少判对 3 题。一般概念解释有所改善，但 QA 与综合研究的语义边界仍是未解决项。

现有 Router 基线门禁：**passed**。明确权限违例数：0。标签是产品候选标注，尤其“复现代码”和“深度分析 vs 论文问答”的边界仍需人工确认；未改标签凑分。单机延迟受负载影响，不是严格性能基准。

## 全部剩余 workflow 错误

| ID | 预期 | 实际 | 模型短理由 |
|---|---|---|---|
| sage-router-v2-014 | direct | clarify | 缺少明确的工作主题或目标 |
| sage-router-v2-018 | direct | clarify | 缺少明确的研究主题或对象。根据规则，当用户请求超出已有执行能力（如需要调研、总结论文等）时，应澄清。此处用户要求定义一个概念，但并未指定具体的上下文来源，因此需要确认意图。 |
| sage-router-v2-022 | direct | clarify | 缺少明确的论文指代对象，且‘泛化误差’为通用学术概念，需澄清具体来源或确认是否接受通用解释 |
| sage-router-v2-028 | direct | clarify | 缺少明确的执行意图和主题聚焦，需澄清选文范围与问题目标的关系 |
| sage-router-v2-058 | qa | deep_research | 用户请求汇总工作区中提到的物理模拟评价指标，但当前 workspace 为空（selected_document_ids=[]），且 catalog 中的文档未包含“物理模拟评价指标”相关内容。需先开展文献调研以识别相关指标来源。 |
| sage-router-v2-060 | qa | deep_research | 用户请求基于整个文献库（workspace）说明使用图神经网络的方法。由于当前没有选中的文档，且这是一个需要综合调研的主题，应启动 deep_research 工作流以开展系统综述或追踪研究进展。 |
| sage-router-v2-064 | qa | direct | 用户要求对已上传的文献进行分类，属于对已有文档内容的直接处理任务。 |
| sage-router-v2-095 | deep_research | qa | 需读取论文摘要、方法、表格、实验以总结数学原理和建模假设 |
| sage-router-v2-098 | deep_research | qa | 需读取两篇文献内容以梳理异同与可改进之处 |
| sage-router-v2-108 | deep_research | qa | 用户要求比较论文与最近两年公网方法的实验表现，需先读取选中的本地论文以获取其实验细节，再结合公网信息进行对比分析。 |
| sage-router-v2-121 | clarify | qa | 用户请求比较两篇论文，且已明确选中了这两篇文献（paper_a 和 paper_b），属于需要读取并分析文献内容的场景。 |
| sage-router-v2-125 | clarify | deep_research | 用户请求进行相关工作综述，属于需要开展文献调研、系统综述的工作类型。由于当前没有指定具体主题或选定论文，且允许使用公网资源，因此判定为公开领域的深度研究工作。 |
| sage-router-v2-141 | deep_research | clarify | 缺少明确的科研主题或任务目标，且上一条消息为系统指令而非用户请求，无法直接确定工作流。 |
| sage-router-v2-143 | clarify | deep_research | 用户要求对比选中的论文与最新公开论文，但明确禁止访问公网。由于不允许联网，无法获取外部最新论文信息，因此需先对当前选中文献进行深度研究分析，并基于现有知识框架进行对比讨论。 |

## 实际验证

- 定向后端测试：57 passed（Clarify、SourcePolicy、QueryGraph、Conversation、Router）。
- 全量后端测试：407 passed、5 failed。四项 MCP 生命周期测试因沙箱禁止执行 ps 失败；另一项 MCP 工具等待耗时 7.0175 秒，超过原有 7 秒断言。未删除、跳过或放宽测试，也未修改 MCP。
- 前端：13 tests passed；lint 与 build 通过。compileall 通过。
- 真实本地模型端到端恢复：3/3 通过，见 live-resume.json。真实 Router + LangGraph + SQLite，QA/Research 回答服务使用替身，不是完整科研质量测试。

```bash
python -m pytest -q tests/test_clarify_resume.py tests/test_router_policy_v2.py tests/test_semantic_query_graph.py tests/test_source_policy.py tests/test_conversation_interactions.py
python -m pytest -q
python -m compileall -q sage_scholar tests
npm --prefix web test
npm --prefix web run lint
npm --prefix web run build
python -m evals run --suite router --live --dataset /Users/pan/Desktop/1/sage-scholar/evals/datasets/gold/sage_router_v2_full.jsonl --baseline evals/reports/router-v2/after-full/20261103T124539232837-router.json --gate evals/gate_configs/router-v2.json
PYTHONPATH=.:tests python evals/reports/router-clarify/check_resume_live.py
```

本机 Python 使用 /opt/miniconda3/envs/deep-research/bin/python；测试设 DYLD_LIBRARY_PATH=/opt/miniconda3/envs/deep-research/lib/python3.14/site-packages/torch/lib、OMP_NUM_THREADS=1、OPENBLAS_NUM_THREADS=1、TOKENIZERS_PARALLELISM=false 规避已有 FAISS/Torch OpenMP 冲突。

## 变更文件

- `sage_scholar/graph/query_graph.py`：配对澄清上下文、更新路由配置、checkpoint 内继续。
- `sage_scholar/runtime/human.py`、`run_manager.py`：可信结构化设置更新、问题绑定、精确保存补丁字段。
- `sage_scholar/routing/classifier.py`：小幅语义提示词修改；`sources.py`、`policy.py`：区分权限冲突原因。
- `web/components/HumanInputCard.tsx`、`web/lib/api.ts`、`web/lib/conversations.ts`：设置确认和结构化请求。
- `tests/test_clarify_resume.py`、`tests/test_router_policy_v2.py`、`web/tests/clarify.test.mjs`：恢复、权限、重启、指代、提交与 UI 测试。
- `README.md`、本目录：操作说明、完整逐题评测与限制。

## 剩余限制

小模型仍会误解语义，不能保证所有充分补充都不再被澄清；现有三轮限制保留。已存在的旧版等待问题没有新设置控件，回答后若仍冲突会生成新卡片；如已到旧三轮上限，需要重新开始。模型给出的自然语言来源意图仍可能错误，但不能修改服务器硬权限。此次未改 Research/RAG/Context 算法，未新增常驻 Agent。

## 未采用的额外边界调整

`full-refined/` 保存了最后一次对分类边界的尝试：总体准确率 92.36%，但 Clarify Recall 降至 70.59%，因此回退，当前源码对应 `full-final/`，并非该尝试。该次命令使用相对数据集路径，现有 Eval 将其判为与基线不兼容（quality gate: not_evaluated），未将其算作门禁通过；上述人工对比核对的是同一 144 条样本。
