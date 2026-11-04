# Agent Observability

## 接入边界

现有 SQLite `run_budget` / `stage_usage`、Journal、SSE 和 Checkpoint 保持原职责。Trace 是它们的性能诊断视图，不回写预算或接管业务恢复。`evals/runners/trace.py` 继续从 SQLite 离线读取事实，评测不依赖 Langfuse 可用性。

代码主要使用自定义 `LLMClient.invoke()`、同步 MCP 适配器和普通 LangGraph Python 节点，非 LangChain ChatModel。自动 LangChain Callback 会收集大量状态/输入且不能覆盖这些供应商调用，因此没有同时安装自动 Callback/OpenAI instrumentor。采用官方 OTel Context 传播和 OTLP 导出，在明确边界手动创建 Span：

```
Agent.Run (trace 与 run_id 关联)
├── router → LLM.router:classify
├── direct / qa → LLM / Tool
└── deep_research
    ├── Supervisor → LLM / Skill Tool
    ├── Researcher [task_id=T1] → LLM / Tool → MCP.transport
    ├── Researcher [task_id=T2] → LLM / Tool
    ├── Supervisor.Review → LLM
    └── Writer → LLM
Agent.Run.Resume (同 Trace，指向首个 Run Span，并 Link 上次执行)
Lifecycle.run_cancelled (在等待用户期间取消，也关联原 Trace)
```

`asyncio.to_thread`、事件循环回调和 LangGraph 的并发执行沿用原生 contextvars 传播；每个 Researcher 在自身节点中新建 Span，不共享其当前 Span。已使用真实 LangGraph `Send` 并行执行验证子 LLM 的 parent ID。

## 安装与连接 Langfuse

```bash
cd /Users/pan/Desktop/1/sage-scholar
conda activate deep-research
python -m pip install -r requirements-observability.txt
```

本次核对并测试的版本：OTel API/SDK/OTLP HTTP Exporter **1.44.0**，现有 LangGraph **1.2.11**、OpenAI **3.16.2**、Anthropic **0.125.0**。使用 Langfuse 官方 OTLP ingestion v4 属性和 `x-langfuse-ingestion-version: 4`，没有依赖旧版 Langfuse Python `trace()/generation()` API，也不需安装 Langfuse Python SDK。

云端：在 Langfuse 建项目、生成公私钥，设置项目 region 对应 Host。EU `https://cloud.langfuse.com`，US `https://us.cloud.langfuse.com`。本地：按官方 Docker Compose 部署 Langfuse，自行选择空闲端口，例如 3001；本仓库不添加 Langfuse 的数据库或其他服务。自托管实例需要支持官方 OTLP ingestion v4。

`.env`：

```dotenv
OBSERVABILITY_ENABLED=true
LANGFUSE_BASE_URL=http://localhost:3001
LANGFUSE_PUBLIC_KEY=pk-lf-your-key
LANGFUSE_SECRET_KEY=sk-lf-your-key
OBSERVABILITY_QUEUE_SIZE=512
OBSERVABILITY_BATCH_SIZE=64
OBSERVABILITY_DELAY_MS=2000
OBSERVABILITY_TIMEOUT_SECONDS=3
OBSERVABILITY_MODEL_PRICES_JSON={}
```

请求实际发到 `{LANGFUSE_BASE_URL}/api/public/otel/v1/traces`，使用项目公私钥组成 Basic Authorization。Host 不要填完整的 `/v1/traces` 路径。不要把密钥提交到 Git。修改配置后重启后端；关闭仅需 `OBSERVABILITY_ENABLED=false`。

价格 JSON 按**精确模型 ID**索引，单位 USD / 百万 Token；每个条目需要 `input` 和 `output`，可选 `cache_read`、`cache_write`。例如 `{"your-model-id":{"input":1.0,"output":2.0,"cache_read":0.1}}` 仅展示格式，**不是任何真实模型报价**。请按自己的供应商/网关合同填入。未配置价格或缺少计算所需缓存用量时，不提交估算费用；没有内置单价。

## 字段与统计口径

| 字段 | 含义 |
| --- | --- |
| `langfuse.trace.metadata.run_id` / `sage.run_id` | 原有 Run ID；Trace ID 由 Run ID 稳定派生 |
| `sage.workflow` / `sage.task_id` | 确定后的 workflow / 子任务标识，不含子问题正文 |
| `sage.attempt` | Run 的执行次数，包括中断恢复；不表示模型重试 |
| Span duration | 该节点/模型操作/工具操作的墙钟耗时；父节点包含子节点时间，不相加当总耗时 |
| `sage.run.wall_seconds` | 从 SQLite created_at 至本次执行结束，包含排队和等待用户；区别于预算 active_seconds |
| `sage.llm.provider_seconds` | 已有 provider 调用计时，不含 Context prepare；模型 Span 本身还可能包括 Context 压缩子调用 |
| `sage.llm.ttft_seconds` | 流式调用首个非空文本 delta 延迟；不为非流式调用虚构 TTFT |
| `gen_ai.usage.input_tokens/output_tokens` | 总输入/输出 Token，包含供应商报告的输入缓存 Token |
| `sage.usage.source` | `provider` / `estimated` / `unknown`；中断流缺失 Usage 时保留本地估算标记 |
| `langfuse.observation.usage_details` | generation 叶节点用量；input 与 cache_read/cache_write 互斥，避免重复计费 |
| `sage.cost.source` | `provider`（明确 cost_usd）、`configured_estimate` 或 `unknown`；未报告的账单不可算成 0 |
| `sage.retry_count` | 模型 SDK request header 或工具连续失败重试计数，具体定义见 source 字段 |
| `provider.response` 事件 | 仅 HTTP 状态码与到响应头的耗时；不含 URL/headers/body，能看到 429 → 200 等重试原因 |
| `sage.mcp.retry_count` | MCP 已有 borrowed 重试循环的实际次数 |
| `sage.tool.execution_count` | 真实逻辑工具派发为 1，Journal 已完成结果复用为 0 |
| `sage.tool.replayed/recovery_retry` | 结果缓存复用 / 恢复未知执行状态后的安全重试 |
| `sage.tool.result_chars` | 结果字符数，不上传正文；失败 Envelope 标记 failed |
| `sage.status` / `error.type` | completed / partial / failed / cancelled / interrupted / waiting_for_user / timeout 等及异常类；不含异常消息 |

一次 `LLMClient.invoke`（或一次 Router classify）对应一个 generation；SDK 内部自动重试不创建多个 generation，不重复提交最终 Usage。失败尝试若供应商没有返回用量，其消耗是未知的；成功响应 Usage 不代表已核实所有失败请求的账单。相邻 ReAct 轮次不是重试。

Usage 只放在 generation，不复制到 Agent/Run 父 Span。Langfuse 通过子树汇总查看 Supervisor、Researcher、Review、Writer 的时间和成本。`MCP.transport` 为普通子 Span，不是第二个逻辑 Tool；查看调用量时区分工具观察次数与 `execution_count`，缓存回放不算一次外部执行。

缓存输入计费采用互斥分量。例如输入 100、缓存命中 40、输出 20，会提交 input=60、cache_read=40、output=20，总 Token=120，不是 160。Claude cache creation 独立保存；配置对应费率后才做本地估算。模型名保持真实值，Langfuse 仍可能依据平台模型表推算费用，该推算与本地 `sage.cost.source=unknown` 应区分，均不能声称为供应商账单。

## 隐私、恢复与故障隔离

目前仅实现安全元数据模式，没有上传原始内容的开关。所有异常自动记录关闭，代码只记录类别。不会导出 query、history、Memory、ResearchNote、Evidence 正文、文献标题、URL、工具参数/结果、图片数据、请求 headers 或 API Key。异常堆栈仍按原项目日志策略保存在本地，不进入 Trace。

同一 Run 的 Trace ID、首个 Span ID、上次执行 Span ID 和 attempt 写在原有 ExecutionRequest.metadata._observability，由现有 SQLite 事件持久化。恢复创建新 Span、关联原 Trace，不复用内存 Span；等待用户时执行 Span 已结束，不占用后台工作资源。显式重新开始产生新 run_id/Trace。暂停等待期间取消使用生命周期 Span 记录。

OTel BatchSpanProcessor 使用有界队列和后台导出；队列满时可丢 Trace，不能阻塞 Agent。Exporter 异常和 SDK 初始化失败均降级，导出服务不可用不改变预算、Run/SSE 状态。关闭服务先停止业务任务，再尽力关闭 exporter，等待有上限。

Trace 是 best effort：进程被强杀、队列溢出、网络不可用可能丢失未导出的 Span（包括尚未结束的初始 Run Span）。恢复可关联原 Trace ID，但不会从 Journal 自动重放遥测，也不承诺 Langfuse 包含完整历史。排障与恢复以 SQLite/Checkpoint 为准。等待时间计入 `wall_seconds`，不要把它当模型计算时间。

## 真实链路检查

启用并重启后端后，先在页面提交一个普通问题，再提交已上传论文的 QA，最后按需执行一次正常调研。也可以调用原接口：

```bash
curl -s http://localhost:8000/api/queries \
  -H 'Content-Type: application/json' \
  -d '{"query":"解释一下梯度下降","workflow":"direct"}'
# 将返回的 run_id 替换到下面路径：
curl -s http://localhost:8000/api/runs/RUN_ID
```

这些是业务请求，可能消耗用户配置模型的 API 费用；Observability 本身不新增模型请求。查看返回的 `execution.metadata._observability.trace_id`，或在 Langfuse 按 `run_id` 元数据筛选。一般等待一个 batch delay 即可看到数据。

检查：Router generation 的模型为本地 Qwen；Direct/QA/Research 的 generation 显示实际模型；每个 Researcher 的 LLM/工具 parent 正确；工具有 duration/status；生成的 Usage 与 SQLite 阶段记录可交叉核对。让一次任务等待 Clarify、刷新/重启后回应，应继续同一 Trace；停止任务应出现 cancelled/interrupted，不再派发后续工具。用不同 run_id 对比各阶段耗时，避免把并行节点耗时求和当总时长。

如果 UI 无数据：先检查启用状态、可选 SDK、Host region、项目公私钥、后端到 Langfuse 的网络；本地 `metadata._observability` 只证明建立了关联，**不证明远端已经成功接收**。用 Langfuse UI 的实际记录完成最后验证。

## 测试

```bash
python -m pytest -q tests/test_observability.py
OBSERVABILITY_ENABLED=false python -m pytest -q
python -m compileall -q sage_scholar tests
```

CI 安装可选 OTel 依赖并运行 Mock Exporter 测试，不需要 Langfuse Key、不连接云端、不使用真实 LLM。自动化测试覆盖实际 QueryGraph 与 ResearchGraph、异步/并行 Context、SQLite/Checkpoint 多次恢复、取消、超时、SDK 重试、互斥缓存计费、隐私、有界队列、导出失败及原 SSE 事件。

本次实际结果：Observability 专项 **16 passed**；全量 **422 passed / 5 failed**。五项失败均来自 `tests/test_mcp_lifecycle.py` 中沙箱禁止执行 `ps`（PermissionError），未删除/跳过断言或修改清理逻辑。`compileall` 与 `git diff --check` 通过。配置日本区地址后，已完成真实 metadata-only smoke：OTLP 导出 SUCCESS，v2 observations API 返回 200，并读回同一 Trace 的两个 Span。测试 Trace ID：`e805129c2b92f04681cf15c214452801`。未调用付费 LLM，也未将此连通性测试视为完整业务链路的远端验收。新组织的旧 `/api/public/traces/{id}` 接口可能返回 410；读取验证改用 `/api/public/v2/observations`，带时间范围和 traceId。

主要修改：新增 `observability/{config,backend,tracing,instrumentation}.py`；在 RunManager、Container、QueryGraph、ResearchGraph、LLMClient、Router classify、streaming、工具调用入口、Journal 和 MCP 适配器添加薄接入。新增可选依赖文件、专项测试与 CI 步骤；未增加数据库表或改变预算、Checkpoint schema、Router 分类、RAG 与研究规划算法。

## 官方依据

- [Langfuse OTLP 接入和属性映射](https://langfuse.com/integrations/native/opentelemetry)
- [Langfuse Usage / Cost](https://langfuse.com/docs/observability/features/token-and-cost-tracking)
- [OpenTelemetry Python BatchSpanProcessor](https://opentelemetry-python.readthedocs.io/en/stable/sdk/trace.export.html)
- [Langfuse 本地 Docker Compose 部署](https://langfuse.com/self-hosting/deployment/docker-compose)
