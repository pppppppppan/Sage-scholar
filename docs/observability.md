# Agent observability

OpenTelemetry exports safe structured metadata to Langfuse through OTLP/HTTP. It is optional and disabled by default. SQLite and checkpoints remain authoritative; an unavailable exporter cannot prevent research.

Set `OBSERVABILITY_ENABLED=true`, `LANGFUSE_BASE_URL` and the project's public/secret keys in `.env` (host) or `.env.docker` (Compose), then restart/recreate the backend. Use the exact regional address from Langfuse, such as `https://jp.cloud.langfuse.com`; legacy `LANGFUSE_HOST` remains supported. Optional OTel dependencies are already included in the backend Docker image. No Langfuse server is deployed by this Compose file.

Prompt text, paper contents, tool results, API keys and images are not exported. Token totals use provider usage where available; estimates are marked; missing prices remain unknown rather than zero. Model prices are explicit configuration. Async bounded queues can drop telemetry under failure; they do not replace the Run journal.

[Detailed configuration, metrics and trace validation](../evals/OBSERVABILITY.md) preserves the existing observability documentation and test results.
