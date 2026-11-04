"""Private OTel provider; no global auto-instrumentation or content capture."""
import base64
from contextvars import ContextVar

TRACE_ID = ContextVar("sage_trace_id", default=None)


def build_provider(config, exporter=None):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
    from opentelemetry.sdk.resources import Resource

    class RunIdGenerator(RandomIdGenerator):
        def generate_trace_id(self):
            return TRACE_ID.get() or super().generate_trace_id()

    class SafeExporter(SpanExporter):
        def export(self, spans):
            try:
                return target.export(spans)
            except Exception:
                return SpanExportResult.FAILURE

        def shutdown(self):
            try:
                target.shutdown()
            except Exception:
                pass

    if exporter is None:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        authorization = base64.b64encode(f"{config.public_key}:{config.secret_key}".encode()).decode()
        exporter = OTLPSpanExporter(endpoint=config.host + "/api/public/otel/v1/traces",
            headers={"Authorization": "Basic " + authorization, "x-langfuse-ingestion-version": "4"},
            timeout=config.timeout_seconds)
    target = exporter
    provider = TracerProvider(resource=Resource({"service.name": "sage-scholar"}),
                             id_generator=RunIdGenerator(), shutdown_on_exit=False)
    provider.add_span_processor(BatchSpanProcessor(SafeExporter(), max_queue_size=config.queue_size,
        max_export_batch_size=config.batch_size, schedule_delay_millis=config.delay_ms,
        export_timeout_millis=int(config.timeout_seconds * 1000)))
    return provider
