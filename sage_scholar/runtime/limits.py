"""Workflow execution defaults and hard ceilings; these are not prompt lengths."""
import os
from pydantic import BaseModel, Field


class ExecutionLimits(BaseModel):
    max_total_tokens: int = Field(default=2_000_000, ge=4096, le=10_000_000)
    max_seconds: int = Field(default=3600, ge=1, le=21600)
    max_tool_calls: int = Field(default=240, ge=0, le=1000)
    max_steps: int = Field(default=12, ge=1, le=30)
    max_rounds: int = Field(default=4, ge=0, le=20)
    max_tasks: int = Field(default=24, ge=1, le=100)
    max_writer_revisions: int = Field(default=1, ge=0, le=3)


def execution_budget(workflow, requested=None):
    kind = str(workflow)
    defaults = {'direct': {'max_total_tokens': 50_000, 'max_seconds': 300, 'max_tool_calls': 0},
                'qa': {'max_total_tokens': 300_000, 'max_seconds': 900, 'max_tool_calls': 0},
                'citation_audit': {'max_total_tokens': 1_000_000, 'max_seconds': 1800, 'max_tool_calls': 120}}
    values = defaults.get(kind, {}).copy()
    for field in ExecutionLimits.model_fields:
        configured = os.getenv(f'SAGE_{kind.upper()}_{field.upper()}')
        if configured:
            values[field] = int(configured)
    values.update({k: v for k, v in (requested or {}).items() if k in ExecutionLimits.model_fields})
    return ExecutionLimits(**values).model_dump()
