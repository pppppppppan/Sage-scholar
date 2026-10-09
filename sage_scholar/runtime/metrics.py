"""Small persisted stage counters; cached-input unknown is not a cache miss."""
import time
from sage_scholar.tools.base_tool import BaseTool


def model_stage(tag):
    if tag.endswith((":react", ":note", ":forced_summary")):
        return "research"
    return tag.split(":")[0] or "model"


def record_stage(stage, **values):
    from .control import CURRENT_CONTROL
    control = CURRENT_CONTROL.get()
    if control and control.store:
        control.store.record_stage(control.request.run_id, stage, **values)


class MeasuredTool(BaseTool):
    def __init__(self, tool):
        super().__init__(tool.name, tool.description, tool.parameters)
        self.tool = tool
        self.source_access = tool.source_access

    def to_schema(self):
        return self.tool.to_schema()

    def run_tool(self, parameters):
        from sage_scholar.tools.results import is_error_result
        start, failed = time.monotonic(), True
        try:
            result = self.tool.run_tool(parameters)
            failed = is_error_result(result)
            return result
        finally:
            record_stage("tool:" + self.name, seconds=time.monotonic() - start, failed=failed)
