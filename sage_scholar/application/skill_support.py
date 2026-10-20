"""Optional tool calling for Direct/QA using the existing model and source adapters."""
from sage_scholar.context.manager import request_context
from sage_scholar.context.tools import ContextReadTool
from sage_scholar.research.evidence_context import ResearchEvidenceContext
from sage_scholar.runtime.control import cooperative_thread
from sage_scholar.tools.calling import invoke_tools
from sage_scholar.tools.registry import ToolRegistry
from sage_scholar.tools.tool_conversation import ConversationReadTool
from sage_scholar.tools.tool_rag import RAGTool
from sage_scholar.tools.tool_sections import DocumentSectionsTool
from sage_scholar.tools.tool_pdf_visual import PDFVisualTool


class SkillSupport:
    def __init__(self, service, model, retrieval, context_manager):
        self.skills, self.model, self.retrieval, self.context_manager = service, model, retrieval, context_manager

    def prepare(self, request):
        self.skills.explicit(request)

    async def complete(self, request, messages, *, evidence, tag, images=None):
        self.prepare(request)
        context = ResearchEvidenceContext(self.retrieval.evidence_store, request.source_policy,
                                          self.retrieval.documents, self.context_manager)
        records = {item.evidence_id: item for item in evidence}

        def record(item):
            context.register(item)
            if item.evidence_id not in records:
                records[item.evidence_id] = item
                evidence.append(item)
            if self.skills.store and request.run_id:
                self.skills.store.register_evidence(request.run_id, item.evidence_id)

        for item in list(evidence):
            record(item)
        candidates = [*self.skills.tools(request),
            ConversationReadTool(request, self.skills.store, context, record),
            ContextReadTool(self.context_manager, context, self.context_manager.owner(request))]
        if request.source_policy.permits_local:
            candidates += [RAGTool(self.retrieval, request.source_policy, record),
                           DocumentSectionsTool(self.retrieval, request.source_policy, record)]
            visual = getattr(self.retrieval, "visual", None)
            if visual and visual.config.enabled:
                candidates.append(PDFVisualTool(visual, context, request.source_policy, self.model, record))
        registry = ToolRegistry()
        for tool in candidates:
            registry.register_tool(tool)
        tools = registry.for_policy(list(registry.tools), request.source_policy)
        # Direct cannot turn a Skill into web access. QA uses its existing local retrieval;
        # work requiring public searches belongs to the existing Research workflow.
        with request_context(request):
            response = await cooperative_thread(invoke_tools,
                self.model.vision_client if images else self.model.client, messages, tools,
                tag=tag, max_rounds=6 if tag == "qa" else 3, images=images)
        return response.content or ""
