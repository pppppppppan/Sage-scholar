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
from sage_scholar.tools.base_tool import BaseTool, ToolCallError


class SkillSupport:
    def __init__(self, service, model, retrieval, context_manager):
        self.skills, self.model, self.retrieval, self.context_manager = service, model, retrieval, context_manager

    def prepare(self, request):
        self.skills.explicit(request)

    def can_continue(self, request):
        """A usable source, prior report, or selected diagnostic guide warrants tools."""
        return bool(self.skills.selected(request) or request.history or (
            request.source_policy.permits_local and self.retrieval.permitted_documents(request.source_policy)))

    def allows_unverifiable(self, request):
        return self.skills.allows_unverifiable_report(request)

    async def complete(self, request, messages, *, evidence, tag, images=None, allow_visual=True, visual_reads=None):
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

        def record_visual(item):
            record(item)
            if visual_reads is not None:
                visual_reads.add(item.evidence_id)

        for item in list(evidence):
            record(item)
        initial_empty = not evidence
        supplementary = {"attempted": False}
        # Capability availability is not an image request. The model can opt in
        # after inspecting insufficient text, without an extra classification call.
        visual_mode = request.metadata.get("visual_mode", "auto") if allow_visual else "off"
        candidates = [*self.skills.tools(request),
            ConversationReadTool(request, self.skills.store, context, record),
            ContextReadTool(self.context_manager, context, self.context_manager.owner(request))]
        if request.source_policy.permits_local:
            candidates += [RAGTool(self.retrieval, request.source_policy, record, visual_mode=visual_mode),
                           DocumentSectionsTool(self.retrieval, request.source_policy, record)]
            visual = getattr(self.retrieval, "visual", None)
            if visual and visual.config.enabled and visual_mode != "off":
                candidates.append(PDFVisualTool(visual, context, request.source_policy, self.model, record_visual))
        registry = ToolRegistry()
        for tool in candidates:
            if initial_empty and tool.name in {"rag_search", "document_sections"}:
                tool = SupplementTool(tool, supplementary, lambda: bool(self.skills.selected(request)))
            registry.register_tool(tool)
        tools = registry.for_policy(list(registry.tools), request.source_policy)
        # Direct cannot turn a Skill into web access. QA uses its existing local retrieval;
        # work requiring public searches belongs to the existing Research workflow.
        with request_context(request):
            response = await cooperative_thread(invoke_tools,
                self.model.vision_client if images else self.model.client, messages, tools,
                tag=tag, max_rounds=6 if tag == "qa" else 3, images=images,
                finish_when=lambda: initial_empty and supplementary["attempted"] and not records and not self.skills.selected(request))
        return response.content or ""


class SupplementTool(BaseTool):
    """One extra retrieval for empty ordinary QA; selected Skills keep their bounded loop."""
    def __init__(self, tool, state, selected):
        super().__init__(tool.name, tool.description, tool.parameters)
        self.source_access, self.replay_policy = tool.source_access, tool.replay_policy
        self.tool, self.state, self.selected = tool, state, selected

    def run_tool(self, parameters):
        if self.state["attempted"] and not self.selected():
            raise ToolCallError("本次补充检索已完成，资料不足请明确说明")
        self.state["attempted"] = True
        return self.tool.run_tool(parameters)
