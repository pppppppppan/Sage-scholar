# Deep Research

1. The Router identifies a research request; deterministic source resolution and policy checks establish local/public/mixed boundaries.
2. Supervisor dynamically plans a minimal set of complementary, self-contained questions. Simple research can use one Researcher.
3. LangGraph Send runs independent Researchers. Each owns its ReAct history and uses shared tools/Evidence infrastructure; ContextManager selects task-relevant content.
4. Researchers write Markdown ResearchNotes with task identity, question, findings, source IDs and uncertainties. Notes and committed tool observations survive checkpoint recovery.
5. Supervisor Review considers coverage and important gaps, requesting bounded follow-up work where needed. It does not demand mechanical citation verification of every report.
6. Writer consumes notes, source references and writing requirements and generates one full Markdown report. Uncertainty and missing coverage remain visible.

RunManager persists lifecycle, budget, events and outcomes. LangGraph persists graph progress. Cancel stops new scheduling and cooperatively stops in-flight work; closing SSE or switching conversation does not cancel the Run. Resume reuses compatible checkpoints and confirmed outputs, without claiming exactly-once external tool execution.

Context limits, total Run token/tool/time budgets and source sufficiency are separate constraints. A partial report is not proof all goals were met. Skills such as `citation-check` can be loaded on demand; source existence is not equivalent to claim support.

See [architecture](architecture.md), [configuration](configuration.md), and the detailed historical sections in [implementation notes](implementation-notes.md).
