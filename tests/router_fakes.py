"""Deterministic classifier for orchestration tests, never a production routing fallback."""
from sage_scholar.routing.classifier import SemanticIntent


class FixedClassifier:
    def __init__(self, workflow="direct", scope=None, **fields):
        self.workflow, self.scope, self.fields = workflow, scope, fields
        self.calls = []

    def classify(self, payload):
        self.calls.append(payload)
        workflow = payload.get("requested_workflow") or self.workflow
        docs = payload.get("selected_document_ids", [])
        if workflow == "qa" and not docs and payload.get("catalog"):
            docs = [payload["catalog"][0]["document_id"]]
        scope = payload.get("requested_scope") or self.scope or (
            "selected_documents" if docs and workflow != "direct" else
            "public" if workflow == "deep_research" else "workspace" if workflow == "qa" else "conversation")
        return SemanticIntent(**{"workflow": workflow, "source_scope": scope, "document_ids": docs,
            "allow_web": scope in {"public", "mixed"} and payload.get("allow_web") is not False,
            "confidence": .99, **self.fields})
