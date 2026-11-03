"""Single semantic call → source identity resolution → deterministic permissions."""
from .classifier import LocalClassifier
from .diagnostics import record
from .models import SourceScope, Workflow
from .sources import SourceResolver
from .policy import PolicyValidator


class QueryRouter:
    def __init__(self, classifier=None, *, confidence_threshold=None, confidence_gate=None):
        self.classifier = classifier or LocalClassifier()
        config = getattr(self.classifier,'config',None)
        self.confidence_threshold = confidence_threshold if confidence_threshold is not None else getattr(config,'confidence_threshold',.65)
        self.confidence_gate = confidence_gate if confidence_gate is not None else getattr(config,'confidence_gate',False)
        self.sources, self.policy = SourceResolver(), PolicyValidator()

    def route(self, query, *, requested_workflow=None, requested_scope=None, document_ids=None,
              allow_web=None, history=(), catalog=None, resolution_message=None):
        if requested_workflow is not None:
            requested_workflow = Workflow(requested_workflow)
            if requested_workflow not in {Workflow.DIRECT, Workflow.QA, Workflow.DEEP_RESEARCH}:
                raise ValueError('该独立执行类型未提供，请选择 Direct、QA 或 Deep Research')
        if requested_scope is not None:
            requested_scope = SourceScope(requested_scope)
        selected = list(dict.fromkeys(document_ids or []))
        intent = self.classifier.classify({
            'query':query,'requested_workflow':requested_workflow,'requested_scope':requested_scope,
            'selected_document_ids':selected,'allow_web':allow_web,
            'recent_conversation':[{'role':m['role'],'content':m['content'][:1500]} for m in history[-6:] if m.get('role') in {'user','assistant'}],
            'catalog':[{k:d[k] for k in ('document_id','title','ready') if k in d} for d in list(catalog or ())[:50]],
        })
        # An explicit workflow override changes the operation, not permissions. When the
        # model treated it as direct, resolve the explicitly supplied local QA context.
        effective = intent
        if requested_workflow == Workflow.QA and intent.workflow=='direct' and intent.source_scope==SourceScope.CONVERSATION:
            effective = intent.model_copy(update={'source_scope':SourceScope.SELECTED_DOCUMENTS if selected else SourceScope.WORKSPACE})
        if requested_workflow == Workflow.DIRECT or (not requested_workflow and intent.workflow=='direct'):
            effective = intent.model_copy(update={'source_scope':SourceScope.CONVERSATION})
        sources = self.sources.resolve(effective,requested_scope=requested_scope,selected=selected,
            catalog=catalog,resolution_message=resolution_message)
        decision,trigger = self.policy.validate(intent,sources,requested_workflow=requested_workflow,
            requested_scope=requested_scope,allow_web=allow_web,confidence_gate=self.confidence_gate,
            confidence_threshold=self.confidence_threshold)
        record(final_decision=decision.model_dump(mode='json'),
            source_resolution={'selected':selected,'resolved':sources.document_ids,'unknown':list(sources.unknown),
                               'scope':sources.scope,'available_local':sources.available_local,'question':sources.question},
            policy_overrides={k:{'model':getattr(intent,k,None),'final':getattr(decision,k)}
                for k in ('workflow','source_scope','allow_web','document_ids') if getattr(intent,k,None)!=getattr(decision,k)},
            clarify_trigger=trigger,confidence_gate=self.confidence_gate)
        return decision
