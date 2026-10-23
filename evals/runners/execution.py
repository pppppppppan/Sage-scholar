"""Explicit opt-in calls to production services, with isolated Run/Skill state."""
import asyncio
import json
import tempfile
import time
import uuid
import json_repair

from sage_scholar.config import Config
from sage_scholar.domain.execution import ExecutionRequest
from sage_scholar.runtime.limits import execution_budget
from sage_scholar.runtime.store import RunStore
from sage_scholar.runtime.control import RunControl, run_control
from sage_scholar.context.manager import ContextManager, request_context
from sage_scholar.infrastructure.model_gateway import LegacyModelGateway
from sage_scholar.infrastructure.retrieval import LegacyRetrievalAdapter
from sage_scholar.skills import SkillService
from sage_scholar.application.skill_support import SkillSupport


class ObservedClient:
    def __init__(self, client):
        self.client = client
        self.messages = []
        self.calls = {}
        self.results = set()

    def __getattr__(self, name):
        return getattr(self.client, name)

    def invoke(self, *args, **kwargs):
        messages = kwargs.get('messages', args[0] if args else [])
        for m in messages:
            if m.get('role') == 'tool':
                self.results.add(m.get('tool_call_id'))
        response = self.client.invoke(*args, **kwargs)
        for call in response.tool_calls or []:
            self.calls[call.id] = call.function.name
        return response


def execute(suite, sample, args, retrieval=None, *, skills_enabled=True):
    if not getattr(args,'allow_paid',False):
        raise ValueError('Supervisor/Skills/Citation model execution requires --allow-paid as well as --live')
    with tempfile.TemporaryDirectory(prefix='sage-eval-run-') as directory:
        store = RunStore(directory)
        context = ContextManager(directory)
        context.run_store = store
        service = SkillService(store)
        context.skills = service if skills_enabled else None
        config = Config(data_dir=args.data_dir or directory)
        gateway = LegacyModelGateway(config,context)
        client = ObservedClient(gateway.client)
        gateway._client = client
        retrieval = retrieval or LegacyRetrievalAdapter(config)
        support = SkillSupport(service,gateway,retrieval,context) if skills_enabled else None
        workflow = 'deep_research' if suite=='supervisor' else sample.get('workflow','qa')
        policy = sample.get('policy', {'source_scope':'conversation','allow_web':False})
        query = sample['query']
        if suite=='citations':
            query += '\n请执行引用检查，最终仅输出JSON对象，包含 status（supported/partial/contradicted/insufficient/unverifiable）、reason、evidence_ids。'
        request = ExecutionRequest(query=query,workflow=workflow,execution_mode='run',**policy,
            run_id='eval-'+uuid.uuid4().hex,history=sample.get('history',[]),
            budget=execution_budget(workflow,sample.get('budget')),
            metadata={'skills':sample.get('explicit_skills',[]), 'visual_mode':sample.get('visual_mode','off')})
        started = time.perf_counter()
        try:
            with request_context(request), run_control(RunControl(request,store,lambda:False)):
                if suite=='supervisor':
                    from sage_scholar.agents.supervisor import Supervisor
                    tasks = Supervisor(client,config.max_steps,skills=service).plan(request)
                    observed = {'tasks':[t.model_dump() for t in tasks]}
                else:
                    from sage_scholar.application.direct_service import DirectService
                    from sage_scholar.application.qa_service import QAService
                    if workflow=='direct':
                        answer = asyncio.run(DirectService(gateway,context_manager=context,skill_support=support).answer(request))
                        observed = {'answer':answer,'evidence':[]}
                    else:
                        result = asyncio.run(QAService(retrieval,gateway,context_manager=context,skill_support=support).answer(request))
                        observed = {'answer':result.answer,'evidence':[e.model_dump(mode='json') for e in result.evidence],
                                    'citations':list(result.citations.values())}
                    if suite=='citations':
                        parsed = json_repair.loads(observed['answer'])
                        if not isinstance(parsed,dict) or parsed.get('status') not in {'supported','partial','contradicted','insufficient','unverifiable'}:
                            raise ValueError('Citation Skill did not return a valid verification status')
                        observed['status'] = parsed['status']
                        observed['verification'] = parsed
            selected=service.selected(request)
            observed.update(selected_skills=[s['name'] for s in selected],
                skill_versions={s['name']:s['version'] for s in selected},
                source_tools=sorted({client.calls[k] for k in client.results if k in client.calls and client.calls[k] not in {'skill_view','skills_list'}}),
                source_tool_trace_complete=True, tool_calls=len(client.calls),
                prompt_tokens=client.total_prompt_tokens,completion_tokens=client.total_completion_tokens,
                latency_ms=(time.perf_counter()-started)*1000,model=client.model,
                source_versions=sorted({(e['document_id'],e['source_version']) for e in observed.get('evidence',[]) if e.get('document_id')}))
            return observed
        finally:
            store.close()
