import hashlib
import json
import platform
import random
from pathlib import Path
import statistics
import tempfile
import time
from datetime import datetime, timezone

from evals.metrics import evaluate, classification
from .trace import read_run

SUITES = ('rag','router','deep_research','citations','skills')


def run(args):
    random.seed(args.seed)
    rows, hashes = [], {}
    suites = SUITES if args.suite == 'all' else (args.suite,)
    with tempfile.TemporaryDirectory(prefix='sage-eval-') as temp:
        for suite in suites:
            paths = [Path(args.dataset)] if args.dataset else sorted((Path(__file__).parents[1]/'datasets'/suite).glob('*.jsonl'))
            for path in paths:
                hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                for line in path.read_text().splitlines():
                    if not line.strip():
                        continue
                    sample = json.loads(line)
                    if sample.get('suite',suite) != suite:
                        continue
                    if not all(sample.get(k) for k in ('id','query','dataset_version','provenance')):
                        raise ValueError('Each sample requires id/query/dataset_version/provenance')
                    if sample['id'] in {'.', '..'} or Path(sample['id']).name != sample['id']:
                        raise ValueError('Sample ID must not contain a path')
                    started = time.perf_counter()
                    row = {'suite':suite,'id':sample['id'],'sample':sample}
                    try:
                        if sample.get('run_id'):
                            if not args.data_dir:
                                raise ValueError('--data-dir required for run_id')
                            observed = read_run(args.data_dir,sample['run_id'])
                            observed['annotations'] = sample.get('annotations',{})
                            if suite == 'citations':
                                # Run lifecycle status is not a claim verification status.
                                prediction = sample.get('prediction', {})
                                if not prediction.get('status') or not prediction.get('quote'):
                                    raise ValueError('Citation Run sample requires prediction.status and verbatim prediction.quote from the recorded answer')
                                if prediction['quote'] not in observed.get('answer', ''):
                                    raise ValueError('Citation prediction quote is absent from recorded answer')
                                observed['run_status'] = observed.pop('status')
                                observed['status'] = prediction['status']
                                observed['prediction'] = prediction
                            row['mode'] = 'recorded_run'
                        elif args.live:
                            observed = live(suite,sample,args)
                            row['mode'] = 'live'
                        elif suite == 'rag' and 'documents' in sample:
                            from .fixture import rag
                            observed = rag(sample,Path(temp),args.k)
                            row['mode'] = 'synthetic_index_execution'
                        elif suite == 'router' and sample.get('provenance',{}).get('kind') == 'synthetic':
                            from .fixture import router
                            observed = router(sample)
                            row['mode'] = 'router_contract_fixture'
                        elif 'observed' in sample:
                            observed = sample['observed']
                            row['mode'] = 'synthetic_record' if sample['provenance'].get('kind') == 'synthetic' else 'recorded_prediction'
                        else:
                            raise ValueError('No recorded observations; use --live or provide run_id/observed')
                        observed.setdefault('annotations', sample.get('annotations', {}))
                        row['observed'] = observed
                        if sample.get('required_source_versions'):
                            actual = {tuple(pair) for pair in observed.get('source_versions', [])}
                            required = {tuple(pair) for pair in sample['required_source_versions']}
                            if not required <= actual:
                                raise ValueError(f'Source version mismatch or missing evidence: {sorted(required - actual)}')
                        row['metrics'] = evaluate(suite,sample,observed,args.k)
                        if args.judge_model:
                            from evals.judges.llm import judge
                            row['judge'] = judge(sample,observed,args.judge_model,args.judge_base_url)
                            row['metrics'].update({'llm_judge.'+name:value['score'] for name,value in row['judge']['output']['metrics'].items() if value['score'] is not None})
                        row['errors'] = [name for name,value in row['metrics'].items()
                                         if (name.endswith('accuracy') or name.endswith('exact_match') or name.endswith('within_gold_limit')) and value < 1]
                    except Exception as exc:
                        row['error'] = f'{type(exc).__name__}: {exc}'
                        row['metrics'] = {}
                    row['evaluation_ms'] = (time.perf_counter()-started)*1000
                    rows.append(row)
    summary = {}
    for suite in suites:
        items = [r for r in rows if r['suite']==suite]
        metric_names = sorted({k for r in items for k in r['metrics']})
        means = {key: {'mean':statistics.mean(r['metrics'][key] for r in items if key in r['metrics']),
                       'n':sum(key in r['metrics'] for r in items)} for key in metric_names}
        summary[suite] = {'metrics':means,'samples':len(items),'execution_errors':sum('error' in r for r in items)}
        label = 'workflow' if suite=='router' else 'status' if suite=='citations' else None
        if label:
            pairs=[(r['sample']['expected'][label],r['observed'].get(label,'missing')) for r in items
                   if label in r['sample'].get('expected',{}) and 'observed' in r]
            summary[suite]['classification'] = classification(pairs)
    if not rows:
        raise ValueError('No evaluation samples found')
    report = {'schema_version':1,'created_at':datetime.now(timezone.utc).isoformat(),
        'seed':args.seed,'dataset_hashes':hashes,'platform':platform.platform(),
        'config':vars(args),'prompt_hashes':prompt_hashes(), 'summary':summary,'samples':rows,
        'interpretation':'Synthetic records test metric contracts, not current model accuracy. Missing semantic labels remain unassessed.'}
    if args.baseline:
        baseline=json.loads(Path(args.baseline).read_text())
        report['baseline_compatible'] = (baseline.get('dataset_hashes') == hashes and
            baseline.get('config', {}).get('k') == args.k and
            baseline.get('config', {}).get('live') == args.live and
            baseline.get('config', {}).get('answers') == args.answers and
            baseline.get('config', {}).get('judge_model') == args.judge_model)
        report['baseline_delta']={suite:{key:value['mean']-baseline['summary'][suite]['metrics'][key]['mean']
            for key,value in summary[suite]['metrics'].items() if key in baseline.get('summary',{}).get(suite,{}).get('metrics',{})}
            for suite in summary} if report['baseline_compatible'] else {}
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    name=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'-'+args.suite
    (output/(name+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2))
    lines=['# Sage Scholar Eval',report['interpretation'],f"Seed: {args.seed}", '']
    for suite,data in summary.items():
        lines += [f'## {suite}',f"Samples: {data['samples']}; execution errors: {data['execution_errors']}",
                  '| Metric | Mean | Labeled N |','|---|---:|---:|']
        lines += [f"| {key} | {v['mean']:.6f} | {v['n']} |" for key,v in data['metrics'].items()]
        if 'classification' in data:
            lines += ['```json',json.dumps(data['classification'],ensure_ascii=False,indent=2),'```']
    lines += ['## Error cases']+[f"- {r['suite']}/{r['id']}: {r.get('error') or r.get('errors')}" for r in rows if r.get('error') or r.get('errors')]
    if 'baseline_delta' in report:
        lines += ['## Baseline delta',f"Comparable datasets: {report['baseline_compatible']}",'```json',json.dumps(report['baseline_delta'],indent=2),'```']
    lines += ['## Unassessed semantic quality','Semantic correctness/support/coverage/redundancy require explicit human annotations or an opt-in Judge. Identity/format checks are not semantic proof.']
    (output/(name+'.md')).write_text('\n'.join(lines)+'\n')
    print(output/(name+'.md'))
    return report


def prompt_hashes():
    root=Path(__file__).parents[2]
    paths=[root/'sage_scholar/routing/classifier.py',*list((root/'sage_scholar/prompts').glob('**/*')),*list((root/'sage_scholar/agents').glob('*.py')),*list((root/'skills').glob('*/SKILL.md'))]
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def live(suite,sample,args):
    if suite=='router':
        from sage_scholar.routing.router import QueryRouter
        from sage_scholar.routing.classifier import LocalClassifier, RouterUnavailable
        from sage_scholar.config import RouterConfig
        from sage_scholar.runtime.store import RunStore
        from sage_scholar.runtime.control import RunControl, run_control
        from sage_scholar.domain.execution import ExecutionRequest
        with tempfile.TemporaryDirectory() as directory:
            ledger=RunStore(directory)
            request=ExecutionRequest(query=sample['query'],workflow='direct',source_scope='conversation',allow_web=False,run_id='eval')
            start=time.perf_counter()
            try:
                with run_control(RunControl(request,ledger,lambda:False)):
                    router_config = RouterConfig()
                    result=QueryRouter(LocalClassifier(router_config)).route(sample['query'],**sample.get('route_input',{})).model_dump(mode='json')
            except RouterUnavailable as exc:
                result={'unavailable':True,'error':str(exc)}
            result.update(ledger.budget('eval'),latency_ms=(time.perf_counter()-start)*1000,
                model=router_config.model, model_config={'timeout':router_config.timeout,'confidence_threshold':router_config.confidence_threshold})
            ledger.close()
            return result
    if suite=='rag':
        if not args.data_dir:
            raise ValueError('--data-dir required for live retrieval')
        from sage_scholar.config import Config
        from sage_scholar.infrastructure.retrieval import LegacyRetrievalAdapter
        from sage_scholar.domain.execution import SourcePolicy
        config=Config(data_dir=args.data_dir)
        adapter=LegacyRetrievalAdapter(config)
        if args.answers:
            import asyncio
            from sage_scholar.application.qa_service import QAService
            from sage_scholar.infrastructure.model_gateway import LegacyModelGateway
            from sage_scholar.context.manager import ContextManager
            from sage_scholar.domain.execution import ExecutionRequest
            gateway=LegacyModelGateway(config,ContextManager(args.data_dir))
            request=ExecutionRequest(query=sample['query'],workflow='qa',**sample['policy'],
                metadata={'visual_mode':sample.get('visual_mode','auto')})
            start=time.perf_counter()
            result=asyncio.run(QAService(adapter,gateway).answer(request))
            clients=[c for c in (gateway._client,gateway._vision_client) if c is not None]
            return {'retrieved':[e.evidence_id for e in result.evidence], 'answer':result.answer,
                'citations':list(result.citations.values()),'evidence':[e.model_dump(mode='json') for e in result.evidence],
                'source_versions': sorted({(e.document_id, e.source_version) for e in result.evidence if e.document_id}),
                'latency_ms':(time.perf_counter()-start)*1000,'models':[c.model for c in clients],
                'prompt_tokens':sum(c.total_prompt_tokens for c in clients),
                'completion_tokens':sum(c.total_completion_tokens for c in clients)}
        start=time.perf_counter()
        hits,warnings=adapter.retrieve_with_warnings(sample['query'],SourcePolicy.model_validate(sample['policy']),args.k,visual_mode=sample.get('visual_mode','auto'))
        return {'retrieved':[h.evidence_id for h in hits],'evidence':[h.model_dump(mode='json') for h in hits],
                'retrieval_ms':(time.perf_counter()-start)*1000,'warnings':warnings,
                'source_versions': sorted({(e.document_id, e.source_version) for e in hits if e.document_id})}
    raise ValueError('This suite evaluates existing Run/recorded outputs; do not replay a research workflow for eval')
