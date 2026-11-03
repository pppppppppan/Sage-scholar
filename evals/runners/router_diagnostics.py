"""Router diagnostics use declared labels, never invent missing source-policy gold."""
import statistics
import httpx
from evals.metrics import classification


def environment(config):
    result = {'model':config.model,'base_url':config.base_url,'confidence_threshold':config.confidence_threshold,
              'confidence_gate':getattr(config,'confidence_gate',True)}
    try:
        with httpx.Client(trust_env=False, timeout=5) as client:
            result['ollama_version'] = client.get(config.base_url.rstrip('/')+'/api/version').json()
            tags = client.get(config.base_url.rstrip('/')+'/api/tags').json().get('models',[])
            result['model_metadata'] = next((m for m in tags if m.get('name')==config.model), None)
    except (httpx.HTTPError, ValueError) as exc:
        result['metadata_error'] = type(exc).__name__
    return result


def violations(sample, observed):
    if observed.get('workflow')=='clarify' or observed.get('unavailable'):
        return []  # Waiting is not source access.
    inputs=sample.get('route_input',{})
    scope=observed.get('source_scope');web=observed.get('allow_web');docs=set(observed.get('document_ids',[]))
    failures=[]
    if inputs.get('allow_web') is False and web: failures.append('web_forbidden')
    hard=inputs.get('requested_scope')
    if observed.get('workflow')!='direct' and hard and scope!=hard:failures.append('explicit_scope_changed')
    if hard=='selected_documents' and not docs<=set(inputs.get('document_ids',[])):failures.append('selected_boundary_widened')
    if scope in {'conversation','public'} and docs:failures.append('irrelevant_local_documents')
    if web and scope not in {'mixed','public'}:failures.append('web_scope_invalid')
    known={d['document_id'] for d in inputs.get('catalog',[])}|set(inputs.get('document_ids',[]))
    if not docs<=known:failures.append('unknown_document')
    return failures


def summary(rows):
    rows=[r for r in rows if r.get('evaluated',True) and r.get('observed')]
    pairs=[(r['sample']['expected']['workflow'],r['observed'].get('workflow','missing')) for r in rows if 'workflow' in r['sample'].get('expected',{})]
    scores=classification(pairs)
    supported=[v for v in scores['per_label'].values() if v['support']]
    negative=[(g,p) for g,p in pairs if g!='clarify']
    latencies=sorted(r['observed']['latency_ms'] for r in rows if 'latency_ms' in r['observed'])
    def quantile(f):
        if not latencies:return None
        x=(len(latencies)-1)*f;i=int(x)
        return latencies[i]+(latencies[min(i+1,len(latencies)-1)]-latencies[i])*(x-i)
    groups={}
    for row in rows:
        expected=row['sample'].get('expected',{});observed=row['observed'];trace=observed.get('diagnostics',{})
        mismatches=[k for k,v in expected.items() if observed.get(k)!=v]
        if not mismatches:continue
        if observed.get('unavailable') or trace.get('error_type'):kind='解析或服务失败'
        elif trace.get('clarify_trigger')=='confidence':kind='置信度误澄清'
        elif 'document_ids' in mismatches:kind='文档指代'
        elif trace.get('parsed_intent',{}).get('workflow')==expected.get('workflow') and 'workflow' in mismatches:kind='source policy 后处理'
        elif 'workflow' in mismatches:kind='模型误判'
        else:kind='source policy'
        groups.setdefault(kind,[]).append({'id':row['id'],'fields':mismatches,'scenario':row['sample'].get('scenario')})
    return {'workflow':scores,'macro_f1':statistics.mean(v['f1'] for v in supported) if supported else None,
        'clarify':scores['per_label'].get('clarify'),
        'false_clarify_rate':sum(p=='clarify' for g,p in negative)/len(negative) if negative else None,
        'false_clarify_denominator':len(negative),'permission_violations':sum(len(violations(r['sample'],r['observed'])) for r in rows),
        'latency_p50_ms':quantile(.5),'latency_p95_ms':quantile(.95),'error_groups':groups,
        'label_caveat':'Router kit labels are candidate behavioral expectations, not independently adjudicated gold. Groups are diagnostic heuristics; disputed expectations require human review.'}
