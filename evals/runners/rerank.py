"""Same versioned candidates and gold; real models in isolated subprocesses."""
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime,timezone
from evals.dataset import load,is_labeled
from evals.metrics import retrieval_metrics


def worker(spec):
    from sage_scholar.rag.reranker import Reranker,RerankerConfig
    from sage_scholar.rag.document import Chunk
    config=RerankerConfig(model=spec['model'],device=spec['device'],local_only=True,
        fallback='raise',cache_size=0,max_length=512,batch_size=8)
    reranker=Reranker(config)
    samples=[s for s in load(spec['dataset']) if is_labeled(s)]
    started=time.perf_counter()
    first=samples[0]
    chunks=[Chunk(file_path=d['id'],chunk_idx=0,content=d['text']) for d in first['documents']]
    reranker.rerank(first['query'],chunks,spec['k'])
    cold=(time.perf_counter()-started)*1000
    results=[]
    for sample in samples:
        chunks=[Chunk(file_path=d['id'],chunk_idx=0,content=d['text']) for d in sample['documents']]
        started=time.perf_counter()
        hits=reranker.rerank(sample['query'],chunks,spec['k'])
        elapsed=(time.perf_counter()-started)*1000
        results.append({'id':sample['id'],'language':sample.get('language','unknown'),
            'candidates':[c.file_path for c in chunks],'predicted':[c.file_path for c,_ in hits],
            'gold':sample['gold_evidence'],'metrics':retrieval_metrics([c.file_path for c,_ in hits],sample['gold_evidence'],spec['k']),
            'latency_ms':elapsed})
    summary={}
    for language in ['all',*sorted({r['language'] for r in results})]:
        group=[r for r in results if language=='all' or r['language']==language]
        summary[language]={**{key:statistics.mean(r['metrics'][key] for r in group) for key in group[0]['metrics']},
            'n':len(group),'latency_ms':statistics.mean(r['latency_ms'] for r in group)}
    return {'status':'measured','config':asdict(config),'resolved_revision':reranker.last_status.get('resolved_revision'),
        'cold_load_and_first_query_ms':cold,'summary':summary,'samples':results,
        'peak_process_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024 if sys.platform=='darwin' else 1024)}


def compare(args):
    from .runner import code_version
    import hashlib
    if args.k < 1:
        raise ValueError('k must be positive')
    load(args.dataset)
    results=[]
    environment={**os.environ,'HF_HUB_OFFLINE':'1','OMP_NUM_THREADS':'1',
        'OPENBLAS_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'}
    for model in args.models:
        spec={'model':model,'device':args.device,'dataset':str(Path(args.dataset).resolve()),'k':args.k}
        try:
            proc=subprocess.run([sys.executable,'-m','evals.runners.rerank',json.dumps(spec)],
                capture_output=True,text=True,env=environment,timeout=900)
            if proc.returncode:
                raise RuntimeError(proc.stderr[-3000:])
            measured=json.loads(proc.stdout.splitlines()[-1])
            results.append(measured)
        except (ValueError,RuntimeError,subprocess.TimeoutExpired) as exc:
            results.append({'model':model,'status':'unavailable','error':str(exc)})
    data={'created_at':datetime.now(timezone.utc).isoformat(),'code_version':code_version(),
        'dataset_sha256':hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
        'method':'Identical candidate passages; strict real CrossEncoder, no RRF fallback; CPU thread count 1. Warm query scores not cached. RSS includes interpreter/model. Gold is source-checked paraphrase passages, not full PDF.',
        'results':results}
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    stem='reranker-'+time.strftime('%Y%m%d-%H%M%S')
    (output/(stem+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2))
    lines=['# Reranker comparison',data['method'],'| Model | Language | N | Recall | MRR | nDCG | Mean ms | Peak MB |','|---|---|---:|---:|---:|---:|---:|---:|']
    for row in results:
        if row['status']!='measured':
            lines.append('Unavailable: '+row['model']+' — '+row['error'])
            continue
        for lang,m in row['summary'].items():
            lines.append(f"| {row['config']['model']} | {lang} | {m['n']} | {m[f'recall@{args.k}']:.3f} | {m['mrr']:.3f} | {m['ndcg']:.3f} | {m['latency_ms']:.2f} | {row['peak_process_rss_mb']:.1f} |")
    (output/(stem+'.md')).write_text('\n'.join(lines)+'\n')
    print(output/(stem+'.md'))
    if any(r['status']!='measured' for r in results):
        raise SystemExit(1)
    return data


if __name__=='__main__':
    print(json.dumps(worker(json.loads(sys.argv[1])),ensure_ascii=False))
