"""Isolated old/new processes with identical corpus, query embeddings and reranker.
Legacy VectorStore is used only as a measured migration baseline, never as a live backend.
"""
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import numpy as np
from evals.metrics import retrieval_metrics
from evals.runners.fixture import embedding

TOPICS = [
 ('distillation','On-policy distillation samples student trajectories. Teacher feedback supervises token probabilities.', '学生轨迹蒸馏的教师反馈是什么？'),
 ('generalization','The experiment measures generalization to unseen boundary conditions in fluid dynamics.', '流体力学边界条件变化时的泛化实验'),
 ('formula','The objective minimizes reverse KL divergence between student and teacher distributions.', '反向 KL 散度目标函数'),
 ('table','Table 2 reports accuracy 0.91 on Dataset Alpha and 0.82 on Dataset Beta.', 'Table 2 accuracy Dataset Alpha'),
 ('limitations','Limitations include simulation cost and insufficient validation on turbulent flows.', '湍流验证不足和计算成本局限'),
 ('figure','Figure 3 shows a two-stage encoder decoder architecture with residual connections.', 'Figure 3 encoder decoder architecture'),
]


def worker(spec):
    from sage_scholar.rag.vector_store import VectorStore
    from sage_scholar.rag.persistent_store import PersistentStore
    from sage_scholar.rag.document import Chunk
    from sage_scholar.rag.pipeline import Pipeline
    import faiss
    faiss.omp_set_num_threads(1)
    data=json.loads(Path(spec).read_text())
    directory=Path(data['directory']);directory.mkdir()
    vectors=np.load(data['vectors'])
    documents=data['documents'][:data['size']]
    queries=data['queries']
    if data['models']=='local':
        os.environ['HF_HUB_OFFLINE']='1'
        os.environ['EMBEDDING_LOCAL_ONLY']='true'
        from sage_scholar.rag.local_embedding import LocalEmbedding, DEFAULT_MODEL
        from sage_scholar.rag.reranker import Reranker
        embed=LocalEmbedding(DEFAULT_MODEL)
        reranker=Reranker()
        embed.invoke(['warmup'],task='query')
        gateway=SimpleNamespace(invoke=lambda text:(embed.invoke([text],task='query')[0],0))
        reranker_name=reranker.config.model
    else:
        gateway=SimpleNamespace(invoke=lambda text:(embedding(text),0))
        def rerank(query,chunks,top_k):
            q=np.asarray(embedding(query))
            return sorted([(c,float(q @ np.asarray(c.embedding))) for c in chunks],key=lambda x:x[1],reverse=True)[:top_k]
        reranker=SimpleNamespace(rerank=rerank)
        reranker_name='synthetic-cosine-rerank-v1 (not production model)'
    # Warm tokenization in both baselines; report model initialization separately from index load.
    import jieba
    jieba.lcut('warmup')
    chunks=[Chunk(file_path=d['id'],chunk_idx=0,content=d['text'],embedding=vectors[i].tolist()) for i,d in enumerate(documents)]
    start=time.perf_counter()
    path=directory/('vector_store.json' if data['backend']=='legacy' else 'rag.sqlite3')
    store=VectorStore() if data['backend']=='legacy' else PersistentStore(path)
    store.add_chunks(chunks)
    persist=(lambda:store.save_to_json(str(path))) if data['backend']=='legacy' else store.save
    persist()
    build=time.perf_counter()-start
    if data['backend']=='new':
        store.close()
    del store
    import gc
    gc.collect()
    start=time.perf_counter()
    store=VectorStore.load_json(str(path)) if data['backend']=='legacy' else PersistentStore(path)
    cold=time.perf_counter()-start
    pipeline=Pipeline.__new__(Pipeline)
    pipeline.vector_store,pipeline.embedding,pipeline.reranker,pipeline.mqe=store,gateway,reranker,None
    latencies,raw_latencies,metrics=[],[],[]
    for q in queries:
        start=time.perf_counter()
        hits=pipeline.retrieve_chunks(q['query'],top_k=5)
        latencies.append((time.perf_counter()-start)*1000)
        gold={d['id']:1 for d in documents if d['topic']==q['topic']}
        metrics.append(retrieval_metrics([c.file_path for c,_ in hits],gold,5))
        vector=gateway.invoke(q['query'])[0]
        start=time.perf_counter();store.hybrid_search(q['query'],vector,20)
        raw_latencies.append((time.perf_counter()-start)*1000)
    # Version activation is a Catalog update plus version-scoped retrieval, not
    # re-embedding or overwriting the old version's chunks.
    from sage_scholar.domain.documents import DocumentRecord, DocumentVersion
    from sage_scholar.infrastructure.document_store import JsonDocumentStore
    catalog = JsonDocumentStore(str(directory))
    record = DocumentRecord(title='benchmark',source_identifier='fixture:benchmark',
        converted_path=chunks[0].file_path,active_version='v1',document_version='v1',versions=[
            DocumentVersion(document_version='v1',version_number=1,parser_version='fixture',converted_path=chunks[0].file_path),
            DocumentVersion(document_version='v2',version_number=2,parser_version='fixture',converted_path='new-version')])
    catalog.put(record)
    mutation={}
    for operation in ('add','replace','version_switch','delete'):
        start=time.perf_counter()
        if operation=='add':
            c=chunks[0].model_copy(update={'file_path':'new-version'})
            store.replace_file(c.file_path,[c])
        elif operation=='replace':
            c=chunks[0].model_copy(update={'content':chunks[0].content+' Revised.'})
            store.replace_file(c.file_path,[c])
        elif operation=='version_switch':
            record.active_version = record.document_version = 'v2'
            record.converted_path = 'new-version'
            catalog.put(record)
            restored = catalog.get(record.document_id)
            switched = store.hybrid_search(queries[0]['query'], chunks[0].embedding, 5,
                allowed_file_paths={restored.converted_path})
            assert switched and all(c.file_path == 'new-version' for c,_ in switched)
            assert chunks[0].file_path in store.indexed_paths()
        else:
            store.remove_by_filepath('new-version')
        if operation != 'version_switch':
            if data['backend']=='legacy':
                store.save_to_json(str(path))
            else:
                store.save()
        mutation[operation+'_ms']=(time.perf_counter()-start)*1000
    result={'backend':data['backend'],'chunks':len(chunks),'dimension':vectors.shape[1],
        'build_seconds':build,'cold_index_load_seconds':cold,'query_mean_ms':statistics.mean(latencies),
        'query_p95_ms':float(np.percentile(latencies,95)),'hybrid_mean_ms':statistics.mean(raw_latencies),
        'index_bytes':sum(p.stat().st_size for p in directory.iterdir() if p.is_file()),
        'peak_process_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024 if sys.platform=='darwin' else 1024),
        'metrics':{key:statistics.mean(m[key] for m in metrics) for key in metrics[0]},
        'mutations':mutation,'query_latencies_ms':latencies,'reranker':reranker_name,
        'faiss':getattr(faiss,'__version__','unknown'),'index_kind':type(getattr(store,'_index',None)).__name__}
    if data['backend']=='new':
        store.close()
    print(json.dumps(result))


def benchmark(args):
    sizes=[int(v) for v in args.sizes.split(',')]
    if not sizes or min(sizes)<6 or args.queries<1:
        raise ValueError('Benchmark needs sizes >=6 and queries >=1')
    np.random.seed(args.seed)
    documents=[{'id':f'study-{i}','topic':i%len(TOPICS),'text':f'Study {i}. '+TOPICS[i%len(TOPICS)][1]} for i in range(max(sizes))]
    queries=[{'query':TOPICS[i%len(TOPICS)][2], 'topic':i%len(TOPICS)} for i in range(args.queries)]
    if args.models=='local':
        os.environ['HF_HUB_OFFLINE']='1';os.environ['EMBEDDING_LOCAL_ONLY']='true'
        from sage_scholar.rag.local_embedding import LocalEmbedding, DEFAULT_MODEL
        embed=LocalEmbedding(DEFAULT_MODEL)
        vectors=embed.invoke([d['text'] for d in documents],task='passage')
        model=embed.identity
    else:
        vectors=[embedding(d['text']) for d in documents]
        model={'model':'synthetic-sha256-token-v1'}
    with tempfile.TemporaryDirectory(prefix='sage-benchmark-') as temp:
        root=Path(temp);np.save(root/'vectors.npy',np.asarray(vectors,dtype='float32'))
        results=[]
        for size in sizes:
            for backend in ('legacy','new'):
                spec={**vars(args),'size':size,'backend':backend,'documents':documents,'queries':queries,
                      'vectors':str(root/'vectors.npy'),'directory':str(root/f'{backend}-{size}')}
                path=root/'spec.json';path.write_text(json.dumps(spec))
                proc=subprocess.run([sys.executable,'-m','evals.runners.benchmark',str(path)],capture_output=True,text=True)
                if proc.returncode:
                    raise RuntimeError(proc.stderr[-5000:])
                results.append(json.loads(proc.stdout.splitlines()[-1]))
                print(f"Measured {backend}: {size} chunks",flush=True)
    report={'schema':1,'seed':args.seed,'machine':{'platform':platform.platform(),'processor':platform.processor(),'cpus':os.cpu_count()},
            'embedding':model,'queries':queries,'corpus_sha256':__import__('hashlib').sha256(json.dumps(documents).encode()).hexdigest(),
            'method':'Synthetic topic corpus with multilingual queries. Gold is all chunks in the known topic. Query timing includes query embedding, hybrid retrieval, RRF and same reranker. Corpus embedding is precomputed identically and excluded from index build. RSS includes Python/models/index; cold index load uses a fresh object (OS cache not flushed). One isolated process per backend/size.',
            'source_hashes': {str(path): __import__('hashlib').sha256(path.read_bytes()).hexdigest() for path in (Path('sage_scholar/rag/vector_store.py'),Path('sage_scholar/rag/persistent_store.py'),Path('sage_scholar/rag/pipeline.py'),Path('sage_scholar/rag/reranker.py'))},
            'results':results}
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    stem='benchmark-'+args.models+'-'+time.strftime('%Y%m%d-%H%M%S')
    (output/(stem+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2))
    lines=['# Retrieval benchmark',report['method'],json.dumps(report['machine']),json.dumps(model),
        '| Backend | Chunks | Build s | Cold s | Query mean ms | P95 ms | MB | Peak RSS MB | Recall@5 | MRR | nDCG |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['backend']} | {r['chunks']} | {r['build_seconds']:.3f} | {r['cold_index_load_seconds']:.3f} | {r['query_mean_ms']:.2f} | {r['query_p95_ms']:.2f} | {r['index_bytes']/1e6:.2f} | {r['peak_process_rss_mb']:.1f} | {r['metrics']['recall@5']:.4f} | {r['metrics']['mrr']:.4f} | {r['metrics']['ndcg']:.4f} |")
    lines += ['## Incremental operations','| Backend | Chunks | Add ms | Replace ms | Version switch ms | Delete ms |','|---|---:|---:|---:|---:|---:|']
    for r in results:
        m=r['mutations'];lines.append(f"| {r['backend']} | {r['chunks']} | {m['add_ms']:.2f} | {m['replace_ms']:.2f} | {m['version_switch_ms']:.2f} | {m['delete_ms']:.2f} |")
    lines += ['Version switch measures atomic Catalog activation plus filtered hybrid retrieval with the new version; old version chunks remain. It does not reparse or embed.', '## Quality and limitations']
    for old,new in zip(results[::2],results[1::2]):
        changes={key:new['metrics'][key]-old['metrics'][key] for key in old['metrics']}
        lines.append(f"- {old['chunks']} chunks: quality delta {changes}; query speed ratio {old['query_mean_ms']/max(new['query_mean_ms'],1e-9):.2f}x. Negative quality deltas are regressions, not omitted.")
    lines.append('These are measured synthetic-corpus results, not real-paper accuracy or production SLAs. Dense snapshots rebuild on mutations; FTS/metadata updates are incremental. Large-library write throughput remains bounded by FAISS rebuild cost.')
    (output/(stem+'.md')).write_text('\n'.join(lines)+'\n')
    print(output/(stem+'.md'))
    return report


if __name__=='__main__':
    worker(sys.argv[1])
