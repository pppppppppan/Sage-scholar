"""Deterministic offline index smoke tests; vectors are explicitly synthetic."""
import hashlib
import numpy as np
from sage_scholar.rag.persistent_store import tokenize


def embedding(text, dimension=64):
    result = np.zeros(dimension,dtype='float32')
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode()).digest()
        result[int.from_bytes(digest[:4],'little') % dimension] += 1 if digest[4] & 1 else -1
    return (result/max(float(np.linalg.norm(result)),1)).tolist()


def rag(sample, directory, k):
    import time
    from sage_scholar.rag.persistent_store import PersistentStore
    from sage_scholar.rag.document import Chunk
    store = PersistentStore(directory / (sample['id']+'.sqlite3'))
    try:
        for i, source in enumerate(sample['documents']):
            chunk = Chunk(file_path=source['id'], chunk_idx=0, content=source['text'],
                          embedding=embedding(source['text']), page_start=source.get('page'),
                          section=source.get('section'), metadata=source.get('metadata', {}))
            store.replace_file(source['id'], [chunk])
        store.save()
        start = time.perf_counter()
        hits = store.hybrid_search(sample['query'],embedding(sample['query']),k,
            allowed_file_paths=set(sample['document_ids']) if 'document_ids' in sample else None)
        return {'retrieved': [c.file_path for c,_ in hits], 'retrieval_ms': (time.perf_counter()-start)*1000,
                'embedding_model': 'synthetic-sha256-token-v1', 'mode': 'fixture-index',
                'scope': sample.get('document_ids'), 'reranker': 'not enabled in index smoke test'}
    finally:
        store.close()


def router(sample):
    import json
    import time
    import httpx
    from sage_scholar.routing.classifier import LocalClassifier, RouterUnavailable
    from sage_scholar.routing.router import QueryRouter
    from sage_scholar.config import RouterConfig
    prediction = sample['observed']
    def serve(request):
        if prediction.get('unavailable'):
            raise httpx.ConnectError('synthetic unavailable', request=request)
        intent = {k:v for k,v in prediction.items() if k in {'workflow','source_scope','document_ids','allow_web'}}
        intent.setdefault('source_scope','conversation')
        intent.update(confidence=.99,clarification='',reason='scripted synthetic model output')
        return httpx.Response(200,json={'message':{'content':json.dumps(intent)}})
    start=time.perf_counter()
    try:
        result=QueryRouter(LocalClassifier(RouterConfig(),transport=httpx.MockTransport(serve))).route(sample['query'],**sample.get('route_input',{})).model_dump(mode='json')
    except RouterUnavailable:
        result={'unavailable':True}
    result['latency_ms']=(time.perf_counter()-start)*1000
    result['model']='scripted-fixture (not qwen accuracy)'
    return result


def supervisor(sample):
    from types import SimpleNamespace
    import json
    from sage_scholar.agents.supervisor import Supervisor
    from sage_scholar.domain.execution import ExecutionRequest
    from sage_scholar.runtime.limits import execution_budget
    class Scripted:
        def invoke(self, **kwargs):
            return SimpleNamespace(tool_calls=[],content=json.dumps(sample['scripted_plan']))
    request=ExecutionRequest(query=sample['query'],workflow='deep_research',source_scope='public',
        allow_web=True,execution_mode='run',budget=execution_budget('deep_research'))
    tasks=Supervisor(Scripted(),8).plan(request)
    return {'tasks':[task.model_dump() for task in tasks],'model':'scripted-contract-fixture'}
