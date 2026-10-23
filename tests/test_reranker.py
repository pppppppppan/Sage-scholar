from types import SimpleNamespace
import pytest
from sage_scholar.rag.document import Chunk
from sage_scholar.rag.reranker import Reranker,RerankerConfig,RerankerUnavailable


def chunks():
    return [Chunk(file_path=str(i),chunk_idx=0,content=text) for i,text in enumerate(['English passage','中文段落','English passage'])]


def test_lazy_shared_model_and_deduplicated_scores(monkeypatch):
    import sentence_transformers
    calls=[];loads=[]
    class Model:
        def __init__(self,*args,**kwargs): loads.append(kwargs)
        def predict(self,pairs,**kwargs):
            calls.append(pairs)
            return [i+1 for i in range(len(pairs))]
    monkeypatch.setattr(sentence_transformers,'CrossEncoder',Model)
    config=RerankerConfig(model='unit-shared',max_length=256,device='cpu',batch_size=2)
    a,b=Reranker(config),Reranker(config)
    assert not loads
    result=a.rerank('问题',chunks(),3)
    assert loads[0]['max_length']==256 and loads[0]['device']=='cpu'
    assert len(calls[0])==2 and result[0][0].content=='中文段落'
    b.rerank('问题',chunks(),3)
    assert len(loads)==len(calls)==1 and b.last_status['cache_hits']==3
    changed=chunks();changed[0].content='Different revision'
    b.rerank('问题',changed,3)
    assert len(calls)==2 and len(calls[1])==1


def test_failed_load_preserves_rrf_scope_and_cooldown(monkeypatch):
    import sentence_transformers
    loads=[]
    def fail(*a,**k): loads.append(1);raise OSError('weights missing')
    monkeypatch.setattr(sentence_transformers,'CrossEncoder',fail)
    reranker=Reranker(RerankerConfig(model='missing-test'))
    selected=chunks()[:2]
    assert [c for c,s in reranker.rerank('q',selected)]==selected
    reranker.rerank('q',selected)
    assert len(loads)==1 and reranker.last_status['state']=='fallback'
    with pytest.raises(RerankerUnavailable,match='weights missing'):
        Reranker(RerankerConfig(model='strict-missing',fallback='raise')).rerank('q',selected)


def test_failed_prediction_does_not_cache_or_invent_scores(monkeypatch):
    import sentence_transformers
    monkeypatch.setattr(sentence_transformers,'CrossEncoder',lambda *a,**kw:SimpleNamespace(predict=lambda *a,**kw:[float('nan')]))
    r=Reranker(RerankerConfig(model='bad-score'))
    assert r.rerank('q',chunks())[0][0].file_path=='0'
    assert not r._shared.cache
    assert r.rerank('q',[])==[]
