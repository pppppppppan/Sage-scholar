import json
from pathlib import Path
from types import SimpleNamespace
import sqlite3

import pytest

from evals.metrics import retrieval_metrics,classification,evaluate
from evals.runners.runner import run


def options(tmp_path, **kwargs):
    return SimpleNamespace(suite='all',dataset=None,data_dir=None,output=str(tmp_path/'reports'),
        baseline=None,seed=42,k=5,live=False,answers=False,judge_model=None,judge_base_url=None,**kwargs)


def test_rank_metrics_and_missing_gold():
    result=retrieval_metrics(['x','a','b'],{'a':2,'b':1},3)
    assert result['recall@3']==1 and result['precision@3']==pytest.approx(2/3)
    assert result['mrr']==.5 and 0<result['ndcg']<1
    assert retrieval_metrics(['x'],None)=={}
    assert retrieval_metrics(['x'],{})=={}


def test_classification_confusion_and_errors():
    result=classification([('supported','partial'),('partial','partial'),('unsupported','supported')])
    assert result['accuracy']==pytest.approx(1/3)
    assert result['per_label']['supported']['recall']==0
    assert result['confusion_matrix']['supported']['partial']==1


def test_semantics_never_inferred_from_citation_presence_or_task_keywords():
    result=evaluate('deep_research',{'query':'test'}, {'tasks':[{'objective':'A'},{'objective':'A'}],
        'notes':[{'source_ids':['ev_missing']}],'evidence':[]},5)
    assert result['exact_duplicate_task_rate']==.5 and result['note_source_resolvable_rate']==0
    assert not {'task_coverage','faithfulness','report_quality'} & result.keys()


def test_human_annotation_requirements_and_scores():
    sample={'expected':{}}
    observation={'annotations':{'faithfulness':[{'method':'human','annotator':'reviewer1','rationale':'Claim goes beyond the source','score':.25}]}}
    assert evaluate('rag',sample,observation,5)['faithfulness']==.25
    observation['annotations']['faithfulness'][0].pop('rationale')
    with pytest.raises(ValueError):
        evaluate('rag',sample,observation,5)


def test_all_offline_suites_run_without_models_and_write_reports(tmp_path,monkeypatch):
    import openai
    monkeypatch.setattr(openai,'OpenAI',lambda *a,**kw:pytest.fail('offline eval called model'))
    result=run(options(tmp_path))
    assert set(result['summary'])=={'rag','router','deep_research','citations','skills'}
    assert len(result['samples'])>=20
    assert not any('error' in row for row in result['samples'])
    assert list((tmp_path/'reports').glob('*.md')) and list((tmp_path/'reports').glob('*.json'))
    assert all('dataset_version' in row['sample'] for row in result['samples'])
    assert all('answer_correctness' not in r['metrics'] for r in result['samples'])
    baseline=next((tmp_path/'reports').glob('*.json'))
    args=options(tmp_path);args.baseline=str(baseline)
    again=run(args)
    assert again['baseline_compatible'] is True
    assert again['baseline_delta']['rag']['mrr']==0


def test_run_reader_is_read_only_and_recovers_durable_notes(tmp_path):
    from sage_scholar.runtime.store import RunStore
    from evals.runners.trace import read_run
    store=RunStore(tmp_path)
    snapshot={'workflow':'deep_research','result':{'report':'report'},'status':'partial'}
    with store.transaction() as db:
        db.execute('INSERT INTO runs VALUES (?,?)',('r',json.dumps(snapshot)))
    store.save_result('r','note:markdown:T1:0',{'task_id':'T1','question':'q','content':'note','source_ids':[]})
    store.save_result('r','react:T1:0',{'history':[{'role':'assistant','tool_calls':[{'function':{'name':'rag_search'}}]}]})
    store.close()
    before=(tmp_path/'runtime.sqlite3').read_bytes()
    result=read_run(tmp_path,'r')
    assert result['notes'][0]['task_id']=='T1' and result['answer']=='report'
    assert result['source_tools']==['rag_search']
    assert (tmp_path/'runtime.sqlite3').read_bytes()==before


def test_missing_prediction_is_reported_not_given_score(tmp_path):
    path=tmp_path/'sample.jsonl'
    path.write_text(json.dumps({'id':'missing','suite':'rag','query':'q','dataset_version':'1','provenance':{'kind':'human'}}))
    args=options(tmp_path);args.dataset=str(path);args.suite='rag'
    result=run(args)
    assert result['samples'][0]['error'] and result['samples'][0]['metrics']=={}


def test_baseline_mismatch_does_not_publish_misleading_deltas(tmp_path):
    first=run(options(tmp_path))
    baseline=next((tmp_path/'reports').glob('*.json'))
    args=options(tmp_path);args.k=1;args.baseline=str(baseline)
    second=run(args)
    assert not second['baseline_compatible'] and second['baseline_delta']=={}


def test_version_mismatch_refuses_to_score_old_gold(tmp_path):
    sample={'id':'version','suite':'rag','query':'q','dataset_version':'v1',
        'provenance':{'kind':'human','basis':'gold for v1'},
        'required_source_versions':[['doc_a','v1']],'gold_evidence':['old'],
        'observed':{'retrieved':['new'],'source_versions':[['doc_a','v2']]}}
    path=tmp_path/'sample.jsonl';path.write_text(json.dumps(sample))
    args=options(tmp_path);args.dataset=str(path);args.suite='rag'
    result=run(args)['samples'][0]
    assert 'version mismatch' in result['error'] and result['metrics']=={}
