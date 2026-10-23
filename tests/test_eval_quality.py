import json
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path
import pytest
from evals.gates import check
from evals.dataset import load,is_labeled
from evals.runners.execution import execute


def report(value=.8,n=4,mode='live'):
    return {'samples':[{'suite':'rag','mode':mode,'metrics':{'mrr':value}} for i in range(n)]}


def test_gate_failure_missing_gold_min_samples_and_mode():
    config={'rules':[{'suite':'rag','metric':'mrr','min':.9,'min_samples':4}]}
    assert check(report(),config)['status']=='failed'
    assert check(report(1,3),config)['status']=='not_evaluated'
    assert check(report(1,4,'synthetic_index_execution'),config)['status']=='not_evaluated'
    assert check(report(1),config)['status']=='passed'
    config['rules'][0]['metric']='faithfulness'
    assert check(report(1),config)['status']=='not_evaluated'


def test_gate_baseline_and_optional_semantics():
    config={'rules':[{'suite':'rag','metric':'mrr','min_samples':4,'min_delta':-.01}]}
    current=report(.7);current['baseline_compatible']=True
    assert check(current,config,report(.9))['status']=='failed'
    current['baseline_compatible']=False
    assert check(current,config,report(.9))['status']=='not_evaluated'
    config={'rules':[{'suite':'rag','metric':'mrr','min_samples':4,'min':.5},
                     {'suite':'router','metric':'accuracy','required':False}]}
    result=check(report(),config)
    assert result['status']=='passed' and result['rules'][1]['status']=='not_evaluated'


def test_pending_gold_and_corpus_hash_validation(tmp_path):
    data=load('evals/datasets/gold/rag.jsonl')
    assert all(is_labeled(s) and s['sources'] for s in data)
    sample={'id':'pending','query':'q','dataset_version':'1','provenance':{'kind':'real_source'},'annotation':{'status':'pending'},'expected':{'answer':'unverified'}}
    assert not is_labeled(sample)
    sample.update(corpus='corpus.json',corpus_sha256='wrong')
    (tmp_path/'corpus.json').write_text('[]');p=tmp_path/'data.jsonl';p.write_text(json.dumps(sample))
    with pytest.raises(ValueError,match='hash mismatch'):load(p)


def test_cli_quality_gate_returns_nonzero_and_report(tmp_path):
    gate=tmp_path/'gate.json';gate.write_text(json.dumps({'rules':[{'suite':'rag','metric':'mrr','min':1.1,'min_samples':1,'modes':['synthetic_index_execution']}]}))
    run=subprocess.run([sys.executable,'-m','evals','run','--suite','rag','--gate',str(gate),'--output',str(tmp_path/'reports')],capture_output=True,text=True)
    assert run.returncode==2,run.stderr
    result=json.loads(next((tmp_path/'reports').glob('*.json')).read_text())
    assert result['quality_gate']['status']=='failed' and 'code_version' in result


def test_paid_modules_need_second_explicit_opt_in():
    with pytest.raises(ValueError,match='allow-paid'):
        execute('supervisor',{'query':'q'},SimpleNamespace(allow_paid=False))


def test_supervisor_executor_calls_production_module(monkeypatch):
    from sage_scholar.infrastructure.model_gateway import LegacyModelGateway
    from sage_scholar.agents.supervisor import Supervisor
    calls=[]
    fake=SimpleNamespace(total_prompt_tokens=8,total_completion_tokens=2,model='fake')
    monkeypatch.setattr(LegacyModelGateway,'client',property(lambda self:fake))
    def plan(self,request): calls.append(request);return []
    monkeypatch.setattr(Supervisor,'plan',plan)
    result=execute('supervisor',{'query':'plan a topic'},SimpleNamespace(allow_paid=True,data_dir=None))
    assert len(calls)==1 and calls[0].workflow=='deep_research'
    assert result['prompt_tokens']==8 and result['tasks']==[]


def test_citation_executor_uses_real_skill_service_path(monkeypatch):
    from sage_scholar.infrastructure.model_gateway import LegacyModelGateway
    from sage_scholar.application.direct_service import DirectService
    fake=SimpleNamespace(total_prompt_tokens=1,total_completion_tokens=2,model='fake')
    monkeypatch.setattr(LegacyModelGateway,'client',property(lambda self:fake))
    async def answer(self,request):
        self._skills.prepare(request)
        return '{"status":"unverifiable","reason":"no source","evidence_ids":[]}'
    monkeypatch.setattr(DirectService,'answer',answer)
    result=execute('citations',{'query':'check','workflow':'direct','explicit_skills':['citation-check']},SimpleNamespace(allow_paid=True,data_dir=None))
    assert result['status']=='unverifiable' and result['selected_skills']==['citation-check']


def test_empty_plan_is_not_successful_coverage():
    from evals.metrics import evaluate
    result=evaluate('supervisor', {'expected':{'max_tasks':2}}, {'tasks':[]},5)
    assert result['task_count_within_gold_limit']==0


def test_citation_gate_uses_classification_not_run_lifecycle():
    rows=[{'suite':'citations','mode':'live','evaluated':True,'sample':{'expected':{'status':'supported'}},
           'observed':{'status':'unsupported'},'metrics':{}}]
    config={'rules':[{'suite':'citations','metric':'classification.supported.recall','min':.8,'min_samples':1}]}
    assert check({'samples':rows},config)['status']=='failed'
