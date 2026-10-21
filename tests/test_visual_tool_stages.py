"""Prove pixel delivery happens at the answer boundary, not on every tool turn."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from test_pdf_visual import library, request, policy
from test_skills import setup, Model, answer, call
from test_agent_context import provider
from sage_scholar.application.qa_service import QAService
from sage_scholar.application.skill_support import SkillSupport
from sage_scholar.context.manager import request_context
from sage_scholar.runtime.control import RunControl, run_control
from sage_scholar.runtime.store import RunStore
from sage_scholar.tools.calling import invoke_tools


def run_qa(adapter, skills, manager, text, vision, req):
    gateway = SimpleNamespace(client=text, vision_client=vision)
    support = SkillSupport(skills, gateway, adapter, manager)
    return asyncio.run(QAService(adapter, gateway, skill_support=support).answer(req))


def test_ordinary_visual_qa_with_skills_available_still_one_call(library, setup):
    adapter, _, _ = library
    skills, manager, _, _ = setup
    text = Model(manager)
    vision = Model(manager, lambda kwargs: answer(f"看见曲线 [{kwargs['images'][0]['evidence_id']}]"))
    result = run_qa(adapter, skills, manager, text, vision, request())
    assert not text.calls and len(vision.calls) == 1
    assert vision.calls[0]['images'] and '看见曲线' in result.answer
    assert not skills.selected(request())


def test_auto_skill_then_many_tools_only_initial_and_final_see_pixels(library, setup):
    adapter, _, _ = library
    skills, manager, _, _ = setup
    text = Model(manager, call('skills_list'), call('document_sections'), answer('资料就绪'))
    vision = Model(manager, call('skill_view', name='paper-review'),
        lambda k: answer(f"根据原图审查 [{k['images'][0]['evidence_id']}]"))
    result = run_qa(adapter, skills, manager, text, vision, request())
    assert len(text.calls) == 3 and all(not c.get('images') for c in text.calls)
    assert len(vision.calls) == 2
    assert vision.calls[0]['images'] == vision.calls[-1]['images']
    assert vision.calls[-1]['tool_choice'] == 'none' and not vision.calls[-1]['tools']
    assert any(m['role'] == 'tool' for m in vision.calls[-1]['messages'])
    assert 'paper-review' in json.dumps(vision.calls[-1]['messages'])
    assert result.citations and '根据原图' in result.answer


@pytest.mark.parametrize('skill', ['paper-review', 'citation-check'])
def test_selected_skill_text_tools_then_one_final_image_request(library, setup, skill):
    adapter, doc, _ = library
    skills, manager, _, _ = setup
    image = adapter.visual.search('第1页', [doc])[0][0]
    req = request().model_copy(update={'metadata': {'skills': [skill]}})
    text = Model(manager, call('skill_view', name=skill),
        call('read_pdf_visual', evidence_id=image.evidence_id, question='曲线趋势'),
        call('skills_list'), answer('资料就绪'))
    vision = Model(manager, lambda k: answer(f"已读原图 [{k['images'][0]['evidence_id']}]"))
    result = run_qa(adapter, skills, manager, text, vision, req)
    assert len(text.calls) == 4 and all(not c.get('images') for c in text.calls)
    assert len(vision.calls) == 1 and image.evidence_id in result.citations.values()
    # A pending read isn't mislabeled a completed vision-tool observation.
    tool_results = [m['content'] for m in text.calls[2]['messages'] if m['role'] == 'tool']
    assert any('当前尚未核验' in content for content in tool_results)
    assert skill in json.dumps(vision.calls[0]['messages'])


def test_final_vision_failure_reuses_tool_history_and_warns(library, setup):
    adapter, _, _ = library
    skills, manager, _, _ = setup
    req = request().model_copy(update={'metadata': {'skills': ['paper-review']}})
    text = Model(manager, call('skill_view', name='paper-review'), call('document_sections'),
        answer('资料就绪'), answer('只依据文字分析'))
    vision = Model(manager, TimeoutError('private endpoint'))
    result = run_qa(adapter, skills, manager, text, vision, req)
    assert len(vision.calls) == 1 and len(text.calls) == 4
    assert not text.calls[-1].get('images') and not text.calls[-1].get('tools')
    assert any(m['role'] == 'tool' for m in text.calls[-1]['messages'])
    assert '尚未核验' in result.answer and 'private endpoint' not in result.answer
    assert '只依据文字' in result.answer


def test_initial_vision_failure_falls_back_without_another_image_attempt(library, setup):
    adapter, _, _ = library
    skills, manager, _, _ = setup
    text = Model(manager, answer('仅文字'))
    vision = Model(manager, TimeoutError('network'))
    result = run_qa(adapter, skills, manager, text, vision, request())
    assert len(text.calls) == len(vision.calls) == 1
    assert not text.calls[0].get('images') and '尚未核验' in result.answer


def test_no_tools_needs_no_extra_text_planning_call(setup):
    _, manager, _, _ = setup
    text, vision = Model(manager), Model(manager, answer('image answer'))
    response = invoke_tools(text, [{'role': 'user', 'content': 'read image'}], [], tag='qa',
        images=[{'evidence_id': 'ev_image'}], vision_llm=vision)
    assert response.content == 'image answer' and not text.calls and len(vision.calls) == 1


def test_final_pixels_at_wire_and_request_metrics_survive_restart(library, setup):
    adapter, _, _ = library
    skills, manager, store, root = setup
    images = adapter.visual.image_inputs(adapter.visual.search('第1页', adapter.documents.list())[0], policy())
    wire = []
    def create(**kwargs):
        wire.append(kwargs)
        assert kwargs['messages'][-1]['content'][1]['image_url']['url'].startswith('data:image/png;base64,')
        return SimpleNamespace(choices=[SimpleNamespace(message=answer('visible pixels'), finish_reason='stop')],
            usage=SimpleNamespace(prompt_tokens=2500, completion_tokens=20))
    vision = provider(manager, create)
    text = Model(manager, call('skills_list'), call('skills_list'), answer('ready'))
    req = request().model_copy(update={"budget": {"max_tool_calls": 10}})
    control = RunControl(req, store, lambda: False)
    with request_context(req), run_control(control):
        response = invoke_tools(text, [{'role': 'user', 'content': 'check paper'}], skills.tools(request()),
            tag='qa', images=images, vision_llm=vision)
    assert response.content == 'visible pixels' and len(wire) == 1 and len(text.calls) == 3
    budget = store.budget(request().run_id)
    assert budget['image_requests'] == 1 and budget['image_count'] == len(images)
    assert budget['image_budget_tokens'] == 1024 * len(images)
    reopened = RunStore(root.parent / 'data')
    assert reopened.budget(request().run_id) == budget
    reopened.close()


def test_failed_image_request_is_counted_but_denied_budget_is_not(library, setup, monkeypatch):
    adapter, _, _ = library
    _, manager, store, _ = setup
    images = adapter.visual.image_inputs(adapter.visual.search('第1页', adapter.documents.list())[0], policy())
    def fail(**kwargs):
        raise TimeoutError('provider failure')
    vision = provider(manager, fail)
    monkeypatch.setenv('VISION_RUN_IMAGE_TOKENS', '1024')
    with request_context(request()), run_control(RunControl(request(), store, lambda: False)):
        with pytest.raises(TimeoutError):
            vision.invoke([{'role': 'user', 'content': 'read'}], images=images)
        with pytest.raises(ValueError, match='独立读图预算'):
            vision.invoke([{'role': 'user', 'content': 'read'}], images=images)
    assert store.budget(request().run_id)['image_requests'] == 1
    assert store.budget(request().run_id)['image_budget_tokens'] == 1024


def test_text_fallback_failure_does_not_restart_completed_tools(library, setup):
    adapter, _, _ = library
    skills, manager, _, _ = setup
    req = request().model_copy(update={'metadata': {'skills': ['paper-review']}})
    text = Model(manager, call('skills_list'), answer('ready'), RuntimeError('text unavailable'))
    vision = Model(manager, TimeoutError('vision unavailable'))
    with pytest.raises(RuntimeError, match='text unavailable'):
        run_qa(adapter, skills, manager, text, vision, req)
    assert len(text.calls) == 3 and len(vision.calls) == 1
