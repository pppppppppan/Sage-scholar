"""Clarification is durable dialogue; permission changes require a bound user action."""
import asyncio

import pytest
from pydantic import ValidationError

from test_semantic_query_graph import Classifier, intent, setup
from sage_scholar.application.query_service import QueryCommand
from sage_scholar.domain.execution import RunOutcome
from sage_scholar.routing.classifier import SemanticIntent
from sage_scholar.runtime import RunStatus
from sage_scholar.runtime.human import HumanResponse, RoutingSettings, validate_routing_response


def research_intent(**kwargs):
    return SemanticIntent(workflow='deep_research', source_scope='public', confidence=.9, **kwargs)


class Research:
    def __init__(self):
        self.calls = []

    async def run(self, request, emit, cancelled):
        self.calls.append(request)
        return RunOutcome(status='completed', result={'report': 'report'})


def response(run, **kwargs):
    return HumanResponse(conversation_id=run.execution.conversation_id, run_id=run.run_id,
                         interrupt_id=run.interrupts[0]['interrupt_id'], **kwargs)


def test_information_answers_are_paired_persisted_and_do_not_relax_permissions(tmp_path):
    async def scenario():
        classifier = Classifier(intent('clarify', clarification='要介绍哪个概念？'),
                                intent('clarify', clarification='需要解释原理还是推导？'), intent())
        c = setup(tmp_path, classifier)
        conversation = c.store.conversations.create().conversation_id
        initial = await c.service.execute(QueryCommand(query='介绍它', conversation_id=conversation, allow_web=False))
        await c.manager._tasks[initial.run_id]
        first = c.manager.get(initial.run_id)
        await c.manager.answer(first.run_id, response(first, answer='梯度下降，允许联网'))
        await c.manager._tasks[first.run_id]
        second = c.manager.get(first.run_id)
        assert second.status == RunStatus.WAITING_FOR_USER
        assert second.interrupts[0]['question'] == '需要解释原理还是推导？'
        await c.manager.answer(second.run_id, response(second, answer='只需要解释原理'))
        await c.manager._tasks[second.run_id]
        final = c.manager.get(second.run_id)
        assert final.status == RunStatus.COMPLETED, final.error
        assert classifier.calls[1]['recent_conversation'][-2:] == [
            {'role': 'assistant', 'content': '要介绍哪个概念？'},
            {'role': 'user', 'content': '梯度下降，允许联网'}]
        assert classifier.calls[2]['recent_conversation'][-1]['content'] == '只需要解释原理'
        assert all(call['allow_web'] is False for call in classifier.calls)
        assert len(c.calls) == 1 and not c.calls[0].allow_web
        turns = c.calls[0].metadata['clarification_turns']
        assert len(turns) == 2 and [t['answer'] for t in turns] == ['梯度下降，允许联网', '只需要解释原理']
        await c.close()
    asyncio.run(scenario())


def test_permission_conflict_requires_explicit_update_and_survives_restart(tmp_path):
    async def scenario():
        research = Research()
        c = setup(tmp_path, Classifier(research_intent()), research)
        conversation = c.store.conversations.create().conversation_id
        initial = await c.service.execute(QueryCommand(query='调研自适应网格', conversation_id=conversation, allow_web=False))
        await c.manager._tasks[initial.run_id]
        run = c.manager.get(initial.run_id)
        assert run.status == RunStatus.WAITING_FOR_USER
        assert run.interrupts[0]['kind'] == 'source_policy'
        with pytest.raises(ValueError, match='资料设置'):
            await c.manager.answer(run.run_id, response(run, answer='可以联网'))
        assert c.manager.get(run.run_id).status == RunStatus.WAITING_FOR_USER and not research.calls
        await c.close()
        classifier = Classifier(research_intent())
        c = setup(tmp_path, classifier, research)
        await c.manager.recover()
        run = c.manager.get(run.run_id)
        approved = response(run, routing_update=RoutingSettings(allow_web=True))
        await c.manager.answer(run.run_id, approved)
        await c.manager._tasks[run.run_id]
        final = c.manager.get(run.run_id)
        assert final.status == RunStatus.COMPLETED, final.error
        assert len(classifier.calls) == 1 and classifier.calls[0]['allow_web'] is True
        assert len(research.calls) == 1 and research.calls[0].source_policy.permits_web
        assert research.calls[0].metadata['_routing']['allow_web'] is True
        assert research.calls[0].metadata['clarification_turns'][0]['routing_update'] == {'allow_web': True}
        assert '已确认并更新' in research.calls[0].metadata['clarification_turns'][0]['answer']
        with pytest.raises(ValueError):
            await c.manager.answer(run.run_id, approved)
        await c.close()
    asyncio.run(scenario())


def test_scope_update_is_not_lost_and_does_not_implicitly_grant_web(tmp_path):
    async def scenario():
        research = Research()
        classifier = Classifier(research_intent(source_explicit=True), research_intent(source_explicit=True), research_intent())
        c = setup(tmp_path, classifier, research)
        conversation = c.store.conversations.create().conversation_id
        initial = await c.service.execute(QueryCommand(query='独立公网调研', conversation_id=conversation,
                                                       source_scope='workspace', allow_web=False))
        await c.manager._tasks[initial.run_id]
        run = c.manager.get(initial.run_id)
        await c.manager.answer(run.run_id, response(run, routing_update=RoutingSettings(source_scope='public')))
        await c.manager._tasks[run.run_id]
        run = c.manager.get(run.run_id)
        assert run.status == RunStatus.WAITING_FOR_USER and not research.calls
        assert run.interrupts[0]['routing_settings']['source_scope'] == 'public'
        assert run.interrupts[0]['routing_settings']['allow_web'] is False
        await c.manager.answer(run.run_id, response(run, routing_update=RoutingSettings(allow_web=True)))
        await c.manager._tasks[run.run_id]
        assert c.manager.get(run.run_id).status == RunStatus.COMPLETED
        assert classifier.calls[-1]['requested_scope'] == 'public'
        assert research.calls[0].metadata['_routing']['source_scope'] == 'public'
        await c.close()
    asyncio.run(scenario())


def test_missing_paper_can_be_selected_on_resume_and_ids_are_checked(tmp_path):
    async def scenario():
        class Docs:
            async def catalog(self):
                return [{'document_id': 'a', 'title': 'A', 'ready': True},
                        {'document_id': 'b', 'title': 'B', 'ready': True}]
            async def resolve(self, query, ids):
                from sage_scholar.application.document_resolver import DocumentResolution
                return DocumentResolution(document_ids=ids)
        qa = SemanticIntent(workflow='qa', source_scope='selected_documents', confidence=.9)
        classifier = Classifier(intent('clarify', clarification='比较哪两篇论文？'), qa, qa)
        c = setup(tmp_path, classifier, docs=Docs())
        conversation = c.store.conversations.create().conversation_id
        initial = await c.service.execute(QueryCommand(query='比较这两篇', conversation_id=conversation))
        await c.manager._tasks[initial.run_id]
        run = c.manager.get(initial.run_id)
        wrong = response(run, routing_update=RoutingSettings(document_ids=['missing']))
        with pytest.raises(ValueError, match='对话/任务'):
            await c.manager.answer(run.run_id, wrong.model_copy(update={'conversation_id': 'other'}))
        with pytest.raises(ValueError):
            await c.manager.answer(run.run_id, wrong.model_copy(update={'interrupt_id': 'stale'}))
        await c.manager.answer(run.run_id, wrong)
        await c.manager._tasks[run.run_id]
        run = c.manager.get(run.run_id)
        assert run.status == RunStatus.WAITING_FOR_USER and not c.calls
        await c.manager.answer(run.run_id, response(run, routing_update=RoutingSettings(document_ids=['a', 'b'])))
        await c.manager._tasks[run.run_id]
        assert c.manager.get(run.run_id).status == RunStatus.COMPLETED
        assert classifier.calls[-1]['selected_document_ids'] == ['a', 'b']
        assert c.calls[0].document_ids == ('a', 'b')
        await c.close()
    asyncio.run(scenario())


def test_settings_update_cannot_be_smuggled_into_other_interrupts_or_skip():
    from sage_scholar.runtime.interaction import Decision
    assert 'routing_settings' not in str(Decision.model_json_schema())
    common = dict(conversation_id='c', run_id='r', interrupt_id='i')
    with pytest.raises(ValidationError):
        RoutingSettings(allow_web='true')
    with pytest.raises(ValidationError):
        RoutingSettings(workflow='direct')
    with pytest.raises(ValueError):
        validate_routing_response({'question': 'Research choice'}, HumanResponse(**common, routing_update=RoutingSettings(allow_web=True)))
    question = {'kind': 'source_policy', 'routing_settings': {'allow_web': False}}
    with pytest.raises(ValueError, match='没有变化'):
        validate_routing_response(question, HumanResponse(**common, routing_update=RoutingSettings(allow_web=False)))
    with pytest.raises(ValueError):
        validate_routing_response(question, HumanResponse(**common, skip=True, routing_update=RoutingSettings(allow_web=True)))
