import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from test_pdf_visual import library, policy, request
from test_skills import setup, Model, answer, call, request as skill_request
from sage_scholar.application.qa_service import QAService, INSUFFICIENT
from sage_scholar.application.skill_support import SkillSupport
from sage_scholar.visual.models import VisualElement
from sage_scholar.infrastructure.json_store import write_json


@pytest.mark.parametrize('query', ['这篇论文的主要创新是什么？', '这篇论文的方法有什么局限？', '总结论文全文'])
def test_text_qa_never_queries_visual_encoder_or_sends_images(library, query):
    adapter, _, encoder = library
    before = len(encoder.inputs)
    class Model:
        async def complete(self, messages, **kwargs):
            assert not kwargs.get('images')
            return '仅依据文字。'
    result = asyncio.run(QAService(adapter, Model()).answer(request().model_copy(update={'query': query})))
    assert result.answer and len(encoder.inputs) == before
    assert all(e.metadata.get('kind') != 'visual_source' for e in result.evidence)


def test_explicit_visual_off_never_uses_pixels(library):
    adapter, _, encoder = library
    before = len(encoder.inputs)
    class Model:
        async def complete(self, messages, **kwargs):
            assert not kwargs.get('images')
            return '图注文字，没有读取原图。'
    req = request().model_copy(update={'metadata': {'visual_mode': 'off'}})
    asyncio.run(QAService(adapter, Model()).answer(req))
    assert len(encoder.inputs) == before


def test_vlm_timeout_falls_back_to_text_and_never_claims_image_verified(library):
    adapter, _, _ = library
    class Model:
        calls = []
        async def complete(self, messages, **kwargs):
            self.calls.append(kwargs)
            if kwargs.get('images'):
                raise TimeoutError('private provider info')
            assert '视觉模型失败' in messages[0]['content']
            return '文字表明 accuracy increases。'
    model = Model()
    result = asyncio.run(QAService(adapter, model).answer(request()))
    assert len(model.calls) == 2 and not model.calls[1].get('images')
    assert '尚未核验' in result.answer and 'private provider info' not in result.answer


def test_visual_inputs_corruption_preserves_text_qa(library, monkeypatch):
    adapter, _, _ = library
    monkeypatch.setattr(adapter.visual, 'image_inputs', lambda *args: (_ for _ in ()).throw(ValueError('damaged')))
    class Model:
        async def complete(self, messages, **kwargs):
            assert not kwargs.get('images')
            return '仅文字'
    result = asyncio.run(QAService(adapter, Model()).answer(request()))
    assert '尚未核验' in result.answer and result.evidence


def test_empty_qa_bounded_supplement_and_final_evidence_gate(library, setup, monkeypatch):
    adapter, _, _ = library
    skills, manager, _, _ = setup
    observed = []
    def empty(*args, **kwargs):
        observed.append(kwargs)
        return [], []
    monkeypatch.setattr(adapter, 'retrieve_with_warnings', empty)
    model = Model(manager, call('rag_search', query='method'), answer('Invented science [ev_fake]'))
    support = SkillSupport(skills, SimpleNamespace(client=model), adapter, manager)
    result = asyncio.run(QAService(adapter, SimpleNamespace(), skill_support=support).answer(
        skill_request('论文的方法', workflow='qa', source_scope='selected_documents', document_ids=['doc_a'])))
    assert result.answer.startswith(INSUFFICIENT) and result.evidence == [] and result.citations == {}
    assert len(observed) == 2 and len(model.calls) == 2
    assert model.calls[-1]['tool_choice'] == 'none'


def test_empty_no_tool_answer_is_also_rejected(library, setup, monkeypatch):
    adapter, _, _ = library
    skills, manager, _, _ = setup
    monkeypatch.setattr(adapter, 'retrieve_with_warnings', lambda *a, **kw: ([], []))
    model = Model(manager, answer('Unsupported fact'))
    support = SkillSupport(skills, SimpleNamespace(client=model), adapter, manager)
    result = asyncio.run(QAService(adapter, SimpleNamespace(), skill_support=support).answer(
        skill_request('paper result', workflow='qa', source_scope='selected_documents', document_ids=['missing'])))
    assert result.answer.startswith(INSUFFICIENT)


def test_citation_skill_can_read_history_even_when_initial_rag_empty(library, setup, monkeypatch):
    adapter, _, _ = library
    skills, manager, _, _ = setup
    monkeypatch.setattr(adapter, 'retrieve_with_warnings', lambda *a, **kw: ([], []))
    model = Model(manager, call('skill_view', name='citation-check'), call('read_conversation'), answer('unverifiable'))
    support = SkillSupport(skills, SimpleNamespace(client=model), adapter, manager)
    req = skill_request('检查报告引用', workflow='qa', source_scope='workspace',
                        history=[{'role': 'assistant', 'content': 'Report: A outperforms B [1].'}])
    result = asyncio.run(QAService(adapter, SimpleNamespace(), skill_support=support).answer(req))
    assert len(model.calls) == 3 and 'unverifiable' in result.answer
    assert skills.selected(req)[0]['name'] == 'citation-check'
    assert not result.evidence


def ranked_fixture(adapter, doc):
    # Real content-addressed PNGs, synthetic layout metadata to cover locator combinations.
    manifest = adapter.visual._manifest(doc)
    base = manifest.elements[0]
    directory = adapter.visual._directory(doc.converted_path)
    elements = []
    for page, kind, label in [(5, 'page', ''), (5, 'figure', 'Figure 2'),
                               (5, 'figure', 'Figure 3'), (5, 'table', 'Table 3'),
                               (7, 'page', ''), (7, 'figure', 'Figure 2')]:
        identity = f'{len(elements) + 1:032x}'
        (directory / f'{identity}.png').write_bytes((directory / base.image).read_bytes())
        elements.append(base.model_copy(update={'element_id': identity, 'image': f'{identity}.png',
            'kind': kind, 'page': page, 'label': label, 'text': 'Figure 2 Table 3', 'caption': label}))
    manifest.elements = elements
    write_json(directory / 'manifest.json', manifest.model_dump(mode='json'))
    return directory, elements


@pytest.mark.parametrize('query,page,kind,label', [
    ('Figure 2', 5, 'figure', 'Figure 2'), ('Figure 2 on page 5', 5, 'figure', 'Figure 2'),
    ('第 5 页的图 2', 5, 'figure', 'Figure 2'), ('Table 3', 5, 'table', 'Table 3'),
    ('第 7 页', 7, 'page', ''), ('Figure 3 on page 5', 5, 'figure', 'Figure 3')])
def test_joint_visual_locator_ordering(library, query, page, kind, label):
    adapter, doc, _ = library
    ranked_fixture(adapter, doc)
    hits, _ = adapter.visual.search(query, [doc], 1)
    assert len(hits) == 1 and hits[0].page == page
    assert hits[0].metadata['visual_kind'] == kind
    if label:
        assert hits[0].section == label


def test_damaged_crop_falls_back_to_correct_page_even_without_encoder(library):
    adapter, doc, encoder = library
    directory, elements = ranked_fixture(adapter, doc)
    (directory / elements[1].image).unlink()
    encoder.encode = lambda *a: (_ for _ in ()).throw(RuntimeError('offline'))
    hits, warnings = adapter.visual.search('Figure 2 on page 5', [doc], 1)
    assert hits[0].page == 5 and hits[0].metadata['visual_kind'] == 'page' and warnings


def test_forbidden_document_never_reaches_visual_search(library, monkeypatch):
    adapter, _, encoder = library
    monkeypatch.setattr(adapter.visual, 'search', lambda *a: pytest.fail('forbidden source read'))
    assert adapter.retrieve_with_warnings('Figure 2', policy('not-allowed')) == ([], [])


def test_text_empty_can_read_visual_evidence_without_extra_classifier(library, monkeypatch):
    adapter, _, _ = library
    monkeypatch.setattr(adapter.pipeline, 'retrieve_chunks', lambda *args, **kwargs: [])
    class Model:
        calls = 0
        async def complete(self, messages, **kwargs):
            self.calls += 1
            assert kwargs.get('images')
            return f"已读原图 [{kwargs['images'][0]['evidence_id']}]"
    model = Model()
    result = asyncio.run(QAService(adapter, model).answer(request().model_copy(update={'query': '论文的内容'})))
    assert model.calls == 1 and '已读原图' in result.answer


def test_same_number_in_other_paper_is_excluded_before_visual_search(library, tmp_path):
    import shutil
    adapter, doc, _ = library
    ranked_fixture(adapter, doc)
    folder = Path(doc.converted_path).parent
    other = folder.parent / 'other'
    shutil.copytree(folder, other)
    version = doc.versions[0].model_copy(update={'original_path': str(other / 'original.pdf'), 'converted_path': str(other / 'document.md')})
    second = doc.model_copy(update={'document_id': 'doc_b', 'source_identifier': 'upload:b',
        'original_path': version.original_path, 'converted_path': version.converted_path, 'versions': [version]})
    adapter.documents.put(second)
    hits, _ = adapter.retrieve_with_warnings('Figure 2 on page 5', policy())
    assert hits and {e.document_id for e in hits} == {'doc_a'}
    assert any(e.metadata.get('visual_kind') == 'figure' for e in hits)


def test_model_can_request_missing_visual_detail_after_reading_text(library, setup):
    adapter, doc, _ = library
    skills, manager, _, _ = setup
    image = adapter.visual.search('第1页', [doc])[0][0]
    vision_calls = []
    class Vision:
        def invoke(self, messages, **kwargs):
            vision_calls.append(kwargs)
            assert kwargs['images'][0]['evidence_id'] == image.evidence_id
            return SimpleNamespace(content='已看原图，曲线向上。')
    model = Model(manager, call('rag_search', query='请定位模型细节', visual_mode='on'),
        call('read_pdf_visual', evidence_id=image.evidence_id, question='曲线趋势'),
        answer(f'视觉细节 [{image.evidence_id}]'))
    gateway = SimpleNamespace(client=model, vision_client=Vision())
    support = SkillSupport(skills, gateway, adapter, manager)
    req = skill_request('解释方法细节', workflow='qa', source_scope='selected_documents', document_ids=['doc_a'])
    result = asyncio.run(QAService(adapter, gateway, skill_support=support).answer(req))
    assert len(vision_calls) == 1 and image.evidence_id in result.citations.values()
    assert all(not c.get('images') for c in model.calls)  # Main text agent doesn't receive pixels again.
