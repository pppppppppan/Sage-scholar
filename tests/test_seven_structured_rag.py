import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import pymupdf

from sage_scholar.library.library_manager import LibraryManager
from sage_scholar.library.pdf_structure import extract_pdf
from sage_scholar.rag.document import Document, Chunk
from sage_scholar.rag.chunker import TextChunker
from sage_scholar.rag.pipeline import Pipeline
from sage_scholar.rag.vector_store import VectorStore
from sage_scholar.infrastructure.retrieval import LegacyRetrievalAdapter
from sage_scholar.domain.execution import SourcePolicy, ExecutionRequest
from sage_scholar.config import Config
from sage_scholar.application.qa_service import QAService


@pytest.fixture
def paper(tmp_path):
    path = tmp_path / 'structured.pdf'
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=600, height=800)
        page.insert_text((40, 40), '1 Introduction', fontsize=18)
        page.insert_textbox((40, 70, 280, 240), 'LEFT COLUMN: An introduction to the research.\n' * 7, fontsize=10)
        page.insert_textbox((320, 70, 560, 240), 'RIGHT COLUMN: Previous work and limitations.\n' * 7, fontsize=10)
        page = pdf.new_page(width=600, height=800)
        page.insert_text((40, 40), '2 Method', fontsize=18)
        page.insert_text((40, 80), '2.1 Objective', fontsize=14)
        page.insert_text((40, 110), 'L = x + y', fontsize=11)
        page.insert_text((40, 130), 'where x is the input and y is the target.', fontsize=10)
        page.insert_text((40, 180), 'Figure 2: Model architecture with two stages.', fontsize=10)
        page.draw_rect((40, 200, 200, 260))
        page.insert_text((40, 310), '3 Experiments', fontsize=18)
        page.insert_text((40, 340), 'Table 1: Experimental results.', fontsize=10)
        # Actual ruled PDF table with multi-line cells; find_tables must retain columns.
        for y in range(360, 481, 40):
            page.draw_line((40, y), (550, y))
        for x in (40, 200, 380, 550):
            page.draw_line((x, 360), (x, 480))
        for row, values in enumerate([['Method', 'Dataset', 'Accuracy'], ['Model A', 'Alpha\nsubset', '0.91'], ['Model B', 'Beta', '0.82']]):
            for x, text in zip((45, 205, 385), values):
                page.insert_text((x, 377 + row * 40), text, fontsize=10)
        page = pdf.new_page(width=600, height=800)
        page.insert_text((40, 40), '4 Limitations', fontsize=18)
        page.insert_textbox((40, 65, 560, 750), 'Generalization requires validation on another dataset.\n' * 35, fontsize=10)
        pdf.save(path)
    return path


class Embedding:
    identity = {'provider': 'fake', 'model': 'token-limited'}
    fail = False
    def token_policy(self):
        return 64, lambda text: len(text.split()) + 2
    def invoke(self, text):
        if self.fail:
            raise RuntimeError('embedding failure')
        return ([1., 0.] if isinstance(text, str) else [[1., 0.] for _ in text]), 0


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    from sage_scholar.rag import pipeline as module
    monkeypatch.setattr(module, 'Embedding', Embedding)
    def rerank(query, chunks, top_k):
        ranked = sorted(chunks, key=lambda c: query.lower() in (str(c.section) + c.content).lower(), reverse=True)
        return [(c, .9) for c in ranked[:top_k]]
    monkeypatch.setattr(module, 'Reranker', lambda: SimpleNamespace(rerank=rerank))
    return Pipeline(str(tmp_path / 'data'))


def test_real_pdf_layout_formula_table_caption_and_pages(paper):
    elements, warnings = extract_pdf(paper)
    assert not warnings
    assert {e.page_start for e in elements} == {1, 2, 3}
    page1 = '\n'.join(e.content for e in elements if e.page_start == 1)
    assert page1.rfind('LEFT COLUMN') < page1.find('RIGHT COLUMN')
    formula = next(e for e in elements if 'L = x + y' in e.content)
    assert formula.page_start == 2 and 'where x' in formula.content and formula.section == '2.1 Objective'
    table = next(e for e in elements if e.element_type == 'table')
    assert '| Method | Dataset | Accuracy |' in table.content
    assert '| Model A | Alpha<br>subset | 0.91 |' in table.content
    assert table.metadata['label'] == 'Table 1' and table.page_start == 2
    caption = next(e for e in elements if e.metadata.get('label') == 'Figure 2')
    assert caption.element_type == 'caption' and caption.metadata['bbox']


def test_import_index_restart_retrieval_evidence_qa_and_idempotence(paper, pipeline):
    class PDFTool:
        def run_tool(self, params):
            return '# Converted PDF\nA fallback Markdown conversion.'
    manager = LibraryManager(pipeline.data_dir, None, PDFTool(), pipeline)
    ingested = manager.ingest(str(paper))
    doc = manager.store.get(ingested.document_id)
    assert doc.parser_version == 'pymupdf-structure-v2'
    chunks = pipeline.vector_store.chunks
    assert chunks and all(c.schema_version == 2 and c.page_start for c in chunks)
    assert all(pipeline.embedding.token_policy()[1](c.content) <= 64 for c in chunks)
    before = [c.model_dump() for c in chunks]
    restored = Pipeline(pipeline.data_dir)
    assert [c.model_dump() for c in restored.vector_store.chunks] == before
    restored.add_document(doc.converted_path)
    assert [c.model_dump() for c in restored.vector_store.chunks] == before
    adapter = LegacyRetrievalAdapter(Config(data_dir=pipeline.data_dir), restored)
    policy = SourcePolicy(source_scope='selected_documents', document_ids=(doc.document_id,))
    hits = adapter.retrieve_sync('Objective', policy)
    formula = next(e for e in hits if 'L = x + y' in e.original_text)
    assert formula.page == 2 and formula.section == '2.1 Objective' and formula.location['bbox']
    assert adapter.evidence_store.get(formula.evidence_id) == formula
    assert adapter.retrieve_sync('Objective', SourcePolicy(source_scope='selected_documents', document_ids=('forbidden',))) == []
    class Model:
        async def complete(self, messages, **kwargs):
            assert not kwargs.get('images')
            return f'目标函数及解释 [{formula.evidence_id}]'
    req = ExecutionRequest(query='Objective', workflow='qa', source_scope='selected_documents', document_ids=(doc.document_id,), allow_web=False, execution_mode='run')
    result = asyncio.run(QAService(adapter, Model()).answer(req))
    assert formula.evidence_id in result.citations.values()
    assert not result.citation_issues
    # Failed same-path rebuild cannot mutate the usable in-memory or disk index.
    saved = [c.model_dump() for c in Pipeline(pipeline.data_dir).vector_store.chunks]
    restored.embedding.fail = True
    with pytest.raises(RuntimeError, match='embedding failure'):
        restored.add_document(doc.converted_path)
    assert [c.model_dump() for c in Pipeline(pipeline.data_dir).vector_store.chunks] == saved
    assert [c.model_dump() for c in restored.vector_store.chunks] == before
    manager.delete_doc(doc.document_id)
    assert not manager.pipeline.vector_store.chunks
    assert adapter.evidence_store.get(formula.evidence_id) == formula


def test_markdown_structure_token_limits_formula_association_and_table_headers():
    text = '# Method\n\n$$\nL = x + y\n$$\n\nwhere x is input and y is output.\n\n## Results\n\n| Method | Score |\n| --- | --- |\n' + ''.join(f'| model {i} | {i} |\n' for i in range(30))
    chunks = TextChunker(70, count_tokens=lambda text: len(text.split())).chunk(Document(file_path='p.md', text=text))
    formula = next(c for c in chunks if c.element_type == 'formula')
    assert 'where x' in formula.content
    tables = [c for c in chunks if c.element_type == 'table']
    assert len(tables) > 1 and all(c.content.startswith('| Method | Score |\n| --- | --- |') for c in tables)
    assert all(c.page_start is None and len(c.content.split()) <= 70 for c in chunks)
    assert tables[-1].metadata['row_end'] == 30


def test_legacy_index_safe_load_unknown_schema_rejected_and_metadata_roundtrip(tmp_path):
    path = tmp_path / 'index.json'
    path.write_text(json.dumps([{'chunk_idx': 0, 'content': 'legacy', 'file_path': 'old.md', 'embedding': [1, 0]}]))
    store = VectorStore.load_json(str(path))
    assert store.chunks[0].page_start is None and store.chunks[0].schema_version == 1
    structured = Chunk(chunk_idx=0, content='new', file_path='new.md', page_start=5, page_end=6, section='Method',
        element_type='table', element_id='table1', metadata={'bbox': [1, 2, 3, 4]}, schema_version=2, parser_version='pymupdf-structure-v2', embedding=[1, 0])
    store.replace_file('new.md', [structured], save_path=str(path))
    restored = VectorStore.load_json(str(path))
    assert restored.chunks[1] == structured and restored.index_revision == 1
    data = json.loads(path.read_text()); data['schema_version'] = 999
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='未知'):
        VectorStore.load_json(str(path))


def test_bad_sidecar_never_fabricates_pages_or_overwrites_good_index(paper, pipeline, monkeypatch):
    from sage_scholar.library.pdf_structure import write_pdf_structure
    converted = Path(pipeline.data_dir) / 'text.md'
    converted.write_text('parsed')
    write_pdf_structure(paper, converted)
    pipeline.add_document(str(converted))
    before = list(pipeline.vector_store.chunks)
    converted.write_text('changed behind the sidecar')
    with pytest.raises(ValueError, match='校验'):
        pipeline.add_document(str(converted))
    assert pipeline.vector_store.chunks == before
    from sage_scholar.library import pdf_structure
    monkeypatch.setattr(pdf_structure, 'extract_pdf', lambda *a: (_ for _ in ()).throw(ValueError('broken parser')))
    other = Path(pipeline.data_dir) / 'other.md'; other.write_text('converted OCR')
    parser, warnings = write_pdf_structure(paper, other)
    assert warnings and not other.with_suffix('.structure.json').exists()
    assert all(c.page_start is None for c in TextChunker().chunk(Document.read_file(str(other))))


def test_local_tokenizer_budget_includes_passage_prefix_and_special_tokens():
    from sage_scholar.rag.local_embedding import LocalEmbedding
    local = LocalEmbedding('fake')
    local._encoder = SimpleNamespace(max_seq_length=64, tokenizer=SimpleNamespace(
        encode=lambda text, **kwargs: [0] + list(text) + [1]))
    maximum, count = local.token_policy()
    assert maximum == 64 and count('x') == len('passage: x') + 2
    chunks = TextChunker(maximum, count_tokens=count).chunk(Document(file_path='test', text='word ' * 100))
    assert all(count(c.content) <= maximum for c in chunks)


def test_failed_structured_rebuild_keeps_active_version_and_old_evidence(paper, pipeline, monkeypatch):
    class PDFTool:
        def run_tool(self, params):
            return 'converted'
    manager = LibraryManager(pipeline.data_dir, None, PDFTool(), pipeline)
    first = manager.ingest(str(paper))
    old = manager.store.get(first.document_id)
    before = [c.model_dump() for c in pipeline.vector_store.chunks]
    monkeypatch.setattr('sage_scholar.library.pdf_structure.extract_pdf',
        lambda *a: (_ for _ in ()).throw(ValueError('parser unavailable')))
    with pytest.raises(ValueError, match='保留之前'):
        manager.ingest(str(paper), document_id=first.document_id)
    current = manager.store.get(first.document_id)
    assert current.active_version == old.active_version and current.versions == old.versions
    assert [c.model_dump() for c in pipeline.vector_store.chunks] == before
