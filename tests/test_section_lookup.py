from types import SimpleNamespace
from pathlib import Path

from sage_scholar.config import Config
from sage_scholar.infrastructure.retrieval import LegacyRetrievalAdapter
from sage_scholar.domain.documents import DocumentRecord, DocumentVersion
from sage_scholar.domain.execution import SourcePolicy
from sage_scholar.rag.document import Chunk, Document
from sage_scholar.rag.sections import SectionLookup
from sage_scholar.domain.evidence import EvidenceRef


def setup(tmp_path):
    path=tmp_path/'v1.md';path.write_text('# Method\nAlpha result. Beta result.\n')
    doc=DocumentRecord(document_id='p',title='paper',source_identifier='paper',converted_path=str(path),
        active_version='v1',document_version='v1',versions=[DocumentVersion(document_version='v1',version_number=1,parser_version='test',converted_path=str(path))])
    pipeline=SimpleNamespace(retrieve_chunks=lambda *a,**k:[(Chunk(file_path=str(path),chunk_idx=i,content=text),1) for i,text in enumerate(['Alpha result.','Beta result.'])])
    adapter=LegacyRetrievalAdapter(Config(data_dir=str(tmp_path)),pipeline=pipeline)
    adapter.documents.put(doc)
    return adapter,doc,pipeline


def test_one_parse_for_multiple_chunks_same_results_as_legacy(tmp_path,monkeypatch):
    adapter,doc,_=setup(tmp_path)
    original=Document.read_file
    calls=[]
    monkeypatch.setattr(Document,'read_file',lambda path:(calls.append(path),original(path))[1])
    result,_=adapter.retrieve_with_warnings('result',SourcePolicy(source_scope='workspace'),visual_mode='off')
    assert calls==[doc.converted_path]
    expected=adapter._sections(doc)[0]
    assert {e.original_text for e in result}=={'Alpha result.','Beta result.',expected.original_text}
    assert expected.evidence_id in {e.evidence_id for e in result}


def test_version_change_reparse_and_delete_or_scope_cannot_reuse(tmp_path):
    adapter,doc,pipeline=setup(tmp_path)
    policy=SourcePolicy(source_scope='workspace')
    old,_=adapter.retrieve_with_warnings('result',policy,visual_mode='off')
    new=tmp_path/'v2.md';new.write_text('# Method\nGamma result. More detail.')
    doc.converted_path=str(new);doc.document_version=doc.active_version='v2'
    adapter.documents.put(doc)
    pipeline.retrieve_chunks=lambda *a,**k:[(Chunk(file_path=str(new),chunk_idx=0,content='Gamma result.'),1)]
    current,_=adapter.retrieve_with_warnings('result',policy,visual_mode='off')
    assert all(e.source_version=='v2' for e in current)
    assert not {e.evidence_id for e in old}&{e.evidence_id for e in current}
    denied,_=adapter.retrieve_with_warnings('result',SourcePolicy(source_scope='selected_documents',document_ids=('other',)),visual_mode='off')
    assert denied==[]
    adapter.documents.delete('p')
    assert adapter.retrieve_with_warnings('result',policy,visual_mode='off')[0]==[]


def test_element_locator_chooses_correct_original_without_fulltext_ambiguity():
    def evidence(identity,page):
        return EvidenceRef.create(source_id='p',original_text='Same text with detail',page=page,location={'element_id':identity})
    a,b=evidence('a',1),evidence('b',2)
    lookup=SectionLookup([a,b])
    c=Chunk(file_path='p',chunk_idx=0,content='Same text',element_id='b',page_start=2)
    assert lookup.containing(c).evidence_id==b.evidence_id
    assert len(lookup._normalized)==1
