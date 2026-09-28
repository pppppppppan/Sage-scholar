import asyncio
from types import SimpleNamespace

from sage_scholar.application.document_resolver import DocumentResolver


def resolve(query, docs, ids=None):
    store = SimpleNamespace(list=lambda: docs)
    return asyncio.run(DocumentResolver(store).resolve(query, ids or []))


def document(identity="doc_a", title="34552", ready=True):
    return SimpleNamespace(document_id=identity, title=title,
                           active_version="v1" if ready else None, converted_path="paper.md")


def test_uploaded_filename_matches_before_clarification():
    result = resolve("34552这篇论文使用了哪些数据集？训练集和测试集是怎么划分的？", [document()])
    assert result.document_ids == ["doc_a"] and result.message is None
    assert resolve("34552.pdf 的实验", [document(title="34552.pdf")]).document_ids == ["doc_a"]


def test_numeric_substrings_and_unknown_names_do_not_match():
    assert resolve("134552这篇论文", [document()]).document_ids == []
    assert resolve("345520这篇论文", [document()]).document_ids == []
    assert resolve("今天是星期几", [document()]).document_ids == []


def test_same_title_needs_selection_and_selection_disambiguates():
    docs = [document(), document("doc_b")]
    assert "同名" in resolve("34552这篇论文", docs).message
    assert resolve("这篇论文", docs, ["doc_b"]).document_ids == ["doc_b"]


def test_failed_upload_is_reported_as_uploaded_but_not_ready():
    result = resolve("34552这篇论文", [document(ready=False)])
    assert result.document_ids == [] and "已上传" in result.message and "索引" in result.message
    assert "不存在" in resolve("这篇论文", [], ["deleted"]).message
