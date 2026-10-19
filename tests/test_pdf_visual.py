"""Real PDF rendering/storage, deterministic encoder/provider doubles; no API billing."""
import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletionMessage

from sage_scholar.application.qa_service import QAService
from sage_scholar.config import Config
from sage_scholar.context.manager import ContextManager, request_context
from sage_scholar.domain.documents import DocumentRecord, DocumentVersion, IngestionStatus
from sage_scholar.domain.execution import ExecutionRequest, SourcePolicy
from sage_scholar.infrastructure.retrieval import LegacyRetrievalAdapter
from sage_scholar.research.evidence_context import ResearchEvidenceContext
from sage_scholar.runtime.control import RunControl, run_control
from sage_scholar.runtime.store import RunStore
from sage_scholar.tools.tool_pdf_visual import PDFVisualTool
from sage_scholar.visual.config import VisualConfig
from sage_scholar.visual.embedding import VisualEmbedding
from sage_scholar.visual.fusion import fuse
from sage_scholar.visual.service import VisualService
from test_agent_context import provider
from test_document_evidence import LocalPipeline


class Encoder:
    identity = {"model": "aligned-test-model", "revision": "1"}

    def __init__(self):
        self.inputs = []

    def encode(self, inputs):
        self.inputs.extend(inputs)
        return [[1., 0.] for _ in inputs]


@pytest.fixture
def library(tmp_path):
    fitz = pytest.importorskip("pymupdf")
    root = tmp_path / "data"
    folder = root / "converted" / "doc_a" / "v1"
    folder.mkdir(parents=True)
    original = folder / "original.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=300, height=400)
    page.insert_text((30, 40), "Figure 1: Accuracy")
    page.draw_line((30, 200), (250, 70), color=(1, 0, 0))
    pdf.save(original)
    pdf.close()
    converted = folder / "document.md"
    converted.write_text("# Results\nAccuracy increases. See Figure 1.")
    pipeline = LocalPipeline()
    pipeline.add_document(str(converted))
    adapter = LegacyRetrievalAdapter(Config(data_dir=str(root)), pipeline)
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    version = DocumentVersion(document_version="v1", version_number=1, content_hash=digest,
        parser_version="test-text-v1", original_path=str(original), converted_path=str(converted))
    doc = DocumentRecord(document_id="doc_a", title="Paper A", source_identifier="upload:a",
        original_path=str(original), converted_path=str(converted), document_version="v1", active_version="v1",
        content_hash=digest, parser_version=version.parser_version, versions=[version], ingestion_status=IngestionStatus.READY)
    adapter.documents.put(doc)
    encoder = Encoder()
    adapter.visual = VisualService(root, adapter.documents, adapter.evidence_store,
        config=VisualConfig(parser="page"), encoder=encoder)
    assert adapter.visual.rebuild(doc.document_id)["status"] == "ready"
    return adapter, doc, encoder


def policy(doc_id="doc_a"):
    return SourcePolicy(source_scope="selected_documents", document_ids=(doc_id,))


def request():
    return ExecutionRequest(query="解释第1页的图形趋势", workflow="qa", source_scope="selected_documents",
        document_ids=("doc_a",), allow_web=False, execution_mode="run", run_id="visual-run")


def test_render_index_fuse_scope_and_source_integrity(library):
    adapter, doc, encoder = library
    hits, warnings = adapter.retrieve_with_warnings("第1页的曲线", policy())
    assert not warnings
    assert any(e.metadata.get("kind") != "visual_source" for e in hits)
    image = next(e for e in hits if e.metadata.get("kind") == "visual_source")
    assert encoder.inputs[0].keys() == {"image"}  # Actual pixels, not an image caption embedding.
    assert isinstance(encoder.inputs[-1], str)  # Query uses the same aligned encoder.
    assert image.page == 1 and image.source_version == "v1" and image.location["bbox"]
    path = adapter.visual.resolve_image(image, policy())
    assert path.read_bytes().startswith(b"\x89PNG")
    assert adapter.retrieve_sync("curve", policy("other")) == []
    with pytest.raises(ValueError):
        adapter.visual.resolve_image(image, policy("other"))
    # Rebuilding does not break the old evidence/image reference.
    adapter.visual.rebuild(doc.document_id)
    assert adapter.visual.resolve_image(image, policy()) == path
    forged = image.model_copy(update={"location": {"element_id": "../../secret"}})
    with pytest.raises(ValueError):
        adapter.visual.resolve_image(forged, policy())
    adapter.documents.delete(doc.document_id)
    with pytest.raises(ValueError):
        adapter.visual.resolve_image(image)


def test_visual_failure_keeps_text_available_and_reports_it(library):
    adapter, doc, encoder = library
    def unavailable(_):
        raise RuntimeError("weights missing")
    encoder.encode = unavailable
    status = adapter.visual.rebuild(doc.document_id)
    assert status["status"] == "failed"
    hits, warnings = adapter.retrieve_with_warnings("Accuracy", policy())
    assert hits and warnings
    assert all(e.metadata.get("kind") != "visual_source" for e in hits)
    assert adapter.documents.get(doc.document_id).ingestion_status == IngestionStatus.READY


def test_encoder_preserves_modality_and_normalizes():
    import numpy as np
    encoder = VisualEmbedding(VisualConfig())
    calls = []
    def encode(inputs, **kwargs):
        calls.append(inputs)
        return np.array([[3., 4.]])
    encoder._model = SimpleNamespace(encode=encode)
    assert encoder.encode([{"image": "/fixture.png"}, "curve"])[0] == pytest.approx([.6, .8])
    assert calls == [[{"image": "/fixture.png"}], ["curve"]]


def test_qa_sends_original_pixels_and_keeps_image_evidence(library):
    adapter, _, _ = library
    class Model:
        async def complete(self, messages, **kwargs):
            images = kwargs.get("images")
            assert images and Path(images[0]["path"]).is_file()
            assert "base64" not in json.dumps(messages)
            return f"图中曲线向上 [{images[0]['evidence_id']}]。"
    result = asyncio.run(QAService(adapter, Model()).answer(request()))
    image_id = next(e.evidence_id for e in result.evidence if e.metadata.get("kind") == "visual_source")
    number = next(n for n, identity in result.citations.items() if identity == image_id)
    assert f"[{number}]" in result.answer
    assert any(e.metadata.get("kind") == "visual_source" for e in result.evidence)


def test_research_visual_tool_is_scoped_and_uses_existing_evidence(library):
    adapter, _, _ = library
    evidence = adapter.visual.search("curve", adapter.documents.list())[0][0]
    context = ResearchEvidenceContext(adapter.evidence_store, policy(), adapter.documents)
    observed = []
    class Client:
        def invoke(self, messages, **kwargs):
            assert kwargs["images"][0]["evidence_id"] == evidence.evidence_id
            return SimpleNamespace(content="The curve increases.")
    tool = PDFVisualTool(adapter.visual, context, policy(), SimpleNamespace(vision_client=Client()), observed.append)
    with pytest.raises(ValueError):
        tool.run_tool({"question": "curve", "evidence_id": evidence.evidence_id})
    context.register(evidence)
    result = tool.run_tool({"question": "curve", "evidence_id": evidence.evidence_id})
    assert "非逐字原文" in result and observed == [evidence]
    assert adapter.evidence_store.get(evidence.evidence_id).original_text == evidence.original_text


def test_pixels_only_at_wire_boundary_and_image_budget_survives_restart(library, tmp_path, monkeypatch):
    adapter, _, _ = library
    images = adapter.visual.image_inputs(adapter.visual.search("curve", adapter.documents.list())[0], policy())
    manager = ContextManager(tmp_path / "context")
    manager.run_store = RunStore(tmp_path / "run")
    sent, prepared = [], []
    original_prepare = manager.prepare
    def prepare(messages, **kwargs):
        prepared.append(messages)
        return original_prepare(messages, **kwargs)
    manager.prepare = prepare
    def create(**kwargs):
        sent.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=ChatCompletionMessage(role="assistant", content="read"), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=2500, completion_tokens=30))
    llm = provider(manager, create)
    control = RunControl(request(), manager.run_store, lambda: False)
    monkeypatch.setenv("VISION_RUN_IMAGE_TOKENS", "1024")
    messages = [{"role": "user", "content": "Read curve"}]
    with request_context(request()), run_control(control):
        llm.invoke(messages, images=images, tag="researcher:visual")
        with pytest.raises(ValueError, match="独立读图预算"):
            llm.invoke(messages, images=images, tag="researcher:visual")
    assert len(sent) == 1
    assert sent[0]["messages"][-1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "base64" not in json.dumps(prepared) and messages == [{"role": "user", "content": "Read curve"}]
    store = manager.run_store
    budget = store.budget(request().run_id)
    assert budget["image_budget_tokens"] == 1024 and budget["image_count"] == 1
    assert store.model_load(request().run_id) == 2530 - budget["image_input_tokens_estimated"]
    store.close()
    restored = RunStore(tmp_path / "run")
    assert restored.budget(request().run_id) == budget
    restored.close()


def test_rank_fusion_ignores_incomparable_embedding_scores():
    text = SimpleNamespace(evidence_id="text", score=10000)
    image = SimpleNamespace(evidence_id="image", score=-1)
    assert fuse([text], [image], 2) == [text, image]
    assert fuse([text] * 10, [], 2) == [text] * 10  # Preserve existing text-only behavior.


def test_image_api_returns_pixels_and_refuses_deleted_source(library):
    from fastapi.testclient import TestClient
    from sage_scholar.api.app import create_app
    from test_api import FakeContainer
    adapter, doc, _ = library
    image = adapter.visual.search("curve", adapter.documents.list())[0][0]
    container = FakeContainer()
    container.retrieval, container.evidence_store = adapter, adapter.evidence_store
    with TestClient(create_app(container)) as client:
        response = client.get(f"/api/evidence/{image.evidence_id}/image")
        assert response.status_code == 200 and response.headers["content-type"] == "image/png"
        assert response.content.startswith(b"\x89PNG")
        adapter.documents.delete(doc.document_id)
        assert client.get(f"/api/evidence/{image.evidence_id}/image").status_code == 404


def test_upload_optional_visual_failure_does_not_fail_text_ingestion(library, tmp_path, monkeypatch):
    import sage_scholar.library.library_manager as module
    from sage_scholar.library.library_manager import LibraryManager
    adapter, doc, _ = library
    def convert(src, output, *args, **kwargs):
        path = Path(output) / "document.md"
        path.write_text("# Parsed PDF\nThe textual source remains usable.")
        return SimpleNamespace(output_path=str(path), source_type="pdf", arxiv_id=None)
    monkeypatch.setattr(module, "convert_to_markdown", convert)
    visual = SimpleNamespace(build=lambda *args: {"status": "failed", "warning": "weights unavailable"})
    manager = LibraryManager(str(tmp_path / "ingest"), None, None, LocalPipeline(), visual)
    result = manager.ingest(doc.original_path)
    assert result.ingestion_status == IngestionStatus.READY
    stored = manager.store.get(result.document_id)
    assert manager.store.to_legacy_dict(stored)["visual"]["status"] == "failed"
    assert Path(stored.converted_path).read_text().startswith("# Parsed PDF")


def test_docling_coordinates_create_real_crops(library, monkeypatch, tmp_path):
    import sys
    from sage_scholar.visual import parser as module
    adapter, doc, _ = library
    bbox = SimpleNamespace(to_top_left_origin=lambda **kwargs: SimpleNamespace(l=20, t=50, r=260, b=220))
    item = SimpleNamespace(caption_text=lambda layout: "Figure 1: Accuracy",
        prov=[SimpleNamespace(page_no=1, bbox=bbox)])
    layout = SimpleNamespace(pictures=[item], tables=[], pages={1: SimpleNamespace(size=SimpleNamespace(width=300, height=400))})
    converter = SimpleNamespace(convert=lambda path: SimpleNamespace(document=layout))
    monkeypatch.setitem(sys.modules, "docling.document_converter", SimpleNamespace(DocumentConverter=lambda: converter))
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: object())
    import importlib.metadata
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "test-layout")
    manifest = module.PdfVisualParser(VisualConfig(parser="docling")).parse(doc.original_path, tmp_path / "crops")
    assert [e.kind for e in manifest.elements] == ["page", "figure"]
    crop = manifest.elements[1]
    assert crop.label == "Figure 1" and crop.page == 1 and crop.bbox == (16, 46, 264, 224)
    assert crop.width < manifest.elements[0].width
    assert (tmp_path / "crops" / crop.image).read_bytes().startswith(b"\x89PNG")
