"""Versioned local visual index and source-bound image resolution."""
import hashlib
import json
import logging
import re
from pathlib import Path

from sage_scholar.domain import EvidenceRef
from sage_scholar.infrastructure.json_store import file_lock, write_json
from sage_scholar.runtime.control import check_control
from sage_scholar.runtime.journal import WorkflowInterrupted
from .config import VisualConfig
from .embedding import VisualEmbedding
from .models import VisualManifest
from .parser import PdfVisualParser

logger = logging.getLogger(__name__)


class VisualService:
    def __init__(self, data_dir, documents, evidence_store, *, config=None, encoder=None, parser=None):
        self.data_dir = Path(data_dir).resolve()
        self.documents, self.evidence_store = documents, evidence_store
        self.config = config or VisualConfig()
        self.encoder = encoder or VisualEmbedding(self.config)
        self.parser = parser or PdfVisualParser(self.config)

    def _directory(self, converted):
        directory = Path(converted).resolve().parent / "visual"
        if not directory.is_relative_to(self.data_dir):
            raise ValueError("PDF 资源路径不属于文献目录")
        return directory

    def build(self, original, converted):
        if not self.config.enabled:
            return {"status": "disabled"}
        directory = self._directory(converted)
        path = directory / "manifest.json"
        with file_lock(path):
            manifest = None
            try:
                check_control()
                manifest = self.parser.parse(original, directory)
                write_json(path, manifest.model_dump(mode="json"))
                vectors = self.encoder.encode([{"image": str(directory / item.image)} for item in manifest.elements])
                if len(vectors) != len(manifest.elements) or not vectors:
                    raise ValueError("视觉向量数量与页面/图表数量不一致")
                if len({len(v) for v in vectors}) != 1:
                    raise ValueError("视觉向量维度不一致")
                for item, vector in zip(manifest.elements, vectors):
                    item.vector = vector
                manifest.embedding_identity = {**self.encoder.identity, "dimension": len(vectors[0])}
                manifest.status = "ready"
                write_json(path, manifest.model_dump(mode="json"))
                return {"status": "ready", "elements": len(vectors), "warning": manifest.warning,
                        "parser": manifest.parser, "embedding": manifest.embedding_identity}
            except WorkflowInterrupted:
                raise
            except Exception as exc:
                logger.warning("PDF visual indexing unavailable (%s)", type(exc).__name__, exc_info=True)
                message = "图片索引未就绪；请安装 requirements-visual.txt、下载视觉模型后重建。文本索引仍可使用。"
                if manifest:
                    manifest.status = "failed"
                    manifest.warning = message
                    write_json(path, manifest.model_dump(mode="json"))
                else:
                    # Never leave an old ready manifest after a failed rebuild.
                    path.unlink(missing_ok=True)
                return {"status": "failed", "warning": message, "error_type": type(exc).__name__}

    def rebuild(self, document_id):
        # Same ingestion lock as delete/version changes: cannot resurrect a removed document.
        with file_lock(self.data_dir / ".ingestion"):
            doc = self.documents.get(document_id)
            if doc is None or not doc.active_version or not doc.converted_path:
                raise ValueError("文献不存在或没有可用版本")
            if not doc.original_path or Path(doc.original_path).suffix.lower() != ".pdf":
                raise ValueError("该版本没有保留原始 PDF，需重新上传 PDF")
            status = self.build(doc.original_path, doc.converted_path)
            doc.metadata["visual"] = status
            self.documents.put(doc)
            return status

    def _manifest(self, doc, converted=None):
        path = self._directory(converted or doc.converted_path) / "manifest.json"
        if not path.is_file():
            return None
        manifest = VisualManifest.model_validate_json(path.read_text())
        version = next((v for v in doc.versions if v.converted_path == str(converted or doc.converted_path)), None)
        expected_hash = version.content_hash if version else doc.content_hash
        if manifest.schema_version != 1 or manifest.source_hash != expected_hash:
            raise ValueError("视觉索引与 PDF 版本不匹配，请重建图片索引")
        return manifest

    def search(self, query, docs, top_k=3):
        if not self.config.enabled:
            return [], []
        import numpy as np
        candidates, warnings = [], []
        for doc in docs:
            if not doc.original_path or Path(doc.original_path).suffix.lower() != ".pdf":
                continue
            try:
                manifest = self._manifest(doc)
                if not manifest:
                    warnings.append(f"《{doc.title}》尚无图片索引；可重建图片索引。")
                    continue
                if manifest.warning:
                    warnings.append(f"《{doc.title}》：{manifest.warning}")
                if manifest.status != "ready":
                    warnings.append(f"《{doc.title}》图片向量未就绪，目前仅使用文字。")
                    continue
                identity = {k: v for k, v in manifest.embedding_identity.items() if k != "dimension"}
                if identity != self.encoder.identity:
                    raise ValueError("视觉 Embedding 配置已变化，请重建图片索引")
                candidates.extend((doc, manifest, item) for item in manifest.elements if item.vector)
            except (ValueError, OSError):
                warnings.append(f"《{doc.title}》图片索引不可用，请重建。")
        if not candidates:
            return [], list(dict.fromkeys(warnings))
        try:
            vector = np.asarray(self.encoder.encode([query])[0])
        except WorkflowInterrupted:
            raise
        except Exception:
            logger.warning("Visual query encoding failed", exc_info=True)
            return [], [*warnings, "视觉检索模型不可用，本次只检索文字，未读取原图。"]
        ranks = []
        # Figure/page references are source locators, not semantic intent classification.
        label = re.search(r"(?:figure|fig\.?|图|table|表)\s*(\d+(?:[.\-]\d+)?)", query, re.I)
        page_ref = re.search(r"(?:第\s*(\d+)\s*页|\bpage\s+(\d+)\b)", query, re.I)
        for doc, manifest, item in candidates:
            if len(item.vector) != len(vector):
                warnings.append("视觉索引维度不匹配，已跳过；请重建。")
                continue
            exact = False
            if label:
                target_kind = "table" if re.match(r"table|表", label.group(), re.I) else "figure"
                number = re.search(r"\d+(?:[.\-]\d+)?", item.label)
                exact = item.kind == target_kind and number is not None and number.group() == label.group(1)
            if page_ref:
                exact = item.kind == "page" and item.page == int(page_ref.group(1) or page_ref.group(2))
            score = float(vector @ np.asarray(item.vector))
            ranks.append((exact, score, doc, manifest, item))
        ranks.sort(key=lambda r: (r[0], r[1]), reverse=True)
        selected, pages = [], set()
        for exact, score, doc, manifest, item in ranks:
            key = (doc.document_id, item.page)
            # Avoid whole-page duplicates of retrieved crops; multiple independent figures survive.
            if item.kind == "page" and key in pages:
                continue
            if item.kind != "page":
                selected = [e for e in selected if not (e.document_id == doc.document_id and e.page == item.page
                            and e.metadata.get("visual_kind") == "page")]
            evidence = self._evidence(doc, manifest, item, score)
            selected.append(evidence)
            pages.add(key)
            if len(selected) >= top_k:
                break
        return selected, list(dict.fromkeys(warnings))

    def _evidence(self, doc, manifest, item, score=None):
        # The description identifies an original image; it does not claim to describe its pixels.
        original = f"PDF 原图：第 {item.page} 页，{item.label or item.kind}。需读取图片后才能判断视觉细节。"
        if item.caption:
            original += "\n原图注：" + item.caption
        if item.text:
            original += "\n解析文字（可能存在 OCR/表格识别误差）：\n" + item.text[:8000]
        evidence = EvidenceRef.create(source_id=doc.source_identifier, document_id=doc.document_id,
            source_version=doc.document_version, parser_version=doc.parser_version, original_text=original,
            page=item.page, section=item.label or f"PDF page {item.page}", score=score,
            location={"element_id": item.element_id, "bbox": list(item.bbox), "coordinate_system": "page_points_top_left"},
            metadata={"title": doc.title, "converted_path": doc.converted_path,
                      "kind": "visual_source", "visual_kind": item.kind, "visual_parser": manifest.parser,
                      "image_sha256": item.image_sha256, "width": item.width, "height": item.height})
        self.evidence_store.save(evidence)
        return evidence

    def resolve_image(self, evidence, policy=None):
        if evidence.metadata.get("kind") != "visual_source":
            raise ValueError("Evidence 不是 PDF 原图")
        doc = self.documents.get(evidence.document_id)
        if not doc or (policy is not None and not policy.permits_document(doc.document_id)):
            raise ValueError("文献已删除或不在本次资料范围")
        version = next((v for v in doc.versions if v.document_version == evidence.source_version), None)
        if not version or version.converted_path != evidence.metadata.get("converted_path"):
            raise ValueError("图片来源版本不匹配")
        # Content-addressed images are immutable and retained across index rebuilds.
        # Historical evidence must not depend on the *current* visual manifest.
        element_id = evidence.location.get("element_id", "")
        if not isinstance(element_id, str) or not re.fullmatch(r"[a-f0-9]{32}", element_id):
            raise ValueError("图片来源标识无效")
        directory = self._directory(version.converted_path)
        path = (directory / f"{element_id}.png").resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            raise ValueError("图片资源路径无效")
        if hashlib.sha256(path.read_bytes()).hexdigest() != evidence.metadata.get("image_sha256"):
            raise ValueError("图片内容校验失败")
        return path

    def image_inputs(self, evidence, policy):
        images = []
        seen = set()
        for item in evidence:
            if item.metadata.get("kind") != "visual_source" or item.metadata.get("image_sha256") in seen:
                continue
            if len(images) >= self.config.max_images:
                break
            path = self.resolve_image(item, policy)
            images.append({"evidence_id": item.evidence_id, "path": str(path), "page": item.page})
            seen.add(item.metadata["image_sha256"])
        return images
