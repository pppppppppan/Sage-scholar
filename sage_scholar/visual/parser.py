"""Layout crops plus page fallback. Never equate embedded bitmaps with PDF figures."""
import hashlib
import importlib.util
import re
import threading
from pathlib import Path

from sage_scholar.runtime.control import check_control
from .models import VisualElement, VisualManifest

# MuPDF and document layout inference are serialized across ingestion/rebuild requests.
PDF_LOCK = threading.RLock()


class PdfVisualParser:
    def __init__(self, config):
        self.config = config

    def parse(self, original, output):
        import pymupdf
        original, output = Path(original), Path(output)
        output.mkdir(parents=True, exist_ok=True)
        with PDF_LOCK, pymupdf.open(original) as pdf:
            if pdf.needs_pass:
                raise ValueError("加密 PDF 需要先解密")
            if len(pdf) > self.config.max_pages:
                raise ValueError(f"PDF 共 {len(pdf)} 页，超过视觉索引上限 {self.config.max_pages}；未截断冒充完整索引")
            manifest = VisualManifest(source_hash=hashlib.sha256(original.read_bytes()).hexdigest(),
                parser=f"pymupdf-pages-v1:{pymupdf.VersionBind}", page_count=len(pdf))
            layout = None
            if self.config.parser == "docling":
                if importlib.util.find_spec("docling") is None:
                    manifest.warning = "Docling 未安装，使用整页视觉检索；没有图表区域分割/OCR。"
                else:
                    from docling.document_converter import DocumentConverter
                    check_control()
                    layout = DocumentConverter().convert(str(original)).document
                    from importlib.metadata import version
                    manifest.parser += f";docling:{version('docling')}"

            def render(page_no, rect, kind, text="", caption=""):
                page = pdf[page_no - 1]
                rect = pymupdf.Rect(rect) & page.rect
                if rect.is_empty or rect.width < 3 or rect.height < 3:
                    return
                scale = min(2.0, self.config.max_side / max(rect.width, rect.height))
                pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=rect, alpha=False)
                image = pix.tobytes("png")
                digest = hashlib.sha256(image).hexdigest()
                identity = hashlib.sha256(f"{page_no}:{kind}:{tuple(rect)}:{digest}".encode()).hexdigest()[:32]
                filename = f"{identity}.png"
                (output / filename).write_bytes(image)
                label = re.search(r"(?:Figure|Fig\.?|Table|图|表)\s*\d+(?:[.\-]\d+)?", caption, re.I)
                manifest.elements.append(VisualElement(element_id=identity, kind=kind, page=page_no,
                    bbox=tuple(rect), image=filename, image_sha256=digest, text=text, caption=caption,
                    label=label.group() if label else "", width=pix.width, height=pix.height))

            for index, page in enumerate(pdf):
                check_control()
                render(index + 1, page.rect, "page", page.get_text("text", sort=True))
            if layout:
                for kind, items in (("figure", layout.pictures), ("table", layout.tables)):
                    for item in items:
                        check_control()
                        caption = item.caption_text(layout)
                        text = item.export_to_markdown(doc=layout) if kind == "table" else ""
                        for prov in item.prov:
                            page_no = prov.page_no
                            # Rotated/CropBox pages keep a correct full-page fallback rather than
                            # trusting a coordinate transform that cannot be established.
                            page = pdf[page_no - 1]
                            size = layout.pages[page_no].size
                            if page.rotation or abs(size.width - page.rect.width) > 1 or abs(size.height - page.rect.height) > 1:
                                manifest.warning = "部分页面坐标或旋转不一致，仅使用对应整页，未猜测裁剪位置。"
                                continue
                            box = prov.bbox.to_top_left_origin(page_height=size.height)
                            render(page_no, (box.l - 4, box.t - 4, box.r + 4, box.b + 4), kind, text, caption)
            return manifest
