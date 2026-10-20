"""Optional PyMuPDF layout adapter: real page/bbox provenance, no guessed pages.

The original PDF and pdfmux Markdown remain immutable. A sidecar supplies text
layout when available; OCR-only/unsupported PDFs keep the converted text and an
explicit unknown-page warning rather than fabricated source coordinates.
"""
import hashlib
import logging
import re
from pathlib import Path
from statistics import median

from sage_scholar.infrastructure.json_store import write_json
from sage_scholar.rag.structure import element
from sage_scholar.visual.parser import PDF_LOCK
from sage_scholar.runtime.control import check_control
from sage_scholar.runtime.journal import WorkflowInterrupted

logger = logging.getLogger(__name__)
PARSER_VERSION = 'pymupdf-structure-v2'
_CAPTION = re.compile(r'^\s*((?:Figure|Fig\.?|Table|图|表)\s*\d+(?:[.\-]\d+)?)', re.I)


def reading_order(blocks, width):
    # Full-width blocks delimit vertical bands; read each column top-to-bottom
    # within a band. Page coordinates, never text length, determine the order.
    spanning = sorted([b for b in blocks if b['bbox'][0] < width * .3 and b['bbox'][2] > width * .7], key=lambda b: b['bbox'][1])
    rest = [b for b in blocks if b not in spanning]
    ordered = []
    for boundary in [*spanning, None]:
        band = [b for b in rest if boundary is None or b['bbox'][1] < boundary['bbox'][1]]
        left = [b for b in band if b['bbox'][2] <= width * .56]
        right = [b for b in band if b['bbox'][0] >= width * .44]
        if left and right and len(left) + len(right) == len(band):
            ordered.extend(sorted(left, key=lambda b: b['bbox'][1]))
            ordered.extend(sorted(right, key=lambda b: b['bbox'][1]))
        else:
            ordered.extend(sorted(band, key=lambda b: (b['bbox'][1], b['bbox'][0])))
        rest = [b for b in rest if b not in band]
        if boundary:
            ordered.append(boundary)
    return ordered


def extract_pdf(original):
    import pymupdf
    elements, section, warnings = [], None, []
    with PDF_LOCK, pymupdf.open(original) as pdf:
        if pdf.needs_pass:
            raise ValueError('PDF 已加密，无法读取结构')
        for page in pdf:
            check_control()
            blocks = [b for b in page.get_text('dict')['blocks'] if b.get('type') == 0]
            if not blocks:
                raise ValueError(f'第 {page.number + 1} 页无文本层；保留完整转换文字，不生成不完整页码映射')
            sizes = [s['size'] for b in blocks for line in b['lines'] for s in line['spans'] if s['text'].strip()]
            normal = median(sizes) if sizes else 10
            tables = []
            try:
                tables = page.find_tables().tables
            except Exception:
                warnings.append(f'第 {page.number + 1} 页未能可靠识别表格，保留原文并需回读原图核验行列。')
            page_items = []
            for b in reading_order(blocks, page.rect.width):
                text = '\n'.join(''.join(s['text'] for s in line['spans']) for line in b['lines']).strip()
                if not text:
                    continue
                rect = pymupdf.Rect(b['bbox'])
                if any((rect & pymupdf.Rect(t.bbox)).get_area() > rect.get_area() * .6 for t in tables):
                    continue  # Table cells have one structured representation below.
                caption = _CAPTION.match(text)
                largest = max(s['size'] for line in b['lines'] for s in line['spans'])
                heading = not caption and len(text) < 160 and (largest > normal * 1.15 or bool(re.match(r'^\d+(?:\.\d+)*\s+[A-Z][^.!?\n]{1,100}$', text)))
                formula = not caption and not heading and bool(re.search(r'[=∑∫≤≥]|\\frac|\\sum', text))
                kind = 'caption' if caption else 'heading' if heading else 'formula' if formula else 'paragraph'
                if heading:
                    section = text
                meta = {'bbox': list(rect), 'coordinate_system': 'page_points_top_left', 'block_index': b['number']}
                if caption:
                    meta['label'] = caption.group(1)
                page_items.append(element(text, section=section, kind=kind, page=page.number + 1, metadata=meta))
            # Join a formula and its adjacent explanation when in the same column.
            joined = []
            for item in page_items:
                if (joined and joined[-1].element_type == 'formula' and item.element_type == 'paragraph'
                        and abs(joined[-1].metadata['bbox'][0] - item.metadata['bbox'][0]) < 35
                        and 0 <= item.metadata['bbox'][1] - joined[-1].metadata['bbox'][3] < 40):
                    previous = joined[-1]
                    previous.content += '\n' + item.content
                    previous.metadata.setdefault('associated_bboxes', []).append(item.metadata['bbox'])
                else:
                    joined.append(item)
            elements.extend(joined)
            for table in tables:
                rows = table.extract()
                if not rows:
                    continue
                clean = lambda value: str(value or '').replace('|', r'\|').replace('\n', '<br>')
                width = max(map(len, rows))
                rows = [list(row) + [''] * (width - len(row)) for row in rows]
                header = '| ' + ' | '.join(map(clean, rows[0])) + ' |'
                lines = [header, '| ' + ' | '.join(['---'] * width) + ' |']
                lines += ['| ' + ' | '.join(map(clean, row)) + ' |' for row in rows[1:]]
                captions = [e for e in page_items if e.element_type == 'caption' and re.match(r'Table|表', e.metadata['label'], re.I)]
                caption = min(captions, key=lambda e: abs(e.metadata['bbox'][3] - table.bbox[1]), default=None)
                nearby = [e for e in page_items if e.metadata['bbox'][1] <= table.bbox[1]]
                table_section = nearby[-1].section if nearby else section
                elements.append(element('\n'.join(lines), section=table_section, kind='table', page=page.number + 1,
                    metadata={'bbox': list(table.bbox), 'coordinate_system': 'page_points_top_left', 'header_lines': 2,
                              'label': caption.metadata['label'] if caption else None,
                              'caption': caption.content if caption else None}))
    if not elements:
        raise ValueError('PDF 没有可提取的文本层；使用转换结果，页码未知')
    return elements, warnings


def write_pdf_structure(original, converted):
    try:
        elements, warnings = extract_pdf(original)
        parser = PARSER_VERSION
    except WorkflowInterrupted:
        raise
    except Exception as exc:
        logger.warning('PDF structured parsing unavailable: %s', type(exc).__name__)
        return 'pdfmux-adapter-v1', ['结构化 PDF 解析不可用，保留转换文字；无法可靠定位的页码为空。']
    path = Path(converted)
    write_json(path.with_suffix('.structure.json'), {
        'schema_version': 2, 'parser_version': parser,
        'source_sha256': hashlib.sha256(Path(original).read_bytes()).hexdigest(),
        'converted_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'elements': [e.model_dump(mode='json') for e in elements], 'warnings': warnings,
    })
    return parser, warnings
