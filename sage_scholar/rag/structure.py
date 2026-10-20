"""Source-preserving Markdown blocks. Offsets always refer to the original text."""
import hashlib
import re
from .document import SourceElement


def element(text, *, section=None, kind='paragraph', page=None, metadata=None):
    meta = metadata or {}
    identity = hashlib.sha256(f'{page}:{kind}:{meta}:{text}'.encode()).hexdigest()[:32]
    return SourceElement(content=text, section=section, page_start=page, page_end=page,
                         element_type=kind, element_id=identity, metadata=meta)


def markdown_elements(text):
    lines = text.splitlines(keepends=True)
    result, section, i, offset = [], None, 0, 0
    while i < len(lines):
        if not lines[i].strip():
            offset += len(lines[i]); i += 1
            continue
        start, first = offset, lines[i]
        heading = re.match(r'^#{1,6}\s+(.+)', first)
        if heading:
            section = heading.group(1).strip()
        kind = 'heading' if heading else 'paragraph'
        if re.match(r'^\s*\|.*\|', first):
            kind = 'table'
        elif re.match(r'^\s*(?:Figure|Fig\.?|Table|图|表)\s*\d+', first, re.I):
            kind = 'caption'
        elif first.lstrip().startswith(('$$', r'\[', r'\begin{equation')):
            kind = 'formula'
        fence = re.match(r'^\s*(```|~~~)', first)
        if fence:
            kind = 'code'
        end = i + 1
        if kind == 'formula':
            # Keep the display and immediately adjacent variable explanation together.
            while end < len(lines) and not re.match(r'^#{1,6}\s', lines[end]):
                if not lines[end].strip() and end > i + 1:
                    break
                end += 1
            while end < len(lines) and not lines[end].strip():
                end += 1
            while end < len(lines) and lines[end].strip() and not re.match(r'^#{1,6}\s', lines[end]):
                end += 1
        elif fence:
            while end < len(lines):
                closing = lines[end].lstrip().startswith(fence.group(1))
                end += 1
                if closing:
                    break
        elif not heading:
            while end < len(lines) and lines[end].strip() and not re.match(r'^#{1,6}\s', lines[end]):
                if kind == 'table' and not lines[end].lstrip().startswith('|'):
                    break
                end += 1
        content = ''.join(lines[i:end])
        offset += len(content)
        meta = {'char_start': start, 'char_end': offset}
        if kind == 'caption':
            label = re.match(r'^\s*((?:Figure|Fig\.?|Table|图|表)\s*\d+)', content, re.I)
            meta['label'] = label.group(1)
        result.append(element(content, section=section, kind=kind, metadata=meta))
        i = end
    return result
