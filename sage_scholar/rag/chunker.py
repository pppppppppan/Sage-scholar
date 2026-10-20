"""Structure-first, token-bounded chunking with stable source locators."""
from .document import Chunk
from .structure import markdown_elements


class TextChunker:
    def __init__(self, chunk_size=480, chunk_overlap=32, *, count_tokens=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = min(chunk_overlap, chunk_size // 4)
        # UTF-8 bytes are a conservative upper bound for unknown remote tokenizers.
        self.count_tokens = count_tokens or (lambda text: len(text.encode('utf-8')))

    def _split(self, text):
        start = 0
        while start < len(text):
            if self.count_tokens(text[start:]) <= self.chunk_size:
                yield start, len(text), text[start:]
                return
            lo, hi = start + 1, len(text)
            while lo <= hi:
                mid = (lo + hi) // 2
                if self.count_tokens(text[start:mid]) <= self.chunk_size:
                    lo = mid + 1
                else:
                    hi = mid - 1
            if hi <= start:
                raise ValueError('Embedding token 上限不足以容纳单个字符')
            cut = hi
            # Prefer paragraph/sentence boundaries without joining away whitespace.
            for separator in ('\n\n', '\n', '。', '. ', ' '):
                found = text.rfind(separator, start + (hi - start) // 2, hi)
                if found >= 0:
                    cut = found + len(separator)
                    break
            yield start, cut, text[start:cut]
            overlap = cut
            while overlap > start and self.count_tokens(text[overlap - 1:cut]) <= self.chunk_overlap:
                overlap -= 1
            start = max(start + 1, overlap)

    def _table(self, source):
        lines = source.content.splitlines()
        # Structured PDF tables and Markdown tables retain header + separator in each part.
        header_count = int(source.metadata.get('header_lines', 2))
        header = '\n'.join(lines[:header_count])
        if source.metadata.get('caption'):
            header = source.metadata['caption'] + '\n' + header
        if self.count_tokens(header) > self.chunk_size:
            raise ValueError('表格列名超过 Embedding 上限，不能截断表头后入库')
        rows, start = [], 0
        for index, row in enumerate(lines[header_count:]):
            content = '\n'.join([header, *rows, row])
            if self.count_tokens(content) > self.chunk_size:
                if rows:
                    yield '\n'.join([header, *rows]), {'row_start': start, 'row_end': index}
                rows, start = [], index
                if self.count_tokens(header + '\n' + row) > self.chunk_size:
                    raise ValueError('单行表格超过 Embedding 上限，不能破坏行列对应；请提高模型输入上限')
            rows.append(row)
        if rows or not lines[header_count:]:
            yield '\n'.join([header, *rows]), {'row_start': start, 'row_end': len(lines) - header_count}

    def chunk(self, document):
        result = []
        elements = document.elements if document.elements is not None else markdown_elements(document.text)
        for source in elements:
            if not source.content.strip():
                continue
            if source.element_type == 'table':
                parts = self._table(source)
            else:
                parts = ((text, {'element_char_start': start, 'element_char_end': end})
                         for start, end, text in self._split(source.content))
            for content, part in parts:
                metadata = {**source.metadata, **part}
                if 'char_start' in source.metadata and 'element_char_start' in part:
                    metadata['char_start'] = source.metadata['char_start'] + part['element_char_start']
                    metadata['char_end'] = source.metadata['char_start'] + part['element_char_end']
                result.append(Chunk(**source.model_dump(exclude={'content', 'metadata'}), content=content,
                    metadata=metadata, chunk_idx=len(result), file_path=document.file_path,
                    schema_version=2, parser_version=document.parser_version))
        return result
