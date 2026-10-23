"""Request-local section lookup. No cross-request cache or stale source ownership."""
import re
from collections import defaultdict


def normalized(text):
    return re.sub(r'\s+', '', text)


class SectionLookup:
    def __init__(self, sections):
        self.sections = sections
        self.by_element = defaultdict(list)
        self.by_location = defaultdict(list)
        self._normalized = {}
        for section in sections:
            location = section.location
            ids = [location.get('element_id'), *[s.get('element_id') for s in location.get('source_locations', [])]]
            for identity in set(ids) - {None}:
                self.by_element[identity].append(section)
            self.by_location[(section.page, section.section)].append(section)

    def containing(self, chunk):
        candidates = self.by_element.get(chunk.element_id, []) if chunk.element_id else []
        if not candidates and (chunk.page_start is not None or chunk.section):
            candidates = self.by_location.get((chunk.page_start, chunk.section), [])
        # Legacy/missing locators need text fallback. Normalize each section once.
        candidates = candidates or self.sections
        needle = normalized(chunk.content)
        found = []
        for section in candidates:
            key = section.evidence_id
            if key not in self._normalized:
                self._normalized[key] = normalized(section.original_text)
            if needle and needle in self._normalized[key]:
                found.append(section)
        return found[0] if len(found) == 1 else None


def version_key(doc):
    return (doc.document_id, doc.document_version, doc.parser_version,
            doc.content_hash, doc.converted_path)
