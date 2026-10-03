"""Conservative bibliographic identity matching; metadata is never source Evidence."""
from dataclasses import dataclass
import re
import unicodedata
from urllib.parse import unquote

_DOI = re.compile(r"\b10\.\d{4,9}/[^\s<>\"\]]+", re.I)
_ARXIV_ID = r"(?:\d{4}\.\d{4,5}|[a-z][a-z.-]*/\d{7})(?:v\d+)?"
_ARXIV = re.compile(r"(?:arxiv\s*:\s*|arxiv\.org/(?:abs|pdf)/)(" + _ARXIV_ID + r")", re.I)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def normalize(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def doi_id(text):
    found = _DOI.search(unquote(text))
    if not found:
        return None
    value = found[0].rstrip(".,;}")
    # Parentheses can belong to older DOI suffixes; only remove unmatched closing delimiters.
    while value.endswith(")") and value.count(")") > value.count("("):
        value = value[:-1]
    return value.casefold()


def arxiv_id(text):
    found = _ARXIV.search(text)
    if found:
        return found[1]
    return text if re.fullmatch(_ARXIV_ID, text, re.I) else None


@dataclass(frozen=True)
class Reference:
    raw: str
    title: str = ""
    authors: tuple[str, ...] = ()
    year: str | None = None
    doi: str | None = None
    arxiv: str | None = None
    url: str | None = None

    @property
    def locator(self):
        if self.doi:
            return "https://doi.org/" + self.doi
        if self.arxiv:
            return "arxiv:" + self.arxiv
        return self.url


def parse_reference(text):
    """Extract common title-only, author/title/year, APA and quoted-title formats.

    Unknown formatting stays conservative: a failed exact metadata match asks for a locator.
    No model invents missing authors, identifiers or publication years.
    """
    text = text.strip()
    doi, arxiv = doi_id(text), arxiv_id(text)
    url_match = re.search(r"https?://[^\s<>\]]+", text)
    url = url_match[0].rstrip(".,;)") if url_match else None
    if doi or arxiv or url or text.startswith(("ev_", "doc_")):
        return Reference(text, doi=doi, arxiv=arxiv, url=url)
    years = _YEAR.findall(text)
    year = years[-1] if years else None
    authors = ""
    title = text
    quoted = re.search(r'[“"]([^”"]+)[”"]', text)
    et_al = re.search(r"\bet\s+al\.?\s*[,.:]?\s*", text, re.I)
    apa = re.search(r"\((?:19|20)\d{2}[a-z]?\)\.?\s*", text)
    if quoted:
        title, authors = quoted[1], text[:quoted.start()]
    elif apa:
        authors, title = text[:apa.start()], text[apa.end():]
    elif et_al:
        authors, title = text[:et_al.start()], text[et_al.end():]
    else:
        parts = re.split(r"\.\s+", text, maxsplit=1)
        if len(parts) == 2 and ("," in parts[0] or len(parts[0].split()) <= 2) and year:
            authors, title = parts
    if not quoted:
        title = re.sub(r"^(?:19|20)\d{2}[.,:;]?\s+", "", title)
        title = re.split(r"\.\s+", title, maxsplit=1)[0]
        title = re.sub(r"\s*[,;(]?\s*(?:19|20)\d{2}\)?\.?$", "", title)
    title = title.strip(" \t\n.,;:")
    # The first author's surname is sufficient as a constraint, never as an identity by itself.
    author_words = re.findall(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", authors, re.UNICODE)
    first_author = next((word for word in author_words if len(word) > 1), None)
    return Reference(text, title=title, authors=(first_author,) if first_author else (), year=year)


def paper_metadata(paper):
    """Accept the installed paper-search MCP's Paper.to_dict format and common aliases."""
    title = paper.get("title", "")
    if isinstance(title, list):
        title = title[0] if title else ""
    authors = paper.get("authors", paper.get("author", []))
    if isinstance(authors, str):
        authors = re.split(r";|\s+and\s+", authors)
    authors = [str(a.get("name") or " ".join(filter(None, [a.get("given"), a.get("family")])))
               if isinstance(a, dict) else str(a) for a in (authors or [])]
    date = str(paper.get("year") or paper.get("published_date") or paper.get("publicationDate") or "")
    year = _YEAR.search(date)
    doi = doi_id(str(paper.get("doi") or paper.get("DOI") or ""))
    url = str(paper.get("url") or "")
    external_ids = paper.get("externalIds") or {}
    arxiv = arxiv_id(str(paper.get("arxiv_id") or external_ids.get("ArXiv") or "")) or arxiv_id(url)
    if doi and doi.startswith("10.48550/arxiv."):
        arxiv = arxiv or arxiv_id(doi.removeprefix("10.48550/arxiv."))
    if str(paper.get("source", "")).casefold() == "arxiv":
        arxiv = arxiv or arxiv_id(str(paper.get("paper_id", "")))
    locator = "https://doi.org/" + doi if doi else "arxiv:" + arxiv if arxiv else url
    if not locator.startswith(("http://", "https://", "arxiv:")):
        locator = ""
    return {"title": str(title), "authors": authors, "year": year[0] if year else None,
            "doi": doi, "arxiv": arxiv, "source": locator}


def match_reference(reference, papers):
    matches = []
    for paper in papers:
        candidate = paper_metadata(paper)
        if not reference.title or normalize(candidate["title"]) != normalize(reference.title):
            continue
        if reference.year and candidate["year"] != reference.year:
            continue
        author_tokens = set(normalize(" ".join(candidate["authors"])).split())
        if any(not set(normalize(author).split()) <= author_tokens for author in reference.authors):
            continue
        if candidate["source"]:
            # Multiple hits for one identifier are one identity, not an arbitrary top result.
            aliases = {candidate["source"]}
            if candidate["doi"]:
                aliases.add("https://doi.org/" + candidate["doi"])
            if candidate["arxiv"]:
                aliases.add("arxiv:" + re.sub(r"v\d+$", "", candidate["arxiv"]))
            overlaps = [entry for entry in matches if entry[0] & aliases]
            for entry in overlaps:
                aliases.update(entry[0])
                matches.remove(entry)
            matches.append((aliases, candidate))
    if len(matches) == 1:
        return matches[0][1], "题名及已提供的作者、年份与检索元数据一致。"
    if matches:
        return None, "存在多个符合元数据条件的不同文献身份，请补充 DOI/arXiv 或完整书目信息。"
    return None, "未找到题名及已提供作者、年份均匹配的记录，当前无法确认来源。"


class AmbiguousLocalReference(ValueError):
    pass


def match_local_reference(reference, documents):
    """Exact identifiers first, then exact normalized titles. Never fuzzy-select a document."""
    matches = []
    for doc in documents:
        metadata = paper_metadata({**doc.metadata, "title": doc.title})
        identifiers = [doc.source_identifier, str(doc.metadata.get("doi", "")),
                       str(doc.metadata.get("DOI", "")), str(doc.metadata.get("arxiv_id", "")),
                       str(doc.metadata.get("url", ""))]
        dois = {doi_id(value) for value in identifiers} | {metadata["doi"]}
        arxivs = {re.sub(r"v\d+$", "", value) for text in identifiers if (value := arxiv_id(text))}
        if metadata["arxiv"]:
            arxivs.add(re.sub(r"v\d+$", "", metadata["arxiv"]))
        if reference.raw == doc.document_id:
            matched = True
        elif reference.doi:
            matched = reference.doi in dois
        elif reference.arxiv:
            matched = re.sub(r"v\d+$", "", reference.arxiv) in arxivs
        elif reference.url:
            matched = reference.url in identifiers
        else:
            matched = bool(reference.title and normalize(reference.title) == normalize(doc.title))
            # Missing metadata is unknown; present contradictory author/year rules a candidate out.
            if reference.year and metadata["year"] and reference.year != metadata["year"]:
                matched = False
            if reference.authors and metadata["authors"]:
                tokens = set(normalize(" ".join(metadata["authors"])).split())
                if any(not set(normalize(author).split()) <= tokens for author in reference.authors):
                    matched = False
        if matched:
            matches.append(doc)
    if len(matches) > 1:
        raise AmbiguousLocalReference("本地存在多个符合条件的来源（ambiguous），请指定 document_id 或 DOI/arXiv；不自动选择。")
    return matches[0] if matches else None
