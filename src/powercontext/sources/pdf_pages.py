# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Page-level captured Sources for complete PDF documents.

Every physical page of the original PDF becomes one independently addressable
Source, so a retrieval citation can be traced back to the exact document
revision and physical page. Page identity depends only on the document
identity, the digest of the complete original bytes, and the physical page
number, which keeps it stable across re-parsing and separates revisions of the
same document.
"""

from __future__ import annotations

import hashlib
import io
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote, unquote

from pydantic import BaseModel, field_validator

from powercontext.artifacts import MemoryCitation
from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

PDF_PAGE_SOURCE_NAME = "pdf-page"
PDF_PAGE_SOURCE_VERSION = "1"
_PDF_IDENTITY_PREFIX = "pdf:"
_LOWER_HEX = re.compile(r"[0-9a-f]{64}\Z")

# Query-side lexical analysis. Only the question is analysed here, never document
# content, so these lists describe how a question is phrased rather than a corpus.
_QUESTION_TOKEN = re.compile(r"[a-z]+|\d+")
_QUERY_FUNCTION_WORDS = frozenset({
    "a",
    "about",
    "all",
    "also",
    "an",
    "and",
    "any",
    "are",
    "as",
    "assume",
    "assuming",
    "at",
    "be",
    "been",
    "being",
    "by",
    "can",
    "could",
    "define",
    "did",
    "do",
    "does",
    "for",
    "from",
    "give",
    "has",
    "have",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "more",
    "no",
    "not",
    "of",
    "on",
    "only",
    "or",
    "our",
    "please",
    "respond",
    "response",
    "should",
    "show",
    "shown",
    "state",
    "such",
    "than",
    "that",
    "the",
    "their",
    "then",
    "these",
    "they",
    "this",
    "those",
    "to",
    "use",
    "using",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "within",
    "would",
    "you",
    "your",
})
# A question may describe the answering task instead of the evidence. Those
# sentences are dropped so the remaining terms describe the document.
_INSTRUCTION_SENTENCE = re.compile(
    r"(?:respond|answer|reply|give (?:a|the) (?:response|answer))"
    r"|(?:assuming|assume|take) the perspective"
    r"|(?:relying|based) (?:on|upon) the details"
    r"|(?:shown|stated|reported) (?:in|within) the"
)
_NUMBER = re.compile(r"\b\d[\d,]{2,}\b")
# Suffix rewrites that keep a financial term's meaning; see _stem.
_SUFFIX_RULES: tuple[tuple[str, str], ...] = (
    ("ies", "y"),
    ("sses", "ss"),
    ("ches", "ch"),
    ("shes", "sh"),
    ("xes", "x"),
    ("zes", "z"),
)
# Pseudo-relevance feedback: after a first pass, the wording of the highest ranked
# pages is offered back as extra query evidence. This is how a question phrased with
# "legal battles" reaches a filing that files the same subject under "Legal
# Proceedings", without a hand-written map from questions to document wording.
_PRF_PAGES = 8
_PRF_TERMS = 8
_PRF_WEIGHT = 0.6
_PRF_MIN_TERM_LENGTH = 4
_BM25_K1 = 1.2
_BM25_B = 0.75
_COVERAGE_WEIGHT = 2.0
# Financial statements are number-dense; a page that carries the figures a question
# asks about is usually a table, so local numeric density is a small positive prior.
_NUMBER_WEIGHT = 8.0
_VARIANT_WEIGHT = 1.0
# General finance-report wording: a question's short form against the phrasing a
# filing uses. This is domain knowledge about financial statements, never a
# document identity, page number, or answer.
_REPORT_WORDING: dict[str, tuple[str, ...]] = {
    "acquisition": ("acquisitions", "acquired"),
    "battle": ("legal", "proceedings", "litigation", "claims"),
    "capex": ("capital", "expenditures", "spending"),
    "categor": ("segment", "product"),
    "customer": ("customers", "clients"),
    "cybersecur": ("security", "breach"),
    "cyclical": ("cyclical", "seasonal"),
    "declin": ("decrease", "decline"),
    "disput": ("litigation", "claims", "proceedings"),
    "drove": ("increase", "decrease", "change"),
    "ebitda": ("operating", "income", "depreciation", "amortization"),
    "employee": ("headcount",),
    "forecast": ("forecast", "estimate"),
    "geograph": ("geographic", "regions", "segment", "international"),
    "headcount": ("employees",),
    "healthy": ("liquidity", "ratio"),
    "improv": ("increase", "improvement"),
    "legal": ("proceedings", "litigation", "claims"),
    "liabilit": ("liabilities",),
    "margin": ("gross", "profit", "operating"),
    "quick": ("liquidity", "current", "ratio"),
    "retain": ("retention", "retained"),
    "revenue": ("revenues", "net", "sales"),
    "sales": ("revenue", "revenues"),
    "tax": ("income", "effective"),
}


class PDFDocumentInput(BaseModel):
    """One complete original PDF to split into physical-page Sources."""

    document_id: str
    content: bytes


class PDFPageInput(BaseModel):
    """One physical page together with the identity of its document revision."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("PDF document_id must not be blank")  # noqa: TRY003
        return value

    @field_validator("document_sha256")
    @classmethod
    def validate_document_sha256(cls, value: str) -> str:
        if _LOWER_HEX.fullmatch(value) is None:
            raise ValueError("PDF document_sha256 must be 64 lowercase hexadecimal characters")  # noqa: TRY003
        return value

    @field_validator("page_number")
    @classmethod
    def validate_page_number(cls, value: int) -> int:
        if value < 1:
            raise ValueError("PDF page_number must start at 1")  # noqa: TRY003
        return value


class PDFPageSource(Source):
    """A captured physical page that can be read back as its exact page input."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Parse every physical page of one original PDF without returning partial pages.

    The complete document is decoded and every page is extracted and validated,
    including its encoded Source identity, before any page is returned, so a
    document that cannot be read whole fails as one ValueError.
    """

    _require_document_id(value.document_id)
    document_sha256 = hashlib.sha256(value.content).hexdigest()
    texts = _extract_page_texts(value.content)
    pages = tuple(
        PDFPageInput(
            document_id=value.document_id,
            document_sha256=document_sha256,
            page_number=page_number,
            text=text,
        )
        for page_number, text in enumerate(texts, start=1)
    )
    for page in pages:
        pdf_page_source_id(page.document_id, page.document_sha256, page.page_number)
    return pages


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    """Return the stable, delimiter-safe identity of one captured physical page."""

    _require_document_id(document_id)
    _require_sha256(document_sha256)
    _require_page_number(page_number)
    source_id = f"{_PDF_IDENTITY_PREFIX}{quote(document_id, safe='')}:{document_sha256}:{page_number}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"PDF page source id exceeds {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
    return source_id


def parse_pdf_page_source_id(source_id: str) -> tuple[str, str, int] | None:
    """Return (document_id, document_sha256, page_number) for one page identity."""

    if not isinstance(source_id, str) or not source_id.startswith(_PDF_IDENTITY_PREFIX):
        return None
    encoded_document_id, separator, remainder = source_id[len(_PDF_IDENTITY_PREFIX) :].partition(":")
    if not separator:
        return None
    document_sha256, separator, encoded_page_number = remainder.rpartition(":")
    if not separator or _LOWER_HEX.fullmatch(document_sha256) is None:
        return None
    try:
        page_number = int(encoded_page_number)
    except ValueError:
        return None
    if page_number < 1:
        return None
    return unquote(encoded_document_id), document_sha256, page_number


class PDFPageSourceAdapter:
    """Materialize and read one captured physical page."""

    input_class = PDFPageInput
    name = PDF_PAGE_SOURCE_NAME
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        source_id = pdf_page_source_id(value.document_id, value.document_sha256, value.page_number)
        return PDFPageSource(
            name=source_id,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


class PDFPageTextEvidenceProjection:
    """Expose one captured page through the shared text-evidence capability."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PDF_PAGE_SOURCE_NAME,
            source_id=source.name,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


PDF_PAGE_SOURCE_ADAPTER = PDFPageSourceAdapter()
PDF_PAGE_SOURCE_DEFINITION = AdapterSourceDefinition(
    PDF_PAGE_SOURCE_ADAPTER,
    version=PDF_PAGE_SOURCE_VERSION,
    projections=(PDFPageTextEvidenceProjection(),),
)


async def search_pdf_pages(
    memory: Any,
    question: str,
    limit: int = 10,
) -> tuple[MemoryCitation, ...]:
    """Return up to limit distinct evidence-page citations for one question.

    The scoped Memory head is read through the public listing API, every stored page
    is scored against the question offline, and each returned citation is re-read
    through memory.get so a citation can only point at a page that is actually stored
    in that Memory revision.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("PDF question must contain non-whitespace content")  # noqa: TRY003
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("PDF search limit must be a positive integer")  # noqa: TRY003

    from powercontext.builtin.runtime import GetMemoryEntryRequest

    entries = await _listed_pages(memory)
    if not entries:
        return ()
    ranked = _rank_pages(entries, _question_terms(question))
    return await _distinct_page_citations(memory, GetMemoryEntryRequest, ranked, limit)


async def _listed_pages(memory: Any) -> tuple[Any, ...]:
    """Return the active page entries of the scoped Memory head, in stored order."""

    listing = await memory.list()
    return tuple(entry for entry in listing.entries if entry.state == "active" and _entry_page(entry.entry) is not None)


def _rank_pages(entries: Sequence[Any], terms: Sequence[str]) -> tuple[Any, ...]:
    """Order stored page entries by lexical evidence for the question.

    Statistics are computed over the pages this scoped Memory actually holds, so a page
    is ranked against the collection the caller scoped to rather than a fixed corpus.
    Ranking runs twice: a first pass over the question terms, then a second pass that
    also weighs the distinctive wording of the first pass's best pages. That feedback
    is what lets a question phrased in everyday words reach a filing that uses its own
    terminology.
    """

    if not terms:
        return tuple(entries)
    documents = [(entry, _page_tokens(entry.entry.text)) for entry in entries]
    index = _PageIndex(documents)
    weights = _term_weights(terms)
    ranked = index.rank(weights)
    if not ranked:
        return ()
    feedback = index.feedback_terms(ranked[:_PRF_PAGES], set(weights))
    if feedback:
        for term in feedback:
            weights.setdefault(term, _PRF_WEIGHT)
        ranked = index.rank(weights)
    return tuple(entry for entry, _score in ranked)


class _PageIndex:
    """Lexical statistics over one scoped collection of captured pages."""

    __slots__ = ("_average_length", "_documents", "_frequencies", "_maximum_numbers")

    def __init__(self, documents: Sequence[tuple[Any, tuple[str, ...]]]) -> None:
        self._documents = documents
        self._frequencies: Counter[str] = Counter()
        for _entry, tokens in documents:
            self._frequencies.update(set(tokens))
        count = max(len(documents), 1)
        self._average_length = sum(len(tokens) for _entry, tokens in documents) / count
        self._maximum_numbers = (
            max(
                (len(_NUMBER.findall(entry.entry.text)) for entry, _tokens in documents),
                default=1,
            )
            or 1
        )

    def rank(self, weights: Mapping[str, float]) -> list[tuple[Any, float]]:
        """Score every page that shares at least one weighted term, best first."""

        document_count = max(len(self._documents), 1)
        total_weight = sum(weights.values()) or 1.0
        scored: list[tuple[float, int, Any]] = []
        for position, (entry, tokens) in enumerate(self._documents):
            counts = Counter(tokens)
            present = [term for term in weights if term in counts]
            if not present:
                continue
            length = len(tokens)
            score = 0.0
            for term in present:
                occurrences = counts[term]
                document_frequency = self._frequencies[term]
                inverse_frequency = math.log(
                    1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                score += (
                    weights[term]
                    * inverse_frequency
                    * (occurrences * (_BM25_K1 + 1))
                    / (occurrences + _BM25_K1 * (1 - _BM25_B + _BM25_B * length / max(self._average_length, 1.0)))
                )
            score += _COVERAGE_WEIGHT * sum(weights[term] for term in present) / total_weight
            score += _NUMBER_WEIGHT * len(_NUMBER.findall(entry.entry.text)) / self._maximum_numbers
            scored.append((-score, position, entry))
        scored.sort(key=lambda item: (item[0], item[1]))
        return [(entry, -score) for score, _position, entry in scored]

    def feedback_terms(self, top: Sequence[tuple[Any, float]], exclude: set[str]) -> tuple[str, ...]:
        """Pick distinctive wording from the best first-pass pages.

        A term qualifies when it appears on several of those pages and is rare enough in
        the collection to carry signal, so boilerplate shared by every page is skipped.
        """

        candidates: Counter[str] = Counter()
        by_entry = {id(entry): tokens for entry, tokens in self._documents}
        for entry, _score in top:
            tokens = by_entry.get(id(entry), ())
            for term in set(tokens):
                if term in exclude or term in _QUERY_FUNCTION_WORDS:
                    continue
                if len(term) < _PRF_MIN_TERM_LENGTH or not term.isalpha():
                    continue
                if self._frequencies[term] < 2:
                    continue
                candidates[term] += 1
        scored = [
            (count * math.log(1 + len(self._documents) / self._frequencies[term]), term)
            for term, count in candidates.items()
        ]
        scored.sort(reverse=True)
        return tuple(term for _weight, term in scored[:_PRF_TERMS])


def _term_weights(terms: Sequence[str]) -> dict[str, float]:
    """Weight question terms and add lower-weighted finance-report wording variants."""

    weights: dict[str, float] = {}
    for term in terms:
        weights[_stem(term)] = 1.0
        for variant in _report_wording(term):
            weights[_stem(variant)] = max(weights.get(_stem(variant), 0.0), _VARIANT_WEIGHT)
    return weights


async def _distinct_page_citations(
    memory: Any,
    request_class: Any,
    ranked: Sequence[Any],
    limit: int,
) -> tuple[MemoryCitation, ...]:
    """Resolve ranked entries into distinct, verified page citations."""

    citations: list[MemoryCitation] = []
    seen: set[tuple[str, int]] = set()
    for entry in ranked:
        if len(citations) >= limit:
            break
        page = _entry_page(entry.entry)
        if page is None or page in seen:
            continue
        citation = MemoryCitation(
            memory_ref=entry.memory_ref,
            entry_id=entry.entry.entry_id,
            entry_version_id=entry.entry.entry_version_id,
        )
        verified = (await memory.get(request_class(citation=citation))).entry
        if _entry_page(verified) != page:
            continue
        seen.add(page)
        citations.append(citation)
    return tuple(citations)


def _entry_page(entry: Any) -> tuple[str, int] | None:
    """Return the single page an entry evidences, or None when it is ambiguous."""

    pages: list[tuple[str, int]] = []
    for reference in entry.sources:
        if reference.source_type != PDF_PAGE_SOURCE_NAME:
            continue
        parsed = parse_pdf_page_source_id(reference.source_id)
        if parsed is None:
            continue
        document_id, _document_sha256, page_number = parsed
        pages.append((document_id, page_number))
    distinct = tuple(dict.fromkeys(pages))
    if len(distinct) != 1:
        return None
    return distinct[0]


def _question_terms(question: str) -> tuple[str, ...]:
    """Return the document-facing content terms of one natural-language question."""

    kept: list[str] = []
    for sentence in re.split(r"(?<=[.!?;])\s+|\n+", question):
        if _INSTRUCTION_SENTENCE.search(sentence.casefold()):
            continue
        kept.append(sentence)
    text = " ".join(kept) if kept else question
    tokens = _QUESTION_TOKEN.findall(text.casefold())
    terms = [token for token in tokens if token not in _QUERY_FUNCTION_WORDS]
    if not terms:
        terms = tokens
    return tuple(dict.fromkeys(terms))


def _report_wording(term: str) -> tuple[str, ...]:
    """Return the wording a financial statement uses for one question term.

    These are general finance-report conventions, applied to the question only: a
    filing says "capital expenditures" where a question says "capex", and "legal
    proceedings" where a question says "legal battles". They never name a document,
    a page, or an answer.
    """

    for stem, wording in _REPORT_WORDING.items():
        if term.startswith(stem) or stem.startswith(term):
            return wording
    return ()


def _page_tokens(text: str) -> tuple[str, ...]:
    """Tokenize one page into comparable word forms."""

    return tuple(_stem(token) for token in _QUESTION_TOKEN.findall(text.casefold()))


def _stem(token: str) -> str:
    """Reduce common English inflections so question and filing share one form.

    Only suffixes that never change the meaning of a financial term are stripped, and
    the rules are deliberately conservative: an over-stemmed term would match unrelated
    pages, which is worse than missing a plural.
    """

    if len(token) <= 3 or not token.isalpha():
        return token
    for suffix, replacement in _SUFFIX_RULES:
        if token.endswith(suffix):
            return token[: -len(suffix)] + replacement
    if token.endswith(("ss", "us", "is")):
        return token
    if token.endswith("s"):
        return token[:-1]
    if token.endswith("ing") and len(token) > 6:
        return token[:-3]
    if token.endswith("ed") and len(token) > 5:
        return token[:-2]
    return token


def _extract_page_texts(content: bytes) -> tuple[str, ...]:
    """Return one extracted text per physical page, failing whole on any error."""

    if not content:
        raise ValueError("PDF content must not be empty")  # noqa: TRY003
    reader = _open_pdf(content)
    try:
        page_count = len(reader.pages)
    except Exception as error:
        raise ValueError("PDF page tree cannot be read") from error  # noqa: TRY003
    if page_count < 1:
        raise ValueError("PDF must contain at least one physical page")  # noqa: TRY003
    texts: list[str] = []
    for index in range(page_count):
        try:
            text = reader.pages[index].extract_text()
        except Exception as error:
            raise ValueError(f"PDF page {index + 1} cannot be parsed") from error  # noqa: TRY003
        texts.append("" if text is None else text)
    return tuple(texts)


def _open_pdf(content: bytes) -> Any:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(content))
    except Exception as error:
        raise ValueError("PDF content cannot be parsed") from error  # noqa: TRY003
    if reader.is_encrypted:
        try:
            decrypted = reader.decrypt("")
        except Exception as error:
            raise ValueError("PDF permissions cannot be read") from error  # noqa: TRY003
        if not decrypted:
            raise ValueError("PDF requires a user password that was not supplied")  # noqa: TRY003
    return reader


def _require_document_id(document_id: str) -> None:
    if not isinstance(document_id, str):
        raise TypeError("PDF document_id must be a string")  # noqa: TRY003
    if not document_id.strip():
        raise ValueError("PDF document_id must not be blank")  # noqa: TRY003


def _require_sha256(document_sha256: str) -> None:
    if not isinstance(document_sha256, str) or _LOWER_HEX.fullmatch(document_sha256) is None:
        raise ValueError("PDF document_sha256 must be 64 lowercase hexadecimal characters")  # noqa: TRY003


def _require_page_number(page_number: int) -> None:
    if isinstance(page_number, bool) or not isinstance(page_number, int) or page_number < 1:
        raise ValueError("PDF page_number must start at 1")  # noqa: TRY003


__all__ = [
    "PDF_PAGE_SOURCE_ADAPTER",
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDF_PAGE_SOURCE_NAME",
    "PDF_PAGE_SOURCE_VERSION",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "PDFPageSourceAdapter",
    "PDFPageTextEvidenceProjection",
    "parse_pdf_document",
    "parse_pdf_page_source_id",
    "pdf_page_source_id",
    "search_pdf_pages",
]
