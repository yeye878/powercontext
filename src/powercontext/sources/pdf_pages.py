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

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import io
import math
import re
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

_WHITESPACE = re.compile(r"\s+")
_QUESTION_TOKEN = re.compile(r"[a-z]+|\d+")
_ALNUM_TOKEN = re.compile(r"[a-z0-9]+")
_NUMBER = re.compile(r"\b\d[\d,]{2,}\b")

_FILING_META_TOKENS = frozenset({
    "10",
    "10k",
    "10q",
    "20f",
    "40f",
    "8k",
    "annual",
    "dated",
    "earnings",
    "form",
    "fy",
    "k",
    "q",
    "q1",
    "q2",
    "q3",
    "q4",
    "report",
})

# Standard corporate ticker / shorthand forms used in financial questions.
_ENTITY_SHORTHANDS: dict[str, tuple[str, ...]] = {
    "amex": ("americanexpress", "american", "express"),
    "amzn": ("amazon",),
    "ba": ("boeing",),
    "jnj": ("johnson",),
    "jpm": ("jpmorgan", "morgan", "chase"),
    "mgm": ("mgmresorts",),
    "mmm": ("3m",),
    "pep": ("pepsico",),
}

# Meta-instruction clauses that describe how to format an answer or handle N/A
# cases without naming the target financial statements.
_BOILERPLATE_CLAUSE = re.compile(
    r"if\s+[^.?!]*?\b(?:not\s+(?:a\s+)?(?:useful|relevant|meaningful|explicitly\s+outlined)"
    r"|not\s+something\s+that\s+a\s+financial\s+analyst)[^.?!]*[.?!]?"
    r"|respond\s+to\s+the\s+question\s+by\s+assuming\s+the\s+perspective\s+of\s+an\s+investment\s+analyst"
    r"\s+who\s+can\s+only\s+use\s+the\s+details\s+shown\s+(?:within|in)"
    r"|give\s+a\s+response\s+to\s+the\s+question\s+by\s+relying\s+on\s+the\s+details\s+shown\s+(?:within|in)"
    r"|address\s+the\s+question\s+by\s+using\s+the\s+line\s+items\s+and\s+information\s+shown\s+(?:within|in)"
    r"|calculate\s+what\s+was\s+asked\s+by\s+utilizing\s+the\s+line\s+items\s+clearly\s+shown\s+(?:within|in)"
    r"|round\s+your\s+answer\s+to\s+[^.?!]*[.?!]?"
    r"|\(in\s+units\s+of\s+percents\s+and\s+round\s+to\s+[^)]*\)"
    r"|answer\s+in\s+usd\s+\w+[.?!]?"
)

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
    "based",
    "be",
    "been",
    "being",
    "between",
    "by",
    "calculate",
    "can",
    "clearly",
    "company",
    "could",
    "data",
    "define",
    "defined",
    "details",
    "did",
    "do",
    "does",
    "during",
    "each",
    "ended",
    "ending",
    "explain",
    "explicitly",
    "for",
    "from",
    "give",
    "had",
    "has",
    "have",
    "how",
    "if",
    "in",
    "information",
    "into",
    "is",
    "it",
    "items",
    "its",
    "like",
    "line",
    "many",
    "may",
    "meaningful",
    "measure",
    "mentioned",
    "metric",
    "more",
    "most",
    "much",
    "no",
    "not",
    "of",
    "on",
    "only",
    "or",
    "other",
    "our",
    "outlined",
    "overall",
    "period",
    "perspective",
    "place",
    "places",
    "please",
    "primarily",
    "profile",
    "provided",
    "question",
    "quite",
    "reasonably",
    "relevant",
    "relying",
    "report",
    "reported",
    "respond",
    "response",
    "roughly",
    "round",
    "should",
    "show",
    "shown",
    "state",
    "statement",
    "such",
    "than",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "three",
    "time",
    "times",
    "to",
    "two",
    "under",
    "units",
    "use",
    "useful",
    "using",
    "utilizing",
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
    "year",
    "years",
    "you",
    "your",
})

_SUFFIX_RULES: tuple[tuple[str, str], ...] = (
    ("ies", "y"),
    ("sses", "ss"),
    ("ches", "ch"),
    ("shes", "sh"),
    ("xes", "x"),
    ("zes", "z"),
)

_PRF_PAGES = 6
_PRF_TERMS = 6
_PRF_WEIGHT = 0.45
_PRF_MIN_TERM_LENGTH = 4
_BM25_K1 = 1.2
_BM25_B = 0.75
_COVERAGE_WEIGHT = 2.5
_NUMBER_WEIGHT = 3.0
_VARIANT_WEIGHT = 0.85
_PHRASE_WEIGHT = 0.95
_MIN_STATEMENT_NUMBERS = 8

_INCOME_HEADINGS = (
    "consolidated statements of operations",
    "consolidated statement of operations",
    "consolidated statements of income",
    "consolidated statement of income",
    "consolidated statements of earnings",
    "consolidated statement of earnings",
    "statements of consolidated income",
    "statement of consolidated income",
    "statements of consolidated operations",
)
_BALANCE_HEADINGS = (
    "consolidated balance sheets",
    "consolidated balance sheet",
    "consolidated statements of financial position",
    "consolidated statement of financial position",
    "balance sheets",
)
_CASHFLOW_HEADINGS = (
    "consolidated statements of cash flows",
    "consolidated statement of cash flows",
    "statements of consolidated cash flows",
    "statement of consolidated cash flows",
)

_INCOME_QUERY_MARKERS = (
    "statement of income",
    "statements of income",
    "income statement",
    "income statements",
    "statement of operations",
    "statements of operations",
    "p&l",
    "gross margin",
    "gross margins",
    "operating margin",
    "ebitda",
    "restructuring",
    "return on assets",
    "roa",
    "days payable outstanding",
    "dpo",
    "inventory turnover",
    "capital-intensive",
    "capital intensive",
    "change in revenue",
    "revenue change",
    "top line",
    "topline",
    "effective tax rate",
    "tax rate",
    "net income",
    "net earnings",
    "operating income",
    "operating profit",
)
_BALANCE_QUERY_MARKERS = (
    "balance sheet",
    "balance sheets",
    "statement of financial position",
    "quick ratio",
    "liquidity profile",
    "working capital",
    "inventory turnover",
    "days payable outstanding",
    "dpo",
    "return on assets",
    "roa",
    "capital-intensive",
    "capital intensive",
    "ppne",
    "pp&e",
    "property, plant",
    "largest liability",
    "debt on balance sheet",
    "cash & cash equivalents",
    "cash and cash equivalents",
)
_CASHFLOW_QUERY_MARKERS = (
    "statement of cash flows",
    "cash flow statement",
    "cash flows",
    "cash flow",
    "capex",
    "capital expenditures",
    "depreciation and amortization",
    "d&a",
    "ebitda",
    "capital-intensive",
    "capital intensive",
    "operating, investing, and financing",
    "operations, investing, and financing",
    "paid dividends",
)
_BUSINESS_OVERVIEW_MARKERS = (
    "geographies",
    "primarily operates",
    "primarily operate",
    "what industry",
    "major products and services",
    "primary customers",
    "customer concentration",
    "subject to cyclicality",
)
_LEGAL_MARKERS = (
    "legal battles",
    "legal proceedings",
    "litigation",
    "lawsuits",
)

_PHRASE_EXPANSIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("p&l", ("income", "operations", "earnings", "revenue", "revenues")),
    ("d&a", ("depreciation", "amortization")),
    ("m&a", ("acquisitions", "divestitures", "acquired")),
    ("sg&a", ("selling", "general", "administrative")),
    ("r&d", ("research", "development")),
    ("ppne", ("property", "plant", "equipment")),
    ("pp&e", ("property", "plant", "equipment")),
    ("cogs", ("cost", "sales", "goods", "products", "sold")),
    ("days payable outstanding", ("accounts", "payable", "cost", "sales", "inventories")),
    ("dpo", ("accounts", "payable", "cost", "sales", "inventories")),
    ("return on assets", ("net", "income", "total", "assets")),
    ("roa", ("net", "income", "total", "assets")),
    ("working capital", ("current", "assets", "liabilities", "cash", "receivable", "inventories")),
    (
        "capital-intensive",
        ("capital", "expenditures", "property", "plant", "equipment", "total", "assets", "revenues"),
    ),
    (
        "capital intensive",
        ("capital", "expenditures", "property", "plant", "equipment", "total", "assets", "revenues"),
    ),
    ("quick ratio", ("current", "assets", "cash", "equivalents", "receivable", "liabilities")),
    ("inventory turnover", ("inventories", "inventory", "cost", "sales", "goods", "sold")),
    ("gross margin", ("gross", "profit", "margin", "cost", "sales", "products", "sold", "revenues")),
    ("operating margin", ("operating", "income", "profit", "margin", "expenses", "cost", "sales", "revenues")),
    ("effective tax rate", ("income", "taxes", "tax", "effective", "rate", "provision", "earnings")),
    ("legal battles", ("legal", "proceedings", "litigation", "claims", "actions", "contingencies", "dispute")),
    ("customer concentration", ("customer", "customers", "accounted", "consolidated", "revenue", "segment")),
    ("primary customers", ("customers", "customer", "revenues", "government", "commercial", "airlines", "contracts")),
    ("products and services", ("business", "overview", "products", "services", "offering", "segments")),
    ("what industry", ("item", "business", "company", "global", "leader", "developing", "producing", "products")),
    ("retain card members", ("retention", "card", "member", "members", "acquisition", "prospect", "engagement")),
)

_REPORT_WORDING: dict[str, tuple[str, ...]] = {
    "acquisition": ("acquisitions", "acquired", "divestitures"),
    "battle": ("legal", "proceedings", "litigation", "claims", "actions"),
    "capex": ("capital", "expenditures", "spending", "property", "equipment"),
    "categor": ("segment", "segments", "product", "services"),
    "concentration": ("customer", "accounted", "revenue", "consolidated"),
    "customer": ("customers", "clients", "accounted", "contracts"),
    "cybersecur": ("security", "breach"),
    "cyclical": ("cyclical", "seasonal", "economic", "demand"),
    "declin": ("decrease", "decline"),
    "disput": ("litigation", "claims", "proceedings"),
    "dividend": ("dividends", "common", "stock", "paid", "cash"),
    "drove": ("increase", "decrease", "change", "driven", "offset"),
    "ebitda": ("operating", "income", "profit", "depreciation", "amortization"),
    "employee": ("headcount", "employees"),
    "forecast": ("forecast", "estimate", "production", "rate", "deliveries"),
    "geograph": ("geographic", "regions", "countries", "international", "operations", "segment"),
    "headcount": ("employees",),
    "healthy": ("liquidity", "current", "assets", "liabilities"),
    "improv": ("increase", "improvement"),
    "industry": ("business", "company", "global", "products", "leader"),
    "inventory": ("inventories", "cost", "sales", "goods"),
    "legal": ("proceedings", "litigation", "claims", "actions", "contingencies"),
    "liabilit": ("liabilities", "deposits", "debt"),
    "liquidity": ("current", "assets", "cash", "equivalents", "liabilities"),
    "margin": ("gross", "profit", "operating", "cost", "sales", "products", "sold"),
    "payable": ("accounts", "payable", "liabilities", "cost", "sales"),
    "ppne": ("property", "plant", "equipment"),
    "quick": ("current", "assets", "cash", "receivable", "liabilities"),
    "retain": ("retention", "retained", "member", "members"),
    "revenue": ("revenues", "net", "sales"),
    "sales": ("revenue", "revenues", "net"),
    "segment": ("segments", "business", "operating", "revenue"),
    "tax": ("income", "taxes", "effective", "rate", "provision"),
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

    The scoped Memory head is read through the public listing API, narrowed to
    any filing whose document identity is explicitly referenced by the question,
    scored offline against the question's financial terminology and section
    intents, and re-read through memory.get so every returned citation is
    guaranteed to point at a stored page in that Memory revision.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("PDF question must contain non-whitespace content")  # noqa: TRY003
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("PDF search limit must be a positive integer")  # noqa: TRY003

    from powercontext.builtin.runtime import GetMemoryEntryRequest

    entries = await _listed_pages(memory)
    if not entries:
        return ()
    scoped = _scope_entries_to_question(entries, question)
    terms, weights, intents = _extract_query_info(question)
    ranked = _rank_pages(scoped, weights=weights, intents=intents)
    selected = _select_pages(ranked, scoped=scoped, intents=intents, limit=limit)
    return await _distinct_page_citations(memory, GetMemoryEntryRequest, selected, limit)


async def _listed_pages(memory: Any) -> tuple[Any, ...]:
    """Return the active page entries of the scoped Memory head, in stored order."""

    listing = await memory.list()
    return tuple(entry for entry in listing.entries if entry.state == "active" and _entry_page(entry.entry) is not None)


def _document_signature(document_id: str) -> tuple[tuple[str, ...], set[str], set[str], set[str]]:
    """Decompose one document_id into (entity_parts, years, quarters, forms)."""

    parts = _ALNUM_TOKEN.findall(document_id.casefold())
    entity: list[str] = []
    years: set[str] = set()
    quarters: set[str] = set()
    forms: set[str] = set()
    for part in parts:
        if part in _FILING_META_TOKENS:
            if part in {"10k", "10q", "8k", "earnings"}:
                forms.add(part)
            elif part in {"q1", "q2", "q3", "q4"}:
                quarters.add(part)
            continue
        year_quarter = re.fullmatch(r"(19\d\d|20\d\d)(q[1-4])?", part)
        if year_quarter is not None:
            years.add(year_quarter.group(1))
            if year_quarter.group(2):
                quarters.add(year_quarter.group(2))
            continue
        if part.isdigit():
            continue
        entity.append(part)
    return tuple(dict.fromkeys(entity)), years, quarters, forms


def _scope_entries_to_question(entries: Sequence[Any], question: str) -> tuple[Any, ...]:
    """Narrow a multi-document Memory pool when the question names a specific filing entity."""

    by_document: dict[str, list[Any]] = {}
    for entry in entries:
        page = _entry_page(entry.entry)
        if page is not None:
            by_document.setdefault(page[0], []).append(entry)
    if len(by_document) <= 1:
        return tuple(entries)

    signatures = {doc_id: _document_signature(doc_id) for doc_id in by_document}
    shared_entity = (
        set.intersection(*(set(sig[0]) for sig in signatures.values()))
        if signatures
        else set()
    )

    folded = question.casefold()
    alnum_tokens = _ALNUM_TOKEN.findall(folded)
    word_tokens = _QUESTION_TOKEN.findall(folded)
    question_tokens = set(alnum_tokens) | set(word_tokens)
    for seq in (alnum_tokens, word_tokens):
        for index in range(len(seq) - 1):
            question_tokens.add(seq[index] + seq[index + 1])
            if index + 2 < len(seq):
                question_tokens.add(seq[index] + seq[index + 1] + seq[index + 2])
    for token in list(question_tokens):
        if token in _ENTITY_SHORTHANDS:
            question_tokens.update(_ENTITY_SHORTHANDS[token])

    question_years = set(re.findall(r"(?:^|\b|fy\s*['’]?)(19\d\d|20\d\d)\b", folded))
    for short_year in re.findall(r"\bfy\s*['’]?(\d{2})\b", folded):
        question_years.add("20" + short_year)
    question_quarters = set(re.findall(r"\b(q[1-4])\b", folded))
    for quarter_code, quarter_phrase in (
        ("q1", "first quarter"),
        ("q2", "second quarter"),
        ("q3", "third quarter"),
        ("q4", "fourth quarter"),
    ):
        if quarter_phrase in folded:
            question_quarters.add(quarter_code)

    matched_documents: list[str] = []
    for doc_id, (entity, _years, _quarters, _forms) in signatures.items():
        distinguishing = [part for part in entity if part not in shared_entity]
        if distinguishing and all(part in question_tokens for part in distinguishing):
            matched_documents.append(doc_id)

    if not matched_documents:
        return tuple(entries)

    if len(matched_documents) > 1 and (question_years or question_quarters):
        scored_documents: list[tuple[int, str]] = []
        for doc_id in matched_documents:
            _entity, years, quarters, _forms = signatures[doc_id]
            period_score = 0
            if question_years and (years & question_years):
                period_score += 4
                if max(question_years) in years:
                    period_score += 2
            if question_quarters and (quarters & question_quarters):
                period_score += 3
            scored_documents.append((period_score, doc_id))
        best_score = max(score for score, _doc_id in scored_documents)
        if best_score > 0:
            matched_documents = [doc_id for score, doc_id in scored_documents if score == best_score]

    scoped: list[Any] = []
    for doc_id in matched_documents:
        scoped.extend(by_document[doc_id])
    return tuple(scoped)


def _question_terms(question: str) -> tuple[str, ...]:
    """Return the document-facing content terms of one natural-language question."""

    terms, _weights, _intents = _extract_query_info(question)
    return terms


def _extract_query_info(question: str) -> tuple[tuple[str, ...], dict[str, float], set[str]]:
    """Extract normalized query terms, weighted expansions, and section intents."""

    folded = _WHITESPACE.sub(" ", question.casefold()).strip()
    cleaned = _BOILERPLATE_CLAUSE.sub(" ", folded)
    if not cleaned.strip():
        cleaned = folded

    extra_years = ["20" + match for match in re.findall(r"\bfy\s*['’]?(\d{2})\b", cleaned)]
    raw_tokens = _QUESTION_TOKEN.findall(cleaned) + extra_years
    terms = [
        token
        for token in raw_tokens
        if token not in _QUERY_FUNCTION_WORDS and (len(token) > 1 or token.isdigit())
    ]
    if not terms:
        terms = raw_tokens
    unique_terms = tuple(dict.fromkeys(terms))

    weights: dict[str, float] = {}
    for term in unique_terms:
        weights[_stem(term)] = 1.0
        for variant in _report_wording(term):
            weights[_stem(variant)] = max(weights.get(_stem(variant), 0.0), _VARIANT_WEIGHT)

    for phrase, variants in _PHRASE_EXPANSIONS:
        if phrase in folded:
            for variant in variants:
                weights[_stem(variant)] = max(weights.get(_stem(variant), 0.0), _PHRASE_WEIGHT)

    intents: set[str] = set()
    if any(marker in folded for marker in _INCOME_QUERY_MARKERS):
        intents.add("income")
    if any(marker in folded for marker in _BALANCE_QUERY_MARKERS):
        intents.add("balance")
    if any(marker in folded for marker in _CASHFLOW_QUERY_MARKERS):
        intents.add("cashflow")
    if any(marker in folded for marker in _BUSINESS_OVERVIEW_MARKERS):
        intents.add("business")
    if any(marker in folded for marker in _LEGAL_MARKERS):
        intents.add("legal")
    return unique_terms, weights, intents


def _rank_pages(
    entries: Sequence[Any],
    *,
    weights: Mapping[str, float],
    intents: set[str],
) -> tuple[Any, ...]:
    """Order stored page entries by lexical and structural evidence for the question."""

    if not weights:
        return tuple(entries)
    documents = []
    for entry in entries:
        body = _entry_body_text(entry.entry.text)
        documents.append((entry, _page_tokens(body), body))
    index = _PageIndex(documents, intents=intents)
    ranked = index.rank(weights)
    if not ranked:
        return ()
    feedback = index.feedback_terms(ranked[:_PRF_PAGES], set(weights))
    if feedback:
        expanded = dict(weights)
        for term in feedback:
            expanded.setdefault(term, _PRF_WEIGHT)
        ranked = index.rank(expanded)
    return tuple(entry for entry, _score in ranked)


class _PageIndex:
    """Lexical and filing-structure statistics over a scoped collection of captured pages."""

    __slots__ = ("_average_length", "_documents", "_frequencies", "_intents", "_maximum_numbers")

    def __init__(
        self,
        documents: Sequence[tuple[Any, tuple[str, ...], str]],
        *,
        intents: set[str],
    ) -> None:
        self._documents = documents
        self._intents = intents
        self._frequencies: Counter[str] = Counter()
        for _entry, tokens, _body in documents:
            self._frequencies.update(set(tokens))
        count = max(len(documents), 1)
        self._average_length = sum(len(tokens) for _entry, tokens, _body in documents) / count
        self._maximum_numbers = (
            max(
                (len(_NUMBER.findall(body)) for _entry, _tokens, body in documents),
                default=1,
            )
            or 1
        )

    def rank(self, weights: Mapping[str, float]) -> list[tuple[Any, float]]:
        """Score every page that shares at least one weighted term, best first."""

        document_count = max(len(self._documents), 1)
        total_weight = sum(weights.values()) or 1.0
        is_numeric_query = bool(self._intents & {"income", "balance", "cashflow"})
        scored: list[tuple[float, int, Any]] = []
        for position, (entry, tokens, body) in enumerate(self._documents):
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
            number_count = len(_NUMBER.findall(body))
            if is_numeric_query:
                score += _NUMBER_WEIGHT * number_count / self._maximum_numbers
            score += self._section_bonus(entry, body, number_count)
            scored.append((-score, position, entry))
        scored.sort(key=lambda item: (item[0], item[1]))
        return [(entry, -score) for score, _position, entry in scored]

    def _section_bonus(self, entry: Any, body: str, number_count: int) -> float:
        """Reward pages whose filing section heading matches the question's intent."""

        if not self._intents:
            return 0.0
        normalized = _WHITESPACE.sub(" ", body.casefold()).strip()
        header = normalized[:420]
        page = _entry_page(entry.entry)
        page_number = page[1] if page is not None else 999
        bonus = 0.0
        if "business" in self._intents:
            if re.search(r"\bitem\s+1\.?\s*[-–—]?\s*business\b", normalized) and "item 1a" not in normalized[:300]:
                bonus += 6.0
            if page_number <= 18:
                bonus += 2.5
        if "legal" in self._intents:
            if "item 3" in normalized and "legal proceedings" in normalized:
                bonus += 8.0
            elif "legal proceedings" in normalized or "legal actions" in normalized or "litigation" in normalized:
                bonus += 4.0
        if number_count >= _MIN_STATEMENT_NUMBERS and "notes to consolidated" not in header:
            if "income" in self._intents and any(heading in header for heading in _INCOME_HEADINGS):
                bonus += 5.0
            if "balance" in self._intents and any(heading in header for heading in _BALANCE_HEADINGS):
                bonus += 5.0
            if "cashflow" in self._intents and any(heading in header for heading in _CASHFLOW_HEADINGS):
                bonus += 5.0
        return bonus

    def feedback_terms(self, top: Sequence[tuple[Any, float]], exclude: set[str]) -> tuple[str, ...]:
        """Pick distinctive wording from the best first-pass pages."""

        candidates: Counter[str] = Counter()
        by_entry = {id(entry): tokens for entry, tokens, _body in self._documents}
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


def _detect_primary_statements(entries: Sequence[Any]) -> dict[str, list[Any]]:
    """Locate the primary Income, Balance Sheet, and Cash Flow statement pages."""

    income: list[tuple[int, int, Any]] = []
    balance: list[tuple[int, int, Any]] = []
    cashflow: list[tuple[int, int, Any]] = []
    for position, entry in enumerate(entries):
        body = _entry_body_text(entry.entry.text)
        normalized = _WHITESPACE.sub(" ", body.casefold()).strip()
        header = normalized[:420]
        if "notes to consolidated" in header or "notes to financial" in header:
            continue
        number_count = len(_NUMBER.findall(body))
        if number_count < _MIN_STATEMENT_NUMBERS:
            continue
        if any(heading in header for heading in _INCOME_HEADINGS) and any(
            marker in normalized
            for marker in (
                "revenue",
                "revenues",
                "net sales",
                "operating income",
                "gross profit",
                "cost of sales",
                "cost of goods",
            )
        ):
            income.append((number_count, -position, entry))
        if any(heading in header for heading in _BALANCE_HEADINGS) and any(
            marker in normalized
            for marker in (
                "cash and cash equivalents",
                "current assets",
                "total assets",
                "accounts receivable",
                "total liabilities",
            )
        ):
            balance.append((number_count, -position, entry))
        if any(heading in header for heading in _CASHFLOW_HEADINGS) and any(
            marker in normalized
            for marker in (
                "operating activities",
                "depreciation",
                "investing activities",
                "financing activities",
            )
        ):
            cashflow.append((number_count, -position, entry))
    return {
        "income": [entry for _numbers, _pos, entry in sorted(income, reverse=True)[:2]],
        "balance": [entry for _numbers, _pos, entry in sorted(balance, reverse=True)[:2]],
        "cashflow": [entry for _numbers, _pos, entry in sorted(cashflow, reverse=True)[:2]],
    }


def _select_pages(
    ranked: Sequence[Any],
    *,
    scoped: Sequence[Any],
    intents: set[str],
    limit: int,
) -> tuple[Any, ...]:
    """Combine top-ranked lexical pages with targeted primary statements and immediate neighbours."""

    if limit < 4 or not ranked:
        return tuple(ranked[:limit])
    by_page: dict[tuple[str, int], Any] = {}
    ordered: list[tuple[str, int]] = []
    for entry in ranked:
        page = _entry_page(entry.entry)
        if page is not None and page not in by_page:
            by_page[page] = entry
            ordered.append(page)
    for entry in scoped:
        page = _entry_page(entry.entry)
        if page is not None and page not in by_page:
            by_page[page] = entry
    if len(ordered) <= 1:
        return tuple(ranked[:limit])

    statements = _detect_primary_statements(scoped)
    must_pages: list[tuple[str, int]] = []
    for kind in ("income", "balance", "cashflow"):
        if kind in intents:
            for entry in statements[kind][:2]:
                page = _entry_page(entry.entry)
                if page is not None and page not in must_pages:
                    must_pages.append(page)

    neighbour_candidates: list[tuple[str, int]] = []
    for document_id, page_number in ordered[:2]:
        for neighbour in (page_number + 1, page_number - 1):
            candidate = (document_id, neighbour)
            if neighbour >= 1 and candidate in by_page and candidate not in neighbour_candidates:
                neighbour_candidates.append(candidate)

    max_reserved = min(6, max(1, limit // 3))
    reserved: list[tuple[str, int]] = [
        page for page in must_pages[:4] if page not in ordered[: max(1, limit - max_reserved)]
    ]
    for neighbour in neighbour_candidates[:2]:
        if (
            len(reserved) < max_reserved
            and neighbour not in ordered[: max(1, limit - len(reserved) - 2)]
            and neighbour not in reserved
        ):
            reserved.append(neighbour)

    main_slots = max(1, limit - len(reserved))
    chosen: list[tuple[str, int]] = []
    used: set[tuple[str, int]] = set()
    for page in ordered[:main_slots]:
        if page not in used:
            used.add(page)
            chosen.append(page)
    for page in reserved:
        if len(chosen) < limit and page not in used:
            used.add(page)
            chosen.append(page)
    for page in ordered:
        if len(chosen) >= limit:
            break
        if page not in used:
            used.add(page)
            chosen.append(page)
    return tuple(by_page[page] for page in chosen[:limit])


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
        try:
            verified = (await memory.get(request_class(citation=citation))).entry
        except Exception:
            continue
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


def _entry_body_text(text: str) -> str:
    """Return the page body text without the leading document_id projection line."""

    if "\n" in text:
        return text.partition("\n")[2]
    return text


def _report_wording(term: str) -> tuple[str, ...]:
    """Return general financial-statement terminology variants for one question term."""

    for stem, wording in _REPORT_WORDING.items():
        if term.startswith(stem) or stem.startswith(term):
            return wording
    return ()


def _page_tokens(text: str) -> tuple[str, ...]:
    """Tokenize one page into comparable stemmed word forms."""

    return tuple(_stem(token) for token in _QUESTION_TOKEN.findall(text.casefold()))


def _stem(token: str) -> str:
    """Reduce common English inflections so question and filing share one form."""

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
    "parse_pdf_document",
    "parse_pdf_page_source_id",
    "pdf_page_source_id",
    "search_pdf_pages",
]
