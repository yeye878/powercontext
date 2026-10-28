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

"""Contract edges for physical-page PDF capture not covered by the public sample."""

from __future__ import annotations

import asyncio
import hashlib
import io

import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from powercontext.artifacts import ArtifactRef
from powercontext.sources import (
    TEXT_EVIDENCE_PROJECTION_KEY,
    SourceDefinitionRegistry,
    TextEvidence,
    manifest_for_definition,
    project_source_for_transport,
)
from powercontext.sources.pdf_pages import (
    PDF_PAGE_SOURCE_DEFINITION,
    PDFDocumentInput,
    PDFPageInput,
    PDFPageSource,
    parse_pdf_document,
    parse_pdf_page_source_id,
    pdf_page_source_id,
    search_pdf_pages,
)


def _blank_pdf(pages: int, *, title: str | None = None, width: int = 100, height: int = 200) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=width, height=height)
    if title is not None:
        writer.add_metadata({"/Title": title})
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _text_pdf(texts: list[str]) -> bytes:
    """Build one PDF page per entry with a standard font and a single text run."""

    writer = PdfWriter()
    for text in texts:
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
        })
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content = DecodedStreamObject()
        content.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode())
        page[NameObject("/Contents")] = content
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _parse(document_id: str, content: bytes):
    return parse_pdf_document(PDFDocumentInput(document_id=document_id, content=content))


@pytest.mark.parametrize("document_id", ["", "   ", "\t\n "])
def test_blank_document_id_fails(document_id: str) -> None:
    with pytest.raises(ValueError):
        _parse(document_id, _blank_pdf(1))


def test_padded_document_id_is_preserved_not_trimmed() -> None:
    pages = _parse(" doc ", _blank_pdf(1))
    assert pages[0].document_id == " doc "
    assert pdf_page_source_id(" doc ", pages[0].document_sha256, 1).startswith("pdf:%20doc%20:")


def test_every_physical_page_is_returned_in_order_including_blank_pages() -> None:
    pages = _parse("doc", _blank_pdf(4))
    assert [page.page_number for page in pages] == [1, 2, 3, 4]
    assert [page.text for page in pages] == ["", "", "", ""]


def test_blank_pages_stay_empty_and_do_not_become_placeholders() -> None:
    pages = _parse("doc", _blank_pdf(3))
    assert all(page.text == "" for page in pages)
    registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
    source = asyncio.run(registry.resolve(pages[0]))
    evidence = TextEvidence.model_validate(registry.project(source, TEXT_EVIDENCE_PROJECTION_KEY))
    assert evidence.content == "doc\n"
    assert evidence.metadata["page_number"] == 1


def test_digest_covers_the_complete_original_bytes() -> None:
    content = _blank_pdf(2)
    expected = hashlib.sha256(content).hexdigest()
    pages = _parse("doc", content)
    assert {page.document_sha256 for page in pages} == {expected}
    assert expected == expected.lower() and len(expected) == 64
    # A byte-level revision changes the digest even when the page count is unchanged.
    revised = _blank_pdf(2, title="New revision")
    assert _parse("doc", revised)[0].document_sha256 != expected


def test_revision_produces_new_page_identity_while_old_identity_survives() -> None:
    original = _parse("doc", _blank_pdf(2))
    revised = _parse("doc", _blank_pdf(2, title="New revision"))
    assert original[0].page_number == revised[0].page_number == 1
    original_id = pdf_page_source_id(original[0].document_id, original[0].document_sha256, 1)
    revised_id = pdf_page_source_id(revised[0].document_id, revised[0].document_sha256, 1)
    assert original_id != revised_id
    assert parse_pdf_page_source_id(original_id) == ("doc", original[0].document_sha256, 1)
    assert parse_pdf_page_source_id(revised_id) == ("doc", revised[0].document_sha256, 1)


def test_empty_password_permission_marker_is_a_legal_input() -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=200)
    writer.add_blank_page(width=100, height=200)
    writer.encrypt(user_password="", owner_password="owner")  # noqa: S106 - permission marker, not a credential
    buffer = io.BytesIO()
    writer.write(buffer)
    pages = _parse("permissions", buffer.getvalue())
    assert [page.page_number for page in pages] == [1, 2]
    assert [page.text for page in pages] == ["", ""]


def test_a_required_user_password_fails_atomically() -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=200)
    writer.encrypt(user_password="secret", owner_password="owner")  # noqa: S106 - test-only password
    buffer = io.BytesIO()
    writer.write(buffer)
    with pytest.raises(ValueError):
        _parse("locked", buffer.getvalue())


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"   ",
        b"not a PDF",
        b"%PDF-1.7\ntruncated",
        b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF",
    ],
)
def test_corrupt_documents_fail_with_value_error(raw: bytes) -> None:
    with pytest.raises(ValueError):
        _parse("invalid", raw)


def test_a_document_that_cannot_be_read_whole_returns_no_pages() -> None:
    good = _blank_pdf(2)
    truncated = good[: len(good) // 3]
    for raw in (truncated, good[:-20]):
        with pytest.raises(ValueError):
            _parse("partial", raw)


def test_parse_is_deterministic_and_repeatable() -> None:
    content = _blank_pdf(3)
    first = _parse("doc", content)
    second = _parse("doc", content)
    assert first == second
    assert len({page.document_sha256 for page in first}) == 1


def test_identity_is_stable_unbounded_and_collision_free() -> None:
    pages = _parse("doc", _blank_pdf(3))
    registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
    names = [asyncio.run(registry.resolve(page)).name for page in pages]
    assert len(set(names)) == 3
    assert all(len(name) <= 256 for name in names)
    assert [asyncio.run(registry.resolve(page)).name for page in pages] == names
    # Different documents with the same bytes stay distinguishable.
    other = _parse("other", _blank_pdf(3))
    assert asyncio.run(registry.resolve(other[0])).name != names[0]


def test_document_identity_components_cannot_collide_across_the_separator() -> None:
    digest = hashlib.sha256(b"x").hexdigest()
    assert pdf_page_source_id("doc:a", digest, 1) != pdf_page_source_id("doc", digest, 1)
    assert pdf_page_source_id("doc", digest, 1) != pdf_page_source_id("doc", digest, 11)


def test_identity_limit_is_enforced_before_any_page_is_returned() -> None:
    digest = hashlib.sha256(b"x").hexdigest()
    assert len(pdf_page_source_id("d" * 180, digest, 1)) <= 256
    with pytest.raises(ValueError):
        pdf_page_source_id("d" * 190, digest, 1)
    # 30 raw characters encode to 90, so the identity cannot be represented.
    with pytest.raises(ValueError):
        pdf_page_source_id("\u6587" * 70, digest, 1)


@pytest.mark.parametrize("digest", ["", "ABC", "z" * 64, "a" * 63, "a" * 65])
def test_identity_rejects_a_malformed_digest(digest: str) -> None:
    with pytest.raises(ValueError):
        pdf_page_source_id("doc", digest, 1)


@pytest.mark.parametrize("page_number", [0, -1])
def test_identity_rejects_a_page_number_below_one(page_number: int) -> None:
    with pytest.raises(ValueError):
        pdf_page_source_id("doc", hashlib.sha256(b"x").hexdigest(), page_number)


def test_page_input_rejects_an_invalid_identity() -> None:
    digest = hashlib.sha256(b"x").hexdigest()
    with pytest.raises(ValueError):
        PDFPageInput(document_id=" ", document_sha256=digest, page_number=1, text="")
    with pytest.raises(ValueError):
        PDFPageInput(document_id="doc", document_sha256=digest.upper(), page_number=1, text="")
    with pytest.raises(ValueError):
        PDFPageInput(document_id="doc", document_sha256=digest, page_number=0, text="")


def test_foreign_source_ids_are_not_parsed_as_page_identities() -> None:
    assert parse_pdf_page_source_id("csv:doc:1") is None
    assert parse_pdf_page_source_id("pdf:doc:nothex:1") is None
    assert parse_pdf_page_source_id("pdf:doc") is None
    assert parse_pdf_page_source_id("pdf:doc:" + "a" * 64 + ":0") is None
    assert parse_pdf_page_source_id("pdf:doc:" + "a" * 64 + ":x") is None


def test_definition_materializes_captured_pages_and_reads_input_back() -> None:
    registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
    page = _parse("doc", _blank_pdf(1))[0]
    source = asyncio.run(registry.resolve(page))
    assert isinstance(source, PDFPageSource)
    assert source.materialization.value == "captured"
    assert source.name == pdf_page_source_id("doc", page.document_sha256, 1)
    assert asyncio.run(registry.read(source)) == page
    assert PDF_PAGE_SOURCE_DEFINITION.name == "pdf-page"
    assert PDF_PAGE_SOURCE_DEFINITION.version == "1"
    assert PDF_PAGE_SOURCE_DEFINITION.input_class is PDFPageInput


def test_text_projection_uses_document_id_then_page_text_and_exact_metadata() -> None:
    registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
    page = _parse("doc", _blank_pdf(1))[0]
    source = asyncio.run(registry.resolve(page))
    evidence = TextEvidence.model_validate(registry.project(source, TEXT_EVIDENCE_PROJECTION_KEY))
    assert evidence.source_type == "pdf-page"
    assert evidence.source_id == source.name
    assert evidence.content == f"{page.document_id}\n{page.text}"
    assert evidence.metadata == {
        "document_id": page.document_id,
        "document_sha256": page.document_sha256,
        "page_number": page.page_number,
    }
    assert set(evidence.metadata) == {"document_id", "document_sha256", "page_number"}


def test_transport_observation_round_trips_without_plugin_state() -> None:
    registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
    page = _parse("doc", _blank_pdf(1))[0]
    source = asyncio.run(registry.resolve(page))
    observation = project_source_for_transport(registry, source)
    assert observation.name == source.name
    assert observation.source_type == "pdf-page"
    assert observation.definition_fingerprint == manifest_for_definition(PDF_PAGE_SOURCE_DEFINITION).fingerprint
    evidence = TextEvidence.model_validate(observation.projection(TEXT_EVIDENCE_PROJECTION_KEY))
    assert evidence.metadata["page_number"] == 1


def test_page_text_is_preserved_exactly_without_layout_reflow() -> None:
    pages = _parse("doc", _text_pdf(["Alpha 2022 revenue 1234", "Second page 5678"]))
    assert len(pages) == 2
    assert "Alpha" in pages[0].text and "1234" in pages[0].text
    assert "5678" in pages[1].text
    assert "5678" not in pages[0].text


def test_reader_accepts_any_seekable_source_of_the_original_bytes() -> None:
    content = _blank_pdf(2)
    assert len(PdfReader(io.BytesIO(content)).pages) == 2
    assert len(_parse("doc", content)) == 2


def test_search_returns_distinct_pages_and_respects_the_limit() -> None:
    pages = _parse("doc", _blank_pdf(6))
    memory = _StubMemory(pages)
    citations = asyncio.run(search_pdf_pages(memory, "geographic regions", limit=4))
    assert len(citations) <= 4
    resolved = [memory.page_for(citation) for citation in citations]
    assert len(set(resolved)) == len(resolved)


def test_search_never_cites_an_entry_it_cannot_read_back() -> None:
    memory = _StubMemory(_parse("doc", _blank_pdf(3)), unreadable=True)
    assert asyncio.run(search_pdf_pages(memory, "geographic regions", limit=5)) == ()


def test_search_rejects_blank_questions_and_invalid_limits() -> None:
    memory = _StubMemory(_parse("doc", _blank_pdf(2)))
    for question in ("", "   ", "\n\t "):
        with pytest.raises(ValueError):
            asyncio.run(search_pdf_pages(memory, question, limit=5))
    for limit in (0, -1):
        with pytest.raises(ValueError):
            asyncio.run(search_pdf_pages(memory, "geographic regions", limit=limit))


def test_search_on_an_empty_scope_returns_no_citations() -> None:
    memory = _StubMemory(())
    assert asyncio.run(search_pdf_pages(memory, "geographic regions", limit=5)) == ()


def test_search_ranks_the_page_that_uses_the_filing_wording() -> None:
    pages = _parse(
        "doc",
        _text_pdf([
            "Table of Contents Item 3. Legal Proceedings The Company is involved in various claims.",
            "Table of Contents Marketing and brand investment continued in the period.",
            "Table of Contents Our employees are located across many countries.",
        ]),
    )
    memory = _StubMemory(pages)
    citations = asyncio.run(
        search_pdf_pages(memory, "Has the company reported materially important legal battles?", limit=1)
    )
    assert len(citations) == 1
    assert memory.page_for(citations[0]) == ("doc", 1)


def test_search_uses_feedback_wording_absent_from_the_question() -> None:
    pages = _parse(
        "doc",
        _text_pdf([
            "Table of Contents Consolidated Statements of Operations Revenue and operating profit.",
            "Table of Contents Depreciation and amortization is added back to operating income.",
            "Table of Contents The company sells beverages in many markets.",
        ]),
    )
    memory = _StubMemory(pages)
    citations = asyncio.run(search_pdf_pages(memory, "What is unadjusted EBITDA?", limit=1))
    assert len(citations) == 1
    assert memory.page_for(citations[0]) == ("doc", 2)


class _StubMemory:
    """Minimal scoped Memory surface used by the ranking tests."""

    def __init__(self, pages, *, unreadable: bool = False) -> None:
        self._entries = []
        self._by_citation = {}
        self._unreadable = unreadable
        digest = hashlib.sha256(b"stub").hexdigest()
        for page in pages:
            source_id = pdf_page_source_id(page.document_id, digest, page.page_number)
            entry = _StubEntry(
                entry_id=f"entry-{page.page_number}",
                text=f"{page.document_id}\n{page.text}".strip(),
                source_type="pdf-page",
                source_id=source_id,
            )
            record = _StubRecord(memory_ref=_StubRef(), state="active", entry=entry)
            self._entries.append(record)
            self._by_citation[(entry.entry_id,)] = record
        self._digest = digest

    async def list(self):
        return _StubListing(tuple(self._entries))

    async def get(self, request):
        if self._unreadable:
            raise LookupError("entry is not readable")  # noqa: TRY003  # noqa: TRY003
        citation = request.citation
        for record in self._entries:
            if record.entry.entry_id == citation.entry_id:
                return record
        raise LookupError("entry is not readable")  # noqa: TRY003

    def page_for(self, citation) -> tuple[str, int]:
        for record in self._entries:
            if record.entry.entry_id == citation.entry_id:
                parsed = parse_pdf_page_source_id(record.entry.sources[0].source_id)
                assert parsed is not None
                return (parsed[0], parsed[2])
        raise AssertionError("citation does not belong to this Memory")  # noqa: TRY003


class _StubRef:
    """Real ArtifactRef so citations validate against the public contract."""

    def __new__(cls):
        return ArtifactRef(family="memory", artifact_id="mem-1", revision=1)


class _StubSourceRef:
    def __init__(self, source_type: str, source_id: str) -> None:
        self.source_type = source_type
        self.source_id = source_id


class _StubEntry:
    def __init__(self, *, entry_id: str, text: str, source_type: str, source_id: str) -> None:
        self.entry_id = entry_id
        self.entry_version_id = f"{entry_id}-v1"
        self.text = text
        self.sources = (_StubSourceRef(source_type, source_id),)


class _StubRecord:
    def __init__(self, *, memory_ref, state: str, entry) -> None:
        self.memory_ref = memory_ref
        self.state = state
        self.entry = entry


class _StubListing:
    def __init__(self, entries) -> None:
        self.entries = entries
