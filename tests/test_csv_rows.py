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

"""Contract edges for row-level CSV capture not covered by the public sample."""

from __future__ import annotations

import asyncio
import json

import pytest

from powercontext.sources import (
    TEXT_EVIDENCE_PROJECTION_KEY,
    SourceDefinitionRegistry,
    TextEvidence,
    manifest_for_definition,
    project_source_for_transport,
)
from powercontext.sources.csv_rows import (
    CSV_ROW_SOURCE_DEFINITION,
    CSVDocumentInput,
    CSVRowSource,
    csv_row_source_id,
    parse_csv_document,
)


def _parse(document_id: str, primary_key: str, content: bytes):
    return parse_csv_document(CSVDocumentInput(document_id=document_id, primary_key=primary_key, content=content))


@pytest.mark.parametrize("document_id", ["", "   ", " doc", "doc ", "\tdoc"])
def test_document_id_must_be_present_and_unpadded(document_id: str) -> None:
    with pytest.raises(ValueError):
        _parse(document_id, "id", b"id,name\n1,one\n")


def test_primary_key_must_be_non_empty_and_match_a_header_exactly() -> None:
    with pytest.raises(ValueError):
        _parse("doc", "", b"id,name\n1,one\n")
    with pytest.raises(ValueError):
        _parse("doc", "id", b" id ,name\n1,one\n")
    rows = _parse("doc", " id ", b" id ,name\n1,one\n")
    assert rows[0].primary_key_value == "1"


def test_padded_and_unpadded_headers_are_distinct_columns() -> None:
    rows = _parse("doc", "id", b"id, id \n1,2\n")
    assert rows[0].columns == ("id", " id ")
    assert rows[0].values == ("1", "2")


@pytest.mark.parametrize(
    "content",
    [
        b"\n",
        b"\xef\xbb\xbf",
        b'"",name\n1,one\n',
        b" \t ,name\n1,one\n",
        b'id,"id"\n1,1\n',
        b"id,id\n1,1\n",
    ],
)
def test_invalid_header_records_fail(content: bytes) -> None:
    with pytest.raises(ValueError):
        _parse("doc", "id", content)


def test_wide_records_and_trailing_commas_fail() -> None:
    with pytest.raises(ValueError):
        _parse("doc", "id", b"id,name\n1,one,extra\n")
    with pytest.raises(ValueError):
        _parse("doc", "id", b"id,name\n1,one,\n")


@pytest.mark.parametrize("key_cell", [b"", b"   ", b'" "'])
def test_blank_primary_key_values_fail(key_cell: bytes) -> None:
    with pytest.raises(ValueError):
        _parse("doc", "id", b"id,name\n" + key_cell + b",one\n")


def test_empty_records_only_skip_when_the_reader_yields_no_fields() -> None:
    content = b"id,name\n1,one\n\n2,two\r\n\n3,three\n"
    rows = _parse("doc", "id", content)
    assert [row.values for row in rows] == [("1", "one"), ("2", "two"), ("3", "three")]

    # A quoted empty field and a single space are field records, not blank lines.
    with pytest.raises(ValueError):
        _parse("doc", "id", b'id\n""\n')
    with pytest.raises(ValueError):
        _parse("doc", "id", b"id\n \n")
    # Two empty fields still need a blank primary key to be rejected.
    with pytest.raises(ValueError):
        _parse("doc", "id", b"id,name\n,\n")


def test_empty_non_key_fields_and_embedded_blank_lines_are_preserved() -> None:
    rows = _parse("doc", "id", b'id,name,note\n1,,"a\n\nb"\n')
    assert rows[0].values == ("1", "", "a\n\nb")


def test_original_string_values_are_not_normalized() -> None:
    rows = _parse("doc", "id", b"id,flag,amount\n01,false,1.50\n")
    assert rows[0].primary_key_value == "01"
    assert rows[0].values == ("01", "false", "1.50")


def test_distinct_raw_key_values_coexist_but_duplicates_fail() -> None:
    rows = _parse("doc", "id", b"id,name\n key ,first\nkey ,second\nkey,third\n")
    assert [row.primary_key_value for row in rows] == [" key ", "key ", "key"]
    with pytest.raises(ValueError):
        _parse("doc", "id", b"id,name\n key ,first\n key ,second\n")


def test_header_only_documents_are_valid_with_or_without_trailing_newline() -> None:
    assert _parse("doc", "id", b"id,name") == ()
    assert _parse("doc", "id", b"id,name\n") == ()


def test_parse_failures_are_atomic_after_valid_rows() -> None:
    with pytest.raises(ValueError):
        _parse("doc", "id", b"id,name\n1,one\n2,two\n3\n")


def test_unicode_and_delimiter_components_are_percent_encoded() -> None:
    assert csv_row_source_id("doc:one", "key/one") == "csv:doc%3Aone:key%2Fone"
    assert csv_row_source_id("d", "a b+c") == "csv:d:a%20b%2Bc"
    assert csv_row_source_id("d", "%3A") == "csv:d:%253A"
    assert csv_row_source_id("d", "文") == "csv:d:%E6%96%87"
    assert csv_row_source_id("d", "\U0001f600") == "csv:d:%F0%9F%98%80"
    assert csv_row_source_id("d", "\u00e9") != csv_row_source_id("d", "e\u0301")


def test_identity_components_cannot_collide_across_the_separator() -> None:
    assert csv_row_source_id("doc:a", "b:c") != csv_row_source_id("doc:a:b", "c")


def test_source_id_limit_is_enforced_after_encoding() -> None:
    assert len(csv_row_source_id("d", "x" * 250)) == 256
    with pytest.raises(ValueError):
        csv_row_source_id("d", "x" * 251)
    assert len(csv_row_source_id("d", ":" * 83 + "x")) == 256
    with pytest.raises(ValueError):
        csv_row_source_id("d", "文" * 30)
    with pytest.raises(ValueError):
        csv_row_source_id(" doc ", "key")
    with pytest.raises(ValueError):
        csv_row_source_id("doc", "   ")


def test_definition_materializes_captured_rows_and_reads_input_back() -> None:
    registry = SourceDefinitionRegistry((CSV_ROW_SOURCE_DEFINITION,))
    row = _parse("doc", "id", b"id,name\n1,one\n")[0]
    source = asyncio.run(registry.resolve(row))

    assert isinstance(source, CSVRowSource)
    assert source.materialization.value == "captured"
    assert source.source_id == source.name == "csv:doc:1"
    assert asyncio.run(registry.read(source)) == row
    assert CSV_ROW_SOURCE_DEFINITION.name == "csv-row"
    assert CSV_ROW_SOURCE_DEFINITION.version == "1"


def test_text_evidence_keeps_column_order_and_exactly_five_metadata_keys() -> None:
    registry = SourceDefinitionRegistry((CSV_ROW_SOURCE_DEFINITION,))
    row = _parse("doc", "id", b"id,name,note\n1,,\xe6\x96\x87\n")[0]
    source = asyncio.run(registry.resolve(row))
    evidence = TextEvidence.model_validate(registry.project(source, TEXT_EVIDENCE_PROJECTION_KEY))

    expected_row = {"id": "1", "name": "", "note": "文"}
    assert evidence.source_type == "csv-row"
    assert evidence.source_id == "csv:doc:1"
    assert evidence.content == json.dumps(expected_row, ensure_ascii=False, separators=(",", ":"))
    assert set(evidence.metadata) == {"document_id", "primary_key", "primary_key_value", "columns", "row"}
    assert evidence.metadata["columns"] == ["id", "name", "note"]
    assert evidence.metadata["row"] == expected_row


def test_transport_observation_round_trips_without_plugin_state() -> None:
    registry = SourceDefinitionRegistry((CSV_ROW_SOURCE_DEFINITION,))
    row = _parse("doc", "id", b"id,name\n1,one\n")[0]
    source = asyncio.run(registry.resolve(row))
    observation = project_source_for_transport(registry, source)

    assert observation.name == "csv:doc:1"
    assert observation.source_type == "csv-row"
    assert observation.definition_fingerprint == manifest_for_definition(CSV_ROW_SOURCE_DEFINITION).fingerprint
    assert (
        TextEvidence.model_validate(observation.projection(TEXT_EVIDENCE_PROJECTION_KEY)).content
        == '{"id":"1","name":"one"}'
    )
