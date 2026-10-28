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

"""Row-level captured Sources for offline CSV documents.

Each data record becomes one independently addressable Source so that a
retrieval citation can be traced back to the exact original row. Row identity
depends only on the document and the primary key value, which keeps it stable
across reordering and edits to non-key columns.
"""

from __future__ import annotations

import csv
import io
import json
from urllib.parse import quote

from pydantic import BaseModel, JsonValue

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

CSV_ROW_SOURCE_NAME = "csv-row"
CSV_ROW_SOURCE_VERSION = "1"
_CSV_IDENTITY_PREFIX = "csv:"
_UTF8_BOM = "\ufeff"


class CSVDocumentInput(BaseModel):
    """One complete offline CSV document to split into row Sources."""

    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    """One validated data record with its original column order and values."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    """A captured row that can be read back as its exact original record."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


def parse_csv_document(value: CSVDocumentInput, /) -> tuple[CSVRowInput, ...]:
    """Parse and fully validate one CSV document without returning partial rows."""

    _require_document_id(value.document_id)
    _require_primary_key(value.primary_key)
    records = _read_records(_decode_document(value.content))

    if not records:
        raise ValueError("CSV document must contain a header record")  # noqa: TRY003
    header = records[0]
    _require_header(header)
    try:
        key_index = header.index(value.primary_key)
    except ValueError:
        raise ValueError(f"CSV header does not contain the primary key {value.primary_key!r}") from None  # noqa: TRY003

    columns = tuple(header)
    rows: list[CSVRowInput] = []
    key_values: set[str] = set()
    for record in records[1:]:
        if not record:
            # ``csv.reader`` reports a physically empty line as no fields at all.
            continue
        if len(record) != len(header):
            raise ValueError(  # noqa: TRY003
                f"CSV record has {len(record)} fields, expected {len(header)}"
            )
        primary_key_value = record[key_index]
        if not primary_key_value.strip():
            raise ValueError("CSV primary key value must not be blank")  # noqa: TRY003
        if primary_key_value in key_values:
            raise ValueError(f"duplicate CSV primary key value {primary_key_value!r}")  # noqa: TRY003
        key_values.add(primary_key_value)
        rows.append(
            CSVRowInput(
                document_id=value.document_id,
                primary_key=value.primary_key,
                primary_key_value=primary_key_value,
                columns=columns,
                values=tuple(record),
            )
        )
    return tuple(rows)


def csv_row_source_id(document_id: str, primary_key_value: str, /) -> str:
    """Return the stable, delimiter-safe identity of one captured row."""

    _require_document_id(document_id)
    _require_primary_key_value(primary_key_value)
    source_id = f"{_CSV_IDENTITY_PREFIX}{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(  # noqa: TRY003
            f"CSV row source id exceeds {MAX_SOURCE_ID_LENGTH} characters"
        )
    return source_id


class CSVRowSourceAdapter:
    """Materialize and read one captured CSV row."""

    input_class = CSVRowInput
    name = CSV_ROW_SOURCE_NAME
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        source_id = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=source_id,
            materialization=SourceMaterialization.CAPTURED,
            source_id=source_id,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class CSVRowTextEvidenceProjection:
    """Expose one captured row through the shared text-evidence capability."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row: dict[str, JsonValue] = dict(zip(source.columns, source.values, strict=True))
        return TextEvidence(
            source_type=CSV_ROW_SOURCE_NAME,
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


CSV_ROW_SOURCE_ADAPTER = CSVRowSourceAdapter()
CSV_ROW_SOURCE_DEFINITION = AdapterSourceDefinition(
    CSV_ROW_SOURCE_ADAPTER,
    version=CSV_ROW_SOURCE_VERSION,
    projections=(CSVRowTextEvidenceProjection(),),
)


def _decode_document(content: bytes) -> str:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("CSV content must be valid UTF-8") from error  # noqa: TRY003
    # Only a leading byte-order mark is an encoding signature; later ones are data.
    return text.removeprefix(_UTF8_BOM)


def _read_records(text: str) -> list[list[str]]:
    try:
        return list(
            csv.reader(
                io.StringIO(text, newline=""),
                delimiter=",",
                quotechar='"',
                doublequote=True,
                strict=True,
            )
        )
    except csv.Error as error:
        raise ValueError(f"malformed CSV document: {error}") from error  # noqa: TRY003


def _require_header(header: list[str]) -> None:
    if not header:
        raise ValueError("CSV document must start with a non-empty header record")  # noqa: TRY003
    seen: set[str] = set()
    for column in header:
        if not column.strip():
            raise ValueError("CSV header fields must not be blank")  # noqa: TRY003
        if column in seen:
            raise ValueError(f"duplicate CSV header field {column!r}")  # noqa: TRY003
        seen.add(column)


def _require_document_id(document_id: str) -> None:
    if not document_id.strip():
        raise ValueError("CSV document_id must not be blank")  # noqa: TRY003
    if document_id != document_id.strip():
        raise ValueError("CSV document_id must not contain leading or trailing whitespace")  # noqa: TRY003


def _require_primary_key(primary_key: str) -> None:
    if not primary_key:
        raise ValueError("CSV primary_key must not be empty")  # noqa: TRY003


def _require_primary_key_value(primary_key_value: str) -> None:
    if not primary_key_value.strip():
        raise ValueError("CSV primary key value must not be blank")  # noqa: TRY003


__all__ = [
    "CSV_ROW_SOURCE_ADAPTER",
    "CSV_ROW_SOURCE_DEFINITION",
    "CSV_ROW_SOURCE_NAME",
    "CSV_ROW_SOURCE_VERSION",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "CSVRowSourceAdapter",
    "CSVRowTextEvidenceProjection",
    "csv_row_source_id",
    "parse_csv_document",
]
