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

"""seekdb hybrid candidate retrieval resolves both channels in one statement.

The embedded seekdb binding ships Linux and macOS wheels only, so these tests drive
the adapter through a recording connection. They pin the observable contract that can
be checked without the engine: how many statements one hybrid search issues, which
predicates and limits those statements carry, and how the returned rows become the two
ordered candidate channels. The engine-side contract (EXPLAIN showing both native
indexes) is documented in the task notes and must be verified on Linux.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.dialects import mysql

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory import (
    CapabilityNotSupportedError,
    EmbeddingProfile,
    MemorySearchRequest,
)
from powercontext.builtin.persistence.memory_index import CompositeMemoryIndex
from powercontext.builtin.persistence.oceanbase.profile import OceanBaseConfig
from powercontext.builtin.persistence.seekdb import SeekDBConfig
from powercontext.builtin.persistence.seekdb.memory_index import SeekDBMemoryIndex
from powercontext.builtin.runtime.composition import _server_memory_indexes
from powercontext.builtin.tags import TagFilter

_PROFILE = EmbeddingProfile(
    profile_id="seekdb-hybrid-v1",
    model="deterministic",
    dimension=3,
    distance="l2",
    normalization="unit",
)
_ALPHA = ArtifactRef(family="memory", artifact_id="alpha-memory", revision=3)
_BETA = ArtifactRef(family="memory", artifact_id="beta-memory", revision=7)


def _request(**overrides: Any) -> MemorySearchRequest:
    values: dict[str, Any] = {
        "query": "alpha",
        "analyzed_query": "alpha",
        "memories": (_ALPHA, _BETA),
        "candidate_limit": 32,
        "mode": "hybrid",
        "query_vector": (1.0, 0.0, 0.0),
        "embedding_profile": _PROFILE,
    }
    values.update(overrides)
    return MemorySearchRequest(**values)


def _row(channel: str, memory: ArtifactRef, entry: str, text: str, score: float) -> dict[str, Any]:
    return {
        "channel": channel,
        "memory_artifact_id": memory.artifact_id,
        "entry_id": entry,
        "entry_version_id": f"{entry}-v1",
        "text": text,
        "channel_score": score,
    }


class _Rows:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def mappings(self) -> list[dict[str, Any]]:
        return self._rows


def _render(statement: Any) -> str:
    """Render a statement without compiling dialect-specific constructs."""

    text_attribute = getattr(statement, "text", None)
    if isinstance(text_attribute, str):
        return text_attribute
    return type(statement).__name__


class _RecordingConnection:
    """Record every statement and answer the hybrid query with canned rows."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.statements: list[tuple[str, dict[str, Any]]] = []

    async def execute(self, statement: Any, parameters: dict[str, Any] | None = None) -> _Rows:
        rendered = _render(statement)
        self.statements.append((rendered, dict(parameters or {})))
        return _Rows(self.rows if _is_hybrid(rendered) else [])

    def hybrid_statements(self) -> list[tuple[str, dict[str, Any]]]:
        return [entry for entry in self.statements if _is_hybrid(entry[0])]


def _is_hybrid(statement: str) -> bool:
    """One statement that carries both candidate channels, as the challenge requires."""

    upper = statement.upper()
    return (
        "pc_memory_entry_heads" in statement
        and "pc_memory_vector_entries" in statement
        and "AGAINST" in upper
        and "l2_distance" in statement.lower()
    )


def _search(rows: list[dict[str, Any]], request: MemorySearchRequest, *, complete: bool = True):
    connection = _RecordingConnection(rows)
    index = SeekDBMemoryIndex(_PROFILE)

    async def complete_always(*_args: Any) -> bool:
        return complete

    async def scenario():
        index.vector_complete = complete_always  # type: ignore[method-assign]
        return await index.search(connection, "scope-a", request)

    return asyncio.run(scenario()), connection, index


def test_hybrid_search_issues_exactly_one_native_statement() -> None:
    channels, connection, _index = _search([], _request())

    assert len(connection.hybrid_statements()) == 1
    statement, _parameters = connection.hybrid_statements()[0]
    assert "UNION ALL" in statement
    # Both channels are ranked by the database, not by Python.
    assert "ORDER BY MATCH" in statement.replace("\n", " ")
    assert "ORDER BY l2_distance" in statement.replace("\n", " ")
    assert "APPROXIMATE" in statement
    assert channels.fts == ()
    assert channels.vector == ()


def test_hybrid_statement_filters_scope_artifact_and_candidate_limit_in_the_database() -> None:
    _channels, connection, _index = _search([], _request(candidate_limit=9))

    statement, parameters = connection.hybrid_statements()[0]
    assert statement.count("h.scope_id = :scope_id") == 1
    assert statement.count("m.scope_id = :scope_id") == 1
    assert statement.count("memory_artifact_id IN :memory_artifact_ids") == 2
    assert statement.count("LIMIT :candidate_limit") == 2
    assert parameters["scope_id"] == "scope-a"
    assert parameters["memory_artifact_ids"] == ("alpha-memory", "beta-memory")
    assert parameters["candidate_limit"] == 9
    assert parameters["query"] == "alpha"
    assert parameters["query_vector"] == (1.0, 0.0, 0.0)


def test_hybrid_search_splits_channels_and_keeps_database_ranking() -> None:
    rows = [
        _row("fts", _ALPHA, "alpha-1", "Alpha semantic record.", 1.5),
        _row("fts", _ALPHA, "alpha-2", "Alpha second record.", 0.5),
        _row("vector", _ALPHA, "alpha-1", "Alpha semantic record.", 0.0),
        _row("vector", _ALPHA, "alpha-2", "Alpha second record.", 0.75),
    ]
    channels, _connection, _index = _search(rows, _request())

    assert [hit.entry_id for hit in channels.fts] == ["alpha-1", "alpha-2"]
    assert [hit.entry_id for hit in channels.vector] == ["alpha-1", "alpha-2"]
    assert [hit.distance for hit in channels.vector] == [0.0, 0.75]
    assert all(hit.distance is None for hit in channels.fts)
    # The requested, already validated head is the authority for the returned ref.
    assert [hit.memory_ref for hit in channels.fts] == [_ALPHA, _ALPHA]
    assert all(hit.memory_ref.revision == _ALPHA.revision for hit in channels.fts)


def test_hybrid_search_drops_rows_outside_the_requested_heads() -> None:
    stray = ArtifactRef(family="memory", artifact_id="other-memory", revision=1)
    rows = [
        _row("fts", _ALPHA, "alpha-1", "Alpha semantic record.", 1.0),
        _row("fts", stray, "stray-1", "Alpha decoy in another scope.", 0.9),
    ]
    channels, _connection, _index = _search(rows, _request())

    assert [hit.entry_id for hit in channels.fts] == ["alpha-1"]


def test_repeated_hybrid_searches_are_stable() -> None:
    rows = [
        _row("fts", _BETA, "beta-1", "Beta record.", 0.25),
        _row("fts", _ALPHA, "alpha-1", "Alpha record.", 0.25),
        _row("vector", _BETA, "beta-1", "Beta record.", 0.5),
        _row("vector", _ALPHA, "alpha-1", "Alpha record.", 0.5),
    ]
    first, _connection, _index = _search(rows, _request())
    second, _connection2, _index2 = _search(rows, _request())

    assert [hit.entry_id for hit in first.fts] == [hit.entry_id for hit in second.fts]
    assert [hit.entry_id for hit in first.vector] == [hit.entry_id for hit in second.vector]
    # Equal scores fall back to the same identity order the statement orders by.
    assert [hit.memory_ref.artifact_id for hit in first.fts] == ["alpha-memory", "beta-memory"]


def test_hybrid_search_applies_the_tag_filter_inside_the_statement() -> None:
    _channels, connection, _index = _search([], _request(tag_filter=TagFilter(tags=("selected",))))

    statement, parameters = connection.hybrid_statements()[0]
    assert statement.count("pc_artifact_tags") == 2
    assert parameters["tag_keys"] == ("selected",)
    assert parameters["tag_minimum"] == 1
    # The filtered vector branch keeps its native approximate top-k.
    assert "APPROXIMATE" in statement


def test_hybrid_search_requires_the_matching_embedding_profile() -> None:
    index = SeekDBMemoryIndex(_PROFILE)
    other = _PROFILE.model_copy(update={"profile_id": "other-profile"})
    connection = _RecordingConnection([])

    with pytest.raises(CapabilityNotSupportedError, match=r"embedding-profile"):
        asyncio.run(index.search(connection, "scope-a", _request(embedding_profile=other)))
    assert connection.hybrid_statements() == []


def test_hybrid_search_requires_complete_vector_projections() -> None:
    connection = _RecordingConnection([])
    index = SeekDBMemoryIndex(_PROFILE)

    async def incomplete(*_args: Any) -> bool:
        return False

    async def scenario() -> None:
        index.vector_complete = incomplete  # type: ignore[method-assign]
        await index.search(connection, "scope-a", _request())

    with pytest.raises(CapabilityNotSupportedError, match=r"not supported: vector$"):
        asyncio.run(scenario())
    assert connection.statements == []


def test_single_channel_modes_keep_the_frozen_statements() -> None:
    for mode in ("fts", "vector"):
        _channels, connection, _index = _search([], _request(mode=mode))
        assert connection.hybrid_statements() == []
        assert connection.statements != []


def test_hybrid_statement_is_compilable_with_bound_parameters() -> None:
    index = SeekDBMemoryIndex(_PROFILE)
    compiled = index._hybrid_sql.compile(dialect=mysql.dialect())

    assert compiled.string.count("%s") >= 8
    assert "UNION ALL" in compiled.string
    assert "pc_memory_entry_heads" in compiled.string
    assert "pc_memory_vector_entries" in compiled.string


class _Embedding:
    profile = _PROFILE


def test_seekdb_deployment_uses_one_hybrid_index_instead_of_two_channels() -> None:
    indexes = _server_memory_indexes(SeekDBConfig(path=Path("seekdb-runtime")), _Embedding())

    assert [type(index).__name__ for index in indexes] == ["SeekDBMemoryIndex"]
    capabilities = CompositeMemoryIndex(*indexes).capabilities
    assert (capabilities.fts, capabilities.vector, capabilities.hybrid) == (True, True, True)


def test_oceanbase_deployment_keeps_the_two_channel_composition() -> None:
    database = OceanBaseConfig(url="mysql+aoceanbase://root@localhost:2881/test?charset=utf8mb4")
    indexes = _server_memory_indexes(database, _Embedding())

    assert [type(index).__name__ for index in indexes] == [
        "OceanBaseMemoryFTSIndex",
        "OceanBaseMemoryVectorIndex",
    ]


@pytest.mark.parametrize(
    "database",
    [
        SeekDBConfig(path=Path("seekdb-runtime")),
        OceanBaseConfig(url="mysql+aoceanbase://root@localhost:2881/test?charset=utf8mb4"),
    ],
)
def test_deployment_without_an_embedding_model_stays_full_text_only(database: object) -> None:
    indexes = _server_memory_indexes(database, None)

    assert [type(index).__name__ for index in indexes] == ["OceanBaseMemoryFTSIndex"]
    capabilities = CompositeMemoryIndex(*indexes).capabilities
    assert (capabilities.fts, capabilities.vector, capabilities.hybrid) == (True, False, False)
