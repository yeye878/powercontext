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

"""seekdb Memory search index resolving both candidate channels in one statement.

The seekdb deployment reuses the OceanBase schema, full-text index and vector
index. Its hybrid path differs in one respect: instead of running one statement
per channel and fusing the two rankings in the application, a single native
statement returns both ordered candidate lists. Everything that narrows the
candidate set - scope, Memory artifact, tag filter and the candidate limit -
stays inside that statement, so no match or distance is computed in Python.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pyobvector import VECTOR
from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.builtin.artifacts.memory import (
    CapabilityNotSupportedError,
    EmbeddingProfile,
    MemoryCapabilities,
    MemoryChannelHit,
    MemorySearchChannels,
    MemorySearchRequest,
)
from powercontext.builtin.artifacts.memory.canonical import validate_embedding
from powercontext.builtin.persistence.memory_index import memory_channel_hits
from powercontext.builtin.persistence.oceanbase.memory_index import (
    OceanBaseMemoryFTSIndex,
    OceanBaseMemoryVectorIndex,
)
from powercontext.builtin.persistence.tags import memory_tag_parameters, memory_tag_sql

# Both branches share one projection so UNION ALL keeps a single column list.
# Each branch ranks its own channel and applies its own candidate limit; the
# application only restores that order from the scores the database returned.
_SEEKDB_HYBRID_SEARCH_SQL = """
(
SELECT
    'fts' AS channel,
    h.memory_artifact_id,
    h.entry_id,
    h.entry_version_id,
    v.text,
    MATCH (h.searchable_text) AGAINST (:query) AS channel_score
FROM pc_memory_entry_heads AS h
JOIN pc_memory_entry_versions AS v
  ON v.scope_id = h.scope_id
 AND v.memory_artifact_id = h.memory_artifact_id
 AND v.entry_version_id = h.entry_version_id
WHERE h.scope_id = :scope_id
  AND h.memory_artifact_id IN :memory_artifact_ids
  AND MATCH (h.searchable_text) AGAINST (:query)
/* fts-tag-filter */
ORDER BY MATCH (h.searchable_text) AGAINST (:query) DESC,
         h.memory_artifact_id,
         h.entry_id,
         h.entry_version_id
LIMIT :candidate_limit
)
UNION ALL
(
SELECT
    'vector' AS channel,
    m.memory_artifact_id,
    m.entry_id,
    m.entry_version_id,
    v.text,
    l2_distance(m.embedding, :query_vector) AS channel_score
FROM pc_memory_vector_entries AS m
JOIN pc_memory_entry_versions AS v
  ON v.scope_id = m.scope_id
 AND v.memory_artifact_id = m.memory_artifact_id
 AND v.entry_version_id = m.entry_version_id
WHERE m.scope_id = :scope_id
  AND m.memory_artifact_id IN :memory_artifact_ids
/* vector-tag-filter */
ORDER BY l2_distance(m.embedding, :query_vector) APPROXIMATE
LIMIT :candidate_limit
)
"""

# A tag filter only changes the WHERE clause: the vector branch keeps its native
# APPROXIMATE top-k, which seekdb plans as a filtered vector index scan rather
# than a table scan followed by application-side distance computation.
_SEEKDB_TAGGED_HYBRID_SEARCH_SQL = _SEEKDB_HYBRID_SEARCH_SQL.replace(
    "/* fts-tag-filter */", memory_tag_sql("h")
).replace("/* vector-tag-filter */", memory_tag_sql("m"))

_IDENTITY_COLUMNS = ("memory_artifact_id", "entry_id", "entry_version_id")


class SeekDBMemoryIndex(OceanBaseMemoryVectorIndex):
    """One native hybrid statement over the seekdb full-text and vector indexes."""

    def __init__(self, profile: EmbeddingProfile) -> None:
        super().__init__(profile)
        self._fts_index = OceanBaseMemoryFTSIndex()
        self.capabilities = MemoryCapabilities(
            fts=True,
            vector=True,
            hybrid=True,
            tag_filter=True,
            embedding_profile=profile,
        )
        self._hybrid_sql = text(_SEEKDB_HYBRID_SEARCH_SQL).bindparams(
            bindparam("memory_artifact_ids", expanding=True),
            bindparam("query_vector", type_=VECTOR(profile.dimension)),
        )
        self._tagged_hybrid_sql = text(_SEEKDB_TAGGED_HYBRID_SEARCH_SQL).bindparams(
            bindparam("memory_artifact_ids", expanding=True),
            bindparam("tag_keys", expanding=True),
            bindparam("tag_hashes", expanding=True),
            bindparam("query_vector", type_=VECTOR(profile.dimension)),
        )

    async def initialize(self, connection: AsyncConnection, /) -> None:
        await self._fts_index.initialize(connection)
        await super().initialize(connection)

    async def search(
        self,
        connection: AsyncConnection,
        scope_id: str,
        request: MemorySearchRequest,
        /,
    ) -> MemorySearchChannels:
        # Single-channel requests keep the frozen per-channel statements so their
        # observable behavior does not change with this index.
        if request.mode == "fts":
            return await self._fts_index.search(connection, scope_id, request)
        if request.mode == "vector":
            return await super().search(connection, scope_id, request)
        if request.mode != "hybrid":
            return MemorySearchChannels()
        if request.embedding_profile != self.profile or request.query_vector is None:
            raise CapabilityNotSupportedError("embedding-profile")
        if not await self.vector_complete(connection, scope_id, request.memories, self.profile):
            raise CapabilityNotSupportedError("vector")
        rows = (
            await connection.execute(
                self._hybrid_sql if request.tag_filter is None else self._tagged_hybrid_sql,
                {
                    "scope_id": scope_id,
                    "memory_artifact_ids": tuple(memory.artifact_id for memory in request.memories),
                    "query": request.analyzed_query,
                    "query_vector": validate_embedding(
                        request.query_vector,
                        dimension=self.profile.dimension,
                    ),
                    "candidate_limit": request.candidate_limit,
                    **memory_tag_parameters(request.tag_filter),
                },
            )
        ).mappings()
        lexical, semantic = _split_channels(rows)
        return MemorySearchChannels(
            fts=_lexical_hits(lexical, request.memories),
            vector=_semantic_hits(semantic, request.memories),
        )


def _split_channels(
    rows: Iterable[Mapping[Any, Any]],
    /,
) -> tuple[list[Mapping[Any, Any]], list[Mapping[Any, Any]]]:
    lexical: list[Mapping[Any, Any]] = []
    semantic: list[Mapping[Any, Any]] = []
    for row in rows:
        channel = str(row["channel"])
        if channel == "fts":
            lexical.append(row)
        elif channel == "vector":
            semantic.append(row)
    return lexical, semantic


def _lexical_hits(
    rows: Iterable[Mapping[Any, Any]],
    memories: tuple[Any, ...],
    /,
) -> tuple[MemoryChannelHit, ...]:
    ordered = sorted(rows, key=lambda row: (-float(row["channel_score"]), *_identity_key(row)))
    return memory_channel_hits(({**dict(row), "distance": None} for row in ordered), memories)


def _semantic_hits(
    rows: Iterable[Mapping[Any, Any]],
    memories: tuple[Any, ...],
    /,
) -> tuple[MemoryChannelHit, ...]:
    ordered = sorted(rows, key=lambda row: (float(row["channel_score"]), *_identity_key(row)))
    return memory_channel_hits(
        ({**dict(row), "distance": float(row["channel_score"])} for row in ordered),
        memories,
    )


def _identity_key(row: Mapping[Any, Any], /) -> tuple[bytes, bytes, bytes]:
    return tuple(str(row[column]).encode("utf-8") for column in _IDENTITY_COLUMNS)
