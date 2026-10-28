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

"""Relational wiring for Source lifecycle propagation.

The service owns the graph algebra; this adapter owns durability plus the two
native visibility effects. Memory entry state is written through the Memory
service's own immutable-revision transition, so the citation captured at
registration keeps resolving to that exact revision. Withdrawals and restorations
are diffed against the current head, which is what makes a replayed successful
event revision-free and lets a retry after a failed native write finish only the
work that is still missing.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence

from sqlalchemy import select

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.artifacts.memory import MemoryEntryVersion, MemoryService
from powercontext.builtin.persistence.database import AsyncDatabase
from powercontext.builtin.persistence.lifecycle import (
    LifecycleLedgerRepository,
    StoredLifecycleEvent,
    StoredLifecycleNode,
)
from powercontext.builtin.persistence.tables import LIFECYCLE_NODES_TABLE
from powercontext.builtin.runtime.source_lifecycle import SourceLifecycleError
from powercontext.sources import SourceRef

MemoryContext = Callable[[str], Awaitable[tuple[MemoryService, str]]]
SourceExists = Callable[[str, SourceRef], Awaitable[bool]]
TopicExists = Callable[[str, ArtifactRef], Awaitable[bool]]


class RelationalLifecycleBackend:
    """Persist lifecycle evidence and mirror it into native family visibility."""

    def __init__(
        self,
        *,
        database: AsyncDatabase,
        ledger: LifecycleLedgerRepository,
        memory_context: MemoryContext,
        source_exists: SourceExists,
        topic_exists: TopicExists,
    ) -> None:
        self._database = database
        self._ledger = ledger
        self._memory_context = memory_context
        self._source_exists = source_exists
        self._topic_exists = topic_exists

    async def load(
        self, scope_id: str, /
    ) -> tuple[dict[str, StoredLifecycleNode], dict[str, tuple[StoredLifecycleEvent, ...]]]:
        async with self._database.transaction() as connection:
            nodes = await self._ledger.nodes(connection, scope_id)
            events = await self._ledger.events(connection, scope_id)
        ledger: dict[str, list[StoredLifecycleEvent]] = {}
        for event in events:
            ledger.setdefault(event.node_id, []).append(event)
        return (
            {node.node_id: node for node in nodes},
            {node_id: tuple(values) for node_id, values in ledger.items()},
        )

    async def append_node(self, scope_id: str, node: StoredLifecycleNode, /) -> None:
        async with self._database.transaction() as connection:
            await self._ledger.add_nodes(connection, scope_id, (node,))

    async def require_source(self, scope_id: str, source_ref: SourceRef, /) -> None:
        if not await self._source_exists(scope_id, source_ref):
            raise SourceLifecycleError("unknown-source-ref", source_ref.source_id)

    async def require_memory(self, scope_id: str, citation: MemoryCitation, /) -> None:
        service, artifact_id = await self._memory_context(scope_id)
        head = await service.head(artifact_id)
        if citation.memory_ref.artifact_id != head.artifact_id:
            raise SourceLifecycleError("unknown-memory-ref", citation.memory_ref.artifact_id)
        entries = {entry.entry_id: entry for entry in await service.entries(head)}
        entry = entries.get(citation.entry_id)
        if entry is None or entry.entry_version_id != citation.entry_version_id:
            raise SourceLifecycleError("unknown-memory-ref", citation.entry_id)

    async def require_topic(self, scope_id: str, reference: ArtifactRef, /) -> None:
        if not await self._topic_exists(scope_id, reference):
            raise SourceLifecycleError("unknown-topic-ref", reference.artifact_id)

    async def set_memory_availability(
        self,
        scope_id: str,
        targets: Sequence[tuple[MemoryCitation, bool]],
        /,
        *,
        reason: str,
    ) -> None:
        """Apply the reached availability as Memory transitions in both directions.

        Only entries whose state disagrees with the target are touched, so a
        replayed successful event writes nothing at all.
        """

        service, artifact_id = await self._memory_context(scope_id)
        head = await service.head(artifact_id)
        entries = {entry.entry_id: entry for entry in await service.entries(head)}
        manifest = {item.entry_id: item for item in head.content.manifest.entries}
        to_withdraw: list[MemoryEntryVersion] = []
        to_restore: list[MemoryEntryVersion] = []
        for citation, available in targets:
            entry = entries.get(citation.entry_id)
            item = manifest.get(citation.entry_id)
            if entry is None or item is None or entry.entry_version_id != citation.entry_version_id:
                raise SourceLifecycleError("stale-citation", citation.entry_id)
            if available and item.state == "inactive":
                to_restore.append(entry)
            elif not available and item.state == "active":
                to_withdraw.append(entry)
        if to_withdraw:
            await service.forget(head, entries=tuple(to_withdraw), reason=reason or None)
        if to_restore:
            current = await service.head(artifact_id)
            await service.reactivate(current, entries=tuple(to_restore), reason=reason or None)

    async def commit_event(
        self,
        scope_id: str,
        event: StoredLifecycleEvent,
        availability: Sequence[tuple[str, bool]],
        /,
    ) -> None:
        async with self._database.transaction() as connection:
            await self._ledger.append_event(connection, scope_id, event)
            await self._ledger.set_availability(connection, scope_id, availability)

    async def unavailable_topics(self, scope_id: str, /) -> frozenset[str]:
        """Return registered Topic artifact ids whose evidence is withdrawn.

        One narrow indexed read; an empty result means the Scope never used the
        lifecycle service and native Topic search keeps its ordinary behavior.
        """

        async with self._database.transaction() as connection:
            rows = await connection.scalars(
                select(LIFECYCLE_NODES_TABLE.c.reference_artifact_id).where(
                    LIFECYCLE_NODES_TABLE.c.scope_id == scope_id,
                    LIFECYCLE_NODES_TABLE.c.node_kind == "topic",
                    LIFECYCLE_NODES_TABLE.c.available.is_(False),
                    LIFECYCLE_NODES_TABLE.c.reference_artifact_id.is_not(None),
                )
            )
        return frozenset(str(value) for value in rows)


def topic_search_visible(
    artifact_id: str,
    unavailable: frozenset[str],
    /,
) -> bool:
    """Return whether one Topic hit may appear in native search and Context."""

    return artifact_id not in unavailable


__all__ = ["RelationalLifecycleBackend", "topic_search_visible"]
