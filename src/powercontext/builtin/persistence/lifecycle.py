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

"""Durable registration graph and event ledger for Source lifecycle evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from sqlalchemy import delete, insert, select
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.builtin.persistence.codec import stored_bytes
from powercontext.builtin.persistence.errors import InvalidRepositoryArgumentError
from powercontext.builtin.persistence.tables import LIFECYCLE_EVENTS_TABLE, LIFECYCLE_NODES_TABLE
from powercontext.limits import MAX_LIFECYCLE_NODE_ID_LENGTH, MAX_LIFECYCLE_REASON_LENGTH, MAX_SCOPE_ID_LENGTH

LifecycleNodeKind: TypeAlias = Literal["source", "memory", "topic"]
LifecyclePolicy: TypeAlias = Literal["all", "any"]
_NodeIds = TypeAdapter(tuple[str, ...])


class _LedgerModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StoredLifecycleNode(_LedgerModel):
    """One registered node and the evidence binding captured at registration."""

    node_id: str
    node_kind: LifecycleNodeKind
    revision: int = Field(ge=1)
    reference_family: str | None = None
    reference_artifact_id: str | None = None
    reference_revision: int | None = None
    source_type: str | None = None
    source_id: str | None = None
    entry_id: str | None = None
    entry_version_id: str | None = None
    parents: tuple[str, ...] = ()
    policy: LifecyclePolicy | None = None
    available: bool


class StoredLifecycleEvent(_LedgerModel):
    """One accepted Source event, keyed by its monotonic per-Source sequence."""

    node_id: str
    sequence: int = Field(ge=1)
    revision: int = Field(ge=1)
    available: bool
    reason: str = Field(min_length=1, max_length=MAX_LIFECYCLE_REASON_LENGTH)


class LifecycleLedgerRepository:
    """Append and read lifecycle evidence inside a caller-owned transaction.

    The ledger is authoritative rather than a rebuildable projection: the nodes
    table holds the declared dependency graph plus each node's current
    availability, and the events table holds every accepted Source event.
    Replaying the events in sequence order reproduces the node states, so a
    restart never has to trust a derived cache.
    """

    async def nodes(self, connection: AsyncConnection, scope_id: str, /) -> tuple[StoredLifecycleNode, ...]:
        _require_scope(scope_id)
        rows = (
            await connection.execute(
                select(LIFECYCLE_NODES_TABLE)
                .where(LIFECYCLE_NODES_TABLE.c.scope_id == scope_id)
                .order_by(LIFECYCLE_NODES_TABLE.c.node_id)
            )
        ).mappings()
        return tuple(_node(row) for row in rows)

    async def node(
        self,
        connection: AsyncConnection,
        scope_id: str,
        node_id: str,
        /,
    ) -> StoredLifecycleNode | None:
        _require_scope(scope_id)
        _require_node(node_id)
        row = (
            (
                await connection.execute(
                    select(LIFECYCLE_NODES_TABLE).where(
                        LIFECYCLE_NODES_TABLE.c.scope_id == scope_id,
                        LIFECYCLE_NODES_TABLE.c.node_id == node_id,
                    )
                )
            )
            .mappings()
            .one_or_none()
        )
        return None if row is None else _node(row)

    async def events(self, connection: AsyncConnection, scope_id: str, /) -> tuple[StoredLifecycleEvent, ...]:
        _require_scope(scope_id)
        rows = (
            await connection.execute(
                select(LIFECYCLE_EVENTS_TABLE)
                .where(LIFECYCLE_EVENTS_TABLE.c.scope_id == scope_id)
                .order_by(LIFECYCLE_EVENTS_TABLE.c.node_id, LIFECYCLE_EVENTS_TABLE.c.sequence)
            )
        ).mappings()
        return tuple(_event(row) for row in rows)

    async def add_nodes(
        self,
        connection: AsyncConnection,
        scope_id: str,
        values: tuple[StoredLifecycleNode, ...],
        /,
    ) -> None:
        _require_scope(scope_id)
        if not values:
            return
        await connection.execute(
            insert(LIFECYCLE_NODES_TABLE),
            [_node_values(scope_id, value) for value in values],
        )

    async def set_availability(
        self,
        connection: AsyncConnection,
        scope_id: str,
        values: tuple[tuple[str, bool], ...],
        /,
    ) -> None:
        """Persist the availability reached by one accepted propagation."""

        _require_scope(scope_id)
        for node_id, available in values:
            _require_node(node_id)
            await connection.execute(
                LIFECYCLE_NODES_TABLE
                .update()
                .where(
                    LIFECYCLE_NODES_TABLE.c.scope_id == scope_id,
                    LIFECYCLE_NODES_TABLE.c.node_id == node_id,
                )
                .values(available=available)
            )

    async def append_event(
        self,
        connection: AsyncConnection,
        scope_id: str,
        value: StoredLifecycleEvent,
        /,
    ) -> None:
        _require_scope(scope_id)
        _require_node(value.node_id)
        await connection.execute(
            insert(LIFECYCLE_EVENTS_TABLE).values(
                scope_id=scope_id,
                node_id=value.node_id,
                sequence=value.sequence,
                revision=value.revision,
                available=value.available,
                reason=value.reason,
            )
        )

    async def delete_scope(self, connection: AsyncConnection, scope_id: str, /) -> None:
        _require_scope(scope_id)
        for table in (LIFECYCLE_EVENTS_TABLE, LIFECYCLE_NODES_TABLE):
            await connection.execute(delete(table).where(table.c.scope_id == scope_id))


def _node_values(scope_id: str, value: StoredLifecycleNode) -> dict[str, object]:
    return {
        "scope_id": scope_id,
        "node_id": value.node_id,
        "node_kind": value.node_kind,
        "revision": value.revision,
        "reference_family": value.reference_family,
        "reference_artifact_id": value.reference_artifact_id,
        "reference_revision": value.reference_revision,
        "source_type": value.source_type,
        "source_id": value.source_id,
        "entry_id": value.entry_id,
        "entry_version_id": value.entry_version_id,
        "parents": _NodeIds.dump_json(value.parents),
        "policy": value.policy,
        "available": value.available,
    }


def _node(row: Mapping[Any, Any]) -> StoredLifecycleNode:
    parents = _NodeIds.validate_json(stored_bytes(row["parents"], column="parents"))
    return StoredLifecycleNode(
        node_id=str(row["node_id"]),
        node_kind=str(row["node_kind"]),
        revision=int(row["revision"]),
        reference_family=_optional_text(row["reference_family"]),
        reference_artifact_id=_optional_text(row["reference_artifact_id"]),
        reference_revision=None if row["reference_revision"] is None else int(row["reference_revision"]),
        source_type=_optional_text(row["source_type"]),
        source_id=_optional_text(row["source_id"]),
        entry_id=_optional_text(row["entry_id"]),
        entry_version_id=_optional_text(row["entry_version_id"]),
        parents=parents,
        policy=_optional_text(row["policy"]),
        available=bool(row["available"]),
    )


def _event(row: Mapping[Any, Any]) -> StoredLifecycleEvent:
    return StoredLifecycleEvent(
        node_id=str(row["node_id"]),
        sequence=int(row["sequence"]),
        revision=int(row["revision"]),
        available=bool(row["available"]),
        reason=str(row["reason"]),
    )


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)


def _require_scope(scope_id: str) -> None:
    if not isinstance(scope_id, str) or not scope_id or len(scope_id) > MAX_SCOPE_ID_LENGTH:
        raise InvalidRepositoryArgumentError("scope_id", "must be a non-empty bounded string")


def _require_node(node_id: str) -> None:
    if not isinstance(node_id, str) or not node_id or len(node_id) > MAX_LIFECYCLE_NODE_ID_LENGTH:
        raise InvalidRepositoryArgumentError("node_id", "must be a non-empty bounded string")


__all__ = [
    "LifecycleLedgerRepository",
    "LifecycleNodeKind",
    "LifecyclePolicy",
    "StoredLifecycleEvent",
    "StoredLifecycleNode",
]
