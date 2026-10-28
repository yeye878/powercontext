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

"""Source lifecycle propagation over an explicit evidence dependency DAG.

One Scope owns one dependency graph. Sources are the roots; Memory entries and
Topic Memory revisions are derived nodes that stay usable only while their
declared parents still support them under the node's all or any policy.
Withdrawal and restoration arrive as per-Source events; every accepted event is
propagated to all descendants and mirrored into the native Memory and Topic
Memory state, so search and Context agree with it while the evidence reference
captured at registration keeps pointing at the same immutable revision.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Annotated, Literal, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.persistence.lifecycle import (
    LifecycleNodeKind,
    LifecyclePolicy,
    StoredLifecycleEvent,
    StoredLifecycleNode,
)
from powercontext.limits import MAX_LIFECYCLE_NODE_ID_LENGTH, MAX_LIFECYCLE_REASON_LENGTH
from powercontext.sources import SourceRef

logger = logging.getLogger(__name__)

LifecyclePurpose: TypeAlias = Literal["generation", "audit"]
LifecycleReference: TypeAlias = MemoryCitation | ArtifactRef | SourceRef
LifecycleBackendFactory: TypeAlias = Callable[[str], "LifecycleBackend"]
_PURPOSES: frozenset[str] = frozenset(("generation", "audit"))
_POLICIES: frozenset[str] = frozenset(("all", "any"))
_TOPIC_FAMILY = "topic-memory"


class SourceLifecycleError(ValueError):
    """Raised when a lifecycle request cannot be honoured under the contract."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(code if not detail else code + ": " + detail)


class SourceLifecycleEvent(BaseModel):
    """One caller-declared Source availability change.

    The revision must equal the Source's registered lifecycle version and the
    sequence must strictly increase per Source. Extra fields are rejected so a
    caller cannot smuggle state the contract does not define.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    node_id: str
    revision: Annotated[int, Field(ge=1, strict=True)]
    sequence: Annotated[int, Field(ge=1, strict=True)]
    available: bool
    reason: str

    @field_validator("node_id", "reason")
    @classmethod
    def require_text(cls, value: str, info) -> str:
        if not isinstance(value, str):
            # Pydantic field validators must signal through ValueError; TypeError
            # would escape the validation envelope instead of becoming ValidationError.
            raise ValueError(info.field_name + " must be a string")  # noqa: TRY004
        if not value.strip():
            raise ValueError(info.field_name + " must contain non-whitespace content")
        limit = MAX_LIFECYCLE_NODE_ID_LENGTH if info.field_name == "node_id" else MAX_LIFECYCLE_REASON_LENGTH
        if len(value) > limit:
            raise ValueError(info.field_name + " must not exceed " + str(limit) + " characters")
        return value


class LifecycleNode(BaseModel):
    """One registered node with the evidence binding captured at registration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    kind: LifecycleNodeKind
    parents: tuple[str, ...] = ()
    policy: LifecyclePolicy | None = None
    available: bool
    reference: ArtifactRef | None = None
    citation: MemoryCitation | None = None
    source_ref: SourceRef | None = None
    revision: int = 1


class LifecycleBackend(Protocol):
    """Durable ledger plus the native writers a lifecycle service propagates into."""

    async def load(
        self, scope_id: str, /
    ) -> tuple[dict[str, StoredLifecycleNode], dict[str, tuple[StoredLifecycleEvent, ...]]]: ...

    async def append_node(self, scope_id: str, node: StoredLifecycleNode, /) -> None: ...

    async def unavailable_topics(self, scope_id: str, /) -> frozenset[str]: ...

    async def require_source(self, scope_id: str, source_ref: SourceRef, /) -> None: ...

    async def require_memory(self, scope_id: str, citation: MemoryCitation, /) -> None: ...

    async def require_topic(self, scope_id: str, reference: ArtifactRef, /) -> None: ...

    async def set_memory_availability(
        self,
        scope_id: str,
        targets: Sequence[tuple[MemoryCitation, bool]],
        /,
        *,
        reason: str,
    ) -> None: ...

    async def commit_event(
        self,
        scope_id: str,
        event: StoredLifecycleEvent,
        availability: Sequence[tuple[str, bool]],
        /,
    ) -> None: ...


class SourceLifecycleService:
    """Register evidence and propagate Source availability through one Scope."""

    def __init__(self, scope_id: str, backend: LifecycleBackend, /) -> None:
        self.scope_id = scope_id
        self._backend = backend

    async def register_source(self, node_id: str, source_ref: SourceRef, /, *, revision: int) -> None:
        """Bind one captured Source as a graph root at an explicit lifecycle version."""

        _require_node_id(node_id)
        _require_revision(revision)
        if not isinstance(source_ref, SourceRef):
            raise SourceLifecycleError("source-ref", "must be a SourceRef")
        node = StoredLifecycleNode(
            node_id=node_id,
            node_kind="source",
            revision=revision,
            source_type=source_ref.source_type,
            source_id=source_ref.source_id,
            available=True,
        )
        await self._register(node)

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        /,
        *,
        parents: Sequence[str],
        policy: LifecyclePolicy,
    ) -> None:
        """Bind one exact Memory entry version under declared parent support."""

        _require_node_id(node_id)
        if not isinstance(citation, MemoryCitation):
            raise SourceLifecycleError("memory-citation", "must be a MemoryCitation")
        parent_ids = _require_parents(parents)
        _require_policy(policy)
        nodes, _ = await self._backend.load(self.scope_id)
        _require_parent_closure(nodes, node_id, parent_ids)
        node = StoredLifecycleNode(
            node_id=node_id,
            node_kind="memory",
            revision=citation.memory_ref.revision,
            reference_family=citation.memory_ref.family,
            reference_artifact_id=citation.memory_ref.artifact_id,
            reference_revision=citation.memory_ref.revision,
            entry_id=citation.entry_id,
            entry_version_id=citation.entry_version_id,
            parents=parent_ids,
            policy=policy,
            available=_derived_availability(nodes, parent_ids, policy),
        )
        if await self._register(node):
            await self._backend.require_memory(self.scope_id, citation)
            await self._sync_native(self.scope_id, ((node.node_id, node.available),), reason="")

    async def register_topic(
        self,
        node_id: str,
        reference: ArtifactRef,
        /,
        *,
        parents: Sequence[str],
        policy: LifecyclePolicy,
    ) -> None:
        """Bind one exact Topic Memory revision under declared parent support."""

        _require_node_id(node_id)
        if not isinstance(reference, ArtifactRef):
            raise SourceLifecycleError("topic-reference", "must be an ArtifactRef")
        if reference.family != _TOPIC_FAMILY:
            raise SourceLifecycleError("topic-reference", "must reference topic-memory")
        parent_ids = _require_parents(parents)
        _require_policy(policy)
        nodes, _ = await self._backend.load(self.scope_id)
        _require_parent_closure(nodes, node_id, parent_ids)
        node = StoredLifecycleNode(
            node_id=node_id,
            node_kind="topic",
            revision=reference.revision,
            reference_family=reference.family,
            reference_artifact_id=reference.artifact_id,
            reference_revision=reference.revision,
            parents=parent_ids,
            policy=policy,
            available=_derived_availability(nodes, parent_ids, policy),
        )
        if await self._register(node):
            await self._backend.require_topic(self.scope_id, reference)

    async def apply(self, event: SourceLifecycleEvent, /) -> list[str]:
        """Accept one Source event, propagate it, and return the available node ids."""

        if not isinstance(event, SourceLifecycleEvent):
            raise SourceLifecycleError("event", "must be a SourceLifecycleEvent")
        nodes, events = await self._backend.load(self.scope_id)
        target = nodes.get(event.node_id)
        if target is None:
            raise SourceLifecycleError("unknown-source", event.node_id)
        if target.node_kind != "source":
            raise SourceLifecycleError("not-a-source", event.node_id)
        if target.revision != event.revision:
            raise SourceLifecycleError(
                "revision-mismatch",
                event.node_id + " is registered at revision " + str(target.revision),
            )
        recorded = events.get(event.node_id, ())
        _require_sequence(recorded, event)
        state = _propagate({**nodes, event.node_id: target.model_copy(update={"available": event.available})})
        available = sorted(node_id for node_id, value in state.items() if value)
        if _is_exact_replay(recorded, event):
            return available
        changed = tuple(
            (node_id, state[node_id]) for node_id in sorted(state) if state[node_id] != nodes[node_id].available
        )
        await self._sync_native(self.scope_id, changed, reason=event.reason)
        await self._backend.commit_event(
            self.scope_id,
            StoredLifecycleEvent(
                node_id=event.node_id,
                sequence=event.sequence,
                revision=event.revision,
                available=event.available,
                reason=event.reason,
            ),
            tuple((node_id, state[node_id]) for node_id in sorted(state)),
        )
        return available

    async def resolve(self, node_id: str, /, *, purpose: LifecyclePurpose) -> LifecycleReference:
        """Return the original evidence reference for generation or audit use."""

        _require_node_id(node_id)
        if purpose not in _PURPOSES:
            raise SourceLifecycleError("purpose", "must be one of " + repr(sorted(_PURPOSES)))
        nodes, _ = await self._backend.load(self.scope_id)
        node = nodes.get(node_id)
        if node is None:
            raise SourceLifecycleError("unknown-node", node_id)
        if purpose == "generation" and not node.available:
            raise SourceLifecycleError("unavailable", node_id)
        return _reference(node)

    async def active(self) -> list[str]:
        """Return every currently available node id in sorted order."""

        nodes, _ = await self._backend.load(self.scope_id)
        return sorted(node_id for node_id, node in nodes.items() if node.available)

    async def nodes(self) -> tuple[LifecycleNode, ...]:
        """Return the registered graph in node-id order."""

        nodes, _ = await self._backend.load(self.scope_id)
        return tuple(_public_node(nodes[node_id]) for node_id in sorted(nodes))

    async def _register(self, node: StoredLifecycleNode, /) -> bool:
        """Insert one binding, tolerating an identical re-registration.

        Returns True when the node was created. An identical registration is a
        no-op; any difference in the binding is a conflict, because a node id
        names one piece of evidence.
        """

        nodes, _ = await self._backend.load(self.scope_id)
        existing = nodes.get(node.node_id)
        if existing is not None:
            if _binding_of(existing) != _binding_of(node):
                raise SourceLifecycleError("conflict", node.node_id + " is already bound to different evidence")
            return False
        if node.node_kind == "source":
            await self._backend.require_source(self.scope_id, _source_ref(node))
        await self._backend.append_node(self.scope_id, node)
        return True

    async def _sync_native(
        self,
        scope_id: str,
        changed: Sequence[tuple[str, bool]],
        /,
        *,
        reason: str,
    ) -> None:
        """Mirror the reached availability of every changed Memory node."""

        if not changed:
            return
        nodes, _ = await self._backend.load(scope_id)
        targets: list[tuple[MemoryCitation, bool]] = []
        for node_id, available in changed:
            node = nodes.get(node_id)
            if node is None or node.node_kind != "memory":
                continue
            citation = _citation(node)
            if citation is None:
                continue
            targets.append((citation, available))
        if targets:
            await self._backend.set_memory_availability(scope_id, tuple(targets), reason=reason)


class SourceLifecycleApplication:
    """Select the lifecycle service for one Scope and expose its visibility filter."""

    def __init__(self, factory: LifecycleBackendFactory, /) -> None:
        self._factory = factory

    def for_scope(self, scope_id: str, /) -> SourceLifecycleService:
        if not isinstance(scope_id, str) or not scope_id.strip():
            raise SourceLifecycleError("scope-id", "must be a non-empty string")
        return SourceLifecycleService(scope_id, self._factory(scope_id))

    async def unavailable_topics(self, scope_id: str, /) -> frozenset[str]:
        """Return the Topic artifact ids whose registered evidence is withdrawn.

        Empty for a Scope that never registered lifecycle evidence, which keeps
        the ordinary native Topic search path unchanged.
        """

        if not isinstance(scope_id, str) or not scope_id.strip():
            return frozenset()
        return await self._factory(scope_id).unavailable_topics(scope_id)


def _public_node(node: StoredLifecycleNode) -> LifecycleNode:
    reference = _artifact_ref(node)
    return LifecycleNode(
        node_id=node.node_id,
        kind=node.node_kind,
        parents=node.parents,
        policy=node.policy,
        available=node.available,
        reference=reference,
        citation=_citation(node),
        source_ref=(
            None
            if node.source_type is None or node.source_id is None
            else SourceRef(source_type=node.source_type, source_id=node.source_id)
        ),
        revision=node.revision,
    )


def _binding_of(node: StoredLifecycleNode) -> tuple[object, ...]:
    return (
        node.node_kind,
        node.revision,
        node.reference_family,
        node.reference_artifact_id,
        node.reference_revision,
        node.source_type,
        node.source_id,
        node.entry_id,
        node.entry_version_id,
        node.parents,
        node.policy,
    )


def _source_ref(node: StoredLifecycleNode) -> SourceRef:
    if node.source_type is None or node.source_id is None:
        raise SourceLifecycleError("source-ref", "a Source root requires a captured SourceRef")
    return SourceRef(source_type=node.source_type, source_id=node.source_id)


def _artifact_ref(node: StoredLifecycleNode) -> ArtifactRef | None:
    if node.reference_family is None or node.reference_artifact_id is None or node.reference_revision is None:
        return None
    return ArtifactRef(
        family=node.reference_family,
        artifact_id=node.reference_artifact_id,
        revision=node.reference_revision,
    )


def _citation(node: StoredLifecycleNode) -> MemoryCitation | None:
    if node.entry_id is None or node.entry_version_id is None:
        return None
    reference = _artifact_ref(node)
    if reference is None:
        return None
    return MemoryCitation(memory_ref=reference, entry_id=node.entry_id, entry_version_id=node.entry_version_id)


def _reference(node: StoredLifecycleNode) -> LifecycleReference:
    if node.node_kind == "source":
        return _source_ref(node)
    citation = _citation(node)
    if citation is not None:
        return citation
    reference = _artifact_ref(node)
    if reference is None:
        raise SourceLifecycleError("reference", "node " + node.node_id + " lost its evidence reference")
    return reference


def _require_node_id(node_id: str) -> None:
    if not isinstance(node_id, str):
        raise SourceLifecycleError("node-id", "must be a string")
    if not node_id.strip():
        raise SourceLifecycleError("node-id", "must contain non-whitespace content")
    if len(node_id) > MAX_LIFECYCLE_NODE_ID_LENGTH:
        raise SourceLifecycleError("node-id", "must not exceed " + str(MAX_LIFECYCLE_NODE_ID_LENGTH) + " characters")


def _require_revision(revision: int) -> None:
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise SourceLifecycleError("revision", "must be a positive integer")


def _require_policy(policy: str) -> None:
    if policy not in _POLICIES:
        raise SourceLifecycleError("policy", "must be all or any")


def _require_parents(parents: Sequence[str]) -> tuple[str, ...]:
    if isinstance(parents, str):
        raise SourceLifecycleError("parents", "must be a sequence of node ids")
    values = tuple(parents)
    if not values:
        raise SourceLifecycleError("parents", "a derived node requires at least one parent")
    for value in values:
        _require_node_id(value)
    if len(set(values)) != len(values):
        raise SourceLifecycleError("parents", "must not repeat a parent node id")
    return values


def _require_parent_closure(
    nodes: dict[str, StoredLifecycleNode],
    node_id: str,
    parents: tuple[str, ...],
) -> None:
    for parent in parents:
        if parent == node_id:
            raise SourceLifecycleError("cycle", node_id + " cannot depend on itself")
        if parent not in nodes:
            raise SourceLifecycleError("unknown-parent", parent)
    if _reaches(nodes, parents, node_id):
        raise SourceLifecycleError("cycle", node_id)


def _reaches(nodes: dict[str, StoredLifecycleNode], starts: Sequence[str], target: str) -> bool:
    pending = list(starts)
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current == target:
            return True
        if current in seen:
            continue
        seen.add(current)
        node = nodes.get(current)
        if node is not None:
            pending.extend(node.parents)
    return False


def _derived_availability(
    nodes: dict[str, StoredLifecycleNode],
    parents: tuple[str, ...],
    policy: LifecyclePolicy,
) -> bool:
    support = [nodes[parent].available for parent in parents]
    return all(support) if policy == "all" else any(support)


def _require_sequence(events: Sequence[StoredLifecycleEvent], event: SourceLifecycleEvent) -> None:
    for recorded in events:
        if recorded.sequence < event.sequence:
            continue
        if recorded.sequence > event.sequence:
            raise SourceLifecycleError("sequence", event.node_id + " already advanced past this sequence")
        if not _same_event(recorded, event):
            raise SourceLifecycleError("sequence", event.node_id + " reused a sequence for different content")


def _is_exact_replay(events: Sequence[StoredLifecycleEvent], event: SourceLifecycleEvent) -> bool:
    return any(recorded.sequence == event.sequence and _same_event(recorded, event) for recorded in events)


def _same_event(recorded: StoredLifecycleEvent, event: SourceLifecycleEvent) -> bool:
    return (
        recorded.revision == event.revision
        and recorded.available == event.available
        and recorded.reason == event.reason
    )


def _propagate(nodes: dict[str, StoredLifecycleNode]) -> dict[str, bool]:
    """Recompute every node's availability from the roots in topological order."""

    state: dict[str, bool] = {}
    for node_id in _topological_order(nodes):
        node = nodes[node_id]
        if node.node_kind == "source":
            state[node_id] = node.available
            continue
        support = [state[parent] for parent in node.parents]
        state[node_id] = all(support) if node.policy == "all" else any(support)
    return state


def _topological_order(nodes: dict[str, StoredLifecycleNode]) -> tuple[str, ...]:
    remaining = {node_id: set(node.parents) for node_id, node in nodes.items()}
    order: list[str] = []
    while remaining:
        ready = sorted(node_id for node_id, parents in remaining.items() if not parents)
        if not ready:
            raise SourceLifecycleError("cycle", "the registered graph is not acyclic")
        for node_id in ready:
            order.append(node_id)
            del remaining[node_id]
        for parents in remaining.values():
            parents.difference_update(ready)
    return tuple(order)


__all__ = [
    "LifecycleBackend",
    "LifecycleBackendFactory",
    "LifecycleNode",
    "LifecyclePurpose",
    "LifecycleReference",
    "SourceLifecycleApplication",
    "SourceLifecycleError",
    "SourceLifecycleEvent",
    "SourceLifecycleService",
]
