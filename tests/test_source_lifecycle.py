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

"""Contract edges of Source lifecycle propagation beyond the published sample.

The published acceptance covers the happy DAG, strict event identity, late
derivatives, and a shared Topic. These tests protect the remaining promises:
idempotent registration, the registration-time derived state, every dependency
rejection, per-Source sequence ordering, cross-Scope isolation, restart
consistency, and retry after an interrupted native write.
"""

from __future__ import annotations

import asyncio
import sqlite3

import pytest
from pydantic import ValidationError
from sqlalchemy.exc import DBAPIError

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.artifacts.memory import MemoryEntryInput
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.records import ArtifactWrite
from powercontext.builtin.runtime import (
    BuiltinConfig,
    CaptureSource,
    RememberMemoryRequest,
    open_builtin_runtime,
)
from powercontext.builtin.runtime.source_lifecycle import (
    SourceLifecycleError,
    SourceLifecycleEvent,
)
from powercontext.builtin.scope import ScopeDraft


async def _scope(runtime, key: str) -> str:
    created = await runtime.scopes.create(
        ScopeDraft(title="Lifecycle", summary="Dependency evidence", idempotency_key=key)
    )
    return created.scope_id


async def _source(runtime, scope: str, name: str):
    receipt = await runtime.sources.for_scope(scope).capture(
        CaptureSource(source_id=name, content="Origin " + name, metadata={})
    )
    return receipt.source_ref


async def _memory(runtime, scope: str, text: str):
    result = await runtime.memory.for_scope(scope).remember(
        RememberMemoryRequest(entries=(MemoryEntryInput(kind="fact", text=text),))
    )
    return result.entry.citation


async def _topic(runtime, scope: str, name: str) -> ArtifactRef:
    created = await runtime.records.for_scope(scope).create_artifact(
        "topic-memory",
        ArtifactWrite(
            content={
                "title": name,
                "summary": name + " supported summary",
                "detail": name + " detailed evidence",
            }
        ),
    )
    return ArtifactRef(family="topic-memory", artifact_id=created.artifact_id, revision=created.revision)


def _event(node_id: str, sequence: int, available: bool, *, revision: int = 1) -> SourceLifecycleEvent:
    return SourceLifecycleEvent(
        node_id=node_id,
        revision=revision,
        sequence=sequence,
        available=available,
        reason="controlled lifecycle",
    )


def test_event_rejects_extra_fields_and_blank_text() -> None:
    with pytest.raises(ValidationError):
        SourceLifecycleEvent(
            node_id="root",
            revision=1,
            sequence=1,
            available=True,
            reason="why",
            extra=True,
        )
    for payload in (
        {"node_id": "", "reason": "why"},
        {"node_id": "   ", "reason": "why"},
        {"node_id": "root", "reason": ""},
        {"node_id": "root", "reason": "\t\n"},
        {"node_id": "root", "reason": "why", "revision": 0},
        {"node_id": "root", "reason": "why", "sequence": 0},
        {"node_id": "root", "reason": "why", "revision": -1},
        {"node_id": "root", "reason": "why", "available": "yes"},
    ):
        base = {"revision": 1, "sequence": 1, "available": True}
        with pytest.raises(ValidationError):
            SourceLifecycleEvent(**{**base, **payload})


def test_identical_registration_is_idempotent_and_conflicting_binding_is_rejected(tmp_path) -> None:
    async def run():
        config = BuiltinConfig(database=SQLiteConfig(url="sqlite+aiosqlite:///" + str(tmp_path / "idem.db")))
        async with open_builtin_runtime(config) as runtime:
            scope = await _scope(runtime, "idem")
            service = runtime.lifecycle.for_scope(scope)
            source = await _source(runtime, scope, "root")
            await service.register_source("root", source, revision=2)
            await service.register_source("root", source, revision=2)
            with pytest.raises(SourceLifecycleError):
                await service.register_source("root", source, revision=3)

            citation = await _memory(runtime, scope, "idem fixture evidence")
            await service.register_memory("fact", citation, parents=("root",), policy="all")
            await service.register_memory("fact", citation, parents=("root",), policy="all")
            with pytest.raises(SourceLifecycleError):
                await service.register_memory("fact", citation, parents=("root",), policy="any")

            # The node id names one piece of evidence across kinds as well.
            with pytest.raises(SourceLifecycleError):
                await service.register_topic("fact", await _topic(runtime, scope, "t"), parents=("root",), policy="all")

            assert await service.active() == ["fact", "root"]

    asyncio.run(run())


def test_registration_computes_state_immediately_from_withdrawn_parents(tmp_path) -> None:
    async def run():
        config = BuiltinConfig(database=SQLiteConfig(url="sqlite+aiosqlite:///" + str(tmp_path / "state.db")))
        async with open_builtin_runtime(config) as runtime:
            scope = await _scope(runtime, "state")
            service = runtime.lifecycle.for_scope(scope)
            await service.register_source("root-a", await _source(runtime, scope, "a"), revision=1)
            await service.register_source("root-b", await _source(runtime, scope, "b"), revision=1)
            await service.apply(_event("root-a", 1, False))

            citation = await _memory(runtime, scope, "state fixture evidence")
            await service.register_memory("needs-a", citation, parents=("root-a",), policy="all")
            await service.register_memory("either", citation, parents=("root-a", "root-b"), policy="any")
            topic = await _topic(runtime, scope, "state-topic")
            await service.register_topic("topic-needs-a", topic, parents=("needs-a",), policy="all")

            assert await service.active() == ["either", "root-b"]
            assert [node.available for node in await service.nodes() if node.node_id == "needs-a"] == [False]
            with pytest.raises(SourceLifecycleError):
                await service.resolve("topic-needs-a", purpose="generation")
            assert await service.resolve("topic-needs-a", purpose="audit") == topic

    asyncio.run(run())


def test_dependency_rejections_are_explicit(tmp_path) -> None:
    async def run():
        config = BuiltinConfig(database=SQLiteConfig(url="sqlite+aiosqlite:///" + str(tmp_path / "dep.db")))
        async with open_builtin_runtime(config) as runtime:
            scope = await _scope(runtime, "dep")
            service = runtime.lifecycle.for_scope(scope)
            await service.register_source("root", await _source(runtime, scope, "root"), revision=1)
            citation = await _memory(runtime, scope, "dep fixture evidence")

            with pytest.raises(SourceLifecycleError):
                await service.register_memory("no-parents", citation, parents=(), policy="all")
            with pytest.raises(SourceLifecycleError):
                await service.register_memory("missing", citation, parents=("absent",), policy="all")
            with pytest.raises(SourceLifecycleError):
                await service.register_memory("dupe", citation, parents=("root", "root"), policy="any")
            with pytest.raises(SourceLifecycleError):
                await service.register_memory("self", citation, parents=("self",), policy="all")
            with pytest.raises(SourceLifecycleError):
                await service.register_memory("root", citation, parents=("root",), policy="all")
            with pytest.raises(SourceLifecycleError):
                await service.register_memory("bad-policy", citation, parents=("root",), policy="most")

            await service.register_memory("child", citation, parents=("root",), policy="all")
            await service.register_memory("grandchild", citation, parents=("child",), policy="all")
            # A parent must already be registered, so a cycle cannot be created;
            # re-binding an existing id to a new parent set is a conflict instead.
            with pytest.raises(SourceLifecycleError):
                await service.register_memory("child", citation, parents=("grandchild",), policy="all")

    asyncio.run(run())


def test_sequence_is_monotonic_per_source_and_replay_is_free(tmp_path) -> None:
    async def run():
        config = BuiltinConfig(database=SQLiteConfig(url="sqlite+aiosqlite:///" + str(tmp_path / "seq.db")))
        async with open_builtin_runtime(config) as runtime:
            scope = await _scope(runtime, "seq")
            service = runtime.lifecycle.for_scope(scope)
            await service.register_source("root-a", await _source(runtime, scope, "a"), revision=1)
            await service.register_source("root-b", await _source(runtime, scope, "b"), revision=1)
            citation = await _memory(runtime, scope, "seq fixture evidence")
            await service.register_memory("fact", citation, parents=("root-a",), policy="all")

            await service.apply(_event("root-a", 3, False))
            before = (await runtime.memory.for_scope(scope).list()).memory_ref
            # An exact replay of an accepted event changes nothing.
            assert await service.apply(_event("root-a", 3, False)) == ["root-b"]
            assert (await runtime.memory.for_scope(scope).list()).memory_ref == before

            with pytest.raises(SourceLifecycleError):
                await service.apply(_event("root-a", 2, True))
            with pytest.raises(SourceLifecycleError):
                await service.apply(_event("root-a", 3, True))
            # Sequences are per Source, so another Source may reuse the number.
            assert await service.apply(_event("root-b", 3, False)) == []
            await service.apply(_event("root-a", 4, True))
            assert await service.active() == ["fact", "root-a"]

    asyncio.run(run())


def test_scopes_are_isolated_for_identical_node_ids(tmp_path) -> None:
    async def run():
        config = BuiltinConfig(database=SQLiteConfig(url="sqlite+aiosqlite:///" + str(tmp_path / "scope.db")))
        async with open_builtin_runtime(config) as runtime:
            left = await _scope(runtime, "left")
            right = await _scope(runtime, "right")
            left_service = runtime.lifecycle.for_scope(left)
            right_service = runtime.lifecycle.for_scope(right)
            for scope, service in ((left, left_service), (right, right_service)):
                await service.register_source("root", await _source(runtime, scope, "root"), revision=1)
                citation = await _memory(runtime, scope, "shared node evidence")
                await service.register_memory("fact", citation, parents=("root",), policy="all")

            assert await left_service.apply(_event("root", 1, False)) == []
            assert await left_service.active() == []
            assert await right_service.active() == ["fact", "root"]
            assert await right_service.apply(_event("root", 1, False)) == []
            assert await right_service.active() == []
            await left_service.apply(_event("root", 2, True))
            assert await left_service.active() == ["fact", "root"]
            assert await right_service.active() == []

    asyncio.run(run())


def test_state_survives_restart_without_new_memory_revisions(tmp_path) -> None:
    database = tmp_path / "restart.db"
    config = BuiltinConfig(database=SQLiteConfig(url="sqlite+aiosqlite:///" + str(database)))

    async def first() -> tuple[str, ArtifactRef, ArtifactRef]:
        async with open_builtin_runtime(config) as runtime:
            scope = await _scope(runtime, "restart")
            service = runtime.lifecycle.for_scope(scope)
            await service.register_source("root", await _source(runtime, scope, "root"), revision=4)
            citation = await _memory(runtime, scope, "restart fixture evidence")
            await service.register_memory("fact", citation, parents=("root",), policy="all")
            topic = await _topic(runtime, scope, "restart-topic")
            await service.register_topic("topic", topic, parents=("fact",), policy="all")
            await service.apply(_event("root", 1, False, revision=4))
            return scope, citation, topic

    async def second(scope: str, citation: MemoryCitation, topic: ArtifactRef) -> None:
        async with open_builtin_runtime(config) as runtime:
            service = runtime.lifecycle.for_scope(scope)
            assert await service.active() == []
            with pytest.raises(SourceLifecycleError):
                await service.resolve("fact", purpose="generation")
            assert await service.resolve("fact", purpose="audit") == await service.resolve("fact", purpose="audit")
            assert (await service.resolve("topic", purpose="audit")) == topic
            # The registered body version is untouched by withdrawal: the head
            # revision advances for the state change, the entry body does not.
            page = await runtime.memory.for_scope(scope).list(include_inactive=True)
            assert page.memory_ref.artifact_id == citation.memory_ref.artifact_id
            assert [record.state for record in page.entries] == ["inactive"]
            assert page.entries[0].entry.entry_version_id == citation.entry_version_id
            assert await service.resolve("fact", purpose="audit") == citation
            # The replay of the accepted event adds no Memory revision.
            assert await service.apply(_event("root", 1, False, revision=4)) == []
            assert (await runtime.memory.for_scope(scope).list(include_inactive=True)).memory_ref == page.memory_ref
            assert await service.apply(_event("root", 2, True, revision=4)) == ["fact", "root", "topic"]
            assert await service.resolve("fact", purpose="generation") is not None

    scope, citation, topic = asyncio.run(first())
    asyncio.run(second(scope, citation, topic))


def test_retry_finishes_propagation_after_a_failed_native_write(tmp_path) -> None:
    database = tmp_path / "retry.db"
    config = BuiltinConfig(database=SQLiteConfig(url="sqlite+aiosqlite:///" + str(database)))

    async def run() -> None:
        async with open_builtin_runtime(config) as runtime:
            scope = await _scope(runtime, "retry")
            service = runtime.lifecycle.for_scope(scope)
            await service.register_source("root", await _source(runtime, scope, "root"), revision=1)
            citation = await _memory(runtime, scope, "retry fixture evidence")
            await service.register_memory("fact", citation, parents=("root",), policy="all")
            before = (await runtime.memory.for_scope(scope).list()).memory_ref

        # Fail only the native Memory write, leaving the ledger at the prior event.
        with sqlite3.connect(database) as connection:
            connection.execute(
                "CREATE TRIGGER interrupt_memory_write BEFORE INSERT ON pc_artifacts "
                "WHEN NEW.family = 'memory' BEGIN SELECT RAISE(ABORT, 'controlled memory failure'); END"
            )
        async with open_builtin_runtime(config) as runtime:
            service = runtime.lifecycle.for_scope(scope)
            with pytest.raises(DBAPIError):
                await service.apply(_event("root", 1, False))
            # The ledger did not advance, so the Scope still reports support.
            assert await service.active() == ["fact", "root"]
            assert (await runtime.lifecycle.for_scope(scope).nodes())[0].node_id == "fact"
            assert (await runtime.memory.for_scope(scope).list()).memory_ref == before

        with sqlite3.connect(database) as connection:
            connection.execute("DROP TRIGGER interrupt_memory_write")
        async with open_builtin_runtime(config) as runtime:
            service = runtime.lifecycle.for_scope(scope)
            assert await service.apply(_event("root", 1, False)) == []
            assert (await runtime.memory.for_scope(scope).list(include_inactive=True)).entries[0].state == "inactive"
            after = (await runtime.memory.for_scope(scope).list(include_inactive=True)).memory_ref
            # A second identical retry is still a no-op.
            assert await service.apply(_event("root", 1, False)) == []
            assert (await runtime.memory.for_scope(scope).list(include_inactive=True)).memory_ref == after

    asyncio.run(run())


def test_resolve_rejects_unknown_node_and_purpose(tmp_path) -> None:
    async def run():
        config = BuiltinConfig(database=SQLiteConfig(url="sqlite+aiosqlite:///" + str(tmp_path / "resolve.db")))
        async with open_builtin_runtime(config) as runtime:
            scope = await _scope(runtime, "resolve")
            service = runtime.lifecycle.for_scope(scope)
            await service.register_source("root", await _source(runtime, scope, "root"), revision=1)
            with pytest.raises(SourceLifecycleError):
                await service.resolve("absent", purpose="generation")
            with pytest.raises(SourceLifecycleError):
                await service.resolve("root", purpose="summary")
            assert (await service.resolve("root", purpose="audit")).source_id == "root"

    asyncio.run(run())
