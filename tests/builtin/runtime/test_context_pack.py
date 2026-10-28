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

"""Context Pack wire contract and incremental transport boundaries.

The public acceptance suite covers the happy path: canonical JSON, digest, byte count,
prefix reconstruction, and a few tampering cases. These tests protect the boundaries it
does not reach — chunk ceilings, determinism, every legal prefix including the extremes,
the full set of rejection codes, and the Runtime integration contract for empty results
and repeated calls.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json

import pytest

from powercontext.builtin.artifacts.memory import MemoryCandidateRequest, MemoryEntryInput
from powercontext.builtin.runtime import BuiltinConfig, CaptureSource, PrepareContextRequest, open_builtin_runtime
from powercontext.builtin.runtime.context_pack import (
    DEFAULT_CHUNK_BYTES,
    DELTA_SCHEMA,
    PACK_SCHEMA,
    ContextPack,
    ContextPackDelta,
    ContextPackError,
    apply_context_pack_delta,
)
from powercontext.builtin.scope import ScopeDraft
from powercontext.builtin.sources import ContentSource

_MEETING_TURNS = (
    "Project Manager: What are your experiences with remote controls ?",
    "Project Manager: You didn't have six remote controls sitting in front of you .",
)


class _CandidatePipeline:
    async def extract(self, request: MemoryCandidateRequest, /) -> tuple[MemoryEntryInput, ...]:
        return tuple(
            MemoryEntryInput(kind="conversation-turn", text=source.content, sources=(source,))
            for source in request.sources
            if isinstance(source, ContentSource)
        )


def _canonical(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _unsigned_digest(items: tuple[bytes, ...]) -> str:
    unsigned = {"schema": PACK_SCHEMA, "items": [base64.b64encode(item).decode("ascii") for item in items]}
    return "sha256:" + hashlib.sha256(_canonical(unsigned)).hexdigest()


def _wire(payload: dict[str, object]) -> bytes:
    return _canonical(payload)


def _pack_payload(pack: ContextPack) -> dict[str, object]:
    return json.loads(pack.to_bytes())


# --- chunking -------------------------------------------------------------


def test_from_content_is_deterministic_and_reversible() -> None:
    content = "alpha \u6d77\u6d0b beta\ngamma \u00e9\u00e8 \u4e2d\u6587\n"

    first = ContextPack.from_content(content, chunk_bytes=7)
    second = ContextPack.from_content(content, chunk_bytes=7)

    assert first == second
    assert first.to_bytes() == second.to_bytes()
    assert b"".join(first.items) == content.encode("utf-8")


@pytest.mark.parametrize(
    ("content", "limit"),
    [
        ("alpha beta gamma\ndelta epsilon\n", 1),
        ("alpha beta gamma\ndelta epsilon\n", 2),
        ("alpha \u6d77\u6d0b beta\n", 3),
        ("alpha \u6d77\u6d0b \u00e9\n\u4e2d\u6587", 4),
        ("alpha \u6d77\u6d0b beta\n", 7),
        ("a\u6d77\u6d0b\u00e9\n\u4e2d\u6587" * 5, 16),
        ("a\u6d77\u6d0b\u00e9\n\u4e2d\u6587" * 20, 64),
        ("a\u6d77\u6d0b\u00e9\n\u4e2d\u6587" * 200, DEFAULT_CHUNK_BYTES),
    ],
)
def test_every_chunk_respects_the_byte_ceiling_and_stays_valid_utf8(content: str, limit: int) -> None:
    pack = ContextPack.from_content(content, chunk_bytes=limit)

    assert pack.items
    for item in pack.items:
        assert isinstance(item, bytes)
        assert 0 < len(item) <= limit
        item.decode("utf-8")
    assert b"".join(pack.items) == content.encode("utf-8")


def test_chunk_ceiling_smaller_than_one_character_is_rejected() -> None:
    with pytest.raises(ContextPackError):
        ContextPack.from_content("\u6d77", chunk_bytes=2)
    with pytest.raises(ContextPackError):
        ContextPack.from_content("\U0001f600", chunk_bytes=3)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "512"])
def test_invalid_chunk_ceilings_are_rejected(limit: object) -> None:
    with pytest.raises(ContextPackError):
        ContextPack.from_content("alpha", chunk_bytes=limit)  # type: ignore[arg-type]


def test_empty_content_is_a_legal_empty_pack() -> None:
    empty = ContextPack.from_content("")

    assert empty.items == ()
    assert ContextPack.from_bytes(empty.to_bytes()) == empty
    assert b"".join(empty.items) == b""
    assert empty.digest == _unsigned_digest(())
    assert empty.byte_count == len(empty.to_bytes())


def test_chunking_prefers_line_boundaries() -> None:
    """The prepared preamble is its own leading chunk, which is what prefix reuse depends on."""

    content = "PowerContext prepared untrusted historical context.\n" + "x" * 200 + "\n" + "y" * 200

    pack = ContextPack.from_content(content, chunk_bytes=64)

    assert pack.items[0] == b"PowerContext prepared untrusted historical context.\n"


# --- wire format ----------------------------------------------------------


def test_pack_wire_fields_are_exactly_the_contract() -> None:
    ocean = "海洋".encode()
    pack = ContextPack.build((b"alpha", ocean))

    payload = _pack_payload(pack)

    assert set(payload) == {"schema", "items", "digest", "byte_count"}
    assert payload["schema"] == PACK_SCHEMA
    assert payload["items"] == ["YWxwaGE=", base64.b64encode(ocean).decode()]
    assert payload["digest"] == _unsigned_digest((b"alpha", ocean))
    assert payload["byte_count"] == len(pack.to_bytes())
    assert pack.to_bytes() == _canonical(payload)


def test_byte_count_resolves_against_its_own_digits() -> None:
    """A pack whose length crosses a digit boundary still reports the final length."""

    for size in (0, 1, 500, 900, 2000):
        pack = ContextPack.build((b"x" * size,) if size else ())
        payload = _pack_payload(pack)
        assert payload["byte_count"] == len(pack.to_bytes())
        assert pack.byte_count == len(pack.to_bytes())


def test_delta_wire_fields_are_exactly_the_contract_and_omit_byte_count() -> None:
    base = ContextPack.build((b"alpha",))
    target = ContextPack.build((b"alpha", b"beta"))

    delta = ContextPackDelta.between(base, target)
    payload = json.loads(delta.to_bytes())

    assert set(payload) == {"schema", "base_digest", "target_digest", "prefix_items", "appended"}
    assert payload["schema"] == DELTA_SCHEMA
    assert payload["base_digest"] == base.digest
    assert payload["target_digest"] == target.digest
    assert payload["prefix_items"] == 1
    assert payload["appended"] == [base64.b64encode(b"beta").decode("ascii")]
    assert delta.byte_count == len(delta.to_bytes())
    assert "byte_count" not in payload


def test_wire_encoding_is_canonical_utf8_json() -> None:
    pack = ContextPack.build(("\u6d77\u6d0b".encode(),))

    raw = pack.to_bytes()

    raw.decode("utf-8")
    assert raw == _canonical(json.loads(raw))
    assert b", " not in raw and b": " not in raw


# --- delta reconstruction -------------------------------------------------


def test_delta_uses_the_longest_shared_prefix() -> None:
    base = ContextPack.build((b"a", b"b", b"c"))

    assert ContextPackDelta.between(base, ContextPack.build((b"a", b"b", b"c", b"d"))).prefix_items == 3
    assert ContextPackDelta.between(base, ContextPack.build((b"a", b"b"))).prefix_items == 2
    assert ContextPackDelta.between(base, ContextPack.build((b"a", b"x"))).prefix_items == 1
    assert ContextPackDelta.between(base, ContextPack.build((b"x",))).prefix_items == 0
    assert ContextPackDelta.between(base, ContextPack.build(())).prefix_items == 0
    assert ContextPackDelta.between(ContextPack.build(()), base).prefix_items == 0


def test_identical_packs_produce_an_empty_delta() -> None:
    pack = ContextPack.build((b"a", b"b"))

    delta = ContextPackDelta.between(pack, pack)

    assert delta.prefix_items == 2
    assert delta.appended == ()
    assert apply_context_pack_delta(pack, delta).to_bytes() == pack.to_bytes()


@pytest.mark.parametrize(
    ("target_items", "prefix"),
    [
        ((b"a", b"b", b"new", b"tail"), 0),
        ((b"a", b"b", b"new", b"tail"), 1),
        ((b"a", b"b", b"new", b"tail"), 2),
        ((b"new", b"b", b"old"), 0),
        ((b"a", b"new", b"old"), 0),
        ((b"a", b"new", b"old"), 1),
        ((b"a", b"b"), 0),
        ((b"a", b"b"), 1),
        ((b"a", b"b"), 2),
        ((), 0),
    ],
)
def test_every_legal_prefix_rebuilds_the_target(target_items: tuple[bytes, ...], prefix: int) -> None:
    """Any shared chunk prefix works, not only the longest one `between` would pick."""

    base = ContextPack.build((b"a", b"b", b"old"))
    target = ContextPack.build(target_items)
    assert prefix <= len(base.items)
    assert base.items[:prefix] == target.items[:prefix]

    payload = {
        "schema": DELTA_SCHEMA,
        "base_digest": base.digest,
        "target_digest": target.digest,
        "prefix_items": prefix,
        "appended": [base64.b64encode(item).decode("ascii") for item in target.items[prefix:]],
    }
    rebuilt = apply_context_pack_delta(base, ContextPackDelta.from_bytes(_wire(payload)))

    assert rebuilt.to_bytes() == target.to_bytes()
    assert b"".join(rebuilt.items) == b"".join(target_items)


def test_rebuild_round_trips_through_serialized_bytes_only() -> None:
    base = ContextPack.from_content("PowerContext prepared untrusted historical context.\n" + "y" * 900)
    target = ContextPack.from_content(
        "PowerContext prepared untrusted historical context.\n" + "y" * 900 + "\ntail \u6d77\u6d0b"
    )

    delta = ContextPackDelta.between(base, target)
    rebuilt = apply_context_pack_delta(
        ContextPack.from_bytes(base.to_bytes()), ContextPackDelta.from_bytes(delta.to_bytes())
    )

    assert rebuilt.to_bytes() == target.to_bytes()
    assert b"".join(rebuilt.items).decode("utf-8") == b"".join(target.items).decode("utf-8")


# --- rejection contract ---------------------------------------------------


@pytest.mark.parametrize("field", ["schema", "items", "digest", "byte_count"])
def test_pack_rejects_missing_fields(field: str) -> None:
    payload = _pack_payload(ContextPack.build((b"alpha",)))
    payload.pop(field)

    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(_wire(payload))


def test_pack_rejects_extra_fields() -> None:
    payload = _pack_payload(ContextPack.build((b"alpha",)))
    payload["extra"] = "value"

    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(_wire(payload))


@pytest.mark.parametrize("field", ["schema", "base_digest", "target_digest", "prefix_items", "appended"])
def test_delta_rejects_missing_fields(field: str) -> None:
    delta = ContextPackDelta.between(ContextPack.build((b"a",)), ContextPack.build((b"a", b"b")))
    payload = json.loads(delta.to_bytes())
    payload.pop(field)

    with pytest.raises(ContextPackError):
        ContextPackDelta.from_bytes(_wire(payload))


def test_delta_rejects_extra_fields_including_a_written_byte_count() -> None:
    delta = ContextPackDelta.between(ContextPack.build((b"a",)), ContextPack.build((b"a", b"b")))
    payload = json.loads(delta.to_bytes())
    payload["byte_count"] = delta.byte_count

    with pytest.raises(ContextPackError):
        ContextPackDelta.from_bytes(_wire(payload))


@pytest.mark.parametrize(
    "payload",
    [
        b"not json",
        b"[]",
        b"\xff\xfe",
        json.dumps({"schema": PACK_SCHEMA}).encode(),
    ],
)
def test_pack_rejects_malformed_payloads(payload: bytes) -> None:
    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(payload)


def test_pack_rejects_wrong_schema_and_bad_base64() -> None:
    payload = _pack_payload(ContextPack.build((b"alpha",)))

    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(_wire({**payload, "schema": "powercontext.context-pack.v2"}))
    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(_wire({**payload, "items": ["not base64!!"]}))
    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(_wire({**payload, "items": [123]}))


def test_pack_rejects_tampered_items_digest_and_count() -> None:
    payload = _pack_payload(ContextPack.build((b"alpha",)))

    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(_wire({**payload, "items": [base64.b64encode(b"beta").decode()]}))
    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(_wire({**payload, "digest": "sha256:" + "0" * 64}))
    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(_wire({**payload, "byte_count": payload["byte_count"] + 1}))


@pytest.mark.parametrize(
    "digest", ["sha256:" + "0" * 63, "sha256:" + "0" * 65, "sha256:" + "G" * 64, "md5:" + "0" * 32]
)
def test_pack_rejects_digests_outside_the_sha256_form(digest: str) -> None:
    payload = _pack_payload(ContextPack.build((b"alpha",)))

    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(_wire({**payload, "digest": digest}))


def test_apply_rejects_a_wrong_base_pack() -> None:
    base = ContextPack.build((b"one",))
    delta = ContextPackDelta.between(base, ContextPack.build((b"one", b"two")))

    with pytest.raises(ContextPackError):
        apply_context_pack_delta(ContextPack.build((b"other",)), delta)


@pytest.mark.parametrize("prefix", [-1, True, 1.5, "1"])
def test_delta_rejects_non_integer_or_negative_prefix(prefix: object) -> None:
    delta = ContextPackDelta.between(ContextPack.build((b"a",)), ContextPack.build((b"a", b"b")))
    payload = json.loads(delta.to_bytes())
    payload["prefix_items"] = prefix

    with pytest.raises(ContextPackError):
        ContextPackDelta.from_bytes(_wire(payload))


def test_apply_rejects_a_prefix_beyond_the_base_chunk_count() -> None:
    base = ContextPack.build((b"a",))
    target = ContextPack.build((b"a", b"b"))

    with pytest.raises(ContextPackError):
        apply_context_pack_delta(base, ContextPackDelta(base.digest, target.digest, 2, (b"b",)))


def test_apply_rejects_a_wrong_target_digest() -> None:
    base = ContextPack.build((b"a",))

    with pytest.raises(ContextPackError):
        apply_context_pack_delta(base, ContextPackDelta(base.digest, "sha256:" + "0" * 64, 1, ()))


def test_apply_rejects_non_pack_operands() -> None:
    base = ContextPack.build((b"a",))

    with pytest.raises(ContextPackError):
        apply_context_pack_delta(base, "not a delta")  # type: ignore[arg-type]
    with pytest.raises(ContextPackError):
        apply_context_pack_delta("not a pack", ContextPackDelta(base.digest, base.digest, 1))  # type: ignore[arg-type]


def test_build_rejects_non_byte_chunks() -> None:
    with pytest.raises(ContextPackError):
        ContextPack.build((b"ok", "text"))  # type: ignore[arg-type]
    with pytest.raises(ContextPackError):
        ContextPack.build((bytearray(b"ok"),))  # type: ignore[arg-type]
    with pytest.raises(ContextPackError):
        ContextPack.build(b"flat bytes")  # type: ignore[arg-type]


def test_direct_construction_is_validated_like_the_factories() -> None:
    """A caller may assemble a pack or delta by hand; both still enforce the contract."""

    with pytest.raises(ContextPackError):
        ContextPack(items=("text",))  # type: ignore[arg-type]
    with pytest.raises(ContextPackError):
        ContextPack(schema=object())  # type: ignore[arg-type]

    good = "sha256:" + "0" * 64
    with pytest.raises(ContextPackError):
        ContextPackDelta("not-a-digest", good, 0)
    with pytest.raises(ContextPackError):
        ContextPackDelta(good, good, -1)
    with pytest.raises(ContextPackError):
        ContextPackDelta(good, good, True)
    with pytest.raises(ContextPackError):
        ContextPackDelta(good, good, 0, ("text",))  # type: ignore[arg-type]
    assert ContextPackDelta(good, good, 0, (b"x",)).appended == (b"x",)


# --- runtime integration --------------------------------------------------


def test_prepare_incremental_matches_prepare_for_empty_and_repeated_rounds() -> None:
    async def scenario() -> None:
        async with open_builtin_runtime(BuiltinConfig(), candidate_pipeline=_CandidatePipeline()) as runtime:
            scope = await runtime.scopes.create(
                ScopeDraft(title="Empty", summary="Empty", idempotency_key="context-pack-empty-runtime")
            )
            context = runtime.context.for_scope(scope.scope_id)
            request = PrepareContextRequest(query="remote", max_bytes=4096)

            prepared = await context.prepare(request)
            first = await context.prepare_incremental(request)

            assert prepared.status == "empty"
            assert isinstance(first, ContextPack)
            assert first.items == ()
            assert b"".join(first.items) == b""

            repeat = await context.prepare_incremental(request, base_pack=first)

            assert isinstance(repeat, ContextPackDelta)
            assert apply_context_pack_delta(first, repeat).items == ()

    asyncio.run(scenario())


def test_prepare_incremental_is_repeatable_and_serialization_independent() -> None:
    async def scenario() -> None:
        async with open_builtin_runtime(BuiltinConfig(), candidate_pipeline=_CandidatePipeline()) as runtime:
            scope = await runtime.scopes.create(
                ScopeDraft(title="Repeat", summary="Repeat", idempotency_key="context-pack-repeat-runtime")
            )
            sources = runtime.sources.for_scope(scope.scope_id)
            memory = runtime.memory.for_scope(scope.scope_id)
            context = runtime.context.for_scope(scope.scope_id)
            request = PrepareContextRequest(query="remote", max_bytes=4096)

            await sources.capture(CaptureSource(source_id="turn-1", content=_MEETING_TURNS[0], metadata={}))
            while (await memory.flush()).processed:
                pass

            first = await context.prepare_incremental(request)
            again = await context.prepare_incremental(request)

            assert isinstance(first, ContextPack) and isinstance(again, ContextPack)
            assert first.to_bytes() == again.to_bytes()

            await sources.capture(CaptureSource(source_id="turn-2", content=_MEETING_TURNS[1], metadata={}))
            while (await memory.flush()).processed:
                pass

            delta = await context.prepare_incremental(request, base_pack=first)
            expected = await context.prepare(request)

            assert isinstance(delta, ContextPackDelta)
            rebuilt = apply_context_pack_delta(
                ContextPack.from_bytes(first.to_bytes()), ContextPackDelta.from_bytes(delta.to_bytes())
            )
            assert b"".join(rebuilt.items).decode("utf-8") == expected.content

    asyncio.run(scenario())


def test_prepare_incremental_rejects_a_non_pack_base() -> None:
    async def scenario() -> None:
        async with open_builtin_runtime(BuiltinConfig(), candidate_pipeline=_CandidatePipeline()) as runtime:
            scope = await runtime.scopes.create(
                ScopeDraft(title="Reject", summary="Reject", idempotency_key="context-pack-reject-runtime")
            )
            context = runtime.context.for_scope(scope.scope_id)
            request = PrepareContextRequest(query="remote", max_bytes=4096)

            with pytest.raises(ContextPackError):
                await context.prepare_incremental(request, base_pack={"items": []})  # type: ignore[arg-type]

    asyncio.run(scenario())


def test_prepare_incremental_keeps_plain_prepare_unchanged() -> None:
    async def scenario() -> None:
        async with open_builtin_runtime(BuiltinConfig(), candidate_pipeline=_CandidatePipeline()) as runtime:
            scope = await runtime.scopes.create(
                ScopeDraft(title="Compat", summary="Compat", idempotency_key="context-pack-compat-runtime")
            )
            sources = runtime.sources.for_scope(scope.scope_id)
            memory = runtime.memory.for_scope(scope.scope_id)
            context = runtime.context.for_scope(scope.scope_id)
            request = PrepareContextRequest(query="remote", max_bytes=4096)

            await sources.capture(CaptureSource(source_id="turn-1", content=_MEETING_TURNS[0], metadata={}))
            while (await memory.flush()).processed:
                pass

            before = await context.prepare(request)
            await context.prepare_incremental(request)
            after = await context.prepare(request)

            assert before == after
            assert before.content is not None
            assert before.content_bytes == len(before.content.encode("utf-8"))

    asyncio.run(scenario())
