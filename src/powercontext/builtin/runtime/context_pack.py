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

"""Deterministic Context Pack chunking and prefix-reuse delta transfer.

A Context Pack is the complete text a client would otherwise receive from
``prepare``, cut into ordered byte chunks and sealed with a SHA-256 digest of its
canonical wire form. A delta keeps the chunk prefix the receiver already holds and
carries only the target suffix, so a client that already owns the previous pack can
rebuild the new one from serialized bytes alone.

The wire format is an interoperability contract: field names, canonical JSON, standard
Base64 chunks, the digest input, and both byte counts are fixed here and nowhere else.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from powercontext.errors import PowerContextError

PACK_SCHEMA = "powercontext.context-pack.v1"
DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
DEFAULT_CHUNK_BYTES = 512

_DIGEST_PREFIX = "sha256:"
_DIGEST_HEX_LENGTH = 64
_BYTE_COUNT_PASSES = 8
_PACK_FIELDS = frozenset({"schema", "items", "digest", "byte_count"})
_DELTA_FIELDS = frozenset({"schema", "base_digest", "target_digest", "prefix_items", "appended"})


class ContextPackError(PowerContextError, ValueError):
    """Raised when a Context Pack, a delta, or a chunk boundary violates the wire contract."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"invalid context pack: {code}")


def _canonical(payload: dict[str, Any]) -> bytes:
    """Encode one wire object: UTF-8, sorted keys, no insignificant whitespace."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _encode_items(items: tuple[bytes, ...]) -> list[str]:
    return [base64.b64encode(item).decode("ascii") for item in items]


def _decode_items(value: object, *, field: str) -> tuple[bytes, ...]:
    if not isinstance(value, list):
        raise ContextPackError(f"{field}-not-an-array")
    decoded: list[bytes] = []
    for entry in value:
        if not isinstance(entry, str):
            raise ContextPackError(f"{field}-entry-not-text")
        try:
            decoded.append(base64.b64decode(entry, validate=True))
        except (binascii.Error, ValueError):
            raise ContextPackError(f"{field}-entry-not-base64") from None
    return tuple(decoded)


def _validate_items(items: Iterable[bytes]) -> tuple[bytes, ...]:
    if isinstance(items, (bytes, bytearray, memoryview, str)):
        raise ContextPackError("items-not-a-chunk-sequence")
    try:
        values = tuple(items)
    except TypeError:
        raise ContextPackError("items-not-iterable") from None
    for item in values:
        if type(item) is not bytes:
            raise ContextPackError("chunk-not-bytes")
    return values


def _validate_digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.startswith(_DIGEST_PREFIX):
        raise ContextPackError(f"{field}-not-a-sha256-digest")
    hex_digest = value[len(_DIGEST_PREFIX) :]
    if len(hex_digest) != _DIGEST_HEX_LENGTH or any(char not in "0123456789abcdef" for char in hex_digest):
        raise ContextPackError(f"{field}-not-a-sha256-digest")
    return value


def _validate_count(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ContextPackError(f"{field}-not-a-non-negative-integer")
    return value


def _decode_payload(data: object, *, fields: frozenset[str]) -> dict[str, Any]:
    if not isinstance(data, (bytes, bytearray)):
        raise ContextPackError("payload-not-bytes")
    try:
        payload = json.loads(bytes(data).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ContextPackError("payload-not-utf8-json") from None
    if not isinstance(payload, dict):
        raise ContextPackError("payload-not-an-object")
    if set(payload) != fields:
        raise ContextPackError("payload-fields-mismatch")
    return payload


def _chunk_text(text: str, limit: int) -> tuple[bytes, ...]:
    """Cut text into ordered UTF-8 chunks that each fit the byte ceiling.

    Line breaks are the preferred boundary and greedy character packing fills whatever a
    single line leaves over. Both passes are deterministic functions of the text, so the
    same leading characters always land in the same leading chunks. Prepared context bodies
    begin with a fixed preamble and only their item tail changes between turns, so keeping
    that preamble in its own leading chunk is what lets a later pack reuse it.
    """

    chunks: list[bytes] = []
    pending: list[str] = []
    pending_bytes = 0

    def flush() -> None:
        nonlocal pending, pending_bytes
        if pending:
            chunks.append("".join(pending).encode("utf-8"))
            pending = []
            pending_bytes = 0

    for line in text.splitlines(keepends=True):
        line_bytes = len(line.encode("utf-8"))
        if pending_bytes + line_bytes <= limit:
            pending.append(line)
            pending_bytes += line_bytes
            continue
        flush()
        if line_bytes <= limit:
            pending.append(line)
            pending_bytes = line_bytes
            continue
        for piece in _split_line(line, limit):
            piece_bytes = len(piece.encode("utf-8"))
            if pending_bytes + piece_bytes <= limit:
                pending.append(piece)
                pending_bytes += piece_bytes
            else:
                flush()
                pending.append(piece)
                pending_bytes = piece_bytes
    flush()
    return tuple(chunks)


def _split_line(line: str, limit: int) -> tuple[str, ...]:
    """Break one over-long line into the longest pieces that fit the ceiling."""

    pieces: list[str] = []
    pending = bytearray()
    for char in line:
        encoded = char.encode("utf-8")
        if len(encoded) > limit:
            raise ContextPackError("chunk-boundary-unreachable")
        if len(pending) + len(encoded) > limit:
            pieces.append(pending.decode("utf-8"))
            pending = bytearray()
        pending.extend(encoded)
    if pending:
        pieces.append(pending.decode("utf-8"))
    return tuple(pieces)


@dataclass(frozen=True)
class ContextPack:
    """One complete context payload as ordered chunks plus its wire identity.

    ``digest`` and ``byte_count`` are derived from the chunks and the schema rather than
    stored, so a pack can never carry a summary that disagrees with its own items.
    """

    items: tuple[bytes, ...] = ()
    schema: str = PACK_SCHEMA

    def __post_init__(self) -> None:
        # Direct construction is part of the public surface, so a pack is validated the same
        # way whether it came from build(), from_content(), or a caller assembling chunks.
        object.__setattr__(self, "items", _validate_items(self.items))
        if not isinstance(self.schema, str):
            raise ContextPackError("schema-not-text")

    @classmethod
    def build(cls, items: Iterable[bytes]) -> ContextPack:
        """Build a complete pack from an ordered chunk sequence."""

        return cls(items=_validate_items(items))

    @classmethod
    def from_content(cls, content: str, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> ContextPack:
        """Chunk one prepared context body deterministically within a byte ceiling.

        Every chunk is valid UTF-8 and at most ``chunk_bytes`` long, and joining the chunks
        reproduces the original body byte for byte. A ceiling that cannot hold even one
        character of the body raises ``ContextPackError``; empty content is a legal empty pack.
        """

        if not isinstance(content, str):
            raise ContextPackError("content-not-text")
        if type(chunk_bytes) is not int or chunk_bytes < 1:
            raise ContextPackError("chunk-bytes-not-a-positive-integer")
        return cls.build(_chunk_text(content, chunk_bytes))

    @property
    def digest(self) -> str:
        """SHA-256 over the canonical JSON of exactly the schema and the encoded items."""

        unsigned = {"schema": self.schema, "items": _encode_items(self.items)}
        return _DIGEST_PREFIX + hashlib.sha256(_canonical(unsigned)).hexdigest()

    @property
    def byte_count(self) -> int:
        """Serialized wire length, including this count field itself."""

        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        """Serialize the pack, resolving the byte count against its own encoded length."""

        encoded_items = _encode_items(self.items)
        digest = self.digest
        byte_count = 0
        # The count's own digits depend on the count, so iterate to the fixed point. It
        # converges in at most a couple of passes; the bound turns a hypothetical
        # non-convergence into a contract failure instead of a hang.
        for _ in range(_BYTE_COUNT_PASSES):
            raw = _canonical({
                "schema": self.schema,
                "items": encoded_items,
                "digest": digest,
                "byte_count": byte_count,
            })
            if len(raw) == byte_count:
                return raw
            byte_count = len(raw)
        raise ContextPackError("byte-count-not-convergent")

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPack:
        """Decode and fully verify one serialized pack."""

        payload = _decode_payload(data, fields=_PACK_FIELDS)
        if payload["schema"] != PACK_SCHEMA:
            raise ContextPackError("schema-mismatch")
        items = _decode_items(payload["items"], field="items")
        digest = _validate_digest(payload["digest"], field="digest")
        byte_count = _validate_count(payload["byte_count"], field="byte_count")
        pack = cls(items=items)
        if pack.digest != digest:
            raise ContextPackError("digest-mismatch")
        if pack.byte_count != byte_count:
            raise ContextPackError("byte-count-mismatch")
        return pack


@dataclass(frozen=True)
class ContextPackDelta:
    """The target suffix of one pack, expressed against a base pack's chunk prefix."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...] = ()
    schema: str = DELTA_SCHEMA

    def __post_init__(self) -> None:
        # Direct construction is public, so a delta is validated the same way whether it came
        # from between(), from_bytes(), or a caller assembling a suffix by hand.
        object.__setattr__(self, "appended", _validate_items(self.appended))
        _validate_digest(self.base_digest, field="base_digest")
        _validate_digest(self.target_digest, field="target_digest")
        _validate_count(self.prefix_items, field="prefix_items")
        if not isinstance(self.schema, str):
            raise ContextPackError("schema-not-text")

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        """Keep the longest chunk prefix the two packs share and carry the rest of the target."""

        if not isinstance(base, ContextPack) or not isinstance(target, ContextPack):
            raise ContextPackError("delta-operands-not-packs")
        prefix_items = 0
        for base_item, target_item in zip(base.items, target.items, strict=False):
            if base_item != target_item:
                break
            prefix_items += 1
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=prefix_items,
            appended=target.items[prefix_items:],
        )

    @property
    def byte_count(self) -> int:
        """Serialized wire length. A delta reports it without writing it as a field."""

        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        """Serialize the delta in canonical wire form."""

        return _canonical({
            "schema": self.schema,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": _encode_items(self.appended),
        })

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPackDelta:
        """Decode and verify one serialized delta."""

        payload = _decode_payload(data, fields=_DELTA_FIELDS)
        if payload["schema"] != DELTA_SCHEMA:
            raise ContextPackError("schema-mismatch")
        base_digest = _validate_digest(payload["base_digest"], field="base_digest")
        target_digest = _validate_digest(payload["target_digest"], field="target_digest")
        prefix_items = _validate_count(payload["prefix_items"], field="prefix_items")
        appended = _decode_items(payload["appended"], field="appended")
        return cls(
            base_digest=base_digest,
            target_digest=target_digest,
            prefix_items=prefix_items,
            appended=appended,
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    """Rebuild the target pack from a base pack and one delta, or raise."""

    if not isinstance(base, ContextPack) or not isinstance(delta, ContextPackDelta):
        raise ContextPackError("delta-operands-not-packs")
    if base.digest != delta.base_digest:
        raise ContextPackError("base-digest-mismatch")
    if not 0 <= delta.prefix_items <= len(base.items):
        raise ContextPackError("prefix-items-out-of-range")
    rebuilt = ContextPack.build((*base.items[: delta.prefix_items], *delta.appended))
    if rebuilt.digest != delta.target_digest:
        raise ContextPackError("target-digest-mismatch")
    return rebuilt


__all__ = [
    "DEFAULT_CHUNK_BYTES",
    "DELTA_SCHEMA",
    "PACK_SCHEMA",
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]
