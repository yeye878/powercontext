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

"""Capture the visible events of one selected Pi session branch.

A Pi session file is a parent-linked graph. The selected leaf and its ancestors
form the branch that was actually visible, and that path is what becomes
evidence. Control records keep the graph connected without producing text, and
summaries never discard the records that precede them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import quote

from pydantic import BaseModel, JsonValue, TypeAdapter

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.connectors import (
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
)
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

PI_EVENT_SOURCE_NAME = "pi-event"
PI_SESSION_CONNECTOR_NAME = "pi-session-history"
SOURCE_VERSION = "1"

_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])
_UTF8_BOM = "\ufeff"
_SUMMARY_TYPES = frozenset({"compaction", "branch_summary"})
_CONTROL_TYPES = frozenset({"thinking_level_change", "model_change", "session"})
_ROLE_SUMMARY = "role"


class PiSessionInput(BaseModel):
    """One complete Pi session archive and the branch leaf to capture."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible event on the selected branch, with its original record."""

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    """A captured Pi event that can be read back as its exact original record."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Return the visible events of the selected branch, ordered root to leaf."""

    records = _read_records(value.content)
    if not records:
        raise ValueError("Pi session archive must contain a session header")  # noqa: TRY003
    session_id = _read_header(records[0])
    nodes = _index_records(records[1:])
    leaf_id = _resolve_leaf(nodes, value.leaf_id)
    path = _ancestor_path(nodes, leaf_id)
    return tuple(event for node in path if (event := _to_event(session_id, node)) is not None)


def pi_event_source_id(session_id: str, event_id: str) -> str:
    """Return the stable, delimiter-safe identity of one captured event."""

    if not session_id.strip() or session_id != session_id.strip():
        raise ValueError("Pi session id must be a non-empty trimmed string")  # noqa: TRY003
    if not event_id.strip() or event_id != event_id.strip():
        raise ValueError("Pi event id must be a non-empty trimmed string")  # noqa: TRY003
    source_id = f"pi:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(  # noqa: TRY003
            f"Pi event source id exceeds {MAX_SOURCE_ID_LENGTH} characters"
        )
    return source_id


class PiEventSourceAdapter:
    """Materialize and read one captured Pi event."""

    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        source_id = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=source_id,
            materialization=SourceMaterialization.CAPTURED,
            source_id=source_id,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class PiEventTextEvidenceProjection:
    """Expose one captured event through the shared text-evidence capability."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


class PiSessionConnector:
    """Import the not-yet-captured visible events of one selected branch."""

    name = PI_SESSION_CONNECTOR_NAME
    version = SOURCE_VERSION
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self.path = Path(path)
        self.leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        events = parse_pi_session(PiSessionInput(content=self.path.read_bytes(), leaf_id=self.leaf_id))
        imported = _read_checkpoint(session.checkpoint)
        pending = tuple(event for event in events if event.event_id not in imported)
        for event in events:
            previous = imported.get(event.event_id)
            if previous is not None and previous != _event_digest(event):
                raise ValueError(  # noqa: TRY003
                    f"Pi event {event.event_id!r} changed after it was imported"
                )
        for event in pending:
            await session.submit(event.event_id, PI_EVENT_SOURCE_NAME, event)
        updated = dict(imported)
        for event in pending:
            updated[event.event_id] = _event_digest(event)
        session_id = events[0].session_id if events else ""
        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint=_write_checkpoint(session_id, updated),
        )


PI_EVENT_SOURCE_ADAPTER = PiEventSourceAdapter()
PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(
    PI_EVENT_SOURCE_ADAPTER,
    version=SOURCE_VERSION,
    projections=(PiEventTextEvidenceProjection(),),
)


def _read_records(content: bytes) -> list[dict[str, JsonValue]]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Pi session archive must be valid UTF-8") from error  # noqa: TRY003
    records: list[dict[str, JsonValue]] = []
    for line in text.removeprefix(_UTF8_BOM).splitlines():
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Pi session archive contains invalid JSON: {error}") from error  # noqa: TRY003
        if not isinstance(decoded, dict):
            # The public contract requires ValueError for every malformed archive.
            raise ValueError("Pi session records must be JSON objects")  # noqa: TRY003, TRY004
        records.append(_JSON_OBJECT.validate_python(decoded))
    return records


def _read_header(record: dict[str, JsonValue]) -> str:
    if record.get("type") != "session":
        raise ValueError("Pi session archive must start with a session header")  # noqa: TRY003
    if record.get("version") != 3:
        raise ValueError("Pi session header must declare version 3")  # noqa: TRY003
    session_id = record.get("id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("Pi session header must declare a non-empty id")  # noqa: TRY003
    return session_id


def _index_records(records: list[dict[str, JsonValue]]) -> dict[str, dict[str, JsonValue]]:
    nodes: dict[str, dict[str, JsonValue]] = {}
    for record in records:
        event_id = record.get("id")
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError("Pi session records must declare a non-empty id")  # noqa: TRY003
        if event_id in nodes:
            raise ValueError(f"duplicate Pi session record id {event_id!r}")  # noqa: TRY003
        nodes[event_id] = record
    for event_id, record in nodes.items():
        parent_id = record.get("parentId")
        if parent_id is not None and (not isinstance(parent_id, str) or parent_id not in nodes):
            raise ValueError(  # noqa: TRY003
                f"Pi session record {event_id!r} references a missing parent {parent_id!r}"
            )
    return nodes


def _resolve_leaf(nodes: dict[str, dict[str, JsonValue]], leaf_id: str | None) -> str:
    if leaf_id is None:
        if not nodes:
            raise ValueError("Pi session archive contains no records to select")  # noqa: TRY003
        return next(reversed(nodes))
    if leaf_id not in nodes:
        raise ValueError(f"Pi session leaf {leaf_id!r} does not exist")  # noqa: TRY003
    return leaf_id


def _ancestor_path(nodes: dict[str, dict[str, JsonValue]], leaf_id: str) -> list[dict[str, JsonValue]]:
    path: list[dict[str, JsonValue]] = []
    seen: set[str] = set()
    current: str | None = leaf_id
    while current is not None:
        if current in seen:
            raise ValueError("Pi session graph contains a cycle")  # noqa: TRY003
        seen.add(current)
        record = nodes[current]
        path.append(record)
        parent_id = record.get("parentId")
        current = parent_id if isinstance(parent_id, str) else None
    path.reverse()
    return path


def _to_event(session_id: str, record: dict[str, JsonValue]) -> PiEventInput | None:
    event_type = record.get("type")
    if not isinstance(event_type, str):
        return None
    if event_type == "message":
        rendered = _render_message(record.get("message"))
    elif event_type == "custom_message":
        rendered = _render_custom(record)
    elif event_type in _SUMMARY_TYPES:
        summary = record.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            return None
        rendered = (event_type, summary)
    else:
        # Control records keep the graph connected but never become evidence.
        return None
    if rendered is None:
        return None
    role, body = rendered
    if not body.strip():
        return None
    return PiEventInput(
        session_id=session_id,
        event_id=str(record["id"]),
        parent_id=record.get("parentId") if isinstance(record.get("parentId"), str) else None,
        event_type=event_type,
        role=role,
        content=f"{role}: {body}",
        record=record,
    )


def _render_message(message: JsonValue) -> tuple[str, str] | None:
    if not isinstance(message, dict):
        return None
    role = message.get("role")
    if not isinstance(role, str) or not role:
        return None
    if role == "bashExecution":
        return _render_bash(role, message)
    if role == "toolResult":
        return _render_tool_result(role, message)
    text = _render_content_text(message.get("content"))
    if text is None or not text.strip():
        return None
    return role, text


def _render_bash(role: str, message: dict[str, JsonValue]) -> tuple[str, str] | None:
    command = message.get("command")
    if not isinstance(command, str):
        return None
    output = message.get("output")
    if isinstance(output, str) and output:
        return role, f"$ {command}\n{output}"
    return role, f"$ {command}"


def _render_tool_result(role: str, message: dict[str, JsonValue]) -> tuple[str, str] | None:
    text = _render_content_text(message.get("content"))
    if text is None:
        return None
    tool_name = _text_or_unknown(message.get("toolName"))
    tool_call_id = _text_or_unknown(message.get("toolCallId"))
    return role, f"[tool_result {tool_name} {tool_call_id}] {text}"


def _render_content_text(content: JsonValue) -> str | None:
    """Render message content given either a plain string or a block list."""

    text = _render_blocks(content)
    return text if text is not None else _render_string(content)


def _render_custom(record: dict[str, JsonValue]) -> tuple[str, str] | None:
    text = _render_content_text(record.get("content"))
    if text is None:
        return None
    return "custom", text


def _render_blocks(content: JsonValue) -> str | None:
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif block_type == "toolCall":
            call = _render_tool_call(block)
            if call:
                parts.append(call)
    if not parts:
        return None
    return "\n".join(parts)


def _render_tool_calls(content: JsonValue) -> list[str]:
    if not isinstance(content, list):
        return []
    calls: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "toolCall":
            call = _render_tool_call(block)
            if call:
                calls.append(call)
    return calls


def _render_tool_call(block: dict[str, JsonValue]) -> str | None:
    name = block.get("name")
    call_id = block.get("id")
    if not isinstance(name, str) or not isinstance(call_id, str):
        return None
    arguments = block.get("arguments")
    rendered = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"[tool_call {name} {call_id}] {rendered}"


def _render_string(value: JsonValue) -> str | None:
    return value if isinstance(value, str) else None


def _text_or_unknown(value: JsonValue) -> str:
    return value if isinstance(value, str) and value else "unknown"


def _event_digest(event: PiEventInput) -> str:
    encoded = json.dumps(event.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    return f"sha256:{hashlib.sha256(encoded.encode()).hexdigest()}"


def _read_checkpoint(checkpoint: JsonValue | None) -> dict[str, str]:
    if checkpoint is None:
        return {}
    if not isinstance(checkpoint, dict):
        # Checkpoints are opaque JSON written by this module; a foreign value is
        # still reported through the same ValueError contract as bad archives.
        raise ValueError("Pi session checkpoint must be a JSON object")  # noqa: TRY003, TRY004
    events = checkpoint.get("events")
    if not isinstance(events, dict):
        return {}
    return {str(event_id): str(digest) for event_id, digest in events.items()}


def _write_checkpoint(session_id: str, events: dict[str, str]) -> dict[str, JsonValue]:
    return {"session_id": session_id, "events": dict(sorted(events.items()))}


__all__ = [
    "PI_EVENT_SOURCE_ADAPTER",
    "PI_EVENT_SOURCE_DEFINITION",
    "PI_EVENT_SOURCE_NAME",
    "PI_SESSION_CONNECTOR_NAME",
    "PiEventInput",
    "PiEventSource",
    "PiEventSourceAdapter",
    "PiEventTextEvidenceProjection",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]
