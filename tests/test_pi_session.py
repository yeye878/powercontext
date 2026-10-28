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

"""Contract edges for Pi session branch capture not covered by the public sample."""

from __future__ import annotations

import asyncio
import json

import pytest

from powercontext.sources import (
    TEXT_EVIDENCE_PROJECTION_KEY,
    SourceDefinitionRegistry,
    TextEvidence,
    manifest_for_definition,
    project_source_for_transport,
)
from powercontext.sources.pi_session import (
    PI_EVENT_SOURCE_DEFINITION,
    PiEventSource,
    PiSessionConnector,
    PiSessionInput,
    parse_pi_session,
    pi_event_source_id,
)

HEADER = {"type": "session", "version": 3, "id": "s-1"}


def _raw(records, header=HEADER) -> bytes:
    return "\n".join(json.dumps(record) for record in [header, *records]).encode()


def _parse(records, leaf=None, header=HEADER):
    return parse_pi_session(PiSessionInput(content=_raw(records, header), leaf_id=leaf))


def test_string_and_block_message_content_are_both_rendered() -> None:
    events = _parse([
        {"type": "message", "id": "a", "parentId": None, "message": {"role": "user", "content": "plain"}},
        {
            "type": "message",
            "id": "b",
            "parentId": "a",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}],
            },
        },
    ])
    assert [event.content for event in events] == ["user: plain", "assistant: one\ntwo"]


def test_thinking_and_image_blocks_produce_no_text_but_keep_chain() -> None:
    events = _parse([
        {"type": "message", "id": "a", "parentId": None, "message": {"role": "user", "content": "q"}},
        {
            "type": "message",
            "id": "b",
            "parentId": "a",
            "message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "x"}]},
        },
        {
            "type": "message",
            "id": "c",
            "parentId": "b",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
        },
    ])
    assert [event.event_id for event in events] == ["a", "c"]
    assert events[1].parent_id == "b"


def test_tool_call_arguments_are_sorted_and_compact_with_unicode() -> None:
    events = _parse([
        {
            "type": "message",
            "id": "a",
            "parentId": None,
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "name": "read",
                        "id": "c1",
                        "arguments": {"z": "文", "a": 1, "m": [1, 2]},
                    }
                ],
            },
        }
    ])
    assert events[0].content == 'assistant: [tool_call read c1] {"a":1,"m":[1,2],"z":"文"}'


def test_tool_result_missing_names_fall_back_to_unknown() -> None:
    events = _parse([
        {
            "type": "message",
            "id": "a",
            "parentId": None,
            "message": {"role": "toolResult", "content": [{"type": "text", "text": "out"}]},
        }
    ])
    assert events[0].content == "toolResult: [tool_result unknown unknown] out"


def test_bash_execution_without_output_keeps_only_the_command() -> None:
    events = _parse([
        {
            "type": "message",
            "id": "a",
            "parentId": None,
            "message": {"role": "bashExecution", "command": "ls", "output": "", "content": []},
        }
    ])
    assert events[0].content == "bashExecution: $ ls"


def test_summaries_do_not_remove_earlier_evidence() -> None:
    events = _parse([
        {"type": "message", "id": "a", "parentId": None, "message": {"role": "user", "content": "before"}},
        {"type": "compaction", "id": "b", "parentId": "a", "summary": "compacted"},
        {"type": "message", "id": "c", "parentId": "b", "message": {"role": "assistant", "content": "after"}},
    ])
    assert [event.event_id for event in events] == ["a", "b", "c"]
    assert events[1].role == "compaction"


def test_default_leaf_is_the_last_record() -> None:
    records = [
        {"type": "message", "id": "a", "parentId": None, "message": {"role": "user", "content": "one"}},
        {"type": "message", "id": "b", "parentId": "a", "message": {"role": "assistant", "content": "two"}},
    ]
    assert [event.event_id for event in _parse(records)] == ["a", "b"]


def test_blank_lines_and_bom_are_tolerated() -> None:
    raw = b"\xef\xbb\xbf" + _raw([
        {"type": "message", "id": "a", "parentId": None, "message": {"role": "user", "content": "x"}},
    ]).replace(b"\n", b"\n\n")
    events = parse_pi_session(PiSessionInput(content=raw))
    assert [event.event_id for event in events] == ["a"]


@pytest.mark.parametrize(
    "records,leaf",
    [
        ([{"type": "message", "id": "a", "parentId": "gone"}], "a"),
        ([{"type": "message", "id": "a", "parentId": "a"}], "a"),
        ([{"type": "message", "id": "a", "parentId": None}] * 2, "a"),
        ([{"type": "message", "id": "a", "parentId": None}], "nope"),
        ([{"type": "message", "id": "", "parentId": None}], None),
    ],
)
def test_invalid_graphs_fail(records, leaf) -> None:
    with pytest.raises(ValueError):
        _parse(records, leaf)


def test_cycle_is_detected() -> None:
    records = [
        {"type": "message", "id": "a", "parentId": "b", "message": {"role": "user", "content": "x"}},
        {"type": "message", "id": "b", "parentId": "a", "message": {"role": "user", "content": "y"}},
    ]
    with pytest.raises(ValueError):
        _parse(records, "a")


@pytest.mark.parametrize(
    "header",
    [
        {"type": "session", "version": 2, "id": "s"},
        {"type": "session", "version": 3, "id": ""},
        {"type": "other", "version": 3, "id": "s"},
    ],
)
def test_invalid_headers_fail(header) -> None:
    with pytest.raises(ValueError):
        _parse([], header=header)


@pytest.mark.parametrize("raw", [b"", b"\xff", b"{}\n", b"[]\n", b'{"type":"session","version":3}\n'])
def test_invalid_archives_fail(raw) -> None:
    with pytest.raises(ValueError):
        parse_pi_session(PiSessionInput(content=raw))


def test_identity_is_stable_delimited_and_bounded() -> None:
    assert pi_event_source_id("s:1", "e/1") == "pi:s%3A1:e%2F1"
    assert pi_event_source_id("s:1", "e/1") != pi_event_source_id("s", "1:e/1")
    assert pi_event_source_id("s", "文") == "pi:s:%E6%96%87"
    # len("pi:" + "s" + ":" + payload) == 5 + payload
    assert len(pi_event_source_id("s", "x" * 251)) == 256
    with pytest.raises(ValueError):
        pi_event_source_id("s", "x" * 252)
    with pytest.raises(ValueError):
        pi_event_source_id(" s ", "e")


def test_source_projection_and_readback() -> None:
    registry = SourceDefinitionRegistry((PI_EVENT_SOURCE_DEFINITION,))
    event = _parse([
        {"type": "message", "id": "a", "parentId": None, "message": {"role": "user", "content": "hi"}},
    ])[0]
    source = asyncio.run(registry.resolve(event))

    assert isinstance(source, PiEventSource)
    assert source.materialization.value == "captured"
    assert source.source_id == source.name == "pi:s-1:a"
    assert asyncio.run(registry.read(source)) == event
    assert PI_EVENT_SOURCE_DEFINITION.name == "pi-event"
    assert PI_EVENT_SOURCE_DEFINITION.version == "1"

    evidence = TextEvidence.model_validate(registry.project(source, TEXT_EVIDENCE_PROJECTION_KEY))
    assert evidence.source_type == "pi-event"
    assert evidence.source_id == "pi:s-1:a"
    assert evidence.content == "user: hi"
    assert set(evidence.metadata) == {"session_id", "event_id", "parent_id", "event_type", "role"}


def test_full_record_is_preserved_in_transport() -> None:
    registry = SourceDefinitionRegistry((PI_EVENT_SOURCE_DEFINITION,))
    records = [
        {"type": "message", "id": "a", "parentId": None, "message": {"role": "user", "content": "hi"}},
    ]
    event = _parse(records)[0]
    source = asyncio.run(registry.resolve(event))
    observation = project_source_for_transport(registry, source)

    assert observation.payload["record"] == records[0]
    assert observation.definition_fingerprint == manifest_for_definition(PI_EVENT_SOURCE_DEFINITION).fingerprint


def test_large_tool_result_is_not_truncated_in_source(tmp_path) -> None:
    big = "x" * 200_000
    events = _parse([
        {
            "type": "message",
            "id": "a",
            "parentId": None,
            "message": {
                "role": "toolResult",
                "toolName": "read",
                "toolCallId": "c1",
                "content": [{"type": "text", "text": big}],
            },
        }
    ])
    assert len(events[0].content) > 200_000
    assert big in events[0].content


def test_connector_declares_expected_identity(tmp_path) -> None:
    connector = PiSessionConnector(tmp_path / "unused.jsonl")
    assert connector.name == "pi-session-history"
    assert connector.version == "1"
    assert connector.source_definitions == frozenset({"pi-event"})
