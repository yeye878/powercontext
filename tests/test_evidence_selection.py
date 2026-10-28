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

"""Behavior of query-driven Memory evidence selection in prepared context."""

from __future__ import annotations

import json
from typing import TypedDict, cast

import pytest
from pydantic import ValidationError

from powercontext.artifacts import ArtifactAddress, ArtifactRef
from powercontext.builtin.artifacts.memory import MemoryCitation, MemoryHit
from powercontext.builtin.runtime import ContextAssembly, PrepareContextRequest
from powercontext.builtin.runtime.errors import PreparedContextInvariantError
from powercontext.builtin.runtime.prepared_context import (
    MemoryEntryAddress,
    MemoryFragment,
    PreparedContextBuild,
    PreparedContextBuilder,
    PreparedMemoryCandidates,
)

MEMORY_REF = ArtifactRef(family="memory", artifact_id="memory", revision=3)
OTHER_MEMORY_REF = ArtifactRef(family="memory", artifact_id="memory", revision=4)
SCOPE = "project:alpha"
PREFIX = "General administrative notes. " * 60


class _Item(TypedDict):
    citation: dict[str, object]
    content: str
    start: int
    end: int


def _hit(entry_id: str, text: str, *, memory_ref: ArtifactRef = MEMORY_REF) -> MemoryHit:
    return MemoryHit(
        memory_ref=memory_ref,
        entry_id=entry_id,
        entry_version_id=f"{entry_id}-v1",
        text=text,
        score=1.0,
        matched_by=("fts",),
    )


def _candidates(*hits: MemoryHit, scope_id: str = SCOPE, memory_ref: ArtifactRef = MEMORY_REF) -> tuple:
    return (PreparedMemoryCandidates(scope_id=scope_id, memory_ref=memory_ref, hits=hits),)


def _build(
    *,
    query: str,
    texts: tuple[str, ...],
    max_bytes: int = 1800,
    scope_id: str = SCOPE,
    memory_ref: ArtifactRef = MEMORY_REF,
) -> PreparedContextBuild:
    hits = tuple(_hit(f"entry-{index}", text, memory_ref=memory_ref) for index, text in enumerate(texts))
    return PreparedContextBuilder().build_scopes_result(
        request=PrepareContextRequest(query=query, max_bytes=max_bytes, evidence_selection=True),
        current_scope_id=scope_id,
        memory_candidates=_candidates(*hits, scope_id=scope_id, memory_ref=memory_ref),
    )


def _items(build: PreparedContextBuild) -> list[_Item]:
    assert build.context.content is not None
    return cast(list[_Item], json.loads(build.context.content.splitlines()[-2])["items"])


def test_request_flag_defaults_off_and_accepts_only_json_booleans() -> None:
    assert PrepareContextRequest(query="evidence").evidence_selection is False
    assert PrepareContextRequest(query="evidence", evidence_selection=True).evidence_selection is True
    for value in ("true", 1, None):
        with pytest.raises(ValidationError):
            PrepareContextRequest.model_validate({"query": "evidence", "evidence_selection": value})


def test_evidence_selection_is_absent_from_the_ordinary_prepare_output() -> None:
    texts = ("The cobalt sensor is calibrated at 37 degrees.",)
    build = _build(query="cobalt sensor", texts=texts)
    assert build.fragments
    ordinary = PreparedContextBuilder().build_scopes_result(
        request=PrepareContextRequest(query="cobalt sensor", max_bytes=1800),
        current_scope_id=SCOPE,
        memory_candidates=_candidates(_hit("entry-0", texts[0])),
    )
    assert ordinary.fragments == ()
    assert ordinary.context.content is not None
    assert "BEGIN_POWERCONTEXT_PREPARED_CONTEXT_V1" in ordinary.context.content
    assert "cobalt sensor" in ordinary.context.content


def test_fragments_are_exact_slices_of_the_original_entry_text() -> None:
    texts = (
        PREFIX + "The cobalt sensor is calibrated at 37 degrees.",
        PREFIX + "The amber valve opens at 19 degrees.",
    )
    build = _build(query="cobalt sensor calibrated", texts=texts)
    assert build.fragments
    assert len(build.fragments) == len(build.origins) == len(_items(build))
    by_entry = {f"entry-{index}": text for index, text in enumerate(texts)}
    for fragment in build.fragments:
        assert isinstance(fragment, MemoryFragment)
        assert isinstance(fragment.address, MemoryEntryAddress)
        assert type(fragment.start) is int and type(fragment.end) is int
        assert 0 <= fragment.start < fragment.end <= len(by_entry[fragment.address.entry_id])
        original = by_entry[fragment.address.entry_id]
        assert fragment.text == original[fragment.start : fragment.end]
        assert fragment.address.memory.scope_id == SCOPE
        assert fragment.address.memory.artifact == MEMORY_REF
        assert fragment.address.entry_version_id == f"{fragment.address.entry_id}-v1"


def test_query_change_moves_the_selection_to_the_matching_late_evidence() -> None:
    texts = (
        PREFIX + "The cobalt sensor is calibrated at 37 degrees.",
        PREFIX + "The amber valve opens at 19 degrees.",
    )
    cobalt = _build(query="cobalt sensor calibrated", texts=texts)
    amber = _build(query="amber valve opens", texts=texts)
    assert any("37" in fragment.text for fragment in cobalt.fragments)
    assert not any("19" in fragment.text for fragment in cobalt.fragments)
    assert any("19" in fragment.text for fragment in amber.fragments)
    assert not any("37" in fragment.text for fragment in amber.fragments)


def test_rendered_items_carry_the_exact_citation_content_and_range() -> None:
    texts = (PREFIX + "The cobalt sensor is calibrated at 37 degrees.",)
    build = _build(query="cobalt sensor calibrated", texts=texts)
    items = _items(build)
    assert len(items) == len(build.fragments)
    for item, fragment, origin in zip(items, build.fragments, build.origins, strict=True):
        assert origin == fragment.address
        assert item["content"] == fragment.text
        assert item["start"] == fragment.start
        assert item["end"] == fragment.end
        assert item["citation"] == {
            "scope_id": SCOPE,
            "memory_ref": MEMORY_REF.model_dump(mode="json"),
            "entry_id": fragment.address.entry_id,
            "entry_version_id": fragment.address.entry_version_id,
        }


def test_content_bytes_charges_the_complete_envelope_including_citations() -> None:
    texts = tuple(PREFIX + f"Sensor {index} reports {index} degrees." for index in range(3))
    build = _build(query="sensor reports degrees", texts=texts, max_bytes=1400)
    content = build.context.content
    assert content is not None
    assert build.context.content_bytes == len(content.encode("utf-8"))
    assert build.context.content_bytes <= 1400
    envelope = json.loads(content.splitlines()[-2])
    assert envelope["trust"] == "untrusted_history"
    assert content.startswith("PowerContext prepared untrusted historical context.")
    assert "BEGIN_POWERCONTEXT_PREPARED_CONTEXT_V1" in content
    assert content.endswith("END_POWERCONTEXT_PREPARED_CONTEXT_V1")
    # The citation and formatting overhead are charged: the delivered body is smaller
    # than the whole budget by at least the fixed envelope and per-item citation size.
    body = sum(len(fragment.text.encode("utf-8")) for fragment in build.fragments)
    assert body < build.context.content_bytes


def test_unicode_text_is_charged_by_utf8_bytes_not_characters() -> None:
    texts = ("海洋观测 station records temperatures. " * 20,)
    build = _build(query="海洋观测 station", texts=texts, max_bytes=1200)
    content = build.context.content
    assert content is not None
    assert build.fragments
    assert build.context.content_bytes == len(content.encode("utf-8")) <= 1200
    for fragment in build.fragments:
        assert fragment.text == texts[0][fragment.start : fragment.end]
        assert len(fragment.text.encode("utf-8")) >= len(fragment.text)


def test_budget_below_the_fixed_envelope_returns_a_legal_empty_context() -> None:
    texts = (PREFIX + "The cobalt sensor is calibrated at 37 degrees.",)
    build = _build(query="cobalt sensor", texts=texts, max_bytes=512)
    assert build.context.status in {"ready", "empty"}
    if build.context.status == "empty":
        assert build.context.content is None
        assert build.context.content_bytes == 0
        assert build.fragments == ()
        assert build.origins == ()
    else:
        assert build.context.content is not None
        assert build.context.content_bytes <= 512


def test_empty_candidate_set_returns_an_empty_context() -> None:
    build = PreparedContextBuilder().build_scopes_result(
        request=PrepareContextRequest(query="anything", evidence_selection=True),
        current_scope_id=SCOPE,
        memory_candidates=(),
    )
    assert build.context.status == "empty"
    assert build.context.content is None
    assert build.context.content_bytes == 0
    assert build.fragments == ()
    assert build.origins == ()


def test_one_entry_can_contribute_several_disjoint_fragments() -> None:
    text = (
        "Introduction paragraph about unrelated administrative matters. "
        + "The cobalt sensor is calibrated at 37 degrees. " * 6
        + "Filler discussion of unrelated administrative matters. " * 20
        + "The amber valve opens at 19 degrees. "
    )
    build = _build(query="cobalt sensor calibrated amber valve opens", texts=(text,), max_bytes=3000)
    assert build.fragments
    spans = sorted((fragment.start, fragment.end) for fragment in build.fragments)
    for index in range(1, len(spans)):
        assert spans[index - 1][1] <= spans[index][0]
    assert all(fragment.text == text[fragment.start : fragment.end] for fragment in build.fragments)


def test_same_body_in_two_scopes_keeps_two_distinct_identities() -> None:
    text = "The cobalt sensor is calibrated at 37 degrees."
    builder = PreparedContextBuilder()
    build = builder.build_scopes_result(
        request=PrepareContextRequest(query="cobalt sensor", max_bytes=2400, evidence_selection=True),
        current_scope_id=SCOPE,
        memory_candidates=(
            PreparedMemoryCandidates(scope_id=SCOPE, memory_ref=MEMORY_REF, hits=(_hit("shared", text),)),
            PreparedMemoryCandidates(
                scope_id="project:beta",
                memory_ref=MEMORY_REF,
                hits=(_hit("shared", text),),
            ),
        ),
    )
    assert len(build.fragments) == 2
    scopes = [fragment.address.memory.scope_id for fragment in build.fragments]
    assert sorted(scopes) == ["project:alpha", "project:beta"]
    assert all(fragment.text == text[fragment.start : fragment.end] for fragment in build.fragments)
    for item, fragment in zip(_items(build), build.fragments, strict=True):
        assert item["citation"]["scope_id"] == fragment.address.memory.scope_id


def test_memory_candidates_without_a_scope_are_rejected() -> None:
    with pytest.raises(PreparedContextInvariantError):
        PreparedContextBuilder().build_scopes_result(
            request=PrepareContextRequest(query="cobalt sensor", evidence_selection=True),
            current_scope_id=SCOPE,
            memory_candidates=(
                PreparedMemoryCandidates(scope_id="", memory_ref=MEMORY_REF, hits=(_hit("entry-0", "Cobalt sensor."),)),
            ),
        )


def test_memory_ref_mismatch_is_rejected_before_selection() -> None:
    with pytest.raises(PreparedContextInvariantError):
        PreparedContextBuilder().build_scopes_result(
            request=PrepareContextRequest(query="cobalt sensor", evidence_selection=True),
            current_scope_id=SCOPE,
            memory_candidates=(
                PreparedMemoryCandidates(
                    scope_id=SCOPE,
                    memory_ref=OTHER_MEMORY_REF,
                    hits=(_hit("entry-0", "Cobalt sensor."),),
                ),
            ),
        )


def test_evidence_selection_ignores_code_topic_and_experience_candidates() -> None:
    text = "The cobalt sensor is calibrated at 37 degrees."
    build = PreparedContextBuilder().build_scopes_result(
        request=PrepareContextRequest(query="cobalt sensor", max_bytes=2400, evidence_selection=True),
        current_scope_id=SCOPE,
        memory_candidates=_candidates(_hit("entry-0", text)),
    )
    assert build.context.content is not None
    assert build.context.content.count("BEGIN_POWERCONTEXT_PREPARED_CONTEXT_V1") == 1
    assert "BEGIN_POWERCONTEXT_CODE_V1" not in build.context.content
    assert "BEGIN_POWERCONTEXT_PREPARED_TEXT_V1" not in build.context.content


def test_evidence_selection_takes_precedence_over_an_explicit_assembly() -> None:
    text = "The cobalt sensor is calibrated at 37 degrees."
    build = PreparedContextBuilder().build_scopes_result(
        request=PrepareContextRequest(
            query="cobalt sensor",
            max_bytes=2400,
            evidence_selection=True,
            assembly=ContextAssembly(),
        ),
        current_scope_id=SCOPE,
        memory_candidates=_candidates(_hit("entry-0", text)),
    )
    assert build.fragments
    assert build.context.content is not None
    assert "BEGIN_POWERCONTEXT_PREPARED_TEXT_V1" not in build.context.content


def test_selection_is_deterministic_for_the_same_inputs() -> None:
    texts = (
        PREFIX + "The cobalt sensor is calibrated at 37 degrees.",
        PREFIX + "The amber valve opens at 19 degrees.",
    )
    first = _build(query="cobalt sensor calibrated", texts=texts)
    second = _build(query="cobalt sensor calibrated", texts=texts)
    assert first == second


def test_identical_entries_are_not_delivered_twice() -> None:
    text = "The cobalt sensor is calibrated at 37 degrees."
    build = PreparedContextBuilder().build_scopes_result(
        request=PrepareContextRequest(query="cobalt sensor", max_bytes=2400, evidence_selection=True),
        current_scope_id=SCOPE,
        memory_candidates=_candidates(_hit("entry-0", text), _hit("entry-0", text)),
    )
    assert len(build.fragments) == 1


def test_fragments_stay_within_the_candidate_bodies_and_never_cross_entries() -> None:
    texts = (
        PREFIX + "The cobalt sensor is calibrated at 37 degrees.",
        PREFIX + "The amber valve opens at 19 degrees.",
        PREFIX + "The cobalt sensor is calibrated at 37 degrees.",
    )
    build = _build(query="cobalt sensor calibrated", texts=texts, max_bytes=1400)
    assert build.fragments
    for fragment in build.fragments:
        original = texts[int(fragment.address.entry_id.split("-")[1])]
        assert fragment.text == original[fragment.start : fragment.end]
        assert fragment.start >= 0
        assert fragment.end <= len(original)


def test_probe_budget_reports_the_same_selection_as_the_build() -> None:
    texts = (PREFIX + "The cobalt sensor is calibrated at 37 degrees.",)
    request = PrepareContextRequest(query="cobalt sensor", max_bytes=1400, evidence_selection=True)
    builder = PreparedContextBuilder()
    build = builder.build_scopes_result(
        request=request,
        current_scope_id=SCOPE,
        memory_candidates=_candidates(_hit("entry-0", texts[0])),
    )
    view = builder.probe_budget(
        request=request,
        current_scope_id=SCOPE,
        memory_candidates=_candidates(_hit("entry-0", texts[0])),
    )
    assert view.delivered_items == len(build.origins)
    assert view.max_bytes == 1400
    assert view.unused_bytes == 1400 - build.context.content_bytes
    assert view.dropped_items == 0


def test_citation_can_be_rebuilt_from_a_fragment_address() -> None:
    text = "The cobalt sensor is calibrated at 37 degrees."
    build = _build(query="cobalt sensor", texts=(text,))
    fragment = build.fragments[0]
    citation = MemoryCitation(
        memory_ref=fragment.address.memory.artifact,
        entry_id=fragment.address.entry_id,
        entry_version_id=fragment.address.entry_version_id,
    )
    assert citation.memory_ref == MEMORY_REF
    assert citation.entry_id == fragment.address.entry_id
    assert isinstance(fragment.address.memory, ArtifactAddress)
