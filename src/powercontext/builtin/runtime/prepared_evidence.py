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

"""Select query-relevant literal Memory spans inside one UTF-8 byte budget.

The selection is pure: it reads only the candidate entry bodies and the request
query, never the Runtime, a database, or a model. Every emitted span is a
half-open character range of one candidate body, so a caller can verify it by
slicing the stored Memory entry text.

The unit of evidence is a contiguous run of whole sentences, not an isolated
sentence. A claim's supporting evidence in a scientific abstract is a short
passage, and a passage is also what the byte budget rewards: consecutive
sentences share one citation, while every separate item pays for its own. The
selector scores sentence windows by their query overlap and picks the window
that buys the most overlap per charged byte, then spends whatever remains on
extending the runs it already holds.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise

_TERM = re.compile(r"[a-z]+|\d+")
_CLOSING_PUNCTUATION = frozenset("\"')]}\u201d\u2019\u00bb")
_ABBREVIATIONS = frozenset({
    "al",
    "approx",
    "ca",
    "cf",
    "dr",
    "e.g",
    "eq",
    "etc",
    "fig",
    "figs",
    "i.e",
    "no",
    "nos",
    "ref",
    "refs",
    "st",
    "vs",
})
# A sentence longer than this is split at a space so a small budget can still
# hold a unit; one enormous run is not a useful evidence passage.
_MAX_SENTENCE_CHARS = 420
_SENTENCE_CHUNK_CHARS = 280
# How strongly a window's value decays with its charged size. A shallow exponent
# keeps long, dense passages competitive with short lucky ones.
_DENSITY = 0.28
# A window in the candidate body that matches the query best is worth slightly
# more: a claim's evidence almost always lives in the abstract it cites.
_DOCUMENT_BONUS = 1.0
# Do not spend the last bytes on a passage too short to be honest evidence.
_MIN_FILL_BYTES = 24
_PHRASE_BONUS = 0.5


@dataclass(frozen=True)
class EvidenceSpan:
    """One selected half-open character range of one candidate body."""

    candidate: int
    start: int
    end: int
    score: float


@dataclass
class _Sentence:
    """One indivisible sentence of one candidate body."""

    start: int
    end: int


@dataclass(frozen=True)
class _Run:
    """One selected contiguous span and the exact size of its rendered item."""

    candidate: int
    start: int
    end: int
    value: float
    size: int


@dataclass
class _Selection:
    """Mutable selection state: the non-overlapping runs kept per candidate."""

    runs: dict[int, list[_Run]] = field(default_factory=dict)

    def ordered(self) -> list[_Run]:
        return [run for candidate in sorted(self.runs) for run in self.runs[candidate]]

    def count(self) -> int:
        return sum(len(runs) for runs in self.runs.values())

    def total(self, fixed_bytes: int) -> int:
        runs = self.ordered()
        if not runs:
            return fixed_bytes
        return fixed_bytes + sum(run.size for run in runs) + len(runs) - 1

    def with_run(self, run: _Run) -> _Selection:
        """Return a copy holding `run`, merged with any run it touches."""

        copy = _Selection({candidate: list(runs) for candidate, runs in self.runs.items()})
        copy.runs[run.candidate] = _merge(copy.runs.get(run.candidate, ()), run)
        return copy


def evidence_item(citation: dict[str, object], content: str, start: int, end: int) -> dict[str, object]:
    """Build one rendered envelope item; the only place its shape is defined."""

    return {"citation": citation, "content": content, "start": start, "end": end}


def item_bytes(citation: dict[str, object], content: str, start: int, end: int) -> int:
    """Return the exact UTF-8 size of one rendered item."""

    rendered = json.dumps(evidence_item(citation, content, start, end), ensure_ascii=False, separators=(",", ":"))
    return len(rendered.encode("utf-8"))


def rendered_bytes(fixed_bytes: int, items: Sequence[int]) -> int:
    """Return the exact rendered envelope size from its item sizes.

    `fixed_bytes` is the envelope with no items and consecutive items are joined
    by the single `,` of the JSON array, mirroring the renderer exactly.
    """

    if not items:
        return fixed_bytes
    return fixed_bytes + sum(items) + len(items) - 1


def select_evidence(
    *,
    query: str,
    texts: Sequence[str],
    citations: Sequence[dict[str, object]],
    max_bytes: int,
    fixed_bytes: int,
    max_items: int = 12,
) -> tuple[EvidenceSpan, ...]:
    """Return the literal spans of `texts` worth delivering for `query`.

    `fixed_bytes` is the exact rendered size of the envelope with no items and
    `citations[i]` is the citation rendered for candidate `i`. Both come from the
    renderer, so the budget is charged for the complete output rather than for
    the excerpt bodies alone.
    """

    if max_bytes <= fixed_bytes or not texts or max_items < 1:
        return ()
    sentences = [_sentences(text) for text in texts]
    if not any(sentences):
        return ()
    scores, document_scores = _score(query, texts, sentences)
    best_document = max(range(len(texts)), key=lambda index: (document_scores[index], -index))
    selection = _Selection()
    while selection.count() < max_items:
        window = _best_window(selection, sentences, scores, texts, citations, max_bytes, fixed_bytes, best_document)
        if window is None:
            break
        selection = selection.with_run(window)
    _extend_runs(selection, sentences, scores, texts, citations, max_bytes, fixed_bytes)
    _extend_partial(selection, sentences, scores, texts, citations, max_bytes, fixed_bytes)
    return _spans(selection)


def _spans(selection: _Selection) -> tuple[EvidenceSpan, ...]:
    spans = [
        EvidenceSpan(candidate=run.candidate, start=run.start, end=run.end, score=run.value)
        for run in selection.ordered()
    ]
    return tuple(sorted(spans, key=lambda span: (-span.score, span.candidate, span.start)))


def _best_window(
    selection: _Selection,
    sentences: Sequence[Sequence[_Sentence]],
    scores: Sequence[Sequence[float]],
    texts: Sequence[str],
    citations: Sequence[dict[str, object]],
    max_bytes: int,
    fixed_bytes: int,
    best_document: int,
) -> _Run | None:
    """Return the window that buys the most query overlap per charged byte."""

    used = selection.total(fixed_bytes)
    best: tuple[float, _Run] | None = None
    for candidate, units in enumerate(sentences):
        if not units:
            continue
        bonus = _DOCUMENT_BONUS if candidate == best_document else 0.0
        for first in range(len(units)):
            if _covered(selection.runs.get(candidate, ()), units[first].start, units[first].end):
                continue
            for last in range(first, len(units)):
                start, end = units[first].start, units[last].end
                if _overlaps(selection.runs.get(candidate, ()), start, end):
                    break
                run = _make_run(
                    candidate, start, end, bonus + sum(scores[candidate][first : last + 1]), texts, citations
                )
                cost = selection.with_run(run).total(fixed_bytes) - used
                if cost <= 0 or used + cost > max_bytes:
                    break
                ratio = run.value / (cost**_DENSITY)
                if best is None or ratio > best[0]:
                    best = (ratio, run)
    if best is None or best[1].value <= 0.0:
        # A window with no query overlap is not evidence: the remaining bytes are better
        # spent extending a run that does match, which the caller does next.
        return None
    return best[1]


def _extend_runs(
    selection: _Selection,
    sentences: Sequence[Sequence[_Sentence]],
    scores: Sequence[Sequence[float]],
    texts: Sequence[str],
    citations: Sequence[dict[str, object]],
    max_bytes: int,
    fixed_bytes: int,
) -> None:
    """Spend leftover bytes on sentences adjacent to an already selected run.

    Growing a run costs only its own text, so it is the cheapest way to deliver
    more literal evidence, and the sentences around a matching passage usually
    belong to the same piece of evidence.
    """

    while True:
        used = selection.total(fixed_bytes)
        remaining = max_bytes - used
        if remaining <= 0:
            return
        best: tuple[float, _Run, _Run] | None = None
        for run in selection.ordered():
            for neighbour in _neighbours(sentences[run.candidate], run, scores[run.candidate]):
                merged = _make_run(
                    run.candidate,
                    min(run.start, neighbour.start),
                    max(run.end, neighbour.end),
                    run.value + neighbour.value,
                    texts,
                    citations,
                )
                cost = selection.with_run(merged).total(fixed_bytes) - used
                if cost <= 0 or cost > remaining:
                    continue
                ratio = neighbour.value / (cost**_DENSITY)
                if best is None or ratio > best[0]:
                    best = (ratio, run, merged)
        if best is None:
            return
        _, run, merged = best
        selection.runs[merged.candidate] = _replace_run(selection.runs.get(merged.candidate, ()), run, merged)


def _replace_run(runs: Sequence[_Run], run: _Run, merged: _Run) -> list[_Run]:
    """Install `merged` in place of the run it grew from, keeping the list ordered."""

    kept = [item for item in runs if item != run]
    kept.append(merged)
    return sorted(kept, key=lambda item: item.start)


def _extend_partial(
    selection: _Selection,
    sentences: Sequence[Sequence[_Sentence]],
    scores: Sequence[Sequence[float]],
    texts: Sequence[str],
    citations: Sequence[dict[str, object]],
    max_bytes: int,
    fixed_bytes: int,
) -> None:
    """Grow the best run by the longest word-aligned prefix that still fits.

    A prefix is still exact original text, and delivering it keeps the envelope
    inside the budget without inventing evidence.
    """

    used = selection.total(fixed_bytes)
    remaining = max_bytes - used
    if remaining < _MIN_FILL_BYTES:
        return
    best: tuple[float, _Run] | None = None
    for run in selection.ordered():
        text = texts[run.candidate]
        for neighbour in _neighbours(sentences[run.candidate], run, scores[run.candidate]):
            low, high = 1, neighbour.end - neighbour.start - 1
            while low <= high:
                middle = (low + high) // 2
                end = _word_aligned_end(text, neighbour.start, neighbour.start + middle)
                if end is None:
                    high = middle - 1
                    continue
                merged = _make_run(
                    run.candidate,
                    min(run.start, neighbour.start),
                    max(run.end, end),
                    run.value + neighbour.value,
                    texts,
                    citations,
                )
                cost = selection.with_run(merged).total(fixed_bytes) - used
                if cost <= remaining:
                    if best is None or neighbour.value > best[0]:
                        best = (neighbour.value, merged)
                    low = middle + 1
                else:
                    high = middle - 1
    if best is None:
        return
    merged = best[1]
    if not texts[merged.candidate][merged.start : merged.end].strip():
        return
    selection.runs[merged.candidate] = _replace_covered(selection.runs.get(merged.candidate, ()), merged)


def _replace_covered(runs: Sequence[_Run], merged: _Run) -> list[_Run]:
    """Replace every run that `merged` grew from by `merged` itself."""

    kept = [run for run in runs if run.end <= merged.start or merged.end <= run.start]
    kept.append(merged)
    return sorted(kept, key=lambda run: run.start)


def _neighbours(units: Sequence[_Sentence], run: _Run, scores: Sequence[float]) -> list[_Run]:
    """Return the whole sentences immediately before and after one run."""

    neighbours: list[_Run] = []
    for index, unit in enumerate(units):
        if unit.end == run.start or unit.start == run.end:
            neighbours.append(_Run(run.candidate, unit.start, unit.end, scores[index], 0))
    return neighbours


def _make_run(
    candidate: int,
    start: int,
    end: int,
    value: float,
    texts: Sequence[str],
    citations: Sequence[dict[str, object]],
) -> _Run:
    text = texts[candidate]
    return _Run(candidate, start, end, value, item_bytes(citations[candidate], text[start:end], start, end))


def _merge(runs: Sequence[_Run], run: _Run) -> list[_Run]:
    """Merge one run into a candidate's runs, joining adjacent or overlapping ones."""

    merged: list[_Run] = []
    pending = run
    for existing in sorted(runs, key=lambda item: item.start):
        if existing.end < pending.start or pending.end < existing.start:
            if pending.start < existing.start:
                merged.append(pending)
                pending = existing
            else:
                merged.append(existing)
            continue
        pending = _Run(
            existing.candidate,
            min(existing.start, pending.start),
            max(existing.end, pending.end),
            existing.value + pending.value,
            existing.size + pending.size,
        )
    merged.append(pending)
    return merged


def _covered(runs: Sequence[_Run], start: int, end: int) -> bool:
    return any(run.start <= start and end <= run.end for run in runs)


def _overlaps(runs: Sequence[_Run], start: int, end: int) -> bool:
    return any(not (run.end < start or end < run.start) for run in runs)


def _sentences(text: str) -> list[_Sentence]:
    spans: list[_Sentence] = []
    for start, end in _sentence_spans(text):
        for chunk_start, chunk_end in _chunk_span(text, start, end):
            spans.append(_Sentence(chunk_start, chunk_end))
    return spans


def _chunk_span(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Split an over-long sentence so a small budget can still hold a unit."""

    if end - start <= _MAX_SENTENCE_CHARS:
        return [(start, end)]
    spans: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        limit = min(cursor + _SENTENCE_CHUNK_CHARS, end)
        if limit >= end:
            spans.append((cursor, end))
            break
        boundary = text.rfind(" ", cursor + 1, limit)
        if boundary <= cursor:
            boundary = limit
        spans.append((cursor, boundary))
        cursor = boundary
        while cursor < end and text[cursor].isspace():
            cursor += 1
    return spans


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Split `text` into contiguous sentence spans.

    The spans tile the whole body: a sentence keeps the whitespace that follows
    it, so consecutive spans touch exactly. Adjacency is what lets the selector
    merge two selected sentences into one citation-bearing run, and it is what
    makes the delivered ranges cover the original body without gaps.
    """

    spans: list[tuple[int, int]] = []
    start = 0
    index = 0
    length = len(text)
    while index < length:
        character = text[index]
        if character == "\n":
            _append_span(spans, text, start, index + 1)
            index += 1
            start = index
            continue
        if character in ".!?":
            after = index + 1
            while after < length and text[after] in _CLOSING_PUNCTUATION:
                after += 1
            if after >= length or text[after].isspace():
                word = _trailing_word(text, start, index)
                if _ends_sentence(text, after, word):
                    end = after
                    while end < length and text[end].isspace():
                        end += 1
                    _append_span(spans, text, start, end)
                    index = end
                    start = end
                    continue
        index += 1
    _append_span(spans, text, start, length)
    return spans


def _append_span(spans: list[tuple[int, int]], text: str, start: int, end: int) -> None:
    if end <= start:
        return
    if not spans and start < end and text[start].isspace():
        while start < end and text[start].isspace():
            start += 1
    if end > start:
        spans.append((start, end))


def _trailing_word(text: str, start: int, end: int) -> str:
    index = end
    while index > start and (text[index - 1].isalnum() or text[index - 1] == "."):
        index -= 1
    return text[index:end].lower()


def _ends_sentence(text: str, after: int, word: str) -> bool:
    if word in _ABBREVIATIONS:
        return False
    if len(word) == 1 and word.isalpha():
        return False
    if after >= len(text):
        return True
    following = text[after]
    if following.isspace():
        cursor = after
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text):
            return True
        return not text[cursor].islower()
    return not following.islower()


def _word_aligned_end(text: str, start: int, length: int) -> int | None:
    """Return a prefix end inside `start:length` at a word boundary."""

    if length >= len(text):
        return len(text) if text[start:] else None
    boundary = text.rfind(" ", start + 1, length)
    while boundary > start and text[boundary - 1].isspace():
        boundary -= 1
    return boundary if boundary > start else None


def _score(
    query: str,
    texts: Sequence[str],
    sentences: Sequence[Sequence[_Sentence]],
) -> tuple[list[list[float]], list[float]]:
    """Score every sentence and every candidate body against the query.

    The weight of a query term is its inverse document frequency across the
    candidate bodies, so a term that every candidate contains cannot dominate the
    choice and a term unique to one abstract can. A repeated term stops adding
    weight after a few occurrences: evidence is a passage, not a keyword list.
    """

    ordered_terms = _TERM.findall(query.lower())
    query_terms = Counter(ordered_terms)
    if not query_terms:
        return [[0.0 for _ in units] for units in sentences], [0.0 for _ in texts]
    document_counts = [Counter(_TERM.findall(text.lower())) for text in texts]
    total = len(document_counts)
    document_frequency: Counter[str] = Counter()
    for counts in document_counts:
        document_frequency.update(counts.keys())
    idf = {
        term: math.log(1.0 + (total - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
        for term in query_terms
    }
    bigrams = [(first, second) for first, second in pairwise(ordered_terms) if first != second]
    scores: list[list[float]] = []
    documents: list[float] = []
    for counts, units, text in zip(document_counts, sentences, texts, strict=True):
        documents.append(sum(query_terms[term] * idf[term] for term in query_terms if counts.get(term)))
        unit_scores = []
        for unit in units:
            sequence = _TERM.findall(text[unit.start : unit.end].lower())
            body = set(sequence)
            # A sentence earns credit for carrying a query term, not for repeating it:
            # one passage mentions its subject once, and a repeated word would otherwise
            # let a keyword-dense sentence outbid the sentence that actually states the finding.
            score = sum(query_terms[term] * idf[term] for term in query_terms if term in body)
            for first, second in bigrams:
                for position in range(len(sequence) - 1):
                    if sequence[position] == first and sequence[position + 1] == second:
                        score += _PHRASE_BONUS * (idf[first] + idf[second])
            unit_scores.append(score)
        scores.append(unit_scores)
    return scores, documents


__all__ = ["EvidenceSpan", "evidence_item", "item_bytes", "rendered_bytes", "select_evidence"]
