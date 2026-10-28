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

"""Behavior tests for cross-call embedding coalescing and vector caching."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence

import pytest

from powercontext.builtin.artifacts.memory.models import EmbeddingProfile
from powercontext.builtin.inference.coalescing import CoalescingEmbeddingModel
from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage

PROFILE = EmbeddingProfile(profile_id="coalescing-v1", model="offline", dimension=2)


class RecordingEmbedding:
    """A deterministic provider that records every batch it is asked to embed."""

    profile = PROFILE

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    async def embed(self, texts: Sequence[str], /) -> EmbeddingResult:
        self.calls.append(tuple(texts))
        return EmbeddingResult(
            vectors=tuple(_vector(text) for text in texts),
            usage=InferenceUsage(requests=1, input_tokens=len(texts) * 3, output_tokens=0),
        )


def _vector(text: str) -> EmbeddingVector:
    """Map a text to a stable two-dimensional vector."""

    return (float(len(text)), float(sum(text.encode("utf-8")) % 97))


def _run(scenario: Callable[[], object]) -> None:
    asyncio.run(scenario())  # type: ignore[arg-type]


def test_embed_protocol_exposes_profile_and_returns_ordered_vectors() -> None:
    base = RecordingEmbedding()
    model = CoalescingEmbeddingModel(base, coalesce_ms=0)

    async def scenario() -> None:
        assert model.profile == base.profile
        result = await model.embed(("beta", "alpha", "beta"))
        assert result.vectors == (_vector("beta"), _vector("alpha"), _vector("beta"))
        assert base.calls == [("beta", "alpha")]
        assert result.usage.requests == 1

    _run(scenario)


def test_empty_input_never_reaches_the_provider() -> None:
    base = RecordingEmbedding()
    model = CoalescingEmbeddingModel(base, coalesce_ms=0, cache_size=0)

    async def scenario() -> None:
        result = await model.embed(())
        assert result.vectors == ()
        assert result.usage.requests == 0
        assert result.usage.input_tokens in (None, 0)
        assert base.calls == []

    _run(scenario)


def test_every_provider_request_stays_within_the_batch_ceiling() -> None:
    base = RecordingEmbedding()
    model = CoalescingEmbeddingModel(base, max_batch_size=3, coalesce_ms=0, cache_size=0)
    texts = tuple(f"text-{index}" for index in range(10))

    async def scenario() -> None:
        result = await model.embed(texts)
        assert result.vectors == tuple(_vector(text) for text in texts)
        assert base.calls
        assert all(len(call) <= 3 for call in base.calls)
        assert sorted(text for call in base.calls for text in call) == sorted(texts)

    _run(scenario)


def test_concurrent_calls_share_one_request_per_text_and_split_usage() -> None:
    base = RecordingEmbedding()
    model = CoalescingEmbeddingModel(base, max_batch_size=8, coalesce_ms=20, cache_size=0)

    async def scenario() -> None:
        first, second = await asyncio.gather(
            model.embed(("alpha", "beta")),
            model.embed(("beta", "gamma")),
        )
        assert first.vectors == (_vector("alpha"), _vector("beta"))
        assert second.vectors == (_vector("beta"), _vector("gamma"))
        assert sorted(text for call in base.calls for text in call) == ["alpha", "beta", "gamma"]
        assert first.usage.requests == 1
        assert second.usage.requests == 1
        assert first.usage.input_tokens == 3 * len(base.calls[0])

    _run(scenario)


def test_a_late_caller_reuses_a_request_that_is_still_running() -> None:
    base = GatedEmbedding()
    model = CoalescingEmbeddingModel(base, coalesce_ms=0, cache_size=0)

    async def scenario() -> None:
        leader = asyncio.create_task(model.embed(("shared",)))
        await base.started.wait()
        follower = asyncio.create_task(model.embed(("shared",)))
        await asyncio.sleep(0)
        base.release.set()
        assert (await leader).vectors == (_vector("shared"),)
        assert (await follower).vectors == (_vector("shared"),)
        assert base.calls == [("shared",)]
        assert follower.done()

    _run(scenario)


def test_lru_keeps_recently_used_vectors_and_bounds_the_cache() -> None:
    base = RecordingEmbedding()
    model = CoalescingEmbeddingModel(base, coalesce_ms=0, cache_size=2)

    async def scenario() -> None:
        await model.embed(("one",))
        await model.embed(("two",))
        await model.embed(("one",))
        assert len(base.calls) == 2, "a cached text must not be requested again"
        await model.embed(("three",))
        await model.embed(("one",))
        assert sum(call.count("one") for call in base.calls) == 1, "recent access must refresh LRU order"
        await model.embed(("two",))
        assert sum(call.count("two") for call in base.calls) == 2, "the least recently used entry must be evicted"

    _run(scenario)


def test_disabling_the_cache_still_shares_requests_already_running() -> None:
    base = GatedEmbedding()
    model = CoalescingEmbeddingModel(base, coalesce_ms=0, cache_size=0)

    async def scenario() -> None:
        leader = asyncio.create_task(model.embed(("shared",)))
        await base.started.wait()
        follower = asyncio.create_task(model.embed(("shared",)))
        await asyncio.sleep(0)
        base.release.set()
        await leader
        await follower
        assert base.calls == [("shared",)]
        await model.embed(("shared",))
        assert len(base.calls) == 2, "a disabled cache must not reuse a completed vector"

    _run(scenario)


def test_cache_hits_report_no_requests_and_keep_provider_tokens() -> None:
    class Metered(RecordingEmbedding):
        async def embed(self, texts: Sequence[str], /) -> EmbeddingResult:
            self.calls.append(tuple(texts))
            return EmbeddingResult(
                vectors=tuple(_vector(text) for text in texts),
                usage=InferenceUsage(requests=1, input_tokens=7, output_tokens=2),
            )

    base = Metered()
    model = CoalescingEmbeddingModel(base, coalesce_ms=0)

    async def scenario() -> None:
        first = await model.embed(("alpha", "beta"))
        assert (first.usage.requests, first.usage.input_tokens, first.usage.output_tokens) == (1, 7, 2)
        cached = await model.embed(("alpha",))
        assert cached.usage.requests == 0
        assert cached.usage.input_tokens in (None, 0)
        assert cached.usage.output_tokens in (None, 0)

    _run(scenario)


def test_one_call_sums_the_usage_of_every_batch_it_waited_for() -> None:
    base = RecordingEmbedding()
    model = CoalescingEmbeddingModel(base, max_batch_size=4, coalesce_ms=0, cache_size=0)
    texts = tuple(f"text-{index}" for index in range(9))

    async def scenario() -> None:
        result = await model.embed(texts)
        assert result.vectors == tuple(_vector(text) for text in texts)
        assert len(base.calls) == 3, "nine distinct texts at four per batch need three requests"
        assert result.usage.requests == 3, "each distinct batch is charged exactly once"
        assert result.usage.input_tokens == 3 * len(texts), "provider tokens are preserved, not recomputed"

    _run(scenario)


def test_a_partially_cached_call_only_charges_the_missing_batches() -> None:
    base = RecordingEmbedding()
    model = CoalescingEmbeddingModel(base, max_batch_size=8, coalesce_ms=0)

    async def scenario() -> None:
        await model.embed(("alpha", "beta"))
        result = await model.embed(("alpha", "beta", "gamma"))
        assert result.vectors == (_vector("alpha"), _vector("beta"), _vector("gamma"))
        assert base.calls == [("alpha", "beta"), ("gamma",)]
        assert result.usage.requests == 1, "cached texts must not be charged again"

    _run(scenario)


def test_a_failed_batch_is_not_cached_and_can_be_retried() -> None:
    class Flaky(RecordingEmbedding):
        async def embed(self, texts: Sequence[str], /) -> EmbeddingResult:
            self.calls.append(tuple(texts))
            if len(self.calls) == 1:
                raise RuntimeError("provider down")  # noqa: TRY003
            return EmbeddingResult(vectors=tuple(_vector(text) for text in texts))

    base = Flaky()
    model = CoalescingEmbeddingModel(base, coalesce_ms=0)

    async def scenario() -> None:
        with pytest.raises(RuntimeError):
            await model.embed(("retry",))
        assert (await model.embed(("retry",))).vectors == (_vector("retry"),)
        assert len(base.calls) == 2

    _run(scenario)


def test_a_failed_batch_does_not_prevent_an_independent_batch() -> None:
    class SplitFailure(RecordingEmbedding):
        async def embed(self, texts: Sequence[str], /) -> EmbeddingResult:
            self.calls.append(tuple(texts))
            if "boom" in texts:
                raise RuntimeError("provider down")  # noqa: TRY003
            return EmbeddingResult(vectors=tuple(_vector(text) for text in texts))

    base = SplitFailure()
    model = CoalescingEmbeddingModel(base, max_batch_size=1, coalesce_ms=0, cache_size=0)

    async def scenario() -> None:
        failing = asyncio.create_task(model.embed(("boom",)))
        healthy = asyncio.create_task(model.embed(("fine",)))
        results = await asyncio.gather(failing, healthy, return_exceptions=True)
        assert isinstance(results[0], RuntimeError)
        assert results[1].vectors == (_vector("fine"),)

    _run(scenario)


def test_a_vector_count_mismatch_surfaces_as_a_value_error() -> None:
    class Short(RecordingEmbedding):
        async def embed(self, texts: Sequence[str], /) -> EmbeddingResult:
            self.calls.append(tuple(texts))
            return EmbeddingResult(vectors=())

    base = Short()
    model = CoalescingEmbeddingModel(base, coalesce_ms=0)

    async def scenario() -> None:
        with pytest.raises(ValueError):
            await model.embed(("broken",))
        assert base.calls == [("broken",)]

    _run(scenario)


def test_cancelling_one_waiter_keeps_the_shared_request_for_the_others() -> None:
    base = GatedEmbedding()
    model = CoalescingEmbeddingModel(base, coalesce_ms=0, cache_size=0)

    async def scenario() -> None:
        cancelled = asyncio.create_task(model.embed(("shared",)))
        survivor = asyncio.create_task(model.embed(("shared",)))
        await base.started.wait()
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        base.release.set()
        assert (await survivor).vectors == (_vector("shared"),)
        assert base.calls == [("shared",)]

    _run(scenario)


def test_cancelling_the_only_waiter_does_not_strand_a_request() -> None:
    base = RecordingEmbedding()
    model = CoalescingEmbeddingModel(base, coalesce_ms=50, cache_size=0)

    async def scenario() -> None:
        abandoned = asyncio.create_task(model.embed(("lonely",)))
        await asyncio.sleep(0)
        abandoned.cancel()
        with pytest.raises(asyncio.CancelledError):
            await abandoned
        assert base.calls == [], "a request with no remaining waiter must not reach the provider"
        assert (await model.embed(("lonely",))).vectors == (_vector("lonely"),)

    _run(scenario)


def test_invalid_construction_arguments_are_rejected() -> None:
    base = RecordingEmbedding()
    with pytest.raises(ValueError):
        CoalescingEmbeddingModel(base, max_batch_size=0)
    with pytest.raises(ValueError):
        CoalescingEmbeddingModel(base, coalesce_ms=-1.0)
    with pytest.raises(ValueError):
        CoalescingEmbeddingModel(base, cache_size=-1)


def test_query_and_document_vectors_keep_separate_caches_for_an_asymmetric_model() -> None:
    class Asymmetric(RecordingEmbedding):
        """A provider whose query and document vectors live in different spaces."""

        def __init__(self) -> None:
            super().__init__()
            self.query_calls: list[tuple[str, ...]] = []

        async def embed(self, texts: Sequence[str], /) -> EmbeddingResult:
            self.calls.append(tuple(texts))
            return EmbeddingResult(vectors=tuple((1.0, 0.0) for _ in texts))

        async def embed_query(self, texts: Sequence[str], /) -> EmbeddingResult:
            self.query_calls.append(tuple(texts))
            return EmbeddingResult(vectors=tuple((0.0, 1.0) for _ in texts))

    base = Asymmetric()
    model = CoalescingEmbeddingModel(base, coalesce_ms=0, cache_size=8)

    async def scenario() -> None:
        assert (await model.embed(("shared",))).vectors == ((1.0, 0.0),)
        assert (await model.embed_query(("shared",))).vectors == ((0.0, 1.0),)
        assert base.calls == [("shared",)]
        assert base.query_calls == [("shared",)]
        assert (await model.embed_query(("shared",))).vectors == ((0.0, 1.0),)
        assert base.query_calls == [("shared",)], "a repeated query must hit the query cache"

    _run(scenario)


def test_a_symmetric_model_shares_one_space_between_queries_and_documents() -> None:
    base = RecordingEmbedding()
    model = CoalescingEmbeddingModel(base, coalesce_ms=0, cache_size=8)

    async def scenario() -> None:
        await model.embed(("shared",))
        assert (await model.embed_query(("shared",))).vectors == (_vector("shared"),)
        assert base.calls == [("shared",)], "a model without query embeddings embeds one space"

    _run(scenario)


def test_a_pending_window_never_mixes_queries_with_documents() -> None:
    class Asymmetric(RecordingEmbedding):
        def __init__(self) -> None:
            super().__init__()
            self.query_calls: list[tuple[str, ...]] = []

        async def embed_query(self, texts: Sequence[str], /) -> EmbeddingResult:
            self.query_calls.append(tuple(texts))
            return EmbeddingResult(vectors=tuple((0.0, 1.0) for _ in texts))

    base = Asymmetric()
    model = CoalescingEmbeddingModel(base, max_batch_size=4, coalesce_ms=20, cache_size=0)

    async def scenario() -> None:
        documents = asyncio.create_task(model.embed(("doc-a", "doc-b")))
        queries = asyncio.create_task(model.embed_query(("query-a", "query-b")))
        await documents
        await queries
        assert base.calls == [("doc-a", "doc-b")]
        assert base.query_calls == [("query-a", "query-b")]

    _run(scenario)


def test_the_wrapper_can_serve_more_than_one_event_loop() -> None:
    base = RecordingEmbedding()
    model = CoalescingEmbeddingModel(base, coalesce_ms=0)

    async def first() -> None:
        assert (await model.embed(("alpha",))).vectors == (_vector("alpha"),)

    async def second() -> None:
        assert (await model.embed(("beta",))).vectors == (_vector("beta"),)

    _run(first)
    _run(second)
    assert list(base.calls) == [("alpha",), ("beta",)]


class GatedEmbedding(RecordingEmbedding):
    """A provider that blocks until the test releases it."""

    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def embed(self, texts: Sequence[str], /) -> EmbeddingResult:
        self.calls.append(tuple(texts))
        self.started.set()
        await self.release.wait()
        return EmbeddingResult(vectors=tuple(_vector(text) for text in texts))
