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

"""Cross-call request coalescing and bounded vector caching for embedding models.

Memory writes and vector search embed the same text more than once: one call can
repeat an entry, concurrent calls can embed identical text, and a call can arrive
while the provider is still working on that text. This wrapper shares one provider
request between every caller that needs the same text, keeps successful vectors in
a bounded LRU cache, and reports each caller's own share of the provider usage.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

from powercontext.builtin.inference.errors import InvalidInferenceOutputError
from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import QueryEmbeddingModel, embed_query

if TYPE_CHECKING:
    from powercontext.builtin.artifacts.memory.models import EmbeddingProfile
    from powercontext.builtin.inference.protocols import EmbeddingModel

DEFAULT_MAX_BATCH_SIZE = 64
DEFAULT_COALESCE_MS = 5.0
DEFAULT_CACHE_SIZE = 1024

_EmbeddingKind = Literal["document", "query"]
_Key = tuple[_EmbeddingKind, str]
_KINDS: tuple[_EmbeddingKind, ...] = ("document", "query")


@dataclass(frozen=True, slots=True)
class _BatchOutcome:
    """One completed provider request: text-keyed vectors plus its usage."""

    vectors: dict[str, EmbeddingVector]
    usage: InferenceUsage


_BatchTask: TypeAlias = asyncio.Task[_BatchOutcome]


@dataclass(frozen=True, slots=True)
class _Waiting:
    """The keys one call is still waiting for, split by how it obtains each vector."""

    fresh_keys: tuple[_Key, ...]
    fresh_futures: tuple[asyncio.Future[_BatchTask], ...]
    joined_keys: tuple[_Key, ...]
    joined_tasks: tuple[_BatchTask, ...]


class CoalescingEmbeddingModel:
    """Share in-flight embedding requests across calls and cache successful vectors.

    Each call deduplicates its own text, joins the texts other calls are already
    waiting for, and only then splits the still-pending texts into provider requests
    of at most ``max_batch_size`` distinct texts. Pending texts wait at most
    ``coalesce_ms`` for late callers before dispatch, and a full batch is dispatched
    immediately. Successful vectors are cached in an LRU bounded by ``cache_size``;
    ``cache_size=0`` disables cross-call caching while in-flight requests are still
    shared. A caller that is cancelled only abandons its own wait: provider work it
    shared with other callers keeps running, and a failed request is never cached.

    Query embeddings keep their own cache and request namespace, so an asymmetric
    model never serves a document vector to a retrieval query or the reverse.
    """

    profile: EmbeddingProfile

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        coalesce_ms: float = DEFAULT_COALESCE_MS,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")  # noqa: TRY003
        if coalesce_ms < 0:
            raise ValueError("coalesce_ms must not be negative")  # noqa: TRY003
        if cache_size < 0:
            raise ValueError("cache_size must not be negative")  # noqa: TRY003
        self._model = model
        self.profile = model.profile
        self._max_batch_size = int(max_batch_size)
        self._coalesce_seconds = coalesce_ms / 1000.0
        self._cache_size = int(cache_size)
        self._cache: OrderedDict[_Key, EmbeddingVector] = OrderedDict()
        self._loop: asyncio.AbstractEventLoop | None = None
        # Key -> future resolved with the provider task that will carry its vector.
        self._pending: dict[_Key, asyncio.Future[_BatchTask]] = {}
        # Key -> number of waiting callers, so a cancelled caller cannot strand work.
        self._pending_waiters: dict[_Key, int] = {}
        # Key -> provider task still running, so a late caller can reuse it.
        self._inflight: dict[_Key, _BatchTask] = {}
        self._flush_task: asyncio.Task[None] | None = None

    async def embed(self, texts: Sequence[str], /) -> EmbeddingResult:
        """Embed document texts in order, sharing provider work with concurrent callers."""

        return await self._embed(texts, kind="document")

    async def embed_query(self, texts: Sequence[str], /) -> EmbeddingResult:
        """Embed retrieval queries in order, sharing provider work with concurrent callers.

        A model that does not distinguish queries from documents embeds both through one
        space, so a search for text already embedded by a write is a cache hit instead of
        a new provider request.
        """

        kind: _EmbeddingKind = "query" if isinstance(self._model, QueryEmbeddingModel) else "document"
        return await self._embed(texts, kind=kind)

    async def _embed(self, texts: Sequence[str], *, kind: _EmbeddingKind) -> EmbeddingResult:
        requested = tuple(texts)
        if not requested:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))
        self._bind_loop()
        vectors, waiting = self._reserve(requested, kind)
        batch_by_key = await self._await_batches(waiting)
        batches = _distinct_tasks(batch_by_key.values())
        outcomes = await self._await_outcomes(batches)
        by_batch = {id(task): outcome for task, outcome in zip(batches, outcomes, strict=True)}
        for key, task in batch_by_key.items():
            vectors[key[1]] = by_batch[id(task)].vectors[key[1]]
        return EmbeddingResult(
            vectors=tuple(vectors[text] for text in requested),
            usage=_merged_usage(outcome.usage for outcome in outcomes),
        )

    def _reserve(self, requested: tuple[str, ...], kind: _EmbeddingKind) -> tuple[dict[str, EmbeddingVector], _Waiting]:
        """Resolve cache hits and join or open the request for every uncached text."""

        vectors: dict[str, EmbeddingVector] = {}
        fresh_keys: list[_Key] = []
        fresh_futures: list[asyncio.Future[_BatchTask]] = []
        joined_keys: list[_Key] = []
        joined_tasks: list[_BatchTask] = []
        for text in dict.fromkeys(requested):
            key = (kind, text)
            cached = self._cached(key)
            if cached is not None:
                vectors[text] = cached
                continue
            inflight = self._inflight.get(key)
            if inflight is not None:
                joined_keys.append(key)
                joined_tasks.append(inflight)
                continue
            future = self._pending.get(key)
            if future is None:
                future = self._loop.create_future()
                self._pending[key] = future
                self._pending_waiters[key] = 0
            self._pending_waiters[key] += 1
            fresh_keys.append(key)
            fresh_futures.append(future)
            if len(self._pending) >= self._max_batch_size:
                self._dispatch_pending()
        if self._pending:
            self._schedule_flush()
        return vectors, _Waiting(
            fresh_keys=tuple(fresh_keys),
            fresh_futures=tuple(fresh_futures),
            joined_keys=tuple(joined_keys),
            joined_tasks=tuple(joined_tasks),
        )

    async def _await_batches(self, waiting: _Waiting) -> dict[_Key, _BatchTask]:
        """Return the provider task that will carry each uncached text of this call."""

        batch_by_key: dict[_Key, _BatchTask] = {}
        try:
            for key, future in zip(waiting.fresh_keys, waiting.fresh_futures, strict=True):
                batch_by_key[key] = await asyncio.shield(future)
        except asyncio.CancelledError:
            self._abandon(waiting.fresh_keys)
            raise
        for key, task in zip(waiting.joined_keys, waiting.joined_tasks, strict=True):
            batch_by_key[key] = task
        return batch_by_key

    async def _await_outcomes(self, batches: Sequence[_BatchTask]) -> tuple[_BatchOutcome, ...]:
        """Await the shared provider work, surfacing the first failure to every waiter."""

        outcomes = await asyncio.gather(*(asyncio.shield(task) for task in batches), return_exceptions=True)
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
        return tuple(outcome for outcome in outcomes if isinstance(outcome, _BatchOutcome))

    def _bind_loop(self) -> None:
        """Reset request bookkeeping when the wrapper is reused on another event loop."""

        loop = asyncio.get_running_loop()
        if self._loop is loop:
            return
        self._loop = loop
        self._pending = {}
        self._pending_waiters = {}
        self._inflight = {}
        self._flush_task = None

    def _cached(self, key: _Key) -> EmbeddingVector | None:
        vector = self._cache.get(key)
        if vector is None:
            return None
        self._cache.move_to_end(key)
        return vector

    def _store(self, kind: _EmbeddingKind, vectors: dict[str, EmbeddingVector]) -> None:
        if self._cache_size <= 0:
            return
        for text, vector in vectors.items():
            key = (kind, text)
            self._cache[key] = vector
            self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _schedule_flush(self) -> None:
        if self._flush_task is not None and not self._flush_task.done():
            return
        self._flush_task = self._loop.create_task(self._flush_after_window())

    async def _flush_after_window(self) -> None:
        try:
            await asyncio.sleep(self._coalesce_seconds)
        finally:
            self._flush_task = None
        self._dispatch_pending()

    def _dispatch_pending(self) -> None:
        """Turn every text waiting for the window into provider batches."""

        if not self._pending:
            return
        pending, self._pending = self._pending, {}
        for key in pending:
            self._pending_waiters.pop(key, None)
        # Document and query vectors come from different provider calls on an asymmetric
        # model, so a batch never mixes the two kinds even when both are pending.
        for kind in _KINDS:
            keys = tuple(key for key in pending if key[0] == kind)
            for start in range(0, len(keys), self._max_batch_size):
                self._dispatch_batch(pending, keys[start : start + self._max_batch_size])

    def _dispatch_batch(self, pending: dict[_Key, asyncio.Future[_BatchTask]], batch: tuple[_Key, ...]) -> None:
        task = self._loop.create_task(self._run_batch(batch))
        task.add_done_callback(_observe_outcome)
        for key in batch:
            self._inflight[key] = task
            future = pending[key]
            if not future.done():
                future.set_result(task)

    def _abandon(self, keys: Iterable[_Key]) -> None:
        """Drop pending texts whose last waiter was cancelled so no request is stranded."""

        for key in keys:
            waiting = self._pending_waiters.get(key)
            if waiting is None:
                continue
            if waiting > 1:
                self._pending_waiters[key] = waiting - 1
                continue
            del self._pending_waiters[key]
            future = self._pending.pop(key, None)
            if future is not None and not future.done():
                future.cancel()

    async def _run_batch(self, keys: tuple[_Key, ...]) -> _BatchOutcome:
        kind = keys[0][0]
        texts = tuple(key[1] for key in keys)
        try:
            result = await self._call_provider(texts, kind)
            if len(result.vectors) != len(texts):
                raise InvalidInferenceOutputError("embedding", "vector-count")
            outcome = _BatchOutcome(vectors=dict(zip(texts, result.vectors, strict=True)), usage=result.usage)
            self._store(kind, outcome.vectors)
            return outcome
        finally:
            # Release before the task settles so a late caller never joins a finished batch.
            self._release(keys)

    async def _call_provider(self, texts: tuple[str, ...], kind: _EmbeddingKind) -> EmbeddingResult:
        if kind == "query" and isinstance(self._model, QueryEmbeddingModel):
            return await embed_query(self._model, texts)
        return await self._model.embed(texts)

    def _release(self, keys: Iterable[_Key]) -> None:
        current = asyncio.current_task()
        for key in keys:
            if self._inflight.get(key) is current:
                del self._inflight[key]


def _distinct_tasks(tasks: Iterable[_BatchTask]) -> tuple[_BatchTask, ...]:
    distinct: list[_BatchTask] = []
    for task in tasks:
        if not any(task is seen for seen in distinct):
            distinct.append(task)
    return tuple(distinct)


def _observe_outcome(task: _BatchTask) -> None:
    """Mark a finished batch as observed so an unawaited failure is not reported twice."""

    if not task.cancelled():
        task.exception()


def _merged_usage(usages: Iterable[InferenceUsage]) -> InferenceUsage:
    """Sum the usage of the distinct provider requests one caller waited for."""

    collected = list(usages)
    if not collected:
        return InferenceUsage(requests=0)
    return InferenceUsage(
        requests=sum(usage.requests for usage in collected),
        input_tokens=_sum_tokens(usage.input_tokens for usage in collected),
        output_tokens=_sum_tokens(usage.output_tokens for usage in collected),
    )


def _sum_tokens(values: Iterable[int | None]) -> int | None:
    provided = [value for value in values if value is not None]
    return sum(provided) if provided else None


__all__ = ["CoalescingEmbeddingModel"]
