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

"""Native Memory write/search coverage for the coalescing embedding wrapper."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from powercontext.builtin.artifacts.memory.models import EmbeddingProfile, MemoryEntryInput
from powercontext.builtin.inference.coalescing import CoalescingEmbeddingModel
from powercontext.builtin.inference.models import EmbeddingResult, InferenceUsage
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import BuiltinConfig, open_builtin_contexts

DIMENSION = 24
PROFILE = EmbeddingProfile(profile_id="coalescing-native-v1", model="offline", dimension=DIMENSION)


class FixtureEmbedding:
    """A deterministic offline provider whose vectors isolate each fixture text."""

    profile = PROFILE

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    async def embed(self, texts: Sequence[str], /) -> EmbeddingResult:
        self.calls.append(tuple(texts))
        return EmbeddingResult(
            vectors=tuple(_vector(text) for text in texts),
            usage=InferenceUsage(requests=1, input_tokens=len(texts) * 7, output_tokens=0),
        )


def _vector(text: str) -> tuple[float, ...]:
    """Return a one-hot vector that only matches its own fixture text."""

    return tuple(float(f"needle{index} fixture" == text) for index in range(DIMENSION))


def _config(tmp_path, name: str) -> BuiltinConfig:
    return BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / name}"))


def test_concurrent_scopes_share_bounded_provider_work_and_cached_search(tmp_path) -> None:
    base = FixtureEmbedding()
    model = CoalescingEmbeddingModel(base, max_batch_size=6, coalesce_ms=0, cache_size=48)
    parties = 6

    class StartTogether:
        """Release every scope into the wrapper at the same interface boundary."""

        profile = model.profile

        def __init__(self) -> None:
            self.arrived = 0
            self.release = asyncio.Event()

        async def embed(self, texts: Sequence[str], /) -> EmbeddingResult:
            if self.arrived < parties:
                self.arrived += 1
                if self.arrived == parties:
                    self.release.set()
                await self.release.wait()
            return await model.embed(texts)

        async def embed_query(self, texts: Sequence[str], /) -> EmbeddingResult:
            return await model.embed_query(texts)

    async def scenario() -> None:
        config = _config(tmp_path, "batching.db")
        async with open_builtin_contexts(config, embedding_model=StartTogether()) as contexts:
            services = [(await contexts.get(f"scope-{index}")).artifacts.memory for index in range(parties)]
            texts = tuple(f"needle{index} fixture" for index in range(DIMENSION))
            entries = tuple(MemoryEntryInput(kind="fact", text=value) for value in texts)
            memories = await asyncio.wait_for(
                asyncio.gather(
                    *(service.remember(memory=None, entries=entries, mode="append") for service in services)
                ),
                timeout=30,
            )
            assert len(base.calls) == DIMENSION // 6
            assert all(1 <= len(call) <= 6 for call in base.calls)
            assert sorted(value for call in base.calls for value in call) == sorted(texts)
            for index, (service, memory) in enumerate(zip(services, memories, strict=True)):
                result = await service.search(texts[index], memories=(memory,), limit=1, mode="vector")
                assert result.hits[0].text == texts[index]
            assert len(base.calls) == DIMENSION // 6, "cached searches must not issue provider requests"

    asyncio.run(scenario())


def test_write_reports_only_the_batches_it_waited_for(tmp_path) -> None:
    base = FixtureEmbedding()
    model = CoalescingEmbeddingModel(base, max_batch_size=4, coalesce_ms=0, cache_size=0)
    texts = tuple(f"needle{index} fixture" for index in range(8))

    async def scenario() -> None:
        config = _config(tmp_path, "usage.db")
        async with open_builtin_contexts(config, embedding_model=model) as contexts:
            service = (await contexts.get("usage")).artifacts.memory
            entries = tuple(MemoryEntryInput(kind="fact", text=value) for value in texts)
            memory = await service.remember(memory=None, entries=entries, mode="append")
            assert memory is not None
            assert len(base.calls) == 2, "eight distinct texts at four per batch need two requests"
            assert all(len(call) <= 4 for call in base.calls)

    asyncio.run(scenario())


def test_search_reuses_vectors_written_under_the_same_profile(tmp_path) -> None:
    base = FixtureEmbedding()
    model = CoalescingEmbeddingModel(base, coalesce_ms=0, cache_size=16)

    async def scenario() -> None:
        config = _config(tmp_path, "reuse.db")
        async with open_builtin_contexts(config, embedding_model=model) as contexts:
            service = (await contexts.get("reuse")).artifacts.memory
            entries = (MemoryEntryInput(kind="fact", text="needle0 fixture"),)
            memory = await service.remember(memory=None, entries=entries, mode="append")
            assert memory is not None
            written = len(base.calls)
            result = await service.search("needle0 fixture", memories=(memory,), limit=1, mode="vector")
            assert result.hits[0].text == "needle0 fixture"
            assert len(base.calls) == written, "an unchanged profile must serve the search from cache"

    asyncio.run(scenario())


def test_a_failed_provider_request_leaves_the_memory_write_retryable(tmp_path) -> None:
    class Flaky(FixtureEmbedding):
        async def embed(self, texts: Sequence[str], /) -> EmbeddingResult:
            self.calls.append(tuple(texts))
            if len(self.calls) == 1:
                raise RuntimeError("provider down")  # noqa: TRY003
            return EmbeddingResult(vectors=tuple(_vector(text) for text in texts))

    base = Flaky()
    model = CoalescingEmbeddingModel(base, coalesce_ms=0, cache_size=0)

    async def scenario() -> None:
        config = _config(tmp_path, "retry.db")
        async with open_builtin_contexts(config, embedding_model=model) as contexts:
            service = (await contexts.get("retry")).artifacts.memory
            entries = (MemoryEntryInput(kind="fact", text="needle0 fixture"),)
            with pytest.raises(RuntimeError):
                await service.remember(memory=None, entries=entries, mode="append")
            memory = await service.remember(memory=None, entries=entries, mode="append")
            assert memory is not None
            result = await service.search("needle0 fixture", memories=(memory,), limit=1, mode="vector")
            assert result.hits[0].text == "needle0 fixture"

    asyncio.run(scenario())


def test_a_provider_returning_the_wrong_vector_count_raises_value_error(tmp_path) -> None:
    class Short(FixtureEmbedding):
        async def embed(self, texts: Sequence[str], /) -> EmbeddingResult:
            self.calls.append(tuple(texts))
            return EmbeddingResult(vectors=())

    model = CoalescingEmbeddingModel(Short(), coalesce_ms=0)

    async def scenario() -> None:
        config = _config(tmp_path, "short.db")
        async with open_builtin_contexts(config, embedding_model=model) as contexts:
            service = (await contexts.get("short")).artifacts.memory
            entries = (MemoryEntryInput(kind="fact", text="needle0 fixture"),)
            with pytest.raises(ValueError):
                await service.remember(memory=None, entries=entries, mode="append")

    asyncio.run(scenario())
