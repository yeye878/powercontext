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

"""Behavior tests for HTTP cross-encoder Memory reranking.

The public challenge acceptance covers the happy path, schema rejection, and one native
search. These tests protect the contract edges the acceptance suite does not reach:
non-dividing batch sizes, cross-batch score ties, protocol boundaries that must stay
ValueError, transport failures that must stay httpx, and identity preservation through a
real Memory search.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory import HTTPMemoryReranker, MemoryEntryInput
from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import BuiltinConfig, open_builtin_contexts

ENDPOINT = "https://reranker.invalid/score"


def _hit(rank: int) -> MemoryHit:
    return MemoryHit(
        memory_ref=ArtifactRef(family="memory", artifact_id="memory", revision=1),
        entry_id=f"entry-{rank}",
        entry_version_id=f"entry-{rank}-v1",
        text=f"candidate {rank}",
        score=1.0 / rank,
        matched_by=("fts",),
    )


def _pool(size: int) -> tuple[MemoryHit, ...]:
    return tuple(_hit(rank) for rank in range(1, size + 1))


def _scoring_transport(scores: dict[int, float], *, reverse: bool = False) -> tuple[Any, list[dict[str, Any]]]:
    """Answer every batch with the scores this test assigned to its global ranks."""

    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        rows = [{"rank": item["rank"], "score": scores[item["rank"]]} for item in body["candidates"]]
        return httpx.Response(200, json={"scores": list(reversed(rows)) if reverse else rows})

    return handler, requests


def _run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


# --- batching, coverage, and global selection -------------------------------------------


def test_non_dividing_batch_size_sends_each_global_rank_exactly_once() -> None:
    handler, requests = _scoring_transport({rank: float(rank) for rank in range(1, 8)})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reranker = HTTPMemoryReranker(ENDPOINT, client=client, batch_size=3)
            decision = await reranker.rerank("query", _pool(7), 3)

        assert decision.usage.requests == 3
        assert [len(body["candidates"]) for body in requests] == [3, 3, 1]
        assert [item["rank"] for body in requests for item in body["candidates"]] == list(range(1, 8))
        # Every request names the text of the global candidate its rank refers to.
        for body in requests:
            for item in body["candidates"]:
                assert item["text"] == _hit(item["rank"]).text
                assert type(item["rank"]) is int
        assert decision.selected_ranks == (7, 6, 5)

    _run(scenario())


def test_batch_size_larger_than_pool_issues_one_request() -> None:
    handler, requests = _scoring_transport(dict.fromkeys(range(1, 4), 0.5))

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            decision = await HTTPMemoryReranker(ENDPOINT, client=client, batch_size=64).rerank("query", _pool(3), 2)

        assert decision.usage.requests == 1
        assert len(requests) == 1
        # An all-tie pool keeps the coarse order.
        assert decision.selected_ranks == (1, 2)

    _run(scenario())


def test_cross_batch_ties_break_on_the_original_global_rank() -> None:
    # The challenge's worked example: rank 3 (batch 2) wins outright, and the equal
    # scores of rank 2 (batch 1) and rank 4 (batch 2) resolve in favour of rank 2.
    handler, _requests = _scoring_transport({1: 0.10, 2: 0.90, 3: 0.95, 4: 0.90})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reranker = HTTPMemoryReranker(ENDPOINT, client=client, batch_size=2)
            decision = await reranker.rerank("query", _pool(4), 2)

        assert decision.selected_ranks == (3, 2)

    _run(scenario())


def test_every_batch_participates_in_one_global_selection() -> None:
    # A per-batch selection that truncated each batch to the requested limit would lose
    # the winner parked in the middle batch, so the pool hides one there.
    scores = dict.fromkeys(range(1, 11), 0.0)
    scores[2] = 0.99
    scores[9] = 0.98
    handler, _requests = _scoring_transport(scores)

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reranker = HTTPMemoryReranker(ENDPOINT, client=client, batch_size=4)
            decision = await reranker.rerank("query", _pool(10), 2)

        assert decision.selected_ranks == (2, 9)

    _run(scenario())


def test_limit_equal_to_pool_returns_every_rank_once() -> None:
    scores = {1: 0.3, 2: 0.1, 3: 0.2}
    handler, _requests = _scoring_transport(scores)

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            decision = await HTTPMemoryReranker(ENDPOINT, client=client, batch_size=2).rerank("query", _pool(3), 3)

        assert decision.selected_ranks == (1, 3, 2)
        assert len(set(decision.selected_ranks)) == 3

    _run(scenario())


# --- response validation ----------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        # A rank that belongs to another batch is not this batch's answer.
        {"scores": [{"rank": 1, "score": 0.5}, {"rank": 3, "score": 0.5}]},
        # A batch answer must not silently drop one of its ranks.
        {"scores": [{"rank": 1, "score": 0.5}]},
        # A rank outside the pool entirely.
        {"scores": [{"rank": 1, "score": 0.5}, {"rank": 99, "score": 0.5}]},
        # Non-numeric and boolean scores are not JSON numbers.
        {"scores": [{"rank": 1, "score": "0.5"}, {"rank": 2, "score": 0.5}]},
        {"scores": [{"rank": 1, "score": True}, {"rank": 2, "score": 0.5}]},
        # The score record itself is a closed schema.
        {"scores": [{"rank": 1, "score": 0.5, "text": "extra"}, {"rank": 2, "score": 0.5}]},
        {"scores": [{"rank": 1.0, "score": 0.5}, {"rank": 2, "score": 0.5}]},
        # The envelope is a closed schema too.
        {"scores": [{"rank": 1, "score": 0.5}, {"rank": 2, "score": 0.5}], "usage": {"requests": 1}},
        [{"rank": 1, "score": 0.5}],
        {},
    ],
)
def test_rejects_cross_batch_missing_and_malformed_batches(payload: Any) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps(payload).encode())

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reranker = HTTPMemoryReranker(ENDPOINT, client=client, batch_size=2)
            with pytest.raises(ValueError):
                await reranker.rerank("query", _pool(2), 1)

    _run(scenario())


def test_rejects_non_finite_scores_even_when_the_body_is_valid_json() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        # Python's JSON writer emits bare NaN/Infinity, which json.loads happily accepts.
        return httpx.Response(200, content=b'{"scores": [{"rank": 1, "score": NaN}]}')

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(ValueError):
                await HTTPMemoryReranker(ENDPOINT, client=client).rerank("query", _pool(1), 1)

    _run(scenario())


def test_rejects_malformed_json_body() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json at all")

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(ValueError):
                await HTTPMemoryReranker(ENDPOINT, client=client).rerank("query", _pool(1), 1)

    _run(scenario())


def test_accepts_integer_scores_and_out_of_order_rows() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"scores": [{"rank": 2, "score": 1}, {"rank": 1, "score": 0}]})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            decision = await HTTPMemoryReranker(ENDPOINT, client=client, batch_size=2).rerank("query", _pool(2), 1)

        assert decision.selected_ranks == (2,)

    _run(scenario())


# --- HTTP and transport failures --------------------------------------------------------


@pytest.mark.parametrize("status", [400, 404, 429, 500, 503])
def test_non_2xx_responses_stay_httpx_status_errors(status: int) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"scores": []})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(httpx.HTTPStatusError):
                await HTTPMemoryReranker(ENDPOINT, client=client).rerank("query", _pool(1), 1)

    _run(scenario())


def test_network_errors_stay_httpx_transport_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("offline", request=request)

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(httpx.ReadTimeout):
                await HTTPMemoryReranker(ENDPOINT, client=client).rerank("query", _pool(1), 1)

    _run(scenario())


def test_a_failing_later_batch_returns_no_partial_decision() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(len(calls) + 1)
        if len(calls) == 2:
            return httpx.Response(500, json={"scores": []})
        return httpx.Response(
            200, json={"scores": [{"rank": item["rank"], "score": 0.1} for item in body["candidates"]]}
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reranker = HTTPMemoryReranker(ENDPOINT, client=client, batch_size=1)
            with pytest.raises(httpx.HTTPStatusError):
                await reranker.rerank("query", _pool(3), 2)

        assert len(calls) == 2

    _run(scenario())


# --- parameter bounds -------------------------------------------------------------------


def test_invalid_parameters_fail_before_any_request_is_sent() -> None:
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={"scores": []})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reranker = HTTPMemoryReranker(ENDPOINT, client=client)
            with pytest.raises(ValueError):
                await reranker.rerank("query", (), 1)
            with pytest.raises(ValueError):
                await reranker.rerank("query", _pool(2), 0)
            with pytest.raises(ValueError):
                await reranker.rerank("query", _pool(2), 3)

        assert sent == []

    _run(scenario())


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True, "2"])
def test_constructor_rejects_invalid_batch_sizes(batch_size: Any) -> None:
    async def scenario() -> None:
        transport = httpx.MockTransport(lambda _request: httpx.Response(200))
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(ValueError):
                HTTPMemoryReranker(ENDPOINT, client=client, batch_size=batch_size)

    _run(scenario())


@pytest.mark.parametrize("endpoint", ["", "   "])
def test_constructor_rejects_blank_endpoints(endpoint: str) -> None:
    async def scenario() -> None:
        transport = httpx.MockTransport(lambda _request: httpx.Response(200))
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(ValueError):
                HTTPMemoryReranker(endpoint, client=client)

    _run(scenario())


def test_constructor_defaults_match_the_declared_contract() -> None:
    async def scenario() -> None:
        transport = httpx.MockTransport(lambda _request: httpx.Response(200))
        async with httpx.AsyncClient(transport=transport) as client:
            reranker = HTTPMemoryReranker(ENDPOINT, client=client)
            assert reranker.policy_id == "http-cross-encoder-v1"
            custom = HTTPMemoryReranker(ENDPOINT, client=client, policy_id="my-cross-encoder")
            assert custom.policy_id == "my-cross-encoder"

    _run(scenario())


# --- usage, determinism, identity -------------------------------------------------------


def test_usage_requests_matches_the_actual_http_call_count() -> None:
    handler, requests = _scoring_transport(dict.fromkeys(range(1, 11), 0.5))

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reranker = HTTPMemoryReranker(ENDPOINT, client=client, batch_size=7)
            decision = await reranker.rerank("query", _pool(10), 5)

        assert decision.usage.requests == 2
        assert len(requests) == 2
        assert decision.usage.input_tokens is None
        assert decision.usage.output_tokens is None

    _run(scenario())


def test_repeated_calls_are_stateless_and_deterministic() -> None:
    handler, requests = _scoring_transport({1: 0.4, 2: 0.9, 3: 0.9})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reranker = HTTPMemoryReranker(ENDPOINT, client=client, batch_size=2)
            first = await reranker.rerank("query", _pool(3), 2)
            second = await reranker.rerank("query", _pool(3), 2)

        assert first.selected_ranks == second.selected_ranks == (2, 3)
        assert first.usage.requests == second.usage.requests == 2
        assert len(requests) == 4

    _run(scenario())


def test_decision_never_carries_candidate_content() -> None:
    handler, _requests = _scoring_transport({1: 0.1, 2: 0.2})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            decision = await HTTPMemoryReranker(ENDPOINT, client=client).rerank("query", _pool(2), 2)

        # A decision is ranks plus usage: reranking never rewrites or re-carries text.
        assert decision.selected_ranks == (2, 1)
        assert decision.discarded_rank_count == 0
        assert decision.used_fallback is False

    _run(scenario())


# --- native Memory search integration ---------------------------------------------------


def test_native_search_keeps_entry_version_identity_and_reports_actual_requests() -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        # Score the later coarse candidates higher so rerank visibly reorders the result.
        return httpx.Response(
            200,
            json={"scores": [{"rank": item["rank"], "score": float(item["rank"])} for item in body["candidates"]]},
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reranker = HTTPMemoryReranker(ENDPOINT, client=client, batch_size=1, policy_id="http-test-v1")
            async with open_builtin_contexts(
                BuiltinConfig(database=SQLiteConfig()), memory_reranker=reranker
            ) as contexts:
                service = (await contexts.get("http-rerank-native-identity")).artifacts.memory
                memory = await service.remember(
                    memory=None,
                    entries=(
                        MemoryEntryInput(kind="fact", text="Project alpha fact."),
                        MemoryEntryInput(kind="fact", text="Project beta fact."),
                    ),
                    mode="append",
                )
                assert memory is not None
                calls.clear()
                result = await service.search("project", memories=(memory,), limit=1, mode="fts")

        assert result.rerank is not None
        assert result.rerank.policy_id == "http-test-v1"
        assert result.generation_calls == 1
        # Usage is the real number of HTTP requests this search issued, not a guess.
        assert result.rerank.usage.requests == len(calls) == len(result.rerank.candidate_hits)

        selected = result.rerank.selected_ranks
        assert len(result.hits) == 1
        chosen = result.rerank.candidate_hits[selected[0] - 1]
        # The hit is the very candidate the rank named: same entry, same version, same text.
        assert result.hits[0].entry_version_id == chosen.entry_version_id
        assert result.hits[0].entry_id == chosen.entry_id
        assert result.hits[0].text == chosen.text
        assert result.hits[0].memory_ref == chosen.memory_ref

    _run(scenario())


def test_native_search_without_hits_never_calls_the_service() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"scores": []})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reranker = HTTPMemoryReranker(ENDPOINT, client=client)
            async with open_builtin_contexts(
                BuiltinConfig(database=SQLiteConfig()), memory_reranker=reranker
            ) as contexts:
                service = (await contexts.get("http-rerank-native-empty")).artifacts.memory
                memory = await service.remember(
                    memory=None,
                    entries=(MemoryEntryInput(kind="fact", text="Project alpha fact."),),
                    mode="append",
                )
                assert memory is not None
                result = await service.search("nothing matches this", memories=(memory,), limit=1, mode="fts")

        # An empty coarse pool never becomes a rerank call, so no request is billed.
        assert result.hits == ()
        assert result.rerank is None
        assert calls == []

    _run(scenario())
