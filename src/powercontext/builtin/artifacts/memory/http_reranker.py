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

"""Cross-encoder Memory reranking through a caller-supplied HTTP scoring service.

The service scores one query against a bounded batch of candidates and answers with the
original candidate ranks. Because the service caps how many candidates one request may
carry, this adapter splits the coarse pool into batches, sends every global rank exactly
once, and then performs a single global selection over the union of all batch scores.

Two invariants drive the implementation:

* A rank is the candidate's position in the *complete* coarse pool. Batching never
  renumbers, so the score a batch returns is always attributed to the candidate that
  rank named on the way out.
* The transport is owned by the caller. Non-2xx responses and network failures are
  httpx exceptions and must stay httpx exceptions; only a well-formed HTTP answer whose
  JSON body or schema violates this protocol becomes a ``ValueError``.
"""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import (
    MemoryRerankDecision,
    validate_rerank_bounds,
)
from powercontext.builtin.inference import InferenceUsage

DEFAULT_HTTP_RERANK_POLICY_ID = "http-cross-encoder-v1"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HTTPRerankScore(_StrictModel):
    """One service score attributed to a candidate's global rank."""

    rank: int = Field(ge=1, strict=True)
    score: float = Field(allow_inf_nan=False)

    @field_validator("score", mode="before")
    @classmethod
    def require_json_number(cls, value: object) -> object:
        """Reject booleans and strings: a score is a JSON number, not a coercion target."""

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("http rerank score must be a JSON number")  # noqa: TRY003, TRY004
        return value


class HTTPRerankResponse(_StrictModel):
    """The exact body one scoring request may answer with."""

    scores: tuple[HTTPRerankScore, ...]


class HTTPMemoryReranker:
    """Score coarse Memory candidates through an HTTP cross-encoder and select globally."""

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = DEFAULT_HTTP_RERANK_POLICY_ID,
    ) -> None:
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("http rerank endpoint must be a non-empty string")  # noqa: TRY003
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("http rerank batch_size must be a positive integer")  # noqa: TRY003
        if not isinstance(policy_id, str) or not policy_id.strip():
            raise ValueError("http rerank policy_id must be a non-empty string")  # noqa: TRY003
        self.policy_id = policy_id
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Score every candidate once, then select the globally best ``limit`` ranks."""

        validate_rerank_bounds(len(candidates), limit)
        scores: dict[int, float] = {}
        requests = 0
        for start in range(0, len(candidates), self._batch_size):
            ranks = tuple(range(start + 1, min(start + self._batch_size, len(candidates)) + 1))
            batch = tuple(candidates[rank - 1] for rank in ranks)
            scores.update(await self._score_batch(query, ranks, batch))
            requests += 1
        return MemoryRerankDecision(
            selected_ranks=_select_global_ranks(scores, limit),
            usage=InferenceUsage(requests=requests),
        )

    async def _score_batch(
        self,
        query: str,
        ranks: tuple[int, ...],
        batch: tuple[MemoryHit, ...],
    ) -> dict[int, float]:
        """Send one batch and return its scores keyed by the original global ranks."""

        payload = {
            "query": query,
            "candidates": [
                {"rank": rank, "text": candidate.text} for rank, candidate in zip(ranks, batch, strict=True)
            ],
        }
        response = await self._client.post(self._endpoint, json=payload)
        # Non-2xx stays an httpx.HTTPStatusError: the transport failed, the protocol did not.
        response.raise_for_status()
        return _validated_scores(_decoded_body(response), expected_ranks=frozenset(ranks))


def _decoded_body(response: httpx.Response) -> Any:
    """Decode a 2xx body, treating malformed JSON as a protocol violation."""

    try:
        return response.json()
    except ValueError as error:
        raise ValueError("http rerank response is not valid JSON") from error  # noqa: TRY003


def _validated_scores(body: Any, *, expected_ranks: frozenset[int]) -> dict[int, float]:
    """Validate that one batch response covers exactly this batch's ranks, once each."""

    try:
        parsed = HTTPRerankResponse.model_validate(body)
    except ValidationError as error:
        raise ValueError("http rerank response does not match the scores schema") from error  # noqa: TRY003
    scores: dict[int, float] = {}
    for entry in parsed.scores:
        if entry.rank not in expected_ranks:
            raise ValueError("http rerank response returned a rank outside the requested batch")  # noqa: TRY003
        if entry.rank in scores:
            raise ValueError("http rerank response returned the same rank more than once")  # noqa: TRY003
        scores[entry.rank] = entry.score
    if expected_ranks.difference(scores):
        raise ValueError("http rerank response omitted a rank from the requested batch")  # noqa: TRY003
    return scores


def _select_global_ranks(scores: dict[int, float], limit: int) -> tuple[int, ...]:
    """Select the top ``limit`` ranks by score, breaking ties on the original rank."""

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return tuple(rank for rank, _score in ordered[:limit])


__all__ = [
    "DEFAULT_HTTP_RERANK_POLICY_ID",
    "HTTPMemoryReranker",
    "HTTPRerankResponse",
    "HTTPRerankScore",
]
