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

"""Report the Context Pack transport cost of a prepared-context sequence.

Runs the real Runtime path: seed Sources, flush Memory, then per round capture any
scheduled turn, run plain `prepare` for the reference body, and run
`prepare_incremental` against the previous pack. Every delta is decoded and applied from
serialized bytes only, so the measurement fails loudly if reconstruction ever disagrees
with the plain body.

Reported totals follow the challenge's measurement contract:

* `raw_full_bytes` — sum of plain body UTF-8 lengths.
* `incremental_bytes` — first complete pack plus every delta, measured with
  `len(to_bytes())`, so JSON, Base64, digest, and protocol fields all count.
* `full_pack_bytes` — the same rounds sent as complete v1 packs, no reuse.
* `gzip_bytes` — `gzip.compress(compresslevel=9, mtime=0)` per round, no cross-request
  state.

The last two are observation controls only: the challenge does not require beating gzip,
and a negative saving is reported as-is.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import sys
from pathlib import Path

from powercontext.builtin.artifacts.memory import MemoryCandidateRequest, MemoryEntryInput
from powercontext.builtin.runtime import BuiltinConfig, CaptureSource, PrepareContextRequest, open_builtin_runtime
from powercontext.builtin.runtime.context_pack import (
    ContextPack,
    ContextPackDelta,
    apply_context_pack_delta,
)
from powercontext.builtin.scope import ScopeDraft
from powercontext.builtin.sources import ContentSource

SEED_ROUNDS = 8
DEFAULT_ROUNDS = 40


class _CandidatePipeline:
    async def extract(self, request: MemoryCandidateRequest, /) -> tuple[MemoryEntryInput, ...]:
        return tuple(
            MemoryEntryInput(kind="conversation-turn", text=source.content, sources=(source,))
            for source in request.sources
            if isinstance(source, ContentSource)
        )


class _ReconstructionError(RuntimeError):
    """A delta that cannot rebuild its body invalidates every total, so the run stops."""

    def __init__(self, round_index: int) -> None:
        self.round_index = round_index
        super().__init__(f"round {round_index}: reconstructed body differs from prepare")


async def measure(
    seed_turns: list[str],
    append_turns: list[str],
    appends: set[int],
    switches: dict[int, str],
    query: str,
    max_bytes: int,
    rounds: int,
) -> dict[str, object]:
    async with open_builtin_runtime(BuiltinConfig(), candidate_pipeline=_CandidatePipeline()) as runtime:
        scope = await runtime.scopes.create(
            ScopeDraft(title="Context Pack cost", summary="Transport cost", idempotency_key="q18-cost")
        )
        sources = runtime.sources.for_scope(scope.scope_id)
        memory = runtime.memory.for_scope(scope.scope_id)
        context = runtime.context.for_scope(scope.scope_id)
        for index, text in enumerate(seed_turns):
            await sources.capture(CaptureSource(source_id=f"turn-{index}", content=text, metadata={}))
        while (await memory.flush()).processed:
            pass

        base_pack: ContextPack | None = None
        raw_full_bytes = full_pack_bytes = gzip_bytes = incremental_bytes = 0
        distinct: set[str] = set()
        rows: list[dict[str, object]] = []
        current_query = query
        pending = list(append_turns)
        for round_index in range(rounds):
            current_query = switches.get(round_index, current_query)
            if round_index in appends and pending:
                await sources.capture(
                    CaptureSource(source_id=f"append-{round_index}", content=pending.pop(0), metadata={})
                )
                while (await memory.flush()).processed:
                    pass
            request = PrepareContextRequest(query=current_query, max_bytes=max_bytes)
            prepared = await context.prepare(request)
            body = prepared.content or ""
            raw_full_bytes += len(body.encode("utf-8"))
            full_pack = ContextPack.from_content(body)
            full_pack_bytes += len(full_pack.to_bytes())
            gzip_bytes += len(gzip.compress(body.encode("utf-8"), compresslevel=9, mtime=0))
            if base_pack is None:
                wire = full_pack.to_bytes()
                kind, prefix_items = "pack", None
            else:
                delta = ContextPackDelta.between(base_pack, full_pack)
                wire = delta.to_bytes()
                kind, prefix_items = "delta", delta.prefix_items
                rebuilt = apply_context_pack_delta(
                    ContextPack.from_bytes(base_pack.to_bytes()),
                    ContextPackDelta.from_bytes(wire),
                )
                if b"".join(rebuilt.items).decode("utf-8") != body:
                    raise _ReconstructionError(round_index)
            incremental_bytes += len(wire)
            base_pack = full_pack
            distinct.add(body)
            rows.append({
                "round": round_index,
                "kind": kind,
                "wire_bytes": len(wire),
                "prefix_items": prefix_items,
                "chunks": len(full_pack.items),
                "body_bytes": len(body.encode("utf-8")),
                "query": current_query,
            })
        return {
            "rounds": rounds,
            "seed_turns": len(seed_turns),
            "raw_full_bytes": raw_full_bytes,
            "incremental_bytes": incremental_bytes,
            "full_pack_bytes": full_pack_bytes,
            "gzip_bytes": gzip_bytes,
            "distinct_contexts": len(distinct),
            "empty_rounds": sum(1 for row in rows if row["body_bytes"] == 0),
            "savings_vs_raw": 1 - incremental_bytes / raw_full_bytes,
            "savings_vs_full_pack": 1 - incremental_bytes / full_pack_bytes,
            "savings_vs_gzip": 1 - incremental_bytes / gzip_bytes,
            "rows": rows,
        }


def load_workload(path: Path) -> tuple[list[str], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [turn["text"] for turn in payload["turns"]], payload["query"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workload", type=Path, default=None, help="public turns.json with real meeting turns")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--max-bytes", type=int, default=4096)
    parser.add_argument(
        "--appends",
        default="",
        help="comma separated round indexes that capture one extra meeting turn before preparing",
    )
    parser.add_argument("--switch", action="append", default=[], metavar="ROUND=QUERY")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    seed_turns: list[str] = []
    query = "remote"
    if args.workload is not None:
        real_turns, query = load_workload(args.workload)
        seed_turns.extend(real_turns)
    seed_turns.extend(_SUPPLEMENTARY_TURNS[: max(0, SEED_ROUNDS - len(seed_turns))])
    append_turns = _APPEND_TURNS
    appends = {int(value) for value in args.appends.split(",") if value.strip()}
    switches = {}
    for spec in args.switch:
        round_text, _, value = spec.partition("=")
        switches[int(round_text)] = value

    result = asyncio.run(measure(seed_turns, append_turns, appends, switches, query, args.max_bytes, args.rounds))
    rows = result.pop("rows")
    print(f"{'round':>5} {'kind':>6} {'wire':>6} {'prefix':>7} {'chunks':>7} {'body':>6}  query")
    for row in rows:
        prefix = "-" if row["prefix_items"] is None else str(row["prefix_items"])
        print(
            f"{row['round']:>5} {row['kind']:>6} {row['wire_bytes']:>6} {prefix:>7} "
            f"{row['chunks']:>7} {row['body_bytes']:>6}  {row['query']}"
        )
    print()
    for key, value in result.items():
        print(f"{key:>22} = {value:.4f}" if isinstance(value, float) else f"{key:>22} = {value}")
    print(f"{'threshold_met':>22} = {result['savings_vs_raw'] >= 0.45}")
    if args.json_out is not None:
        args.json_out.write_text(json.dumps({**result, "rows": rows}, indent=2), encoding="utf-8")
    return 0


# Meeting-style turns that keep the shared product vocabulary, so a query switch still
# matches and every round stays non-empty. Only the turns loaded from --workload are real
# QMSum transcript text; the rest are measurement-protocol inputs for this task.
_SUPPLEMENTARY_TURNS = [
    "Project Manager: Okay . What are your experiences with remote controls ? I mean I've got "
    "{disfmarker} we got um {disfmarker} we had three videos , a T_V_ and a sort of amp thing all set up .",
    "Project Manager: but that was quite good , the fact that you could {disfmarker} You didn't have "
    "six remote controls sitting in front of you .",
    "Marketing: Right . And the thing that matters most is that the remote is obvious . If a customer "
    "picks it up and cannot tell what to press , the whole product feels broken at the design review .",
    "Designer: We looked at a couple of competitor remotes last week . Their layouts were much simpler , "
    "fewer buttons and a clearer hierarchy , and we should show them at the review .",
    "Engineer: The prototype board is finished . The firmware still needs the remote pairing flow and the "
    "power management tuning before the design review .",
    "Marketing: We should also think about the packaging , because the remote is the first thing a customer "
    "sees when they open the box , and the review deck should show that .",
    "Designer: Agreed , and the remote should feel substantial . The weight matters more than people expect "
    "for perceived quality , so the colour and mass decisions belong in the review .",
    "Project Manager: So what do we think the timeline is for the industrial design review next month , and "
    "do we have the remote ready for that date ?",
]

_APPEND_TURNS = [
    "Engineer: The pairing flow on the remote now takes about four seconds , which is inside the target we "
    "agreed at the last design review .",
    "Marketing: Can we get a short video of the remote being set up for the review page without any manual ?",
    "Designer: The remote colour options are still open . We should decide between the two neutrals before "
    "the review so the mockups are consistent .",
    "Engineer: Battery life on the remote is roughly six months with normal use in our measurements , which "
    "is slightly better than the review target .",
    "Marketing: The remote is the main differentiator in our positioning , so the review messaging has to "
    "lead with it rather than the screen .",
    "Designer: The remote silhouette should read clearly from across the room , and the buttons need a "
    "tactile difference so people can find them without looking at the review .",
    "Engineer: We can do a small texture change on the remote without retooling the whole housing , so that "
    "is a cheap experiment for the design review .",
    "Project Manager: Good . Let us close the remote antenna question and the colour question before the "
    "design review , and keep everything else as it is .",
    "Marketing: The remote ergonomics study should include older users as well as the core demographic we "
    "tested last time , and it should be in the review pack .",
    "Designer: I will bring three remote shape mockups to the next session so we can compare them side by "
    "side at the review .",
    "Engineer: The remote firmware update path is already in the prototype and it survived the last "
    "regression run before the review .",
    "Project Manager: Any concerns about the remote supply chain before we commit to the design review date ?",
    "Marketing: The remote supplier lead time was eight weeks last quarter , so we should place the order "
    "early for the review .",
    "Designer: The remote packaging insert needs a revision for the new shape and the new colour options "
    "we agreed at the review .",
    "Engineer: The remote antenna placement is the remaining open item on the hardware side for the review .",
    "Project Manager: Let us close the remote antenna question before the review and then freeze the design .",
    "Marketing: The remote is the main differentiator in our positioning , so the launch page leads with it "
    "and the review deck follows .",
    "Designer: The remote silhouette should read clearly from across the room even at a glance , which is "
    "what the review asked for .",
    "Engineer: The remote firmware can report battery level over the pairing channel already , so the "
    "review can see it live .",
    "Project Manager: Good , that covers the remote for today . Let us book the design review follow up .",
    "Marketing: I will circulate the remote messaging draft for the review tomorrow morning .",
    "Designer: The remote mockups will be ready on Friday for the design review .",
    "Engineer: The remote pairing regression suite is green after the last colour change for the review .",
    "Project Manager: Thanks everyone , the remote plan is on track for the design review .",
    "Marketing: One last thing on the remote , the warranty terms need legal review before launch .",
    "Designer: The remote warranty follows the main unit , so no separate review is needed for it .",
    "Engineer: The remote field failure rate is below the threshold we agreed at the design review .",
    "Project Manager: Great , we are done with the remote review for this week .",
    "Marketing: I will send the remote launch checklist and the colour decision to the group before the next review .",
    "Designer: The remote tooling freeze is scheduled for the end of the month after the design review .",
    "Engineer: The remote certification test starts next Monday after the design review .",
    "Project Manager: Perfect , let us wrap up the remote discussion here and close the review .",
]


if __name__ == "__main__":
    sys.exit(main())
