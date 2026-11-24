# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Shrink a ``privacyguard.jlens_preview/v1`` receipt for sharing, e.g. as tutorial data.

Trimming keeps every cell, every tracked-token rank and every intervention run;
it only shortens the unfiltered top lists, rounds scores and drops patched-pass
readout cells of runs that are not kept. The trimmed receipt records the hash of
its source and what was removed, so a reader can tell it from the full receipt.
"""

import dataclasses
from collections.abc import Collection, Sequence

from privacy_guard.analysis.jacobian_lens.preview_schema import (
    CellReadout,
    InterventionRun,
    JLensPreviewReceipt,
    TokenScore,
)


def trim_receipt(
    receipt: JLensPreviewReceipt,
    top_k: int,
    score_decimals: int | None = None,
    keep_readout_runs: Collection[str] | None = None,
) -> JLensPreviewReceipt:
    """
    args:
        receipt: a validated receipt
        top_k: new top-list length (at most the receipt's ``top_k``)
        score_decimals: round every score to this many decimals (None keeps them)
        keep_readout_runs: names of runs whose ``readout_cells`` are kept; None
            keeps all, an empty collection drops all

    returns:
        a new, re-hashed receipt
    """
    if not 0 < top_k <= receipt.top_k:
        raise ValueError(f"top_k must be in 1..{receipt.top_k}")
    if score_decimals is not None and score_decimals < 0:
        raise ValueError("score_decimals must be >= 0")

    def scores(
        entries: Sequence[TokenScore], limit: int | None = None
    ) -> tuple[TokenScore, ...]:
        kept = entries if limit is None else entries[:limit]
        if score_decimals is None:
            return tuple(kept)
        return tuple(
            dataclasses.replace(e, score=round(e.score, score_decimals)) for e in kept
        )

    def cell(c: CellReadout) -> CellReadout:
        return dataclasses.replace(
            c, top=scores(c.top, top_k), tracked=scores(c.tracked)
        )

    def run(r: InterventionRun) -> InterventionRun:
        keep = keep_readout_runs is None or r.name in keep_readout_runs
        return dataclasses.replace(
            r,
            next_token_top=scores(r.next_token_top, top_k),
            next_token_tracked=scores(r.next_token_tracked),
            readout_cells=tuple(cell(c) for c in r.readout_cells) if keep else (),
        )

    actual = receipt.actual_output
    dropped = sorted(
        r.name
        for r in receipt.interventions
        if r.readout_cells
        and keep_readout_runs is not None
        and r.name not in keep_readout_runs
    )
    display = dict(receipt.display)
    display["trimmed"] = {
        "source_receipt_sha256": receipt.content_sha256(),
        "source_top_k": receipt.top_k,
        "top_k": top_k,
        "score_decimals": score_decimals,
        "readout_cells_dropped_for_runs": dropped,
    }
    trimmed = dataclasses.replace(
        receipt,
        top_k=top_k,
        cells=tuple(cell(c) for c in receipt.cells),
        actual_output=(
            None
            if actual is None
            else dataclasses.replace(
                actual,
                next_token_top=scores(actual.next_token_top, top_k),
                next_token_tracked=scores(actual.next_token_tracked),
            )
        ),
        interventions=tuple(run(r) for r in receipt.interventions),
        display=display,
        receipt_sha256=None,
    )
    return trimmed.with_hash()
