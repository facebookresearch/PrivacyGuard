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
Optional, experimental analysis node summarizing a Jacobian-lens preview receipt.

It reports what the receipt measured; it does not run a model. Rank curves,
head-consistency and answer margins are descriptive. They make no claim about
model intent, a global workspace, or privacy efficacy.
"""

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from privacy_guard.analysis.base_analysis_node import BaseAnalysisNode
from privacy_guard.analysis.base_analysis_output import BaseAnalysisOutput
from privacy_guard.analysis.jacobian_lens.preview_analysis_input import (
    JacobianLensPreviewInput,
)
from privacy_guard.analysis.jacobian_lens.preview_schema import (
    CellReadout,
    JLensPreviewReceipt,
    TokenScore,
)


logger: logging.Logger = logging.getLogger(__name__)

HEAD_SCORE_TOLERANCE = 1e-3


@dataclass
class JacobianLensPreviewNodeOutput(BaseAnalysisOutput):
    """
    Summary of a ``privacyguard.jlens_preview/v1`` receipt.

    rank_curves: per (method, concept, position), the best full-vocabulary
        rank over that concept's tracked token variants at each layer (None
        where the cell is unavailable)
    first_layer_within_top: per (method, concept, position), the first layer
        whose best rank is <= ``top_threshold`` (None if never)
    head_consistency: whether the final-layer logit lens at the last prompt
        position reproduces the actual next-token scores
    answer_margins: best score of the first answer concept minus the second
        (e.g. "6" minus "8") at the next token, for the actual output and each
        intervention run
    intervention_readouts: per run with readout cells, rank curves of the patched
        pass and per-concept best ranks before and after the patch
    labels: caveats that must accompany any display of this receipt
    """

    schema_version: str
    receipt_sha256: str
    run_kind: str
    model_role: str
    model_repo_id: str
    model_revision_sha: str
    lens_id: str | None
    num_cells: dict[str, dict[str, int]]
    rank_curves: list[dict[str, Any]]
    first_layer_within_top: list[dict[str, Any]]
    head_consistency: dict[str, Any]
    actual_next_token: dict[str, Any] | None
    actual_greedy_text: str | None
    answer_margins: list[dict[str, Any]]
    intervention_readouts: list[dict[str, Any]]
    unavailable: list[dict[str, str]]
    labels: list[str]


class JacobianLensPreviewNode(BaseAnalysisNode):
    """
    Summarizes a J-lens layer-by-token preview receipt.

    args:
        analysis_input: ``JacobianLensPreviewInput`` holding a validated receipt
        answer_concepts: (preferred, reference) tracked concepts compared at the
            next token, e.g. ("6", "8") for the spider -> ant experiment
        top_threshold: rank threshold for ``first_layer_within_top``
        head_score_tolerance: max absolute score difference allowed between the
            final-layer logit lens and the actual next-token scores
    """

    def __init__(
        self,
        analysis_input: JacobianLensPreviewInput,
        answer_concepts: tuple[str, str] = ("6", "8"),
        top_threshold: int = 10,
        head_score_tolerance: float = HEAD_SCORE_TOLERANCE,
    ) -> None:
        if top_threshold <= 0 or head_score_tolerance < 0:
            raise ValueError("top_threshold must be positive and tolerance >= 0")
        self._answer_concepts = answer_concepts
        self._top_threshold = top_threshold
        self._head_score_tolerance = head_score_tolerance
        super().__init__(analysis_input=analysis_input)

    @property
    def receipt(self) -> JLensPreviewReceipt:
        analysis_input = self.analysis_input
        if not isinstance(analysis_input, JacobianLensPreviewInput):
            raise TypeError("analysis_input must be a JacobianLensPreviewInput")
        return analysis_input.receipt

    def run_analysis(self) -> JacobianLensPreviewNodeOutput:
        receipt = self.receipt
        curves = _rank_curves(receipt, receipt.cells)
        margins = [
            self._margin(
                "actual_output",
                "actual",
                "native_forward",
                receipt.actual_output.next_token_tracked,
            )
            if receipt.actual_output is not None
            else None
        ] + [
            self._margin(
                run.name,
                run.kind,
                run.method_label,
                run.next_token_tracked,
                primary=run.primary,
            )
            for run in receipt.interventions
            if run.status == "ok"
        ]
        actual = receipt.actual_output
        return JacobianLensPreviewNodeOutput(
            schema_version=receipt.schema_version,
            receipt_sha256=receipt.content_sha256(),
            run_kind=receipt.run_kind,
            model_role=receipt.model_role,
            model_repo_id=str(receipt.model_identity["repo_id"]),
            model_revision_sha=str(receipt.model_identity["revision_sha"]),
            lens_id=None if receipt.lens is None else str(receipt.lens["lens_id"]),
            num_cells=_cell_counts(receipt.cells),
            rank_curves=curves,
            first_layer_within_top=self._first_layers(curves),
            head_consistency=_head_consistency(receipt, self._head_score_tolerance),
            actual_next_token=(
                None
                if actual is None
                else {
                    "token_id": actual.next_token_top[0].token_id,
                    "text": actual.next_token_top[0].text,
                }
            ),
            actual_greedy_text=None if actual is None else actual.greedy_text,
            answer_margins=[m for m in margins if m is not None],
            intervention_readouts=[
                {
                    "name": run.name,
                    "kind": run.kind,
                    "curves": _rank_curves(receipt, run.readout_cells),
                    "rank_changes": _rank_changes(receipt, run.readout_cells),
                }
                for run in receipt.interventions
                if run.readout_cells
            ],
            unavailable=[
                {"scope": u.scope, "reason": u.reason} for u in receipt.unavailable
            ],
            labels=_labels(receipt),
        )

    def _margin(
        self,
        name: str,
        kind: str,
        method_label: str,
        tracked: Sequence[TokenScore],
        primary: bool = False,
    ) -> dict[str, Any] | None:
        best = _best_by_concept(self.receipt, tracked)
        preferred, reference = self._answer_concepts
        if preferred not in best or reference not in best:
            return None
        return {
            "name": name,
            "kind": kind,
            "method_label": method_label,
            "primary": primary,
            "preferred_concept": preferred,
            "reference_concept": reference,
            "margin": best[preferred].score - best[reference].score,
            "preferred_rank": best[preferred].rank,
            "reference_rank": best[reference].rank,
        }

    def _first_layers(self, curves: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for curve in curves:
            hit = next(
                (
                    layer
                    for layer, rank in zip(curve["layers"], curve["best_rank"])
                    if rank is not None and rank <= self._top_threshold
                ),
                None,
            )
            result.append(
                {
                    "method": curve["method"],
                    "concept": curve["concept"],
                    "position": curve["position"],
                    "top_threshold": self._top_threshold,
                    "layer": hit,
                }
            )
        return result


def _best_by_concept(
    receipt: JLensPreviewReceipt, tracked: Iterable[TokenScore]
) -> dict[str, TokenScore]:
    concept_of = {t.token_id: t.concept for t in receipt.tracked_tokens}
    best: dict[str, TokenScore] = {}
    for entry in tracked:
        concept = concept_of[entry.token_id]
        if concept not in best or entry.rank < best[concept].rank:
            best[concept] = entry
    return best


def _rank_curves(
    receipt: JLensPreviewReceipt, cells: Sequence[CellReadout]
) -> list[dict[str, Any]]:
    concepts = sorted({t.concept for t in receipt.tracked_tokens})
    keys = sorted({(c.method, c.position) for c in cells})
    by_key = {(c.method, c.position, c.layer): c for c in cells}
    curves = []
    for method, position in keys:
        layers = sorted(
            layer for (m, p, layer) in by_key if m == method and p == position
        )
        for concept in concepts:
            ranks: list[int | None] = []
            token_ids: list[int | None] = []
            for layer in layers:
                cell = by_key[(method, position, layer)]
                best = _best_by_concept(receipt, cell.tracked).get(concept)
                ranks.append(None if best is None else best.rank)
                token_ids.append(None if best is None else best.token_id)
            curves.append(
                {
                    "method": method,
                    "concept": concept,
                    "position": position,
                    "layers": layers,
                    "best_rank": ranks,
                    "best_token_id": token_ids,
                }
            )
    return curves


def _rank_changes(
    receipt: JLensPreviewReceipt, patched_cells: Sequence[CellReadout]
) -> list[dict[str, Any]]:
    """Best rank per concept in each patched cell next to the unpatched cell."""
    baseline = {(c.method, c.layer, c.position): c for c in receipt.cells}
    changes = []
    for cell in patched_cells:
        if cell.status != "ok":
            continue
        reference = baseline.get((cell.method, cell.layer, cell.position))
        before = (
            _best_by_concept(receipt, reference.tracked)
            if reference is not None and reference.status == "ok"
            else {}
        )
        after = _best_by_concept(receipt, cell.tracked)
        for concept in sorted(after):
            changes.append(
                {
                    "method": cell.method,
                    "layer": cell.layer,
                    "position": cell.position,
                    "concept": concept,
                    "baseline_rank": before[concept].rank
                    if concept in before
                    else None,
                    "patched_rank": after[concept].rank,
                }
            )
    return changes


def _cell_counts(cells: Iterable[CellReadout]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for cell in cells:
        per_method = counts.setdefault(cell.method, {"ok": 0, "unavailable": 0})
        per_method[cell.status] += 1
    return counts


def _head_consistency(receipt: JLensPreviewReceipt, tolerance: float) -> dict[str, Any]:
    final_layer = receipt.num_layers - 1
    last_position = len(receipt.prompt.token_ids) - 1
    cell = next(
        (
            c
            for c in receipt.cells
            if c.method == "logit_lens"
            and c.layer == final_layer
            and c.position == last_position
        ),
        None,
    )
    actual = receipt.actual_output
    if cell is None or cell.status != "ok" or actual is None:
        return {
            "checked": False,
            "reason": "needs an available final-layer logit-lens cell at the last "
            "prompt position and the actual output",
        }
    actual_scores = {t.token_id: t.score for t in actual.next_token_top}
    diffs = [
        abs(t.score - actual_scores[t.token_id])
        for t in cell.top
        if t.token_id in actual_scores
    ]
    top1_match = cell.top[0].token_id == actual.next_token_top[0].token_id
    max_diff = max(diffs) if diffs else None
    consistent = top1_match and max_diff is not None and max_diff <= tolerance
    if not consistent:
        logger.warning("Final-layer logit lens does not reproduce the actual output")
    return {
        "checked": True,
        "top1_match": top1_match,
        "max_abs_score_diff": max_diff,
        "tolerance": tolerance,
        "consistent": consistent,
    }


def _labels(receipt: JLensPreviewReceipt) -> list[str]:
    labels = []
    if receipt.run_kind != "real_model":
        labels.append(
            f"{receipt.run_kind}: not a real-model result and not a Glimmer result"
        )
    if receipt.model_role == "independent_control":
        labels.append(
            f"independent control model {receipt.model_identity['repo_id']}: "
            "not a substitute for the Glimmer target"
        )
    if receipt.lens is None:
        labels.append("no fitted J-lens in this receipt; J-lens cells are unavailable")
    else:
        labels.append(
            "J-lens layers are candidate sites, not validated workspace bands"
        )
    if receipt.prompt.rendering != "raw_completion":
        labels.append(f"prompt rendered with {receipt.prompt.rendering}")
    primary = [r.name for r in receipt.interventions if r.primary]
    if primary:
        labels.append(
            f"predeclared primary intervention arm: {primary[0]}; other "
            "intervention arms are exploratory"
        )
    for run in receipt.interventions:
        if run.method_label == "linear_probe_swap":
            labels.append(
                f"intervention {run.name} uses a linear-probe swap, not a J-lens readout"
            )
    return labels
