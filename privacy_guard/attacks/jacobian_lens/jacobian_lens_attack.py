# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Experimental Jacobian-lens preview as a PrivacyGuard attack.

``JacobianLensAttack`` runs one prompt through a model that satisfies the
``LensModel`` protocol (residual capture, J-lens and logit-lens readouts, the
native greedy continuation, and optional concept-edit arms) and returns a
``JacobianLensPreviewInput`` holding a validated ``privacyguard.jlens_preview/v1``
receipt, ready for ``JacobianLensPreviewNode``. Fitting a lens is a separate,
long-running step (``lens_fitting.fit``); this attack only applies one.
"""

import datetime
import json
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import torch
from privacy_guard.analysis.jacobian_lens.calibration_corpus import sha256_text
from privacy_guard.analysis.jacobian_lens.lens_fitting import JacobianLens, LensModel
from privacy_guard.analysis.jacobian_lens.lens_intervention import (
    Arm,
    assess,
    LayerDirections,
    run_arms,
)
from privacy_guard.analysis.jacobian_lens.lens_readout import (
    capture_residuals,
    greedy_continuation,
    head_parity,
    rank_scores,
    readout_cells,
)
from privacy_guard.analysis.jacobian_lens.preview_analysis_input import (
    JacobianLensPreviewInput,
)
from privacy_guard.analysis.jacobian_lens.preview_schema import JLensPreviewReceipt
from privacy_guard.attacks.base_attack import BaseAttack

logger: logging.Logger = logging.getLogger(__name__)

SCHEMA_VERSION = "privacyguard.jlens_preview/v1"


@dataclass(frozen=True)
class LensPrompt:
    """The prompt text, its label in the receipt and a provenance note."""

    label: str
    text: str
    note: str = ""


@dataclass(frozen=True)
class ReceiptProvenance:
    """Receipt fields that describe the run rather than come out of the model."""

    model_identity: Mapping[str, Any]
    code: Mapping[str, Any]
    score_definition: str
    run_kind: str = "real_model"
    model_role: str = "target"
    lens: Mapping[str, Any] | None = None
    actual_output_scores: str = "native forward logits"


@dataclass(frozen=True)
class ConceptEdit:
    """One set of concept-edit arms on the prompt, all sharing ``directions``.

    A set that contains the primary arm is judged by the predeclared
    ``SUCCESS_RULE``; any other set is reported per arm (answer flips only) and
    enters the receipt only next to a set that holds the primary arm.
    """

    concept_from: str
    concept_to: str
    from_answer: str
    to_answer: str
    directions: Mapping[int, LayerDirections]
    arms: Sequence[Arm]
    answer_forms: Mapping[str, Sequence[int]]
    contrast_set_sha256: str
    downstream_layers: Sequence[int]
    note: str = ""

    @property
    def has_primary(self) -> bool:
        return any(arm.primary for arm in self.arms)


@dataclass
class EditResult:
    """Arm results of one ``ConceptEdit`` and its verdict or flip summary."""

    edit: ConceptEdit
    arms: list[dict[str, Any]]
    assessment: dict[str, Any] | None = None
    flips: dict[str, dict[str, Any]] = field(default_factory=dict)


def flip_summary(
    arms: Sequence[Mapping[str, Any]], from_answer: str, to_answer: str
) -> dict[str, dict[str, Any]]:
    """Per arm: whether the best ``to_answer`` form outranks the best
    ``from_answer`` form at the next token and after a leading whitespace token,
    the greedy answer, and both answer group probabilities."""

    def flip(summary: Mapping[str, Any] | None) -> bool | None:
        if summary is None:
            return None
        return summary[to_answer]["best_rank"] < summary[from_answer]["best_rank"]

    return {
        a["name"]: {
            "next_token_flip": flip(a["answers"]),
            "flip_after_whitespace": flip(a["answers_after_whitespace"]),
            "greedy_answer": a["greedy_answer"],
            "p_from": a["answers"][from_answer]["group_prob"],
            "p_to": a["answers"][to_answer]["group_prob"],
        }
        for a in arms
    }


def intervention_run(
    arm: Mapping[str, Any], edit: ConceptEdit, note: str = ""
) -> dict[str, Any]:
    """Map one arm result onto the receipt's ``InterventionRun`` fields."""
    ranks = ", ".join(
        f"best {k}-form rank {v['best_rank']}" for k, v in arm["answers"].items()
    )
    diff = arm["max_abs_logit_diff_vs_baseline"]
    return {
        "name": arm["name"],
        "kind": arm["kind"],
        "method_label": arm["method_label"],
        "status": arm["status"],
        "layers": arm["layers"],
        "positions": arm["positions"],
        "primary": arm["primary"],
        "strength": arm["strength"],
        "delta_norm": arm["delta_norm"],
        "direction": {
            "concept_from": edit.concept_from,
            "concept_to": edit.concept_to,
            "contrast_set_sha256": edit.contrast_set_sha256,
            "control": arm["control"] or None,
            "restore": arm["restore"] or None,
        },
        "next_token_top": arm["next_token_top"],
        "next_token_tracked": arm["next_token_tracked"],
        "greedy_text": arm["greedy_text"],
        "notes": f"{note}{edit.note}{ranks}; greedy answer {arm['greedy_answer']}; "
        f"max |logit diff| vs baseline {'n/a' if diff is None else format(diff, '.4g')}",
        "readout_cells": arm["readout_cells"],
    }


def _verdict_banner(verdict: Mapping[str, Any]) -> str:
    outcome = "POSITIVE" if verdict["success"] else "NEGATIVE"
    return (
        f"SUCCESS_RULE (predeclared primary): {outcome}; answer flipped "
        f"{verdict['primary_answer_flipped']}, downstream representation moved "
        f"{verdict['representation_moved']}, controls keep baseline "
        f"{verdict['controls_keep_baseline']}; "
    )


class JacobianLensAttack(BaseAttack):
    """
    Layer-by-token J-lens preview of one prompt on a ``LensModel``.

    args:
        model: residual-stream adapter (``LensModel``), e.g. ``GlimmerAdapter``
        prompt: the prompt, rendered as raw completion text
        encode_ids: text -> token IDs including any BOS the run prepends
        decode: token IDs -> text
        tracked_tokens: receipt tracked tokens ``{concept, token_id, text}``
        provenance: model identity, code record, lens record and labels
        native_logits: ``[1, T]`` IDs -> ``[1, T, V]`` native logits; defaults
            to ``model.unembed(model.forward(ids))``
        head: decodes residuals for cells (default ``model.unembed``); pass a
            higher-precision head with the same ranking to resolve ties
        lens: a fitted ``JacobianLens``; without it J-lens cells are unavailable
        bos_id: the BOS ID ``encode_ids`` prepends, recorded in the receipt
        device: device for input IDs (default: ``model.device`` when the adapter
            has one, else the device of the first block)
        top_k: top-k entries per cell (<= 100)
        max_new_tokens / stop_ids: greedy continuation bounds
        concept_edits: optional concept-edit arm sets (``ConceptEdit``)
        output_file: optional path for the receipt JSON
    """

    def __init__(
        self,
        model: LensModel,
        *,
        prompt: LensPrompt,
        encode_ids: Callable[[str], list[int]],
        decode: Callable[[Sequence[int]], str],
        tracked_tokens: Sequence[Mapping[str, Any]],
        provenance: ReceiptProvenance,
        native_logits: Callable[[torch.Tensor], torch.Tensor] | None = None,
        head: Callable[[torch.Tensor], torch.Tensor] | None = None,
        lens: JacobianLens | None = None,
        bos_id: int | None = None,
        device: torch.device | str | None = None,
        top_k: int = 20,
        max_new_tokens: int = 8,
        stop_ids: Sequence[int] = (),
        concept_edits: Sequence[ConceptEdit] = (),
        output_file: str | None = None,
    ) -> None:
        if sum(edit.has_primary for edit in concept_edits) > 1:
            raise ValueError("at most one concept edit may hold the primary arm")
        if lens is not None and provenance.lens is None:
            raise ValueError("a lens needs its lens record in provenance.lens")
        self.model = model
        self.prompt = prompt
        self.encode_ids = encode_ids
        self.decode = decode
        self.tracked_tokens: list[dict[str, Any]] = [dict(t) for t in tracked_tokens]
        self.tracked_ids: list[int] = [int(t["token_id"]) for t in tracked_tokens]
        self.provenance = provenance
        self.native_logits: Callable[[torch.Tensor], torch.Tensor] = (
            native_logits or self._default_logits
        )
        self.head: Callable[[torch.Tensor], torch.Tensor] = head or model.unembed
        self.lens = lens
        self.bos_id = bos_id
        self.device: torch.device = torch.device(
            device
            or getattr(model, "device", None)
            or next(model.layers[0].parameters()).device
        )
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.stop_ids: tuple[int, ...] = tuple(stop_ids)
        self.concept_edits: tuple[ConceptEdit, ...] = tuple(concept_edits)
        self.output_file = output_file
        self.edit_results: list[EditResult] = []

    def _default_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        residual = cast(torch.Tensor, self.model.forward(input_ids))
        return self.model.unembed(residual)

    def _prompt_record(self, ids: list[int]) -> dict[str, Any]:
        record: dict[str, Any] = {
            "text": self.prompt.text,
            "sha256": sha256_text(self.prompt.text),
            "rendering": "raw_completion",
            "token_ids": ids,
            "token_texts": [self.decode([i]) for i in ids],
            "label": self.prompt.label,
            "note": self.prompt.note,
        }
        if self.bos_id is not None:
            record["bos_token_id"] = self.bos_id
        return record

    def _arm_kwargs(self, downstream: Sequence[int]) -> dict[str, Any]:
        return {
            "readout_layers": list(range(self.model.n_layers)),
            "tracked_ids": self.tracked_ids,
            "decode": self.decode,
            "lens": self.lens,
            "head": self.head,
            "top_k": self.top_k,
            "max_new_tokens": self.max_new_tokens,
            "stop_ids": self.stop_ids,
            "cell_layers": list(downstream),
        }

    def _run_edit(self, edit: ConceptEdit, x: torch.Tensor) -> EditResult:
        clean = capture_residuals(self.model, x, list(range(self.model.n_layers)))
        arms = run_arms(
            self.model,
            self.native_logits,
            x,
            edit.arms,
            edit.directions,
            clean,
            answer_forms={k: list(v) for k, v in edit.answer_forms.items()},
            **self._arm_kwargs(edit.downstream_layers),
        )
        result = EditResult(
            edit, arms, flips=flip_summary(arms, edit.from_answer, edit.to_answer)
        )
        if edit.has_primary:
            result.assessment = assess(
                arms, edit.from_answer, edit.to_answer, edit.downstream_layers
            )
            if not result.assessment["hooks_effective"]:
                raise RuntimeError(
                    "primary edit left the native logits unchanged; hooks did not fire"
                )
        return result

    def _interventions(self) -> list[dict[str, Any]]:
        """Receipt runs. A receipt's interventions must include the predeclared
        primary arm, so edits on a prompt without one stay in ``edit_results``."""
        if not any(edit.has_primary for edit in self.concept_edits):
            return []
        runs = []
        for result in self.edit_results:
            banner = _verdict_banner(result.assessment) if result.assessment else ""
            runs += [
                intervention_run(a, result.edit, banner if a["primary"] else "")
                for a in result.arms
                if result.edit.has_primary or a["kind"] != "baseline"
            ]
        return runs

    def run_attack(self) -> JacobianLensPreviewInput:
        ids = self.encode_ids(self.prompt.text)
        x = torch.tensor([ids], device=self.device)
        logger.info("J-lens preview of %r (%d tokens)", self.prompt.label, len(ids))
        self.edit_results = [self._run_edit(edit, x) for edit in self.concept_edits]
        residuals = capture_residuals(self.model, x, list(range(self.model.n_layers)))
        cells = readout_cells(
            self.model,
            self.lens,
            residuals,
            positions=list(range(len(ids))),
            tracked_ids=self.tracked_ids,
            top_k=self.top_k,
            decode=self.decode,
            head=self.head,
        )
        greedy, first = greedy_continuation(
            self.native_logits,
            x,
            max_new_tokens=self.max_new_tokens,
            stop_ids=self.stop_ids,
        )
        top, tracked = rank_scores(first, self.tracked_ids, self.top_k, self.decode)
        provenance = self.provenance
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "run_kind": provenance.run_kind,
            "model_role": provenance.model_role,
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "prompt": self._prompt_record(ids),
            "model_identity": dict(provenance.model_identity),
            "code": dict(provenance.code),
            "num_layers": self.model.n_layers,
            "vocab_size": int(first.shape[-1]),
            "top_k": self.top_k,
            "score_definition": provenance.score_definition,
            "tracked_tokens": self.tracked_tokens,
            "cells": cells,
            "lens": dict(provenance.lens) if provenance.lens else None,
            "interventions": self._interventions(),
            "actual_output": {
                "greedy_ids": greedy,
                "greedy_text": self.decode(greedy),
                "max_new_tokens": self.max_new_tokens,
                "next_token_top": top,
                "next_token_tracked": tracked,
            },
            "unavailable": []
            if self.lens
            else [{"scope": "jlens", "reason": "no fitted lens supplied to this run"}],
            "display": {
                "actual_output_scores": provenance.actual_output_scores,
                "head_parity": head_parity(self.model, self.native_logits, x),
            },
        }
        validated = JLensPreviewReceipt.from_dict(receipt).with_hash()
        if self.output_file is not None:
            directory = os.path.dirname(self.output_file)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self.output_file, "w") as handle:
                json.dump(validated.to_dict(), handle, indent=1, sort_keys=True)
            logger.info("wrote receipt %s", self.output_file)
        return JacobianLensPreviewInput(validated)
