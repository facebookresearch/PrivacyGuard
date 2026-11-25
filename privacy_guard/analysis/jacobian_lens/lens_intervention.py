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
"""Controlled concept-substitution experiment for the experimental J-lens preview.

Concept directions come from separate, number-free contrast sentences: the
post-block residual at the last token of a concept word, averaged over
templates (closed form, no optimizer, frozen weights). The primary edit is a
clamp that moves a residual's ``from`` coefficient onto the ``to`` concept
within ``span{mu_from - mu_pool, mu_to - mu_pool}``::

    delta = strength * relu(c_from - c_to) * (mu_to - mu_from)

At ``strength=1`` it maps the ``from`` class mean onto the ``to`` class mean
and is idempotent, so repeating it at every layer of a band does not compound.
It is labeled ``contrast_mean_clamp`` and is not the reference
``linear_probe_swap``. Controls run through the same hook path: a zero edit
(no-op), matched-norm unrelated and random directions, and restoration of
clean residuals. Answer-token directions are never used to build an edit.
"""

from __future__ import annotations

import functools
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
from privacy_guard.analysis.jacobian_lens import preview_ranking
from privacy_guard.analysis.jacobian_lens.lens_fitting import (
    JacobianLens,
    LensModel,
    ResidualEdit,
    ResidualHooks,
)
from privacy_guard.analysis.jacobian_lens.lens_readout import (
    Decode,
    full_vocab_ranks,
    greedy_continuation,
    native_head,
    rank_scores,
)

CONTRAST_TEMPLATES: tuple[str, ...] = (
    "I saw the {x} near the old barn this morning.",
    "There was a photo of the {x} on the kitchen wall.",
    "My sister wrote a short story about the {x}.",
    "The children watched the {x} for a long time.",
    "Look over there, the {x} is right by the door!",
    "The museum had a whole display about the {x}.",
    "She drew a careful picture of the {x} for school.",
    "Nobody expected to find the {x} in the garage.",
    "The documentary followed the {x} through the forest.",
    "He pointed at the {x} and laughed quietly.",
    "We talked about the {x} during dinner yesterday.",
    "In the story, the {x} was the hero of the village.",
)
CONCEPT_POOL: tuple[str, ...] = (
    "spider",
    "ant",
    "dog",
    "cat",
    "horse",
    "bird",
    "snake",
    "fish",
)
UNRELATED_PAIR: tuple[str, str] = ("violin", "trumpet")

SUCCESS_RULE = (
    "primary arm succeeds iff the baseline keeps the from-answer, (a) the edit moves "
    "the representation downstream of the edited band: at the final prompt position "
    "the to-concept coefficient exceeds the from-concept coefficient on every layer "
    "after the band, (b) the native next token flips: best to-answer form outranks "
    "best from-answer form, (c) no-op, matched-norm unrelated and random controls "
    "keep the baseline answer, and (d) the primary edit changes the logits at all"
)


@dataclass(frozen=True)
class ContrastExample:
    concept: str
    template_index: int
    position: int


def build_contrast_examples(
    encode: Callable[[str], Sequence[int]],
    concepts: Sequence[str],
    templates: Sequence[str] = CONTRAST_TEMPLATES,
) -> tuple[list[ContrastExample], list[dict[str, Any]]]:
    """Locate the last token of each concept word; skip (and report) a template
    whose prefix tokenization is not a prefix of the full tokenization."""
    examples: list[ContrastExample] = []
    skipped: list[dict[str, Any]] = []
    for t_index, template in enumerate(templates):
        head = template.split("{x}")[0]
        for concept in concepts:
            full = list(encode(template.format(x=concept)))
            prefix = list(encode(head + concept))
            if full[: len(prefix)] != prefix:
                skipped.append(
                    {
                        "concept": concept,
                        "template_index": t_index,
                        "reason": "token boundary differs from prefix",
                    }
                )
                continue
            examples.append(ContrastExample(concept, t_index, len(prefix) - 1))
    return examples, skipped


@torch.no_grad()
def concept_means(
    model: LensModel,
    encode_tensor: Callable[[str], torch.Tensor],
    examples: Sequence[ContrastExample],
    layers: Sequence[int],
    templates: Sequence[str] = CONTRAST_TEMPLATES,
) -> dict[str, dict[int, torch.Tensor]]:
    """``{concept: {layer: fp32 CPU [d_model]}}`` mean residual at the concept token."""
    sums: dict[str, dict[int, torch.Tensor]] = {}
    counts: dict[str, int] = {}
    for ex in examples:
        ids = encode_tensor(templates[ex.template_index].format(x=ex.concept))
        with ResidualHooks(model.layers, record=layers) as hooks:
            model.forward(ids)
        per = sums.setdefault(ex.concept, {})
        for layer in layers:
            vec = hooks.activations[layer][0, ex.position].float().cpu()
            per[layer] = per[layer] + vec if layer in per else vec.clone()
        counts[ex.concept] = counts.get(ex.concept, 0) + 1
    return {c: {k: v / counts[c] for k, v in per.items()} for c, per in sums.items()}


@dataclass
class LayerDirections:
    """Concept geometry at one layer (private tensors; only norms are reported)."""

    pool_mean: torch.Tensor
    mu_from: torch.Tensor
    mu_to: torch.Tensor
    unrelated: torch.Tensor
    random: torch.Tensor

    def coefficients(self, h: torch.Tensor) -> torch.Tensor:
        """Least-squares ``(c_from, c_to)`` of ``h - pool_mean`` in the concept span."""
        basis = torch.stack(
            [self.mu_from - self.pool_mean, self.mu_to - self.pool_mean], 1
        )
        basis = basis.to(h.device)
        work = h.to(torch.promote_types(h.dtype, torch.float32))
        basis = basis.to(work.dtype)
        gram = basis.T @ basis
        rhs = (work - self.pool_mean.to(work)) @ basis
        return torch.linalg.solve(gram, rhs.unsqueeze(-1)).squeeze(-1)

    def clamp_delta(self, h: torch.Tensor, strength: float) -> torch.Tensor:
        coeff = self.coefficients(h)
        gate = (coeff[..., 0] - coeff[..., 1]).clamp(min=0)[..., None]
        return strength * gate.to(h.dtype) * (self.mu_to - self.mu_from).to(h)


def _unit_orthogonal(vec: torch.Tensor, span: Sequence[torch.Tensor]) -> torch.Tensor:
    basis, _ = torch.linalg.qr(torch.stack(list(span), 1))
    out = vec - basis @ (basis.T @ vec)
    norm = out.norm()
    if not bool(torch.isfinite(norm)) or float(norm) <= 1e-6 * float(vec.norm()):
        raise ValueError("control direction lies in the concept span")
    return out / norm


def layer_directions(
    means: Mapping[str, Mapping[int, torch.Tensor]],
    *,
    concept_from: str,
    concept_to: str,
    pool: Sequence[str],
    unrelated: tuple[str, str],
    seed: int = 0,
) -> dict[int, LayerDirections]:
    """Per-layer directions; control directions are unit vectors orthogonal to
    the concept span (unrelated pair difference, seeded Gaussian)."""
    generator = torch.Generator().manual_seed(seed)
    out: dict[int, LayerDirections] = {}
    for layer in sorted(means[concept_from]):
        pool_mean = torch.stack([means[c][layer] for c in pool]).mean(0)
        mu_from, mu_to = means[concept_from][layer], means[concept_to][layer]
        span = [mu_from - pool_mean, mu_to - pool_mean]
        noise = torch.randn(pool_mean.shape, generator=generator, dtype=torch.float32)
        out[layer] = LayerDirections(
            pool_mean,
            mu_from,
            mu_to,
            _unit_orthogonal(
                means[unrelated[1]][layer] - means[unrelated[0]][layer], span
            ),
            _unit_orthogonal(noise, span),
        )
    return out


@dataclass(frozen=True)
class Arm:
    """One predeclared condition. ``kind`` follows the preview schema.

    ``norm_source`` names an earlier arm whose realized per-(layer, position)
    edit norms this matched-norm control reuses.
    """

    name: str
    kind: str
    method_label: str
    layers: tuple[int, ...]
    strength: float = 1.0
    control: str = ""
    restore: str = ""
    primary: bool = False
    norm_source: str = ""


def _edit_for(
    arm: Arm,
    layer: int,
    directions: LayerDirections,
    prompt_len: int,
    clean: Mapping[int, torch.Tensor],
    realized: dict[int, torch.Tensor],
    reuse: Mapping[int, torch.Tensor] | None,
    ratios: dict[int, torch.Tensor] | None = None,
) -> ResidualEdit:
    def edit(h: torch.Tensor) -> torch.Tensor:
        span = slice(1, prompt_len)
        work = torch.float64 if h.dtype == torch.float64 else torch.float32
        hp = h[:, span].to(work)
        if arm.method_label == "contrast_mean_add":
            delta = arm.strength * (directions.mu_to - directions.mu_from).to(hp)
            delta = delta.expand_as(hp)
        else:
            delta = directions.clamp_delta(hp, arm.strength)
        if arm.control:
            unit = getattr(directions, arm.control).to(hp)
            if reuse is not None:
                norms = reuse[layer].to(hp)[None, :, None]
            else:
                norms = delta.norm(dim=-1, keepdim=True)
            delta = norms * unit
        if arm.kind == "noop":
            delta = torch.zeros_like(hp)
        if layer not in realized:
            realized[layer] = delta[0].norm(dim=-1).detach().float().cpu()
            if ratios is not None:
                ratios[layer] = realized[layer] / hp[0].norm(dim=-1).float().cpu()
        out = h.clone()
        out[:, span] = (hp + delta).to(h.dtype)
        if arm.restore == "final_position":
            out[:, prompt_len - 1] = clean[layer][prompt_len - 1].to(h)
        if arm.restore == "all_positions" and layer == max(arm.layers):
            out[:, :prompt_len] = clean[layer][:prompt_len].to(h)
        return out

    return edit


def _answer_summary(
    logits: torch.Tensor, answer_forms: Mapping[str, Sequence[int]], decode: Decode
) -> dict[str, Any]:
    scores = logits.float()
    logp = torch.log_softmax(scores, dim=-1)
    groups = {}
    for name, forms in answer_forms.items():
        ids = list(dict.fromkeys(int(i) for i in forms))
        ranks = full_vocab_ranks(scores, ids)
        best = min(range(len(ids)), key=lambda i: ranks[i])
        value = scores[ids[best]]
        groups[name] = {
            "best_token_id": ids[best],
            "best_text": decode([ids[best]]),
            "best_rank": ranks[best],
            "best_logit": float(value),
            "best_ties": int((scores == value).sum()) - 1,
            "group_prob": float(logp[ids].exp().sum()),
        }
    return groups


def _decoded_rows(
    head: Callable[[torch.Tensor], torch.Tensor],
    lens: JacobianLens | None,
    vectors: Mapping[int, torch.Tensor],
    cell_layers: Sequence[int],
) -> dict[tuple[str, int], torch.Tensor]:
    """One batched head call: J-lens scores for every fitted layer and
    logit-lens scores for ``cell_layers``."""
    keys: list[tuple[str, int]] = []
    rows: list[torch.Tensor] = []
    for layer, h in vectors.items():
        if lens is not None and layer in lens.jacobians:
            keys.append(("jlens", layer))
            rows.append(lens.transport(h[None], layer)[0])
        if layer in cell_layers:
            keys.append(("logit_lens", layer))
            rows.append(h.float())
    if not rows:
        return {}
    # Under a multi-GPU device map each block's residual lives on its own device.
    device = rows[0].device
    scores = head(torch.stack([r.to(device) for r in rows])).float()
    return {key: scores[i] for i, key in enumerate(keys)}


def _representation(
    head: Callable[[torch.Tensor], torch.Tensor],
    lens: JacobianLens | None,
    directions: Mapping[int, LayerDirections],
    hooks: ResidualHooks,
    edited: Sequence[int],
    position: int,
    tracked_ids: Sequence[int],
    decode: Decode,
    cell_layers: Sequence[int] = (),
    top_k: int = 10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Concept coefficients (and J-lens tracked ranks) at ``position`` on every
    recorded layer, plus J-lens/logit-lens receipt cells on ``cell_layers``."""
    vectors = {
        k: hooks.activations[k][0, position].detach() for k in sorted(hooks.activations)
    }
    scores = _decoded_rows(head, lens, vectors, cell_layers)
    unique = list(dict.fromkeys(int(t) for t in tracked_ids))
    rows, cells = [], []
    for layer, h in vectors.items():
        coeff = directions[layer].coefficients(h[None])[0]
        row: dict[str, Any] = {
            "layer": layer,
            "edited": layer in edited,
            "coef_from": float(coeff[0]),
            "coef_to": float(coeff[1]),
        }
        if layer in hooks.pre_edit:
            pre = directions[layer].coefficients(
                hooks.pre_edit[layer][0, position][None]
            )
            row["pre_edit_coef_from"], row["pre_edit_coef_to"] = map(float, pre[0])
        if ("jlens", layer) in scores:
            row["jlens_tracked"] = rank_scores(
                scores[("jlens", layer)], unique, 0, decode
            )[1]
        rows.append(row)
        for method in ("jlens", "logit_lens"):
            if layer in cell_layers and (method, layer) in scores:
                cell = preview_ranking.cell_from_scores(
                    method,
                    layer,
                    position,
                    scores[(method, layer)],
                    unique,
                    top_k,
                    lambda t: decode([t]),
                )
                cells.append(asdict(cell))
    return rows, cells


def run_arm(
    model: LensModel,
    native_logits: Callable[[torch.Tensor], torch.Tensor],
    input_ids: torch.Tensor,
    arm: Arm,
    directions: Mapping[int, LayerDirections],
    clean: Mapping[int, torch.Tensor],
    *,
    readout_layers: Sequence[int],
    tracked_ids: Sequence[int],
    answer_forms: Mapping[str, Sequence[int]],
    decode: Decode,
    lens: JacobianLens | None = None,
    head: Callable[[torch.Tensor], torch.Tensor] | None = None,
    reuse_norms: Mapping[int, torch.Tensor] | None = None,
    top_k: int = 10,
    max_new_tokens: int = 6,
    stop_ids: Sequence[int] = (),
    cell_layers: Sequence[int] = (),
) -> dict[str, Any]:
    """Run one arm: native next token, greedy text, and the representation at
    the final prompt position on every readout layer (as run, plus pre-edit
    coefficients on edited layers). ``head`` ranks J-lens readouts (default
    ``model.unembed``); native logits rank the answer."""
    prompt_len = int(input_ids.shape[1])
    realized: dict[int, torch.Tensor] = {}
    ratios: dict[int, torch.Tensor] = {}
    edits = {
        k: _edit_for(
            arm, k, directions[k], prompt_len, clean, realized, reuse_norms, ratios
        )
        for k in arm.layers
    }
    record = sorted(set(readout_layers) & set(directions))
    with ResidualHooks(model.layers, record=record, edits=edits) as hooks:
        with torch.no_grad():
            first = native_logits(input_ids)[0, -1].float()
        representation, cells = _representation(
            head or native_head(model, next(iter(clean.values())).dtype),
            lens,
            directions,
            hooks,
            arm.layers,
            prompt_len - 1,
            tracked_ids,
            decode,
            cell_layers,
            top_k,
        )
        greedy, _ = greedy_continuation(
            native_logits, input_ids, max_new_tokens=max_new_tokens, stop_ids=stop_ids
        )
        after_space = None
        if len(greedy) > 1 and not decode([greedy[0]]).strip():
            extended = torch.cat([input_ids, input_ids.new_tensor([[greedy[0]]])], 1)
            with torch.no_grad():
                step = native_logits(extended)[0, -1].float()
            after_space = _answer_summary(step, answer_forms, decode)
    top, tracked = rank_scores(first, tracked_ids, top_k, decode)
    norms = [realized[k] for k in sorted(realized)]
    return {
        **asdict(arm),
        "layers": list(arm.layers),
        "positions": list(range(1, prompt_len)) if arm.layers else [],
        "status": "ok",
        "unavailable_reason": "",
        "delta_norm": float(torch.cat(norms).mean()) if norms else 0.0,
        "delta_norm_by_layer": {
            k: float(v.mean()) for k, v in sorted(realized.items())
        },
        "next_token_top": top,
        "next_token_tracked": tracked,
        "answers": _answer_summary(first, answer_forms, decode),
        "answers_after_whitespace": after_space,
        "greedy_ids": greedy,
        "greedy_text": decode(greedy),
        "max_new_tokens": max_new_tokens,
        "representation_final_position": representation,
        "readout_cells": cells,
        "max_abs_logit_diff_vs_baseline": None,
        "_first_logits": first,
        "edit_to_residual_ratio": _ratio_summary(realized, ratios),
        "greedy_answer": greedy_answer(decode(greedy)),
        "_delta_norms": realized,
    }


def _ratio_summary(
    norms: Mapping[int, torch.Tensor], ratios: Mapping[int, torch.Tensor]
) -> dict[str, Any]:
    """``||delta|| / ||h||`` over the (layer, position) pairs actually edited."""
    edited = [ratios[k][norms[k] > 0] for k in sorted(ratios)]
    values = torch.cat(edited) if edited else torch.zeros(0)
    return {
        "n_edited": int(values.numel()),
        "n_candidates": int(sum(n.numel() for n in norms.values())),
        "mean": float(values.mean()) if values.numel() else 0.0,
        "max": float(values.max()) if values.numel() else 0.0,
    }


_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


@functools.cache
def _number_pattern() -> re.Pattern[str]:
    return re.compile(r"\s*(\d+|" + "|".join(_NUMBER_WORDS) + r")\b", re.I)


def greedy_answer(text: str) -> str | None:
    """The number (digits or an English number word) the continuation opens
    with, after optional whitespace; ``None`` when it opens with anything else."""
    match = _number_pattern().match(text)
    if match is None:
        return None
    token = match.group(1).lower()
    return _NUMBER_WORDS.get(token, token)


def followup_arms(
    band: Sequence[int], tag: str, strengths: Sequence[float] = (1.0, 2.0)
) -> list[Arm]:
    """Post-primary follow-up arms for one (from, to) pair: clamp, matched-norm
    unrelated and random controls per strength, and both restorations at the
    largest strength. Declared after the primary result, never primary."""
    band_t = tuple(band)
    clamp = "contrast_mean_clamp"
    arms = [Arm(f"baseline_{tag}", "baseline", "none", (), 0.0)]
    for s in strengths:
        source = f"clamp_band_s{s:g}_{tag}"
        arms += [
            Arm(source, "intervention", clamp, band_t, s),
            Arm(
                f"unrelated_matched_norm_band_s{s:g}_{tag}",
                "matched_norm_unrelated",
                clamp,
                band_t,
                s,
                control="unrelated",
                norm_source=source,
            ),
            Arm(
                f"random_matched_norm_band_s{s:g}_{tag}",
                "matched_norm_unrelated",
                clamp,
                band_t,
                s,
                control="random",
                norm_source=source,
            ),
        ]
    top = max(strengths)
    arms += [
        Arm(
            f"restore_all_positions_band_s{top:g}_{tag}",
            "restoration",
            clamp,
            band_t,
            top,
            restore="all_positions",
        ),
        Arm(
            f"restore_final_position_band_s{top:g}_{tag}",
            "restoration",
            clamp,
            band_t,
            top,
            restore="final_position",
        ),
    ]
    return arms


def run_arms(
    model: LensModel,
    native_logits: Callable[[torch.Tensor], torch.Tensor],
    input_ids: torch.Tensor,
    arms: Sequence[Arm],
    directions: Mapping[int, LayerDirections],
    clean: Mapping[int, torch.Tensor],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Run arms in order, feeding each matched-norm control the realized norms
    of its ``norm_source`` arm, then :func:`finalize`."""
    realized: dict[str, Mapping[int, torch.Tensor]] = {}
    results = []
    for arm in arms:
        reuse = None
        if arm.norm_source:
            if arm.norm_source not in realized:
                raise ValueError(f"{arm.name} needs {arm.norm_source} to run first")
            reuse = realized[arm.norm_source]
        result = run_arm(
            model,
            native_logits,
            input_ids,
            arm,
            directions,
            clean,
            reuse_norms=reuse,
            **kwargs,
        )
        realized[arm.name] = result.pop("_delta_norms")
        results.append(result)
    return finalize(results)


def predeclared_arms(
    band: Sequence[int], single_layers: Sequence[int], strengths: Sequence[float]
) -> list[Arm]:
    """The full predeclared condition set; every arm is always reported. The
    primary arm is the strength-1 band clamp."""
    band_t = tuple(band)
    clamp = "contrast_mean_clamp"
    arms = [
        Arm("baseline", "baseline", "none", (), 0.0),
        Arm("noop_zero_edit", "noop", clamp, band_t, 0.0),
    ]
    for s in strengths:
        tag = f"s{s:g}"
        source = f"clamp_band_{tag}"
        arms += [
            Arm(source, "intervention", clamp, band_t, s, primary=s == 1.0),
            Arm(
                f"unrelated_matched_norm_band_{tag}",
                "matched_norm_unrelated",
                clamp,
                band_t,
                s,
                control="unrelated",
                norm_source=source,
            ),
            Arm(
                f"random_matched_norm_band_{tag}",
                "matched_norm_unrelated",
                clamp,
                band_t,
                s,
                control="random",
                norm_source=source,
            ),
        ]
    arms += [
        Arm(
            "restore_all_positions_band_s1",
            "restoration",
            clamp,
            band_t,
            1.0,
            restore="all_positions",
        ),
        Arm(
            "restore_final_position_band_s1",
            "restoration",
            clamp,
            band_t,
            1.0,
            restore="final_position",
        ),
    ]
    arms += [
        Arm(f"add_layer{k}_s1", "intervention", "contrast_mean_add", (k,), 1.0)
        for k in single_layers
    ]
    if sum(a.primary for a in arms) != 1:
        raise ValueError("strengths must include 1.0 exactly once")
    return arms


def assess(
    results: Sequence[Mapping[str, Any]],
    from_answer: str,
    to_answer: str,
    downstream_layers: Sequence[int],
) -> dict[str, Any]:
    """Apply :data:`SUCCESS_RULE` to the arm results (primary arm by flag).

    Every criterion reads one decoding step per arm: the next token, or the
    token after it when the continuation opens with whitespace. An arm's answer
    is the answer concept whose best form is top-1 there; ``None`` when none is.
    """
    by_name = {r["name"]: r for r in results}

    def read(r: Mapping[str, Any]) -> Mapping[str, Any]:
        return r.get("answers_after_whitespace") or r["answers"]

    def flips(r: Mapping[str, Any]) -> bool:
        a = read(r)
        return a[to_answer]["best_rank"] < a[from_answer]["best_rank"]

    def answer_of(r: Mapping[str, Any]) -> str | None:
        top = [name for name, group in read(r).items() if group["best_rank"] == 1]
        return top[0] if top else None

    def to_dominant(r: Mapping[str, Any]) -> list[int]:
        rows = {row["layer"]: row for row in r["representation_final_position"]}
        return [
            k
            for k in downstream_layers
            if k in rows and rows[k]["coef_to"] > rows[k]["coef_from"]
        ]

    (prim,) = [r for r in results if r["primary"]]
    moved = to_dominant(prim) == list(downstream_layers)
    controls = [r for r in results if r["kind"] in ("noop", "matched_norm_unrelated")]
    baseline_answer = answer_of(by_name["baseline"])
    clean_controls = all(answer_of(r) == baseline_answer for r in controls)
    baseline_ok = baseline_answer == from_answer
    hooks_effective = bool(prim["max_abs_logit_diff_vs_baseline"])
    return {
        "rule": SUCCESS_RULE,
        "primary_arm": prim["name"],
        "downstream_layers": list(downstream_layers),
        "baseline_keeps_from_answer": baseline_ok,
        "baseline_to_dominant_downstream_layers": to_dominant(by_name["baseline"]),
        "primary_to_dominant_downstream_layers": to_dominant(prim),
        "representation_moved": moved,
        "primary_answer_flipped": flips(prim),
        "controls_keep_baseline": clean_controls,
        "hooks_effective": hooks_effective,
        "success": baseline_ok
        and moved
        and flips(prim)
        and clean_controls
        and hooks_effective,
        "flipped_arms": sorted(r["name"] for r in results if flips(r)),
        "answers": {r["name"]: answer_of(r) for r in results},
    }


def finalize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Record each arm's max |logit diff| from baseline and drop private logits."""
    base = next(r for r in results if r["kind"] == "baseline")["_first_logits"]
    for r in results:
        diff = float((r["_first_logits"] - base).abs().max())
        r["max_abs_logit_diff_vs_baseline"] = diff if math.isfinite(diff) else None
    for r in results:
        r.pop("_first_logits")
    return results
