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
"""Layer x position readout for the experimental Jacobian lens.

Each cell decodes a post-block residual through the model's own native head,
either after the fitted ``J_l`` transport (J-lens) or directly (logit lens),
and ranks the complete vocabulary with :mod:`preview_ranking`, so cells match
the ``privacyguard.jlens_preview/v1`` receipt. Only top-k entries and the
full-vocabulary ranks of tracked token IDs leave this module; residuals and
full logits stay with the caller.
"""

from __future__ import annotations

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
from privacy_guard.analysis.jacobian_lens.preview_schema import CellReadout

Decode = Callable[[Sequence[int]], str]


@dataclass(frozen=True)
class TrackedToken:
    """One surface form of a tracked concept; multi-token forms track their
    first token and say so."""

    concept: str
    surface: str
    token_id: int
    token_ids: tuple[int, ...]
    single_token: bool


def resolve_tracked(
    encode_no_special: Callable[[str], Sequence[int]],
    concepts: Mapping[str, Sequence[str]],
) -> list[TrackedToken]:
    """Tokenize every surface form (no special tokens) of every concept."""
    tracked: list[TrackedToken] = []
    for concept, surfaces in concepts.items():
        for surface in surfaces:
            ids = tuple(int(i) for i in encode_no_special(surface))
            if not ids:
                raise ValueError(f"surface {surface!r} encodes to no tokens")
            tracked.append(TrackedToken(concept, surface, ids[0], ids, len(ids) == 1))
    return tracked


def full_vocab_ranks(scores: torch.Tensor, token_ids: Sequence[int]) -> list[int]:
    """1-based ranks over the whole vocabulary; equal scores rank by ascending ID."""
    if scores.ndim != 1 or not bool(torch.isfinite(scores).all()):
        raise ValueError("expected finite [V] scores")
    ids = torch.arange(scores.numel(), device=scores.device)
    ranks = []
    for token in token_ids:
        value = scores[token]
        ahead = (scores > value) | ((scores == value) & (ids < token))
        ranks.append(int(ahead.sum()) + 1)
    return ranks


def rank_scores(
    scores: torch.Tensor, tracked_ids: Sequence[int], top_k: int, decode: Decode
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Top-k and tracked ``{token_id, text, score, rank, ties}`` entries ranked by
    :func:`preview_ranking.rank_scores`; ``top_k=0`` returns tracked entries only."""
    unique = list(dict.fromkeys(int(t) for t in tracked_ids))
    top, tracked = preview_ranking.rank_scores(
        scores.float(), unique, max(top_k, 1), lambda t: decode([t])
    )
    return [asdict(t) for t in top[:top_k]], [asdict(t) for t in tracked]


@torch.no_grad()
def capture_residuals(
    model: LensModel,
    input_ids: torch.Tensor,
    layers: Sequence[int],
    *,
    edits: Mapping[int, ResidualEdit] | None = None,
) -> dict[int, torch.Tensor]:
    """Post-block residuals ``{layer: [T, d_model]}`` of one ``[1, T]`` input."""
    with ResidualHooks(model.layers, record=layers, edits=edits) as hooks:
        model.forward(input_ids)
    return {k: hooks.activations[k][0].detach() for k in layers}


def native_head(
    model: LensModel, dtype: torch.dtype
) -> Callable[[torch.Tensor], torch.Tensor]:
    """``model.unembed`` on inputs cast to the residual dtype, so fp32 J-lens
    transports reach a low-precision native head in its own dtype."""
    return lambda residual: model.unembed(residual.to(dtype))


@torch.no_grad()
def readout_cells(
    model: LensModel,
    lens: JacobianLens | None,
    residuals: Mapping[int, torch.Tensor],
    *,
    positions: Sequence[int],
    tracked_ids: Sequence[int],
    top_k: int,
    decode: Decode,
    head: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> list[dict[str, Any]]:
    """J-lens and logit-lens cells for every recorded layer and position.

    ``head`` decodes residuals (default ``model.unembed``); pass a monotone
    higher-precision equivalent of the native head to avoid low-precision ties.
    The target layer (``n_layers - 1``) has the identity transport, so its
    J-lens cell is the head on the final residual.
    """
    target = model.n_layers - 1
    if lens is not None and lens.target_layer not in (None, target):
        raise ValueError(
            f"lens targets layer {lens.target_layer}; readout assumes the final layer"
        )
    dtype = next(iter(residuals.values())).dtype if residuals else torch.float32
    decode_head = head or native_head(model, dtype)
    unique = list(dict.fromkeys(int(t) for t in tracked_ids))
    cells: list[dict[str, Any]] = []
    for layer in sorted(residuals):
        h = residuals[layer][list(positions)]
        heads: dict[str, torch.Tensor | str] = {"logit_lens": decode_head(h).float()}
        if lens is None:
            heads["jlens"] = "no lens supplied"
        elif layer == target:
            heads["jlens"] = heads["logit_lens"]
        elif layer in lens.jacobians:
            heads["jlens"] = decode_head(lens.transport(h, layer)).float()
        else:
            heads["jlens"] = "layer not in the fitted lens"
        for method, scores in heads.items():
            for row, position in enumerate(positions):
                if isinstance(scores, str):
                    cell = CellReadout(
                        method,
                        layer,
                        position,
                        "unavailable",
                        unavailable_reason=scores,
                    )
                else:
                    cell = preview_ranking.cell_from_scores(
                        method,
                        layer,
                        position,
                        scores[row],
                        unique,
                        top_k,
                        lambda t: decode([t]),
                    )
                cells.append(asdict(cell))
    return cells


@torch.no_grad()
def greedy_continuation(
    native_logits: Callable[[torch.Tensor], torch.Tensor],
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    stop_ids: Sequence[int] = (),
) -> tuple[list[int], torch.Tensor]:
    """Greedy tokens from the model's native forward (no teacher forcing).

    ``native_logits`` maps ``[1, T]`` ids to ``[1, T, V]`` logits. Returns the
    new token IDs and the first step's next-token logits ``[V]``.
    """
    ids = input_ids
    new: list[int] = []
    first: torch.Tensor | None = None
    for _ in range(max_new_tokens):
        step = native_logits(ids)[0, -1].float()
        first = step if first is None else first
        token = int(torch.sort(step, descending=True, stable=True).indices[0])
        new.append(token)
        if token in stop_ids:
            break
        ids = torch.cat([ids, ids.new_tensor([[token]])], dim=1)
    if first is None:
        raise ValueError("max_new_tokens must be >= 1")
    return new, first


@torch.no_grad()
def head_parity(
    model: LensModel,
    native_logits: Callable[[torch.Tensor], torch.Tensor],
    input_ids: torch.Tensor,
) -> dict[str, Any]:
    """Compare ``unembed(final residual)`` from hooks with native forward logits."""
    final = capture_residuals(model, input_ids, [model.n_layers - 1])[
        model.n_layers - 1
    ]
    ours = model.unembed(final).float()
    native = native_logits(input_ids)[0].float().to(ours.device)
    top_ours = torch.sort(ours, dim=-1, descending=True, stable=True).indices[:, 0]
    top_native = torch.sort(native, dim=-1, descending=True, stable=True).indices[:, 0]
    return {
        "positions": int(ours.shape[0]),
        "max_abs_logit_diff": float((ours - native).abs().max()),
        "top1_agreement": int((top_ours == top_native).sum()),
    }


def tracked_to_dicts(tracked: Sequence[TrackedToken]) -> list[dict[str, Any]]:
    return [
        {**asdict(t), "token_ids": list(t.token_ids), "text": t.surface}
        for t in tracked
    ]
