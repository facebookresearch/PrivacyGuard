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
Shared full-vocabulary ranking for J-lens and logit-lens readouts.

Rank is 1-based over the whole vocabulary; equal scores are ordered by
ascending token ID, so rank(i) = 1 + #{j: s_j > s_i} + #{j < i: s_j == s_i}.
"""

from collections.abc import Callable, Sequence

import torch
from privacy_guard.analysis.jacobian_lens.preview_schema import CellReadout, TokenScore


class UnavailableScoresError(ValueError):
    """Raised when scores cannot be ranked faithfully (e.g. NaN or inf)."""


def rank_scores(
    scores: torch.Tensor,
    tracked_ids: Sequence[int],
    top_k: int,
    decode: Callable[[int], str],
) -> tuple[list[TokenScore], list[TokenScore]]:
    """
    Rank one position's native-head scores over the full vocabulary.

    args:
        scores: 1-D tensor of length vocab_size, indexed by tokenizer ID, after
            the native head's final norm, output scaling and softcapping
        tracked_ids: token IDs whose full-vocabulary rank is always reported
        top_k: number of top-ranked entries to return
        decode: maps one token ID to its decoded text

    returns:
        (top entries with ranks 1..top_k, tracked entries in tracked_ids order)
    """
    if scores.dim() != 1:
        raise ValueError(f"scores must be 1-D [vocab], got shape {tuple(scores.shape)}")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    values = scores.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(values).all()):
        raise UnavailableScoresError("scores contain NaN or inf")
    vocab_size = values.numel()
    for token_id in tracked_ids:
        if not 0 <= token_id < vocab_size:
            raise ValueError(f"tracked token ID {token_id} is outside the vocabulary")
    # Stable descending sort keeps ascending token ID among equal scores.
    _, order = torch.sort(values, descending=True, stable=True)
    top = [
        _entry(values, int(token_id), rank, decode)
        for rank, token_id in enumerate(order[: min(top_k, vocab_size)].tolist(), 1)
    ]
    tracked = [
        _entry(values, token_id, _full_vocab_rank(values, token_id), decode)
        for token_id in tracked_ids
    ]
    return top, tracked


def _full_vocab_rank(values: torch.Tensor, token_id: int) -> int:
    score = values[token_id]
    higher = int((values > score).sum())
    tied_lower_ids = int((values[:token_id] == score).sum())
    return 1 + higher + tied_lower_ids


def _entry(
    values: torch.Tensor, token_id: int, rank: int, decode: Callable[[int], str]
) -> TokenScore:
    score = values[token_id]
    return TokenScore(
        token_id=token_id,
        text=decode(token_id),
        score=float(score),
        rank=rank,
        ties=int((values == score).sum()) - 1,
    )


def cell_from_scores(
    method: str,
    layer: int,
    position: int,
    scores: torch.Tensor,
    tracked_ids: Sequence[int],
    top_k: int,
    decode: Callable[[int], str],
) -> CellReadout:
    """
    Build one receipt cell from native-head scores; non-finite scores yield an
    unavailable cell with the reason instead of a fabricated ranking.
    """
    try:
        top, tracked = rank_scores(scores, tracked_ids, top_k, decode)
    except UnavailableScoresError as e:
        return CellReadout(
            method=method,
            layer=layer,
            position=position,
            status="unavailable",
            unavailable_reason=str(e),
        )
    return CellReadout(
        method=method,
        layer=layer,
        position=position,
        status="ok",
        top=tuple(top),
        tracked=tuple(tracked),
    )
