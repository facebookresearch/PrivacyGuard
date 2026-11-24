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
Synthetic ``cpu_fixture`` receipt for tests, the tutorial and viewer smoke checks.

Scores are seeded random numbers with fixed boosts; this is not a model result
and the receipt is labeled ``run_kind="cpu_fixture"``, ``model_role="fixture"``.
"""

from typing import Any

import torch
from privacy_guard.analysis.jacobian_lens.preview_ranking import (
    cell_from_scores,
    rank_scores,
)
from privacy_guard.analysis.jacobian_lens.preview_schema import (
    ActualOutput,
    InterventionRun,
    JLensPreviewReceipt,
    PromptRecord,
    sha256_text,
    TrackedToken,
    UnavailableReason,
)


VOCAB: tuple[str, ...] = (
    "<bos>",
    "The",
    " number",
    " of",
    " legs",
    " spider",
    " ant",
    " 8",
    " 6",
    "Spider",
    " eight",
    " six",
    ".",
    "\n",
    " is",
) + tuple(f" w{i}" for i in range(25))
SPIDER, ANT, EIGHT, SIX = 5, 6, 7, 8
TRACKED: tuple[TrackedToken, ...] = (
    TrackedToken("spider", SPIDER, VOCAB[SPIDER]),
    TrackedToken("spider", 9, VOCAB[9]),
    TrackedToken("ant", ANT, VOCAB[ANT]),
    TrackedToken("8", EIGHT, VOCAB[EIGHT]),
    TrackedToken("6", SIX, VOCAB[SIX]),
)
PROMPT_IDS: tuple[int, ...] = (0, 1, 2, 3, 4, 14)
NUM_LAYERS = 4
TOP_K = 8


def decode_fixture_token(token_id: int) -> str:
    return VOCAB[token_id]


def _scores_for(seed: int, boosts: dict[int, float]) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    scores = torch.randn(len(VOCAB), generator=generator, dtype=torch.float64)
    for token_id, value in boosts.items():
        scores[token_id] = value
    return scores


def make_fixture_receipt(with_hash: bool = True) -> JLensPreviewReceipt:
    """
    Fixture: J-lens ranks ``spider`` first at the middle layers of the last
    position and the final-layer logit lens equals the actual output (`` 8``).
    """
    tracked_ids = [t.token_id for t in TRACKED]
    last = len(PROMPT_IDS) - 1
    final_scores = _scores_for(99, {EIGHT: 9.0, SIX: 4.0})
    cells = []
    for layer in range(NUM_LAYERS):
        for position in range(len(PROMPT_IDS)):
            seed = 10 * layer + position
            lens_boost = {SPIDER: 8.0} if position == last and layer in (1, 2) else {}
            if layer == 0:
                cells.append(
                    cell_from_scores(
                        "jlens",
                        layer,
                        position,
                        torch.tensor([float("nan")] * len(VOCAB)),
                        tracked_ids,
                        TOP_K,
                        decode_fixture_token,
                    )
                )
            else:
                cells.append(
                    cell_from_scores(
                        "jlens",
                        layer,
                        position,
                        _scores_for(seed, lens_boost),
                        tracked_ids,
                        TOP_K,
                        decode_fixture_token,
                    )
                )
            logit_scores = (
                final_scores
                if layer == NUM_LAYERS - 1 and position == last
                else _scores_for(seed + 500, {})
            )
            cells.append(
                cell_from_scores(
                    "logit_lens",
                    layer,
                    position,
                    logit_scores,
                    tracked_ids,
                    TOP_K,
                    decode_fixture_token,
                )
            )
    next_top, next_tracked = rank_scores(
        final_scores, tracked_ids, TOP_K, decode_fixture_token
    )
    patched_top, patched_tracked = rank_scores(
        _scores_for(7, {SIX: 9.5, EIGHT: 9.0}), tracked_ids, TOP_K, decode_fixture_token
    )
    noop_top, noop_tracked = rank_scores(
        final_scores, tracked_ids, TOP_K, decode_fixture_token
    )
    text = "".join(VOCAB[i] for i in PROMPT_IDS[1:])
    receipt = JLensPreviewReceipt(
        run_kind="cpu_fixture",
        model_role="fixture",
        created_utc="2026-11-24T00:00:00Z",
        prompt=PromptRecord(
            text=text,
            sha256=sha256_text(text),
            rendering="raw_completion",
            token_ids=PROMPT_IDS,
            token_texts=tuple(VOCAB[i] for i in PROMPT_IDS),
            label="fixture prompt",
            bos_token_id=0,
        ),
        model_identity={
            "repo_id": "fixture/toy-decoder",
            "revision_sha": "0" * 40,
            "safetensors_sha256_index": {"model.safetensors": "1" * 64},
            "tokenizer_sha256": "2" * 64,
            "transformers_version": "none",
            "torch_version": "fixture",
            "dtype": "float64",
        },
        code={"privacy_guard_rev": "fixture", "upstream_jlens_sha": "fixture"},
        num_layers=NUM_LAYERS,
        vocab_size=len(VOCAB),
        top_k=TOP_K,
        score_definition="fixture scores (random normal with fixed boosts)",
        tracked_tokens=TRACKED,
        cells=tuple(cells),
        actual_output=ActualOutput(
            greedy_ids=(EIGHT, 12),
            greedy_text=" 8.",
            max_new_tokens=2,
            next_token_top=tuple(next_top),
            next_token_tracked=tuple(next_tracked),
        ),
        lens={
            "lens_id": "fixture-lens",
            "sha256": "3" * 64,
            "source_layers": [1, 2, 3],
            "target_site": "final_block_output_pre_norm",
            "estimator": "masked-causal-target-sum_source-mean_prompt-mean/v1",
        },
        interventions=(
            InterventionRun(
                name="spider_to_ant",
                kind="intervention",
                method_label="contrast_mean_swap",
                status="ok",
                layers=(2,),
                positions=(last,),
                primary=True,
                strength=1.0,
                delta_norm=3.5,
                direction={
                    "concept_from": "spider",
                    "concept_to": "ant",
                    "contrast_set_sha256": "4" * 64,
                },
                next_token_top=tuple(patched_top),
                next_token_tracked=tuple(patched_tracked),
                greedy_text=" 6.",
                readout_cells=(
                    cell_from_scores(
                        "jlens",
                        3,
                        last,
                        _scores_for(3, {ANT: 8.0}),
                        tracked_ids,
                        TOP_K,
                        decode_fixture_token,
                    ),
                ),
            ),
            InterventionRun(
                name="noop",
                kind="noop",
                method_label="contrast_mean_swap",
                status="ok",
                layers=(2,),
                positions=(last,),
                strength=0.0,
                delta_norm=0.0,
                next_token_top=tuple(noop_top),
                next_token_tracked=tuple(noop_tracked),
                greedy_text=" 8.",
            ),
            InterventionRun(
                name="restoration",
                kind="restoration",
                method_label="contrast_mean_swap",
                status="unavailable",
                unavailable_reason="fixture: not run",
            ),
        ),
        numerical_receipt={
            "schema_version": "privacyguard.jlens_model_receipt/v1",
            "fixture": True,
        },
        unavailable=(UnavailableReason("jlens/layer=0", "no lens fitted for layer 0"),),
        display={"default_filter": "unfiltered"},
    )
    return receipt.with_hash() if with_hash else receipt


def fixture_receipt_dict() -> dict[str, Any]:
    return make_fixture_receipt().to_dict()
