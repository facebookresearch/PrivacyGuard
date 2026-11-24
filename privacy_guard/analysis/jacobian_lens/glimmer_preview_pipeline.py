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
"""Muse Glimmer J-lens preview pipeline (optional, experimental).

Target: the pinned ``meta-models/Muse-Glimmer-30B`` main BF16 checkpoint,
loaded with :func:`glimmer_loading.load_glimmer_adapter`. Subcommands:

* ``measure``: one calibration prompt; forward time, row-VJP seconds and peak
  GPU memory for a single source layer and for all layers.
* ``fit``: resumable J-lens fit on pinned WikiText-103 prompts (shardable).
* ``run``: per predeclared prompt, a ``privacyguard.jlens_preview/v1`` receipt
  (layer x position J-lens and logit-lens cells, the model's own greedy
  output) and, with ``--causal``, the predeclared spider->ant concept clamp
  with its controls plus an experiment record.

Example::

    python -m privacy_guard.analysis.jacobian_lens.glimmer_preview_pipeline run \\
        --checkpoint-dir CKPT --wikitext-dir WT --out-dir OUT --private-dir PRIV \\
        --lens PRIV/lens.pt --causal --code-rev REV

``--private-dir`` holds tensors (lens matrices, checkpoints, directions) and is
never part of a receipt. The WikiText parquet is read from ``--wikitext-dir``;
nothing here downloads data.
"""

from __future__ import annotations

import argparse
import datetime
import functools
import hashlib
import json
import logging
import math
import os
import platform
import shlex
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

import pandas as pd
import torch
import transformers
from privacy_guard.analysis.jacobian_lens import calibration_corpus as corpus
from privacy_guard.analysis.jacobian_lens.glimmer_adapter import GlimmerAdapter
from privacy_guard.analysis.jacobian_lens.glimmer_loading import load_glimmer_adapter
from privacy_guard.analysis.jacobian_lens.lens_fitting import (
    fit,
    JacobianLens,
    RECIPE,
    ResidualHooks,
    resolve_layers,
    UPSTREAM_REFERENCE,
    valid_position_mask,
)
from privacy_guard.analysis.jacobian_lens.lens_intervention import (
    assess,
    build_contrast_examples,
    concept_means,
    CONCEPT_POOL,
    CONTRAST_TEMPLATES,
    followup_arms,
    layer_directions,
    LayerDirections,
    predeclared_arms,
    run_arms,
    SUCCESS_RULE,
    UNRELATED_PAIR,
)
from privacy_guard.analysis.jacobian_lens.lens_readout import (
    capture_residuals,
    greedy_continuation,
    head_parity,
    rank_scores,
    readout_cells,
    resolve_tracked,
)
from privacy_guard.analysis.jacobian_lens.preview_schema import JLensPreviewReceipt

logger: logging.Logger = logging.getLogger(__name__)

PROMPTS: dict[str, dict[str, str]] = {
    "blog_literal": {
        "text": "The number of legs on the animal that spins webs is.",
        "note": "exact string quoted in the global-workspace blog post, period included",
    },
    "blog_no_period": {
        "text": "The number of legs on the animal that spins webs is",
        "note": "blog string without the sentence-final period (completion form)",
    },
    "upstream_probe_swap": {
        "text": "Fact: The number of legs on the animal that spins webs is ",
        "note": "data/experiments/probe-swap.json item spider-legs, verbatim",
    },
    "ant_colonies": {
        "text": "The number of legs on the insect that lives in colonies is",
        "note": "post-primary reverse-direction prompt (expected answer 6)",
    },
    "ant_colonies_fact": {
        "text": "Fact: The number of legs on the insect that lives in colonies is ",
        "note": "post-primary reverse-direction prompt in the reference format",
    },
}
# Whole-word single-token forms only (tokenizer audit): a leading space with
# no letters, or a subword such as "ant" or "sp", is not the concept.
TRACKED: dict[str, list[str]] = {
    "spider": [" spider", " spiders", " Spider"],
    "ant": [" ant", " ants", " Ant"],
    "dog": [" dog", " dogs"],
    "bird": [" bird", " birds"],
    "8": ["8", " eight"],
    "6": ["6", " six"],
    "4": ["4", " four"],
    "2": ["2", " two"],
}
PREDECLARED_CAUSAL_PROMPTS: tuple[str, ...] = (
    "blog_literal",
    "blog_no_period",
    "upstream_probe_swap",
)
# Declared after the primary result (post-primary, exploratory): target
# specificity (spider->dog, spider->bird), restorations at strength 2, and the
# reverse direction on a 6-legged prompt. (prompt, from, to, from_answer, to_answer)
FOLLOWUPS: tuple[tuple[str, str, str, str, str], ...] = (
    ("upstream_probe_swap", "spider", "ant", "8", "6"),
    ("upstream_probe_swap", "spider", "dog", "8", "4"),
    ("upstream_probe_swap", "spider", "bird", "8", "2"),
    ("blog_no_period", "spider", "ant", "8", "6"),
    ("blog_no_period", "spider", "dog", "8", "4"),
    ("blog_no_period", "spider", "bird", "8", "2"),
    ("ant_colonies_fact", "ant", "spider", "6", "8"),
    ("ant_colonies", "ant", "spider", "6", "8"),
)
# Predeclared before any model run. The band is a candidate (middle half of
# 52 blocks), not a validated workspace band.
BAND: tuple[int, ...] = tuple(range(13, 39))
DOWNSTREAM: tuple[int, ...] = tuple(range(39, 52))
SINGLE_LAYERS: tuple[int, ...] = (13, 19, 26, 32, 38)
STRENGTHS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
STOP_IDS: tuple[int, ...] = (200001, 200008)
UPSTREAM_JLENS_SHA = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_FILES: dict[str, str] = {
    "wikitext-103-raw-v1/train-00000-of-00002.parquet": (
        "74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212"
    ),
}
SCORE_DEFINITION = (
    "native head evaluated in float32: cap * tanh(output_multiplier * "
    "lm_head(final_norm(h)) / cap), cap = final_logit_softcapping"
)


class Glue:
    """Tokenization with an explicit BOS policy around a ``GlimmerAdapter``."""

    def __init__(self, adapter: GlimmerAdapter, bos_id: int) -> None:
        self.adapter = adapter
        self.tokenizer: Any = adapter.tokenizer
        self.bos_id = bos_id

    def encode_plain(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def encode_ids(self, text: str) -> list[int]:
        return [self.bos_id, *self.encode_plain(text)]

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.encode_ids(text)], device=self.adapter.device)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(directory: str, name: str, payload: object) -> str:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name)
    with open(path + ".tmp", "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True, allow_nan=False)
    os.replace(path + ".tmp", path)
    logger.info("wrote %s", path)
    return path


PUBLIC_PROGRAM = (
    "python -m privacy_guard.analysis.jacobian_lens.glimmer_preview_pipeline"
)
PATH_PLACEHOLDERS: dict[str, str] = {
    "checkpoint_dir": "CKPT",
    "out_dir": "OUT",
    "private_dir": "PRIVATE",
    "wikitext_dir": "WIKITEXT",
    "lens": "PRIVATE/LENS.pt",
    "fit_manifest": "OUT/FIT_MANIFEST.json",
    "shard_lenses": "PRIVATE/LENS_SHARD.pt",
    "shard_manifests": "OUT/FIT_MANIFEST_SHARD.json",
}


def public_command(args: argparse.Namespace) -> str:
    """The run as a portable command: local paths become placeholders, so no
    host path reaches a receipt (the literal argv stays in the private dir)."""
    parts = [*shlex.split(PUBLIC_PROGRAM), args.command]
    for key, value in sorted(vars(args).items()):
        if key == "command" or value is None or value is False or value == "":
            continue
        flag = "--" + key.replace("_", "-")
        if value is True:
            parts.append(flag)
        elif isinstance(value, list) and key in PATH_PLACEHOLDERS:
            parts += [
                flag,
                *(f"{PATH_PLACEHOLDERS[key]}:{i}" for i in range(len(value))),
            ]
        elif isinstance(value, list):
            parts += [flag, *map(str, value)]
        else:
            parts += [flag, PATH_PLACEHOLDERS.get(key, str(value))]
    return shlex.join(parts)


def environment(args: argparse.Namespace) -> dict[str, Any]:
    gpu = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu.name if gpu else None,
        "gpu_total_gib": round(gpu.total_memory / 2**30, 2) if gpu else None,
        "gpu_count": torch.cuda.device_count(),
        "device_map": args.device,
        "privacy_guard_rev": args.code_rev,
        "run_command": public_command(args),
    }


def code_record(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "privacy_guard_rev": args.code_rev,
        "upstream_jlens_sha": UPSTREAM_JLENS_SHA,
        "estimator": RECIPE,
        "run_command": public_command(args),
    }


def load(args: argparse.Namespace) -> tuple[GlimmerAdapter, Glue, dict[str, Any]]:
    adapter, report = load_glimmer_adapter(args.checkpoint_dir, device=args.device)
    identity = adapter.identity().to_dict()
    tokenizer: Any = adapter.tokenizer
    if tokenizer.bos_token_id is None:
        raise ValueError("tokenizer has no BOS token; the BOS policy needs one")
    bos = int(tokenizer.bos_token_id)
    logger.info("loaded: %s", report.to_dict())
    return adapter, Glue(adapter, bos), {**identity, "load_report": report.to_dict()}


def load_wikitext(directory: str) -> tuple[list[str], dict[str, str]]:
    """Train records in file order from the pinned parquet shard(s)."""
    texts: list[str] = []
    files: dict[str, str] = {}
    for name, expected in WIKITEXT_FILES.items():
        path = os.path.join(directory, name)
        files[name] = _sha256(path)
        if files[name] != expected:
            raise ValueError(f"{name} sha256 {files[name]} != pinned {expected}")
        texts.extend(pd.read_parquet(path, columns=["text"])["text"].tolist())
    return texts, files


def calibration_set(
    glue: Glue, args: argparse.Namespace
) -> tuple[list[torch.Tensor], corpus.CorpusManifest]:
    records, files = load_wikitext(args.wikitext_dir)
    prompts = corpus.select_prompts(records, args.n_prompts)
    eval_texts = [p["text"] for p in PROMPTS.values()] + ["spins webs"]
    separation = corpus.separation_report(prompts, eval_texts)
    ids, lengths, truncated = corpus.tokenize_prompts(
        glue.encode, prompts, max_seq_len=args.max_seq_len
    )
    manifest = corpus.CorpusManifest(
        dataset=corpus.WIKITEXT_DATASET,
        config=corpus.WIKITEXT_CONFIG,
        split="train",
        revision=WIKITEXT_REVISION,
        source_files=files,
        selection_rule=corpus.SELECTION_RULE,
        min_chars=600,
        max_seq_len=args.max_seq_len,
        tokenizer_sha256=_sha256(os.path.join(args.checkpoint_dir, "tokenizer.json")),
        prompt_sha256=[corpus.sha256_text(p) for p in prompts],
        token_lengths=lengths,
        truncated=truncated,
        separation=separation,
    )
    return ids, manifest


def measure_backward(
    adapter: GlimmerAdapter,
    x: torch.Tensor,
    sources: list[int],
    dim_batch: int,
    n_passes: int,
) -> dict[str, Any]:
    """Forward plus ``n_passes`` row-VJP passes of the estimator; timings and peak memory."""
    if n_passes * dim_batch > adapter.d_model:
        raise ValueError("n_passes * dim_batch exceeds the residual width")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sources, target = resolve_layers(sources, None, adapter.n_layers)
    row: dict[str, Any] = {
        "n_source_layers": len(sources),
        "first_source": sources[0],
        "dim_batch": dim_batch,
        "seq_len": int(x.shape[1]),
    }
    started = time.perf_counter()
    with (
        ResidualHooks(
            adapter.layers, record=[*sources, target], graph_root=sources[0]
        ) as hooks,
        torch.enable_grad(),
    ):
        adapter.forward(x.expand(dim_batch, -1))
        z, xs = hooks.activations[target], [hooks.activations[k] for k in sources]
        torch.cuda.synchronize()
        row["forward_seconds"] = round(time.perf_counter() - started, 3)
        cotangent, times = torch.zeros_like(z), []
        for i in range(n_passes):
            cotangent.zero_()
            for b in range(dim_batch):
                cotangent[b, 16 : x.shape[1] - 1, i * dim_batch + b] = 1.0
            started = time.perf_counter()
            grads = torch.autograd.grad(
                z, xs, grad_outputs=cotangent, retain_graph=True
            )
            for grad in grads:
                grad[:, 16:-1].float().mean(1).cpu()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
    steady = sum(times[1:]) / max(1, len(times) - 1)
    passes = math.ceil(adapter.d_model / dim_batch)
    row.update(
        backward_seconds_per_pass=[round(t, 3) for t in times],
        passes_per_prompt=passes,
        peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
        extrapolated_seconds_per_prompt=round(
            row["forward_seconds"] + steady * passes, 1
        ),
    )
    return row


def cmd_measure(
    adapter: GlimmerAdapter,
    glue: Glue,
    identity: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    ids, manifest = calibration_set(glue, args)
    rows = []
    all_layers = list(range(adapter.n_layers - 1))
    configs = dict.fromkeys([(26, 4), (26, 8), (-1, 8), (-1, args.dim_batch)])
    for single, dim_batch in configs:
        sources = all_layers if single < 0 else [single]
        try:
            rows.append(measure_backward(adapter, ids[0], sources, dim_batch, 3))
        except torch.OutOfMemoryError as exc:
            rows.append(
                {
                    "n_source_layers": len(sources),
                    "dim_batch": dim_batch,
                    "status": "oom",
                    "reason": str(exc)[:300],
                }
            )
        logger.info("measure %s", rows[-1])
    _write(
        args.out_dir,
        "measure_receipt.json",
        {
            "schema": "privacyguard.jlens_fit_measure/v1",
            "run_kind": "real_model",
            "model_identity": identity,
            "environment": environment(args),
            "calibration_prompt_sha256": manifest.prompt_sha256[0],
            "rows": rows,
        },
    )


def cmd_fit(
    adapter: GlimmerAdapter,
    glue: Glue,
    identity: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    ids, manifest = calibration_set(glue, args)
    shard = [i for i in range(len(ids)) if i % args.num_shards == args.shard]
    tag = f"shard{args.shard}of{args.num_shards}"
    snapshot = os.path.join(args.private_dir, f"fit_ckpt_after_first_prompt_{tag}.pt")
    started = time.perf_counter()
    lens, fit_manifest = fit(
        adapter,
        [ids[i] for i in shard],
        prompt_hashes=[manifest.prompt_sha256[i] for i in shard],
        dim_batch=args.dim_batch,
        checkpoint_path=os.path.join(args.private_dir, f"fit_ckpt_{tag}.pt"),
        checkpoint_every=args.checkpoint_every,
        first_prompt_snapshot=snapshot,
        on_prompt=lambda r: logger.info(
            "prompt %d %s seq=%d %.0fs", r.index, r.status, r.seq_len, r.seconds
        ),
    )
    lens_path = os.path.join(args.private_dir, f"lens_{tag}.pt")
    fit_manifest.extra = {
        "shard": args.shard,
        "num_shards": args.num_shards,
        "lens_file_sha256": lens.save(lens_path),
        "first_prompt_snapshot_sha256": _sha256(snapshot)
        if os.path.exists(snapshot)
        else None,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    _write(
        args.out_dir,
        f"fit_manifest_{tag}.json",
        {
            "schema": "privacyguard.jlens_fit/v1",
            "run_kind": "real_model",
            "model_identity": identity,
            "environment": environment(args),
            "corpus": manifest.to_dict(),
            "fit": json.loads(fit_manifest.to_json()),
        },
    )


_CHECKPOINT_FIELDS: tuple[str, ...] = (
    "repo_id",
    "revision_sha",
    "safetensors_sha256_index",
    "tokenizer_sha256",
    "config_sha256",
    "modeling_source_sha256",
    "dtype",
)
_RECIPE_FIELDS: tuple[str, ...] = (
    "recipe",
    "upstream_reference",
    "source_layers",
    "target_layer",
    "skip_first",
    "d_model",
)


def _check_shard_coverage(manifests: Sequence[dict[str, Any]]) -> None:
    """Shards must share one corpus selection, be numbered 0..k-1 exactly once,
    and together cover every selected prompt exactly once."""
    corpora = {m["corpus"]["corpus_sha256"] for m in manifests}
    if len(corpora) != 1:
        raise ValueError(
            f"shards come from different corpus selections: {sorted(corpora)}"
        )
    counts = {m["fit"]["extra"]["num_shards"] for m in manifests}
    indices = sorted(m["fit"]["extra"]["shard"] for m in manifests)
    if (
        len(counts) != 1
        or indices != list(range(len(manifests)))
        or counts != {len(manifests)}
    ):
        raise ValueError(
            f"shards {indices} of {sorted(counts)} are not one complete set"
        )
    identities = {
        json.dumps([m["model_identity"].get(k) for k in _CHECKPOINT_FIELDS])
        for m in manifests
    }
    recipes = {json.dumps([m["fit"].get(k) for k in _RECIPE_FIELDS]) for m in manifests}
    if len(identities) != 1 or len(recipes) != 1:
        raise ValueError("shards were fit on different checkpoints or recipes")
    seen = [r["prompt_sha256"] for m in manifests for r in m["fit"]["prompts"]]
    selected = manifests[0]["corpus"]["prompt_sha256"]
    if len(seen) != len(selected) or sorted(seen) != sorted(selected):
        raise ValueError("shard prompts do not cover the corpus selection exactly once")


def cmd_merge(args: argparse.Namespace) -> None:
    """Merge shard lenses fitted on disjoint prompts of one corpus selection
    into a prompt-count-weighted lens; no model is loaded."""
    if len(args.shard_lenses) != len(args.shard_manifests) or not args.shard_lenses:
        raise ValueError("pass one --shard-manifests entry per --shard-lenses entry")
    manifests = []
    for path in args.shard_manifests:
        with open(path) as handle:
            manifests.append(json.load(handle))
    _check_shard_coverage(manifests)
    lenses = []
    for lens_path, manifest in zip(args.shard_lenses, manifests, strict=True):
        if _sha256(lens_path) != manifest["fit"]["extra"]["lens_file_sha256"]:
            raise ValueError(f"{lens_path} does not match its shard manifest")
        lenses.append(JacobianLens.load(lens_path))
    merged = JacobianLens.merge(lenses)
    lens_path = os.path.join(args.private_dir, "lens_merged.pt")
    shards = [
        {
            "manifest_sha256": _sha256(path),
            "lens_file_sha256": m["fit"]["extra"]["lens_file_sha256"],
            "n_prompts_used": m["fit"]["n_prompts_used"],
            "shard": m["fit"]["extra"]["shard"],
            "num_shards": m["fit"]["extra"]["num_shards"],
            "environment": m["environment"],
            "wall_seconds": m["fit"]["extra"]["wall_seconds"],
        }
        for path, m in zip(args.shard_manifests, manifests, strict=True)
    ]
    _write(
        args.out_dir,
        "fit_manifest_merged.json",
        {
            "schema": "privacyguard.jlens_fit/v1",
            "run_kind": "real_model",
            "model_identity": manifests[0]["model_identity"],
            "environment": environment(args),
            "corpus": manifests[0]["corpus"],
            "fit": {
                "recipe": RECIPE,
                "n_prompts_used": merged.n_prompts,
                "source_layers": merged.source_layers,
                "merge": "prompt-count-weighted mean of shard means (JacobianLens.merge)",
                "shards": shards,
                "extra": {"lens_file_sha256": merged.save(lens_path)},
            },
        },
    )


def tracked_tokens(
    glue: Glue,
) -> tuple[list[dict[str, Any]], list[int], dict[str, list[int]], list[dict[str, Any]]]:
    """Receipt tracked tokens, their IDs, the answer forms per digit concept and
    the full surface table. Only surfaces that are one token enter the receipt:
    the first piece of a multi-token form (e.g. the space of " 8") is not the
    concept and would distort its rank curve."""
    resolved = resolve_tracked(glue.encode_plain, TRACKED)
    receipt: dict[int, dict[str, Any]] = {}
    for t in resolved:
        if not t.single_token:
            continue
        receipt.setdefault(
            t.token_id,
            {
                "concept": t.concept,
                "token_id": t.token_id,
                "text": glue.decode([t.token_id]),
            },
        )
    answers = {
        c: sorted({t.token_id for t in resolved if t.concept == c and t.single_token})
        for c in ("8", "6", "4", "2")
    }
    surfaces = [{**asdict(t), "token_ids": list(t.token_ids)} for t in resolved]
    return list(receipt.values()), list(receipt), answers, surfaces


def prompt_record(glue: Glue, key: str) -> dict[str, Any]:
    text = PROMPTS[key]["text"]
    ids = glue.encode_ids(text)
    return {
        "text": text,
        "sha256": corpus.sha256_text(text),
        "rendering": "raw_completion",
        "token_ids": ids,
        "token_texts": [glue.decode([i]) for i in ids],
        "label": key,
        "note": PROMPTS[key]["note"] + "; BOS prepended explicitly",
        "bos_token_id": glue.bos_id,
    }


@functools.cache
def _cached_sha256(path: str) -> str:
    """One hash per file per process: every prompt's receipt names the same lens."""
    return _sha256(path)


def lens_record(
    args: argparse.Namespace, lens: JacobianLens | None
) -> dict[str, Any] | None:
    if lens is None:
        return None
    record: dict[str, Any] = {
        "lens_id": os.path.basename(args.lens),
        "sha256": _cached_sha256(args.lens),
        "source_layers": lens.source_layers,
        "target_site": "output of the final block, pre final norm",
        "estimator": RECIPE,
        "upstream": UPSTREAM_REFERENCE,
        "n_prompts": lens.n_prompts,
        "precision": "fp32 sums, stored fp16, applied fp32",
    }
    if args.fit_manifest:
        with open(args.fit_manifest) as handle:
            fitted = json.load(handle)
        if fitted["fit"]["extra"]["lens_file_sha256"] != record["sha256"]:
            raise ValueError(
                f"{args.lens} is not the lens recorded in {args.fit_manifest}"
            )
        record["corpus_sha256"] = fitted["corpus"]["corpus_sha256"]
        record["fit_manifest_sha256"] = _cached_sha256(args.fit_manifest)
    return record


def build_means(
    adapter: GlimmerAdapter, glue: Glue
) -> tuple[dict[str, dict[int, torch.Tensor]], dict[str, Any]]:
    """Class-mean residuals at the concept token for every contrast concept."""
    concepts = list(dict.fromkeys([*CONCEPT_POOL, *UNRELATED_PAIR]))
    examples, skipped = build_contrast_examples(glue.encode_ids, concepts)
    means = concept_means(adapter, glue.encode, examples, list(range(adapter.n_layers)))
    info = {
        "concepts": concepts,
        "templates_sha256": corpus.sha256_text("\n".join(CONTRAST_TEMPLATES)),
        "n_examples": len(examples),
        "skipped": skipped,
    }
    return means, info


def directions_for(
    means: dict[str, dict[int, torch.Tensor]],
    concept_from: str,
    concept_to: str,
    args: argparse.Namespace,
) -> dict[int, LayerDirections]:
    directions = layer_directions(
        means,
        concept_from=concept_from,
        concept_to=concept_to,
        pool=CONCEPT_POOL,
        unrelated=UNRELATED_PAIR,
        seed=args.seed,
    )
    torch.save(
        {k: vars(v) for k, v in directions.items()},
        os.path.join(args.private_dir, f"directions_{concept_from}_to_{concept_to}.pt"),
    )
    return directions


def answer_leakage(
    adapter: GlimmerAdapter,
    directions: dict[int, LayerDirections],
    from_ids: Sequence[int],
    to_ids: Sequence[int],
    layers: Sequence[int],
    lens: JacobianLens | None = None,
) -> dict[str, dict[int, dict[str, float]]]:
    """Cosine of each edit direction (and its controls) with an answer
    direction ``g * (W[to] - W[from])`` for every answer-form pair; with a lens
    the answer direction is also transported back to layer ``l`` as
    ``J_l^T g * (W[to] - W[from])``. Near zero means the concept edit is not an
    answer-token direction."""
    model: Any = adapter.model
    weight = model.lm_head.weight
    gain = model.model.language_model.norm.weight.float().cpu()
    cos = torch.nn.functional.cosine_similarity
    out: dict[str, dict[int, dict[str, float]]] = {}
    for f in from_ids:
        for t in to_ids:
            answer = gain * (weight[t].float().cpu() - weight[f].float().cpu())
            rows: dict[int, dict[str, float]] = {}
            for k in layers:
                d = directions[k]
                vectors = {
                    "edit": d.mu_to - d.mu_from,
                    "unrelated": d.unrelated,
                    "random": d.random,
                }
                rows[k] = {
                    name: float(cos(v, answer, dim=0)) for name, v in vectors.items()
                }
                if lens is not None and k in lens.jacobians:
                    back = lens.jacobians[k].T @ answer
                    rows[k].update(
                        {
                            f"{name}_transported": float(cos(v, back, dim=0))
                            for name, v in vectors.items()
                        }
                    )
            out[f"{f}->{t}"] = rows
    return out


def intervention_run(
    arm: dict[str, Any],
    contrast_sha: str,
    concept_from: str = "spider",
    concept_to: str = "ant",
    note: str = "",
) -> dict[str, Any]:
    """Map one arm result onto the receipt's ``InterventionRun`` fields."""
    direction = {
        "concept_from": concept_from,
        "concept_to": concept_to,
        "contrast_set_sha256": contrast_sha,
        "control": arm["control"] or None,
        "restore": arm["restore"] or None,
    }
    answers = arm["answers"]
    diff = arm["max_abs_logit_diff_vs_baseline"]
    ranks = ", ".join(
        f"best {k}-form rank {v['best_rank']}" for k, v in answers.items()
    )
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
        "direction": direction,
        "next_token_top": arm["next_token_top"],
        "next_token_tracked": arm["next_token_tracked"],
        "greedy_text": arm["greedy_text"],
        "notes": f"{note}{ranks}; greedy answer {arm['greedy_answer']}; max |logit diff| "
        f"vs baseline {'n/a' if diff is None else format(diff, '.4g')}",
        "readout_cells": arm["readout_cells"],
    }


def _arm_kwargs(
    adapter: GlimmerAdapter,
    glue: Glue,
    lens: JacobianLens | None,
    tracked_ids: list[int],
    answers: dict[str, list[int]],
    args: argparse.Namespace,
    downstream: Sequence[int],
) -> dict[str, Any]:
    return {
        "readout_layers": list(range(adapter.n_layers)),
        "tracked_ids": tracked_ids,
        "answer_forms": answers,
        "decode": glue.decode,
        "lens": lens,
        "head": adapter.head_fp32,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "stop_ids": STOP_IDS,
        "cell_layers": downstream,
    }


def run_causal(
    adapter: GlimmerAdapter,
    glue: Glue,
    x: torch.Tensor,
    lens: JacobianLens | None,
    directions: dict[int, LayerDirections],
    tracked_ids: list[int],
    answers: dict[str, list[int]],
    args: argparse.Namespace,
    band: Sequence[int] = BAND,
    downstream: Sequence[int] = DOWNSTREAM,
    single_layers: Sequence[int] = SINGLE_LAYERS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """All predeclared arms on one prompt and the :data:`SUCCESS_RULE` verdict."""
    clean = capture_residuals(adapter, x, list(range(adapter.n_layers)))
    arms = run_arms(
        adapter,
        adapter.native_logits,
        x,
        predeclared_arms(band, single_layers, STRENGTHS),
        directions,
        clean,
        **_arm_kwargs(adapter, glue, lens, tracked_ids, answers, args, downstream),
    )
    verdict = assess(arms, "8", "6", downstream)
    if not verdict["hooks_effective"]:
        raise RuntimeError(
            "primary edit left the native logits unchanged; hooks did not fire"
        )
    return arms, verdict


def build_receipt(
    adapter: GlimmerAdapter,
    glue: Glue,
    identity: dict[str, Any],
    key: str,
    lens: JacobianLens | None,
    args: argparse.Namespace,
    tracked: list[dict[str, Any]],
    tracked_ids: list[int],
    interventions: list[dict[str, Any]],
    run_kind: str = "real_model",
    model_role: str = "glimmer_target",
) -> dict[str, Any]:
    record = prompt_record(glue, key)
    x = glue.encode(record["text"])
    residuals = capture_residuals(adapter, x, list(range(adapter.n_layers)))
    cells = readout_cells(
        adapter,
        lens,
        residuals,
        positions=list(range(x.shape[1])),
        tracked_ids=tracked_ids,
        top_k=args.top_k,
        decode=glue.decode,
        head=adapter.head_fp32,
    )
    greedy, first = greedy_continuation(
        adapter.native_logits, x, max_new_tokens=args.max_new_tokens, stop_ids=STOP_IDS
    )
    top, tracked_scores = rank_scores(first, tracked_ids, args.top_k, glue.decode)
    unavailable = (
        []
        if lens
        else [{"scope": "jlens", "reason": "no fitted lens supplied to this run"}]
    )
    receipt = {
        "schema_version": "privacyguard.jlens_preview/v1",
        "run_kind": run_kind,
        "model_role": model_role,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "prompt": record,
        "model_identity": {k: v for k, v in identity.items() if k != "load_report"},
        "code": code_record(args),
        "num_layers": adapter.n_layers,
        "vocab_size": adapter.vocab_size,
        "top_k": args.top_k,
        "score_definition": SCORE_DEFINITION,
        "tracked_tokens": tracked,
        "cells": cells,
        "lens": lens_record(args, lens),
        "interventions": interventions,
        "actual_output": {
            "greedy_ids": greedy,
            "greedy_text": glue.decode(greedy),
            "max_new_tokens": args.max_new_tokens,
            "next_token_top": top,
            "next_token_tracked": tracked_scores,
        },
        "unavailable": unavailable,
        "display": {
            "actual_output_scores": "native bf16 forward logits",
            "head_parity": head_parity(adapter, adapter.native_logits, x),
        },
    }
    return JLensPreviewReceipt.from_dict(receipt).with_hash().to_dict()


def run_followup(
    adapter: GlimmerAdapter,
    glue: Glue,
    directions: dict[int, LayerDirections],
    followup: tuple[str, str, str, str, str],
    lens: JacobianLens | None,
    tracked_ids: list[int],
    answers: dict[str, list[int]],
    args: argparse.Namespace,
    band: Sequence[int] = BAND,
    downstream: Sequence[int] = DOWNSTREAM,
) -> dict[str, Any]:
    """One post-primary follow-up: a (prompt, from, to) clamp with its controls."""
    key, concept_from, concept_to, from_answer, to_answer = followup
    x = glue.encode(PROMPTS[key]["text"])
    clean = capture_residuals(adapter, x, list(range(adapter.n_layers)))
    tag = f"{concept_from}_to_{concept_to}_post"
    arms = run_arms(
        adapter,
        adapter.native_logits,
        x,
        followup_arms(band, tag),
        directions,
        clean,
        **_arm_kwargs(adapter, glue, lens, tracked_ids, answers, args, downstream),
    )

    def flip(summary: dict[str, Any] | None) -> bool | None:
        if summary is None:
            return None
        return summary[to_answer]["best_rank"] < summary[from_answer]["best_rank"]

    flips = {
        a["name"]: {
            "next_token_flip": flip(a["answers"]),
            "flip_after_whitespace": flip(a["answers_after_whitespace"]),
            "greedy_answer": a["greedy_answer"],
            "p_from": a["answers"][from_answer]["group_prob"],
            "p_to": a["answers"][to_answer]["group_prob"],
        }
        for a in arms
    }
    leakage = answer_leakage(
        adapter, directions, answers[from_answer], answers[to_answer], band, lens
    )
    return {
        "prompt": key,
        "concept_from": concept_from,
        "concept_to": concept_to,
        "from_answer": from_answer,
        "to_answer": to_answer,
        "arms": arms,
        "summary": flips,
        "answer_leakage_cosine": leakage,
    }


def _causal_prompt(
    adapter: GlimmerAdapter,
    glue: Glue,
    key: str,
    lens: JacobianLens | None,
    directions: dict[int, LayerDirections],
    tracked_ids: list[int],
    answers: dict[str, list[int]],
    args: argparse.Namespace,
    contrast_sha: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    arms, verdict = run_causal(
        adapter,
        glue,
        glue.encode(PROMPTS[key]["text"]),
        lens,
        directions,
        tracked_ids,
        answers,
        args,
    )
    logger.info("causal %s: %s", key, verdict)
    outcome = "POSITIVE" if verdict["success"] else "NEGATIVE"
    banner = (
        f"SUCCESS_RULE (predeclared primary): {outcome}; answer flipped "
        f"{verdict['primary_answer_flipped']}, downstream representation moved "
        f"{verdict['representation_moved']}, controls keep baseline "
        f"{verdict['controls_keep_baseline']}; "
    )
    runs = [
        intervention_run(a, contrast_sha, note=banner if a["primary"] else "")
        for a in arms
    ]
    return runs, {"arms": arms, "assessment": verdict}


def cmd_run(
    adapter: GlimmerAdapter,
    glue: Glue,
    identity: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    lens = JacobianLens.load(args.lens) if args.lens else None
    tracked, tracked_ids, answers, surfaces = tracked_tokens(glue)
    if args.followups and not args.causal:
        raise ValueError("--followups needs --causal")
    means, contrast = build_means(adapter, glue) if args.causal else ({}, {})
    directions = directions_for(means, "spider", "ant", args) if args.causal else {}
    experiment: dict[str, Any] = {
        "schema": "privacyguard.jlens_causal/v1",
        "run_kind": "real_model",
        "model_identity": identity,
        "environment": environment(args),
        "success_rule": SUCCESS_RULE,
        "band": list(BAND),
        "downstream_layers": list(DOWNSTREAM),
        "strengths": list(STRENGTHS),
        "single_layers": list(SINGLE_LAYERS),
        "tracked_surfaces": surfaces,
        "answer_forms": answers,
        "contrast": contrast,
        "prompts": {},
        "followups": [],
    }
    runs: dict[str, list[dict[str, Any]]] = {key: [] for key in args.prompts}
    for key in args.prompts:
        if args.causal and key in PREDECLARED_CAUSAL_PROMPTS:
            runs[key], experiment["prompts"][key] = _causal_prompt(
                adapter,
                glue,
                key,
                lens,
                directions,
                tracked_ids,
                answers,
                args,
                contrast["templates_sha256"],
            )
    pairs: dict[tuple[str, str], dict[int, LayerDirections]] = {
        ("spider", "ant"): directions
    }
    for followup in FOLLOWUPS if args.followups else ():
        pair = (followup[1], followup[2])
        if pair not in pairs:
            pairs[pair] = directions_for(means, *pair, args)
        result = run_followup(
            adapter, glue, pairs[pair], followup, lens, tracked_ids, answers, args
        )
        experiment["followups"].append(result)
        logger.info("followup %s: %s", followup, result["summary"])
        if result["prompt"] not in experiment["prompts"]:
            # A receipt's interventions must include its predeclared primary
            # arm; follow-ups on other prompts stay in the experiment record.
            continue
        runs[result["prompt"]].extend(
            intervention_run(
                a,
                contrast["templates_sha256"],
                result["concept_from"],
                result["concept_to"],
                "post-primary follow-up; ",
            )
            for a in result["arms"]
            if a["kind"] != "baseline"
        )
    receipts = [
        build_receipt(
            adapter, glue, identity, key, lens, args, tracked, tracked_ids, arms
        )
        for key, arms in runs.items()
    ]
    _write(args.out_dir, "jlens_preview_receipts.json", receipts)
    if args.causal:
        _write(args.out_dir, "causal_experiment.json", experiment)


def heldout_quality(
    adapter: GlimmerAdapter,
    glue: Glue,
    lens: JacobianLens,
    texts: Sequence[str],
    max_seq_len: int,
) -> dict[str, Any]:
    """Per source layer, the fraction of valid positions where the J-lens and
    the logit lens predict the model's own top-1 next token (fp32 head)."""
    ids, _, _ = corpus.tokenize_prompts(
        glue.encode, list(texts), max_seq_len=max_seq_len
    )
    final_layer = adapter.n_layers - 1
    hits = {k: {"jlens": 0, "logit_lens": 0} for k in lens.source_layers}
    total = 0
    for x in ids:
        positions = valid_position_mask(x.shape[1]).nonzero(as_tuple=True)[0]
        residuals = capture_residuals(adapter, x, [*lens.source_layers, final_layer])
        target = adapter.head_fp32(residuals[final_layer][positions]).argmax(-1)
        total += int(positions.numel())
        for k in lens.source_layers:
            h = residuals[k][positions]
            hits[k]["jlens"] += int(
                (adapter.head_fp32(lens.transport(h, k)).argmax(-1) == target).sum()
            )
            hits[k]["logit_lens"] += int(
                (adapter.head_fp32(h).argmax(-1) == target).sum()
            )
    return {
        "metric": "top-1 agreement with the model's own next-token top-1 at valid positions",
        "n_prompts": len(ids),
        "n_positions": total,
        "top1_agreement": {
            k: {m: v / max(total, 1) for m, v in row.items()} for k, row in hits.items()
        },
    }


def cmd_evaluate(
    adapter: GlimmerAdapter,
    glue: Glue,
    identity: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """Held-out lens quality on WikiText records outside the lens's corpus
    selection (checked against the selection recorded in ``--fit-manifest``)."""
    if not args.lens or not args.fit_manifest:
        raise ValueError("evaluate needs --lens and --fit-manifest")
    lens = JacobianLens.load(args.lens)
    with open(args.fit_manifest) as handle:
        fitted = set(json.load(handle)["corpus"]["prompt_sha256"])
    records, files = load_wikitext(args.wikitext_dir)
    texts = corpus.select_prompts(records, args.heldout_start + args.heldout)[
        args.heldout_start :
    ]
    if fitted & {corpus.sha256_text(t) for t in texts}:
        raise ValueError("held-out records overlap the lens's corpus selection")
    quality = heldout_quality(adapter, glue, lens, texts, args.max_seq_len)
    _write(
        args.out_dir,
        "lens_quality_heldout.json",
        {
            "schema": "privacyguard.jlens_quality/v1",
            "run_kind": "real_model",
            "model_identity": identity,
            "environment": environment(args),
            "code": code_record(args),
            "lens": lens_record(args, lens),
            "source_files": files,
            "heldout_selection": {"start": args.heldout_start, "count": args.heldout},
            "heldout_prompt_sha256": [corpus.sha256_text(t) for t in texts],
            **quality,
        },
    )


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "command", choices=["measure", "fit", "run", "merge", "evaluate"]
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--device", default="cuda", help="device_map for loading")
    parser.add_argument(
        "--out-dir", required=True, help="receipts and manifests (no tensors)"
    )
    parser.add_argument("--private-dir", required=True, help="tensors; never published")
    parser.add_argument(
        "--wikitext-dir", default="", help="dir holding the pinned parquet"
    )
    parser.add_argument(
        "--code-rev", default="unrecorded", help="source revision of this code"
    )
    parser.add_argument("--lens", default="")
    parser.add_argument("--fit-manifest", default="")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument(
        "--followups", action="store_true", help="post-primary follow-ups"
    )
    parser.add_argument(
        "--prompts", nargs="+", default=list(PROMPTS), choices=list(PROMPTS)
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-prompts", type=int, default=100)
    parser.add_argument(
        "--heldout-start", type=int, default=200, help="first held-out record"
    )
    parser.add_argument(
        "--heldout", type=int, default=8, help="held-out records to score"
    )
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--dim-batch", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=2)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-lenses", nargs="*", default=[])
    parser.add_argument("--shard-manifests", nargs="*", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    os.makedirs(args.private_dir, exist_ok=True)
    recorded = sys.argv if argv is None else [sys.argv[0], *argv]
    _write(args.private_dir, f"argv_{args.command}_{int(time.time())}.json", recorded)
    torch.manual_seed(args.seed)
    if args.command == "merge":
        cmd_merge(args)
        return
    adapter, glue, identity = load(args)
    commands = {
        "measure": cmd_measure,
        "fit": cmd_fit,
        "run": cmd_run,
        "evaluate": cmd_evaluate,
    }
    commands[args.command](adapter, glue, identity, args)


if __name__ == "__main__":
    main()
