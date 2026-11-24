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

"""Numerical receipt for the Muse Glimmer adapter on a real checkpoint.

Loads ``MuseGlimmerForConditionalGeneration`` from a local copy of the pinned
Hugging Face revision, verifies shard SHA-256 against expected values, and
writes a ``privacyguard.jlens_model_receipt/v1`` JSON file containing only
scalars and token-level summaries (no activations).

    python -m privacy_guard.analysis.jacobian_lens.glimmer_numerics \
      --checkpoint_dir=<dir> --expected_sha256_json=<json> --output=<receipt.json>
"""

import argparse
import hashlib
import inspect
import json
import logging
import os
import platform
import sys
import time
from typing import Any

import torch
from privacy_guard.analysis.jacobian_lens import (
    glimmer_adapter,
    glimmer_loading,
    model_numerics,
)
from privacy_guard.analysis.jacobian_lens.glimmer_adapter import (
    GlimmerAdapter,
    sha256_file,
)
from privacy_guard.analysis.jacobian_lens.glimmer_loading import (
    GLIMMER_REPO_ID,
    GLIMMER_REVISION_SHA,
    GLIMMER_SHARD_SHA256,
    load_glimmer_adapter,
)
from privacy_guard.analysis.jacobian_lens.model_numerics import (
    compare_vjp_sketches,
    compute_profile,
    gradient_checks,
    head_parity,
    RECEIPT_SCHEMA,
    resume_parity,
)


logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_PROMPTS: tuple[str, ...] = (
    "The number of legs on the animal that spins webs is",
    "The number of legs on the animal that spins webs is.",
    "Fact: The number of legs on the animal that spins webs is ",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--repo_id", default=GLIMMER_REPO_ID)
    parser.add_argument("--revision_sha", default=GLIMMER_REVISION_SHA)
    parser.add_argument(
        "--expected_sha256_json",
        default=None,
        help="JSON {shard: sha256}; defaults to the pinned GLIMMER_SHARD_SHA256",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--grad_layers", default="12,25,38")
    parser.add_argument("--resume_layers", default="-1,0,12,25,38,50")
    parser.add_argument("--eps_relative", default="0.003,0.01,0.03,0.1")
    parser.add_argument("--profile_batches", default="1,4,8,16")
    parser.add_argument("--profile_seq_len", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument(
        "--code_rev",
        default="",
        help="source revision of this checkout (recorded in the receipt)",
    )
    parser.add_argument(
        "--reference_receipt",
        default=None,
        help="receipt from another precision/device to compare gradient sketches with",
    )
    return parser.parse_args()


def _ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part]


def _token_rows(
    adapter: GlimmerAdapter, scores: torch.Tensor, ids: list[int]
) -> list[dict[str, Any]]:
    decode = adapter.tokenizer.decode
    return [
        {"token_id": i, "text": decode([i]), "score": float(scores[i])} for i in ids
    ]


def _prompt_summary(
    adapter: GlimmerAdapter, prompt: str, max_new_tokens: int
) -> dict[str, Any]:
    ids = adapter.encode(prompt)
    native = adapter.native_logits(ids)[0, -1]
    with torch.no_grad():
        final = adapter.forward_capture(ids, layers=(), compute_logits=False)
        fp32 = adapter.head_fp32(final.final_residual)
    generation_config: Any = adapter.model.generation_config
    eos = generation_config.eos_token_id
    stop_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}
    greedy = adapter.greedy_continuation(ids, max_new_tokens, stop_ids=stop_ids)
    decode = adapter.tokenizer.decode
    top = native.float().topk(10).indices.tolist()
    return {
        "text": prompt,
        "token_ids": ids[0].tolist(),
        "token_texts": [decode([i]) for i in ids[0].tolist()],
        "next_token_top10_native": _token_rows(adapter, native.float(), top),
        "next_token_top10_fp32_head": _token_rows(
            adapter, fp32[0, -1], fp32[0, -1].topk(10).indices.tolist()
        ),
        "greedy_stop_ids": sorted(stop_ids),
        "greedy_token_ids": greedy,
        "greedy_text": decode(greedy),
    }


def redact_path_arguments(argv: list[str]) -> list[str]:
    """Replace local filesystem values (``--flag=/abs/path``) with ``<flag>``."""
    redacted = []
    for arg in argv:
        flag, sep, value = arg.partition("=")
        if sep and (os.path.isabs(value) or value.startswith("~")):
            arg = f"{flag}=<{flag.lstrip('-')}>"
        redacted.append(arg)
    return redacted


def _source_hashes() -> dict[str, str]:
    """SHA-256 of the analysis modules this process actually imported."""
    modules = (glimmer_adapter, glimmer_loading, model_numerics, sys.modules[__name__])
    return {
        os.path.basename(inspect.getsourcefile(module) or module.__name__): (
            sha256_file(inspect.getsourcefile(module) or "")
        )
        for module in modules
    }


def _environment(device: str) -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    if device.startswith("cuda"):
        props = torch.cuda.get_device_properties(0)
        env.update(
            gpu_name=props.name,
            gpu_total_memory_gib=props.total_memory / 2**30,
            gpu_count=torch.cuda.device_count(),
        )
    return env


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    expected = GLIMMER_SHARD_SHA256
    if args.expected_sha256_json is not None:
        with open(args.expected_sha256_json) as handle:
            expected = json.load(handle)
    adapter, report = load_glimmer_adapter(
        args.checkpoint_dir,
        expected,
        revision_sha=args.revision_sha,
        repo_id=args.repo_id,
        device=args.device,
        dtype=getattr(torch, args.dtype),
    )
    receipt = _build_receipt(adapter, args)
    receipt["load_report"] = report.to_dict()
    receipt["code"] = {
        "privacy_guard_rev": args.code_rev,
        "argv": redact_path_arguments(sys.argv[1:]),
        "source_sha256": _source_hashes(),
    }
    if args.reference_receipt is not None:
        with open(args.reference_receipt, "rb") as handle:
            raw = handle.read()
        receipt["reference_receipt_sha256"] = hashlib.sha256(raw).hexdigest()
        receipt["vjp_vs_reference"] = compare_vjp_sketches(receipt, json.loads(raw))
    with open(args.output, "w") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
    logger.info("wrote %s", args.output)


def _build_receipt(adapter: GlimmerAdapter, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    prompts = [
        _prompt_summary(adapter, p, args.max_new_tokens) for p in DEFAULT_PROMPTS
    ]
    ids = adapter.encode(DEFAULT_PROMPTS[0])
    last = ids.shape[1] - 1
    eps = [float(e) for e in args.eps_relative.split(",")]
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "run_kind": "real_model",
        "model_identity": adapter.identity().to_dict(),
        "environment": _environment(args.device),
        "prompts": prompts,
        "head_parity": head_parity(adapter, ids),
        "resume_parity": resume_parity(adapter, ids, _ints(args.resume_layers)),
        "gradient_checks": [
            gradient_checks(adapter, ids, layer, last // 2, last, eps)
            for layer in _ints(args.grad_layers)
        ],
    }
    receipt["compute_profile"] = (
        compute_profile(adapter, args.profile_seq_len, _ints(args.profile_batches))
        if args.profile_batches
        else []
    )
    receipt["timing"] = {"checks_s": time.perf_counter() - started}
    return receipt


if __name__ == "__main__":
    main()
