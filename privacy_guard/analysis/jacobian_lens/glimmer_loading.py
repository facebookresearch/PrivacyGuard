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

"""Load the pinned Muse Glimmer checkpoint into a ``GlimmerAdapter``.

Every J-lens preview job loads the model through ``load_glimmer_adapter`` so
identity, hash verification and loading diagnostics are recorded the same way.
The checkpoint is a local copy of the pinned Hugging Face revision, e.g.
``hf download meta-models/Muse-Glimmer-30B --revision <GLIMMER_REVISION_SHA>``.
"""

import json
import logging
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from privacy_guard.analysis.jacobian_lens.glimmer_adapter import (
    build_identity,
    child_module,
    GlimmerAdapter,
    hash_checkpoint_dir,
)
from safetensors import safe_open
from transformers import (  # @dep=fbsource//third-party/pypi/accelerate:accelerate
    AutoTokenizer,
    MuseGlimmerForConditionalGeneration,
)


logger: logging.Logger = logging.getLogger(__name__)

GLIMMER_REPO_ID: str = "meta-models/Muse-Glimmer-30B"
GLIMMER_REVISION_SHA: str = "1297fd7dfeac4598948a5d15886968435c81c2b7"
# Git LFS SHA-256 of each shard at GLIMMER_REVISION_SHA.
GLIMMER_SHARD_SHA256: dict[str, str] = {
    "model-00001-of-00002.safetensors": (
        "8eef61530e1283642c77ce2e6721feb5c6f348fa055c00e90f2844a136372694"
    ),
    "model-00002-of-00002.safetensors": (
        "b58cc2144ba1ba1af4420f67f4ca3ced7f09298510b80464cc75018a0be14381"
    ),
}

_VISION_MODULES: tuple[str, ...] = (
    "vision_tower",
    "vision_adapter",
    "vision_projection",
)


def checked_weight_names(num_layers: int) -> tuple[str, ...]:
    last = f"model.language_model.layers.{num_layers - 1}"
    return (
        "lm_head.weight",
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        f"{last}.mlp.down_proj.weight",
        f"{last}.post_feedforward_layernorm.weight",
    )


@dataclass
class LoadReport:
    hash_s: float
    load_s: float
    loading_info: dict[str, list[str]]
    checkpoint_weight_parity: dict[str, dict[str, Any]]
    memory_after_load_gib: float | None
    vision_modules_dropped: list[str]
    dtype_conversion: str | None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def verify_shards(checkpoint_dir: str, expected: Mapping[str, str]) -> dict[str, str]:
    """Hash every shard and refuse to continue unless all match."""
    actual = hash_checkpoint_dir(checkpoint_dir)
    if actual != expected:
        raise ValueError(
            f"checkpoint hash mismatch: expected {dict(expected)}, got {actual}"
        )
    return actual


def checkpoint_weight_parity(
    model: torch.nn.Module, checkpoint_dir: str, num_layers: int
) -> dict[str, dict[str, Any]]:
    """Compare loaded parameters with tensors read directly from the shards."""
    with open(os.path.join(checkpoint_dir, "model.safetensors.index.json")) as handle:
        weight_map = json.load(handle)["weight_map"]
    params = dict(model.named_parameters())
    result = {}
    for key in checked_weight_names(num_layers):
        with safe_open(os.path.join(checkpoint_dir, weight_map[key]), "pt") as shard:
            stored = shard.get_tensor(key)
        loaded = params[key].detach().to("cpu", stored.dtype)
        result[key] = {
            "shape": list(stored.shape),
            "dtype": str(stored.dtype).replace("torch.", ""),
            "bitwise_equal": bool(torch.equal(loaded, stored)),
        }
    return result


def _drop_vision_modules(model: torch.nn.Module) -> list[str]:
    """Release the perception encoder; text-only inputs never reach it."""
    dropped = []
    inner = child_module(model, "model")
    for name in _VISION_MODULES:
        if getattr(inner, name, None) is not None:
            setattr(inner, name, None)
            dropped.append(f"model.{name}")
    return dropped


def _cast_parameters_in_place(model: torch.nn.Module, dtype: torch.dtype) -> None:
    """Cast one tensor at a time so peak memory stays near the target size.

    Float32 buffers (rotary inverse frequencies) already hold the precision the
    native model computes them in and are left unchanged.
    """
    for module in model.modules():
        for parameter in module.parameters(recurse=False):
            parameter.data = parameter.data.to(dtype)
        for name, buffer in list(module.named_buffers(recurse=False)):
            if buffer.is_floating_point() and buffer.dtype != torch.float32:
                module._buffers[name] = buffer.to(dtype)


def load_glimmer_adapter(
    checkpoint_dir: str,
    expected_sha256: Mapping[str, str] = GLIMMER_SHARD_SHA256,
    revision_sha: str = GLIMMER_REVISION_SHA,
    repo_id: str = GLIMMER_REPO_ID,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    text_only: bool = True,
) -> tuple[GlimmerAdapter, LoadReport]:
    """Verify, load and wrap the checkpoint.

    The checkpoint is loaded in its stored bfloat16. ``text_only`` drops the
    perception encoder after loading. Another ``dtype`` (float32 for CPU
    reference runs) is applied afterwards, tensor by tensor.
    """
    started = time.perf_counter()
    shard_sha = verify_shards(checkpoint_dir, expected_sha256)
    hash_s = time.perf_counter() - started
    started = time.perf_counter()
    loaded: Any = MuseGlimmerForConditionalGeneration.from_pretrained(
        checkpoint_dir,
        dtype=torch.bfloat16,
        device_map=device,
        output_loading_info=True,
    )
    model, loading_info = loaded
    tokenizer: Any = AutoTokenizer.from_pretrained(checkpoint_dir)
    num_layers = int(model.config.text_config.num_hidden_layers)
    weight_parity = checkpoint_weight_parity(model, checkpoint_dir, num_layers)
    dropped = _drop_vision_modules(model) if text_only else []
    conversion = None
    if dtype != torch.bfloat16:
        _cast_parameters_in_place(model, dtype)
        conversion = f"bfloat16->{str(dtype).replace('torch.', '')} after load"
    load_s = time.perf_counter() - started
    identity = build_identity(model, checkpoint_dir, repo_id, revision_sha, shard_sha)
    report = LoadReport(
        hash_s=hash_s,
        load_s=load_s,
        loading_info={
            key: sorted(str(item) for item in value)
            for key, value in loading_info.items()
        },
        checkpoint_weight_parity=weight_parity,
        memory_after_load_gib=torch.cuda.memory_allocated() / 2**30
        if device.startswith("cuda")
        else None,
        vision_modules_dropped=dropped,
        dtype_conversion=conversion,
    )
    logger.info("loaded %s in %.1fs", identity.model_class, load_s)
    return GlimmerAdapter(model, tokenizer, identity), report
