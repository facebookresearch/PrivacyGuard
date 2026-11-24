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

"""Differentiable residual-stream adapter for Hugging Face Muse Glimmer models.

The adapter wraps an already loaded ``MuseGlimmerForConditionalGeneration``
(transformers >= 5.15) and exposes the text decoder for Jacobian-lens work:

* ``layer i`` is the output of decoder block ``i`` (0-based); ``layer -1`` is
  the normed token embedding that enters block 0.
* ``final_residual`` is the output of the last block, before the final norm.
* ``head`` applies the native output path exactly as ``model.forward`` does:
  final RMSNorm, ``lm_head``, ``output_multiplier`` and tanh softcapping.

Weights are frozen. Gradients flow from a ``requires_grad`` embedding leaf
through every block, so one backward pass yields derivatives with respect to
every captured residual without detaching intermediate activations.
"""

import hashlib
import inspect
import logging
import os
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import torch
import transformers
from torch import nn
from transformers import PreTrainedModel
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)


logger: logging.Logger = logging.getLogger(__name__)

EMBEDDING_LAYER: int = -1
_HASH_CHUNK_BYTES: int = 64 * 1024 * 1024

ResidualOverride = Callable[[torch.Tensor], torch.Tensor]


class Tokenizer(Protocol):
    """The subset of a Hugging Face tokenizer the adapter uses."""

    def __call__(self, text: str, **kwargs: Any) -> Mapping[str, torch.Tensor]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...


@dataclass(frozen=True)
class ModelIdentity:
    """Immutable description of the exact model a receipt was produced with."""

    repo_id: str
    revision_sha: str
    safetensors_sha256_index: dict[str, str]
    tokenizer_sha256: str
    config_sha256: str
    modeling_source_sha256: str
    model_class: str
    transformers_version: str
    torch_version: str
    dtype: str
    device: str
    lm_head_separate_from_embedding: bool
    num_layers: int
    d_model: int
    vocab_size: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ForwardCapture:
    """Residuals captured from one forward pass of the text decoder."""

    input_ids: torch.Tensor
    embeddings: torch.Tensor
    residuals: dict[int, torch.Tensor]
    final_residual: torch.Tensor
    logits: torch.Tensor | None


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def hash_checkpoint_dir(checkpoint_dir: str) -> dict[str, str]:
    """SHA-256 of every safetensors shard, keyed by file name."""
    names = sorted(
        name for name in os.listdir(checkpoint_dir) if name.endswith(".safetensors")
    )
    if not names:
        raise FileNotFoundError(f"no .safetensors files in {checkpoint_dir}")
    return {name: sha256_file(os.path.join(checkpoint_dir, name)) for name in names}


def child_module(module: nn.Module, *names: str) -> nn.Module:
    """Follow attribute ``names`` from ``module``, checking each is a Module."""
    for name in names:
        child = getattr(module, name)
        if not isinstance(child, nn.Module):
            raise TypeError(f"{name!r} is {type(child).__name__}, not an nn.Module")
        module = child
    return module


class GlimmerAdapter:
    """Frozen, graph-connected view of a Muse Glimmer text decoder."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Tokenizer,
        identity: ModelIdentity | None = None,
    ) -> None:
        config: Any = model.config
        text_config: Any = config.text_config
        if text_config.model_type != "muse_glimmer_text":
            raise ValueError(
                f"expected a muse_glimmer_text config, got {text_config.model_type!r}"
            )
        self.model: PreTrainedModel = model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.tokenizer: Tokenizer = tokenizer
        self.text_config: Any = text_config
        self._text_model: nn.Module = child_module(model, "model", "language_model")
        self._embed: nn.Module = child_module(self._text_model, "embed_tokens")
        self._final_norm: nn.Module = child_module(self._text_model, "norm")
        self._rotary: nn.Module = child_module(self._text_model, "rotary_emb")
        layers = child_module(self._text_model, "layers")
        lm_head = child_module(model, "lm_head")
        if not isinstance(layers, nn.ModuleList) or not isinstance(lm_head, nn.Linear):
            raise TypeError("expected ModuleList decoder layers and a Linear lm_head")
        self._layers: nn.ModuleList = layers
        self._lm_head: nn.Linear = lm_head
        self.num_layers: int = int(text_config.num_hidden_layers)
        self.d_model: int = int(text_config.hidden_size)
        self.vocab_size: int = int(text_config.vocab_size)
        self.output_multiplier: float = float(text_config.output_multiplier)
        softcap = text_config.final_logit_softcapping
        self.final_logit_softcapping: float | None = (
            None if softcap is None else float(softcap)
        )
        self._multimodal_token_ids: frozenset[int] = frozenset(
            int(token_id)
            for token_id in (
                getattr(config, "image_token_id", None),
                getattr(config, "video_token_id", None),
            )
            if token_id is not None
        )
        self._identity = identity

    @property
    def device(self) -> torch.device:
        return self._lm_head.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self._lm_head.weight.dtype

    def identity(self) -> ModelIdentity:
        if self._identity is None:
            raise ValueError("adapter was built without a ModelIdentity")
        return self._identity

    def tokenize(self, text: str, add_special_tokens: bool = True) -> torch.Tensor:
        encoded = self.tokenizer(
            text, return_tensors="pt", add_special_tokens=add_special_tokens
        )
        return encoded["input_ids"].to(self.device)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Normed token embeddings entering block 0 (text-only inputs)."""
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {input_ids.shape}")
        if self._multimodal_token_ids and any(
            int(token_id) in self._multimodal_token_ids
            for token_id in input_ids.unique().tolist()
        ):
            raise ValueError("image/video placeholder tokens are not supported")
        return self._embed(input_ids.to(self.device))

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        """Native output head: final norm, lm_head, output scale, softcap.

        Runs the same modules and dtype sequence as
        ``MuseGlimmerForConditionalGeneration.forward``.
        """
        logits = self._lm_head(self._final_norm(hidden))
        return self._scale_and_softcap(logits)

    def final_norm(self, hidden: torch.Tensor) -> torch.Tensor:
        """The final RMSNorm (``language_model.norm``) alone."""
        return self._final_norm(hidden)

    def head_fp32(self, hidden: torch.Tensor, chunk_rows: int = 32768) -> torch.Tensor:
        """The native head evaluated in float32 to resolve bfloat16 ties."""
        normed = self._final_norm(hidden.float())
        weight = self._lm_head.weight
        parts = [
            normed @ weight[start : start + chunk_rows].float().T
            for start in range(0, weight.shape[0], chunk_rows)
        ]
        return self._scale_and_softcap(torch.cat(parts, dim=-1))

    def raw_unembed(self, hidden: torch.Tensor) -> torch.Tensor:
        """``lm_head(norm(h))`` before output scaling and softcapping."""
        return self._lm_head(self._final_norm(hidden))

    def _scale_and_softcap(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits * self.output_multiplier
        cap = self.final_logit_softcapping
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        return logits

    def _block_context(
        self, embeddings: torch.Tensor
    ) -> tuple[dict[str, Any], Any, torch.Tensor]:
        seq_len = embeddings.shape[1]
        position_ids = torch.arange(seq_len, device=embeddings.device).unsqueeze(0)
        mask_kwargs = {
            "config": self.text_config,
            "inputs_embeds": embeddings,
            "attention_mask": None,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        masks = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
        }
        position_embeddings = self._rotary(embeddings, position_ids)
        return masks, position_embeddings, position_ids

    def _run_blocks(
        self,
        hidden: torch.Tensor,
        start: int,
        capture: Iterable[int],
        overrides: Mapping[int, ResidualOverride] | None,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Run blocks ``start..L-1`` exactly as ``MuseGlimmerTextModel.forward``."""
        masks, position_embeddings, position_ids = self._block_context(hidden)
        wanted = set(capture)
        captured: dict[int, torch.Tensor] = {}
        layer_types = self.text_config.layer_types
        rope_theta = self.text_config.layer_rope_theta
        for index in range(start, self.num_layers):
            hidden = self._layers[index](
                hidden,
                attention_mask=masks[layer_types[index]],
                position_embeddings=position_embeddings if rope_theta[index] else None,
                position_ids=position_ids,
                past_key_values=None,
            )
            if overrides is not None and index in overrides:
                hidden = overrides[index](hidden)
            if index in wanted:
                captured[index] = hidden
        return hidden, captured

    def forward_capture(
        self,
        input_ids: torch.Tensor,
        layers: Sequence[int] | None = None,
        requires_grad: bool = False,
        compute_logits: bool = True,
        overrides: Mapping[int, ResidualOverride] | None = None,
    ) -> ForwardCapture:
        """Forward pass returning post-block residuals for ``layers``.

        With ``requires_grad`` the embedding output becomes the only autograd
        leaf; every captured residual stays connected to the final residual,
        so ``torch.autograd.grad(final, [residuals...])`` differentiates them
        all in one backward pass. ``overrides`` maps a block index (or -1)
        to a function applied to that block's output before later blocks
        run; captured residuals are the post-override values.
        """
        capture = list(range(self.num_layers)) if layers is None else list(layers)
        for layer in capture:
            if not EMBEDDING_LAYER <= layer < self.num_layers:
                raise ValueError(f"layer {layer} outside [-1, {self.num_layers})")
        grad_context = torch.enable_grad() if requires_grad else torch.no_grad()
        with grad_context:
            embeddings = self.embed(input_ids)
            if requires_grad:
                embeddings = embeddings.detach().requires_grad_(True)
            hidden = embeddings
            if overrides is not None and EMBEDDING_LAYER in overrides:
                hidden = overrides[EMBEDDING_LAYER](hidden)
            final, residuals = self._run_blocks(hidden, 0, capture, overrides)
            if EMBEDDING_LAYER in capture:
                residuals[EMBEDDING_LAYER] = hidden
            logits = self.head(final) if compute_logits else None
        return ForwardCapture(
            input_ids=input_ids,
            embeddings=embeddings,
            residuals=residuals,
            final_residual=final,
            logits=logits,
        )

    def resume_from(
        self,
        layer: int,
        hidden: torch.Tensor,
        compute_logits: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Continue the forward pass from the output of block ``layer``.

        ``hidden`` is the full ``[batch, seq, d]`` residual after block
        ``layer`` (``-1`` for the embedding). Positions, causal and sliding
        masks and NoPE layers are rebuilt from the sequence length exactly as
        in a full forward pass. Returns ``(final_residual, logits)``.
        """
        if not EMBEDDING_LAYER <= layer < self.num_layers:
            raise ValueError(f"layer {layer} outside [-1, {self.num_layers})")
        final, _ = self._run_blocks(hidden, layer + 1, (), None)
        return final, self.head(final) if compute_logits else None

    @property
    def n_layers(self) -> int:
        return self.num_layers

    @property
    def layers(self) -> nn.ModuleList:
        """The decoder blocks; forward hooks on them fire in every pass."""
        return self._layers

    def encode(self, text: str) -> torch.Tensor:
        """``[1, T]`` token ids with the tokenizer's default special tokens."""
        return self.tokenize(text, add_special_tokens=True)

    def unembed(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(hidden)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run every block and return the final residual (pre final norm).

        When autograd is enabled the embedding output is a ``requires_grad``
        leaf, so hooked block outputs are differentiable. The vocabulary
        head is not evaluated; call ``unembed`` on the result if needed.
        """
        capture = self.forward_capture(
            input_ids,
            layers=(),
            requires_grad=torch.is_grad_enabled(),
            compute_logits=False,
        )
        return capture.final_residual

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.forward(input_ids)

    def native_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Logits from the unmodified ``model.forward`` (reference path)."""
        with torch.no_grad():
            return self.model(
                input_ids=input_ids.to(self.device), use_cache=False
            ).logits

    def greedy_continuation(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        stop_ids: Collection[int] = (),
    ) -> list[int]:
        """Greedy decode with repeated full native forwards (no KV cache).

        Stops after emitting a token in ``stop_ids``; ties go to the lowest id.
        """
        generated: list[int] = []
        current = input_ids.to(self.device)
        for _ in range(max_new_tokens):
            next_id = int(self.native_logits(current)[0, -1].argmax())
            generated.append(next_id)
            if next_id in stop_ids:
                break
            current = torch.cat(
                [current, torch.tensor([[next_id]], device=current.device)], dim=1
            )
        return generated


def build_identity(
    model: PreTrainedModel,
    checkpoint_dir: str,
    repo_id: str,
    revision_sha: str,
    shard_sha256: Mapping[str, str] | None = None,
) -> ModelIdentity:
    """Hash the local checkpoint and describe the loaded model."""
    config: Any = model.config
    text_config: Any = config.text_config
    modeling_path = inspect.getsourcefile(type(model))
    first_param = next(model.parameters())
    embed_weight = child_module(
        model, "model", "language_model", "embed_tokens"
    ).get_parameter("weight")
    lm_head_weight = child_module(model, "lm_head").get_parameter("weight")
    return ModelIdentity(
        repo_id=repo_id,
        revision_sha=revision_sha,
        safetensors_sha256_index=dict(shard_sha256)
        if shard_sha256 is not None
        else hash_checkpoint_dir(checkpoint_dir),
        tokenizer_sha256=sha256_file(os.path.join(checkpoint_dir, "tokenizer.json")),
        config_sha256=sha256_file(os.path.join(checkpoint_dir, "config.json")),
        modeling_source_sha256=sha256_file(modeling_path) if modeling_path else "",
        model_class=type(model).__name__,
        transformers_version=transformers.__version__,
        torch_version=torch.__version__,
        dtype=str(first_param.dtype).replace("torch.", ""),
        device=str(first_param.device),
        lm_head_separate_from_embedding=lm_head_weight.data_ptr()
        != embed_weight.data_ptr(),
        num_layers=int(text_config.num_hidden_layers),
        d_model=int(text_config.hidden_size),
        vocab_size=int(text_config.vocab_size),
    )
