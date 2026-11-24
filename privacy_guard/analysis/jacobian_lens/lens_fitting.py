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
"""Experimental Jacobian lens (J-lens) estimator for decoder language models.

The estimator follows the Apache-2.0 reference implementation
https://github.com/anthropics/jacobian-lens at commit
581d398613e5602a5af361e1c34d3a92ea82ba8e (``jlens/fitting.py``,
``jlens/lens.py``, ``jlens/hooks.py``)::

    lens_l(h) = unembed(J_l @ h),  J_l = E[d h_final / d h_l]

For each target coordinate a one-hot cotangent is placed at every valid target
position at once, so the gradient at source position ``p`` sums the causal
current-and-future target derivatives. Rows are averaged over valid source
positions and then equally over calibration prompts. Model weights are frozen;
only activation gradients are taken and no optimizer is involved.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import torch
from torch import nn

logger: logging.Logger = logging.getLogger(__name__)

RECIPE = "causal-target-sum_source-mean_prompt-mean/v1"
UPSTREAM_REFERENCE = "https://github.com/anthropics/jacobian-lens@581d398613e5602a5af361e1c34d3a92ea82ba8e"
SKIP_FIRST_N_POSITIONS = 16

ResidualEdit = Callable[[torch.Tensor], torch.Tensor]


class LensModel(Protocol):
    """What the estimator needs from a model (the upstream ``LensModel`` shape).

    ``layers`` are the residual blocks; each block output is the post-block
    residual. ``forward`` runs only the residual stack (no LM head) and must
    build an autograd graph when gradients are enabled. ``unembed`` applies the
    model's full native output head (final norm, unembedding and any output
    scaling or soft-capping) to a pre-norm residual.
    """

    @property
    def n_layers(self) -> int: ...

    @property
    def d_model(self) -> int: ...

    @property
    def layers(self) -> nn.ModuleList: ...

    def forward(self, input_ids: torch.Tensor, /) -> object: ...

    def unembed(self, residual: torch.Tensor, /) -> torch.Tensor: ...


class ResidualHooks:
    """Record and optionally edit post-block residuals through forward hooks.

    ``edits`` maps a block index to a function of that block's output; the
    returned tensor replaces the output, so downstream blocks see the edit.
    Recorded tensors are the post-edit values and are not detached; for edited
    blocks the pre-edit output is kept in ``pre_edit``.
    ``graph_root`` marks that block's output ``requires_grad`` so, with frozen
    parameters, the retained graph starts there.
    """

    def __init__(
        self,
        blocks: nn.ModuleList | Sequence[nn.Module],
        record: Iterable[int] = (),
        *,
        edits: Mapping[int, ResidualEdit] | None = None,
        graph_root: int | None = None,
    ) -> None:
        self._blocks = blocks
        self._edits: dict[int, ResidualEdit] = dict(edits or {})
        self._record: set[int] = set(record)
        self._graph_root = graph_root
        if graph_root is not None:
            self._record.add(graph_root)
        self.activations: dict[int, torch.Tensor] = {}
        self.pre_edit: dict[int, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, index: int) -> Callable[..., object]:
        def hook(module: nn.Module, inputs: object, output: Any) -> object:
            is_tensor = isinstance(output, torch.Tensor)
            tensor: torch.Tensor = output if is_tensor else output[0]
            if index == self._graph_root:
                tensor.requires_grad_(True)
            if index in self._edits:
                if index in self._record:
                    self.pre_edit[index] = tensor
                tensor = self._edits[index](tensor)
            if index in self._record:
                self.activations[index] = tensor
            if index not in self._edits:
                return None
            return tensor if is_tensor else (tensor, *output[1:])

        return hook

    def __enter__(self) -> ResidualHooks:
        indices = sorted(self._record | set(self._edits))
        try:
            for index in indices:
                self._handles.append(
                    self._blocks[index].register_forward_hook(self._make_hook(index))
                )
        except Exception:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


class PromptTooShortError(ValueError):
    """The prompt leaves no position for the estimator's mask."""


def valid_position_mask(
    seq_len: int, *, skip_first: int = SKIP_FIRST_N_POSITIONS
) -> torch.Tensor:
    """Positions used by the estimator: skip the first ``skip_first`` positions
    (attention-sink statistics) and the final one (no next-token target)."""
    if skip_first < 0:
        raise ValueError(f"skip_first must be >= 0, got {skip_first}")
    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[skip_first : seq_len - 1] = True
    if not bool(mask.any()):
        raise PromptTooShortError(
            f"prompt too short: seq_len={seq_len}, need > {skip_first + 1} tokens"
        )
    return mask


def resolve_layers(
    source_layers: Sequence[int] | None, target_layer: int | None, n_layers: int
) -> tuple[list[int], int]:
    """Resolve negative indices and require every source to precede the target."""
    if n_layers < 2:
        raise ValueError(f"need at least two layers, got n_layers={n_layers}")
    target = n_layers - 1 if target_layer is None else target_layer
    target = target + n_layers if target < 0 else target
    if not 0 <= target < n_layers:
        raise ValueError(f"target_layer={target_layer} out of range")
    if source_layers is None:
        return list(range(target)), target
    sources = sorted({s + n_layers if s < 0 else s for s in source_layers})
    if not sources or sources[0] < 0 or sources[-1] >= target:
        raise ValueError(f"source layers {sources} must lie in [0, {target})")
    return sources, target


def jacobian_for_prompt(
    model: LensModel,
    input_ids: torch.Tensor,
    source_layers: Sequence[int],
    *,
    target_layer: int | None = None,
    dim_batch: int = 8,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
) -> tuple[dict[int, torch.Tensor], int]:
    """Per-prompt estimator ``J_l`` for one unpadded ``[1, T]`` input.

    One forward on the prompt replicated ``dim_batch`` times along the batch
    axis retains the graph; each of ``ceil(d_model / dim_batch)`` backward
    passes fills ``dim_batch`` rows of every source layer's ``J_l``.

    Returns ``({layer: fp32 CPU [d_model, d_model]}, n_valid_positions)``.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("expected one unpadded input of shape [1, T]")
    if dim_batch < 1:
        raise ValueError("dim_batch must be >= 1")
    sources, target = resolve_layers(source_layers, target_layer, model.n_layers)
    d_model = model.d_model
    mask = valid_position_mask(input_ids.shape[1], skip_first=skip_first)
    jacobians = {
        layer: torch.zeros(d_model, d_model, dtype=torch.float32) for layer in sources
    }
    n_passes = math.ceil(d_model / dim_batch)
    with (
        ResidualHooks(
            model.layers, record=[*sources, target], graph_root=min(sources)
        ) as hooks,
        torch.enable_grad(),
    ):
        model.forward(input_ids.expand(dim_batch, -1))
        z = hooks.activations[target]
        xs = [hooks.activations[layer] for layer in sources]
        if z.grad_fn is None or z.shape[:2] != (dim_batch, input_ids.shape[1]):
            raise ValueError("target residual is not graph-connected [B, T, D]")
        positions = mask.nonzero(as_tuple=True)[0].to(z.device)
        batch = torch.arange(dim_batch, device=z.device)
        cotangent = torch.zeros_like(z)
        for pass_idx, start in enumerate(range(0, d_model, dim_batch)):
            n_rows = min(dim_batch, d_model - start)
            cotangent.zero_()
            cotangent[
                batch[:n_rows, None], positions[None, :], start + batch[:n_rows, None]
            ] = 1.0
            grads = torch.autograd.grad(
                outputs=z,
                inputs=xs,
                grad_outputs=cotangent,
                retain_graph=pass_idx < n_passes - 1,
            )
            for layer, grad in zip(sources, grads, strict=True):
                rows = grad[:n_rows, positions.to(grad.device), :].float().mean(dim=1)
                if not bool(torch.isfinite(rows).all()):
                    raise FloatingPointError(
                        f"nonfinite Jacobian rows at layer {layer}"
                    )
                jacobians[layer][start : start + n_rows].copy_(rows.cpu())
            del grads
    return jacobians, int(mask.sum())


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_save(obj: object, path: str) -> None:
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


class JacobianLens:
    """Fitted per-layer ``J_l`` matrices plus the transport ``J_l @ h``.

    ``target_layer`` and ``skip_first`` record the fit's target block and
    position mask (``None`` for lenses saved before they were recorded).
    """

    def __init__(
        self,
        jacobians: Mapping[int, torch.Tensor],
        *,
        n_prompts: int,
        d_model: int,
        target_layer: int | None = None,
        skip_first: int | None = None,
    ) -> None:
        if n_prompts < 1 or not jacobians:
            raise ValueError("a lens needs at least one prompt and one layer")
        for layer, matrix in jacobians.items():
            if tuple(matrix.shape) != (d_model, d_model):
                raise ValueError(f"layer {layer} matrix is not [{d_model}, {d_model}]")
        self.jacobians: dict[int, torch.Tensor] = {
            int(k): v.float() for k, v in jacobians.items()
        }
        self.source_layers: list[int] = sorted(self.jacobians)
        self.n_prompts = n_prompts
        self.d_model = d_model
        self.target_layer = target_layer
        self.skip_first = skip_first

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        """Map ``[..., d_model]`` residuals at ``layer`` to the target basis."""
        matrix = self.jacobians[layer].to(residual.device)
        return residual.float() @ matrix.T

    def save(self, path: str, *, dtype: torch.dtype = torch.float16) -> str:
        """Save (fp16 by default, as upstream) and return the file SHA-256."""
        _atomic_save(
            {
                "J": {k: v.to(dtype) for k, v in self.jacobians.items()},
                "n_prompts": self.n_prompts,
                "source_layers": self.source_layers,
                "d_model": self.d_model,
                "recipe": RECIPE,
                "target_layer": self.target_layer,
                "skip_first": self.skip_first,
            },
            path,
        )
        return _sha256_file(path)

    @classmethod
    def load(cls, path: str) -> JacobianLens:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if "J" not in state:
            raise ValueError(f"{path} is not a saved JacobianLens")
        return cls(
            state["J"],
            n_prompts=state["n_prompts"],
            d_model=state["d_model"],
            target_layer=state.get("target_layer"),
            skip_first=state.get("skip_first"),
        )

    @classmethod
    def merge(cls, lenses: Sequence[JacobianLens]) -> JacobianLens:
        """Prompt-count-weighted mean of lenses fitted on disjoint prompt sets."""
        if not lenses:
            raise ValueError("merge() needs at least one lens")
        first = lenses[0]

        def key(x: JacobianLens) -> tuple[object, ...]:
            return (x.source_layers, x.d_model, x.target_layer, x.skip_first)

        if any(key(x) != key(first) for x in lenses):
            raise ValueError(
                "lenses disagree on source layers, width, target layer or mask"
            )
        total = sum(x.n_prompts for x in lenses)
        merged = {
            layer: torch.stack([x.jacobians[layer] * x.n_prompts for x in lenses]).sum(
                0
            )
            / total
            for layer in first.source_layers
        }
        return cls(
            merged,
            n_prompts=total,
            d_model=first.d_model,
            target_layer=first.target_layer,
            skip_first=first.skip_first,
        )


@dataclass
class PromptFitRecord:
    """Per-prompt fit diagnostics (no tensors or text)."""

    index: int
    prompt_sha256: str
    seq_len: int
    n_valid: int
    seconds: float
    max_norm_over_sqrt_d: float
    max_rel_mean_change: float | None
    status: str = "ok"
    reason: str = ""


@dataclass
class FitManifest:
    """Provenance for one fit run; JSON-serializable."""

    recipe: str
    upstream_reference: str
    source_layers: list[int]
    target_layer: int
    skip_first: int
    dim_batch: int
    d_model: int
    n_prompts_requested: int
    n_prompts_used: int = 0
    prompts: list[PromptFitRecord] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1, sort_keys=True)


def _resume_state(
    checkpoint_path: str | None, resume: bool, sources: list[int], fixed: dict[str, int]
) -> dict[str, Any] | None:
    if not (resume and checkpoint_path and os.path.exists(checkpoint_path)):
        return None
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected = {"source_layers": sources, **fixed}
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(
                f"checkpoint {checkpoint_path} has {key}={state.get(key)!r}, "
                f"expected {value!r}; pass resume=False to discard it"
            )
    return state


def _initial_state(
    state: dict[str, Any] | None,
    sources: list[int],
    d_model: int,
    prompt_hashes: Sequence[str],
) -> tuple[dict[int, torch.Tensor], int, int, list[PromptFitRecord]]:
    """Fresh running sums, or the resumed ones after checking the prompt order."""
    if state is None:
        zeros = {k: torch.zeros(d_model, d_model, dtype=torch.float32) for k in sources}
        return zeros, 0, 0, []
    records = [PromptFitRecord(**r) for r in state["records"]]
    next_idx = int(state["next_idx"])
    if [r.prompt_sha256 for r in records] != list(prompt_hashes[:next_idx]):
        raise ValueError("checkpoint was fitted on a different prompt sequence")
    return state["jacobian_sum"], int(state["n_done"]), next_idx, records


def _accumulate(
    total: dict[int, torch.Tensor], n_done: int, per_prompt: Mapping[int, torch.Tensor]
) -> float | None:
    """Add one prompt's ``J_l`` to the running sums in place and return the
    largest relative shift of the running mean (``None`` when undefined)."""
    rel_change = None
    if n_done:
        means = {k: total[k] / n_done for k in total}
        norms = {k: float(m.norm()) for k, m in means.items()}
        if min(norms.values()) > 0:
            rel_change = max(
                float((per_prompt[k] - means[k]).norm()) / ((n_done + 1) * norms[k])
                for k in total
            )
    for k in total:
        total[k] += per_prompt[k]
    return rel_change


def fit(
    model: LensModel,
    prompts: Sequence[torch.Tensor],
    *,
    prompt_hashes: Sequence[str],
    source_layers: Sequence[int] | None = None,
    target_layer: int | None = None,
    dim_batch: int = 8,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 1,
    resume: bool = True,
    on_prompt: Callable[[PromptFitRecord], None] | None = None,
    first_prompt_snapshot: str | None = None,
) -> tuple[JacobianLens, FitManifest]:
    """Fit ``J_l`` over pre-tokenized ``[1, T]`` prompts as an equal-weight mean.

    The running fp32 sum is checkpointed atomically every ``checkpoint_every``
    prompts and resumed on restart. A prompt too short for the position mask is
    recorded as skipped, never silently padded or re-masked.
    ``first_prompt_snapshot`` also keeps the state after the first fitted prompt
    as a separate file for independent recomputation.
    """
    if len(prompt_hashes) != len(prompts):
        raise ValueError("one SHA-256 per prompt is required")
    sources, target = resolve_layers(source_layers, target_layer, model.n_layers)
    d_model = model.d_model
    fixed = {"target_layer": target, "skip_first": skip_first}
    manifest = FitManifest(
        recipe=RECIPE,
        upstream_reference=UPSTREAM_REFERENCE,
        source_layers=sources,
        target_layer=target,
        skip_first=skip_first,
        dim_batch=dim_batch,
        d_model=d_model,
        n_prompts_requested=len(prompts),
    )
    total, n_done, next_idx, manifest.prompts = _initial_state(
        _resume_state(checkpoint_path, resume, sources, fixed),
        sources,
        d_model,
        prompt_hashes,
    )

    def write_checkpoint(path: str | None = checkpoint_path) -> None:
        if path is not None:
            _atomic_save(
                {
                    "jacobian_sum": total,
                    "n_done": n_done,
                    "next_idx": next_idx,
                    "source_layers": sources,
                    "records": [asdict(r) for r in manifest.prompts],
                    **fixed,
                },
                path,
            )

    sqrt_d = math.sqrt(d_model)
    for index in range(next_idx, len(prompts)):
        started = time.perf_counter()
        ids = prompts[index]
        try:
            per_prompt, n_valid = jacobian_for_prompt(
                model,
                ids,
                sources,
                target_layer=target,
                dim_batch=dim_batch,
                skip_first=skip_first,
            )
        except PromptTooShortError as exc:
            record = PromptFitRecord(
                index,
                prompt_hashes[index],
                int(ids.shape[-1]),
                0,
                0.0,
                0.0,
                None,
                status="skipped",
                reason=str(exc),
            )
        else:
            rel_change = _accumulate(total, n_done, per_prompt)
            n_done += 1
            record = PromptFitRecord(
                index,
                prompt_hashes[index],
                int(ids.shape[-1]),
                n_valid,
                time.perf_counter() - started,
                max(float(per_prompt[k].norm()) for k in sources) / sqrt_d,
                rel_change,
            )
        manifest.prompts.append(record)
        next_idx = index + 1
        if n_done == 1 and record.status == "ok" and first_prompt_snapshot:
            write_checkpoint(first_prompt_snapshot)
        if on_prompt is not None:
            on_prompt(record)
        if next_idx % max(1, checkpoint_every) == 0:
            write_checkpoint()
    write_checkpoint()
    if n_done == 0:
        raise ValueError("no prompt was long enough to fit on")
    manifest.n_prompts_used = n_done
    mean = {k: v / n_done for k, v in total.items()}
    lens = JacobianLens(
        mean,
        n_prompts=n_done,
        d_model=d_model,
        target_layer=target,
        skip_first=skip_first,
    )
    return lens, manifest
