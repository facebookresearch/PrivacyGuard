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

"""Numerical fidelity checks for a ``GlimmerAdapter``.

Every function returns JSON-serializable scalars (no tensors or activation
vectors), so results can be embedded in a ``privacyguard.jlens_model_receipt``.

* ``head_parity``: adapter head vs ``model.forward`` logits, the softcap
  formula, float32 head agreement, and the ``output_hidden_states`` layout.
* ``resume_parity``: resuming the forward pass from block ``l`` reproduces
  the native logits.
* ``gradient_checks``: activation vector-Jacobian products vs central finite
  differences (perturbation quantization accounted for), causality, and
  agreement between the full graph and a graph cut at the probed layer.
* ``compute_profile``: forward/backward wall time and peak accelerator memory.
"""

import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from privacy_guard.analysis.jacobian_lens.glimmer_adapter import GlimmerAdapter


RECEIPT_SCHEMA: str = "privacyguard.jlens_model_receipt/v1"

Scalar = Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max())


def head_parity(adapter: GlimmerAdapter, input_ids: torch.Tensor) -> dict[str, Any]:
    native = adapter.native_logits(input_ids)
    capture = adapter.forward_capture(input_ids)
    adapter_logits = capture.logits
    assert adapter_logits is not None
    final = capture.final_residual
    with torch.no_grad():
        raw = adapter.raw_unembed(final).float()
        cap = adapter.final_logit_softcapping
        formula = raw * adapter.output_multiplier
        if cap is not None:
            formula = torch.tanh(formula / cap) * cap
        fp32 = adapter.head_fp32(final)
    last_native = native[0, -1].float()
    top_native = int(last_native.argmax())
    return {
        "adapter_vs_native_max_abs": _max_abs(adapter_logits, native),
        "adapter_vs_native_bitwise_equal": bool(torch.equal(adapter_logits, native)),
        "formula_fp32_vs_native_max_abs": _max_abs(formula, native),
        "raw_unembed_max_abs": float(raw.abs().max()),
        "native_logit_max_abs": float(native.float().abs().max()),
        "softcap": cap,
        "output_multiplier": adapter.output_multiplier,
        "native_dtype": str(native.dtype).replace("torch.", ""),
        "fp32_head_vs_native_max_abs": _max_abs(fp32, native),
        "fp32_head_top1_agrees_all_positions": bool(
            torch.equal(fp32.argmax(-1), native.float().argmax(-1))
        ),
        "fp32_head_top10_overlap_last_position": len(
            set(fp32[0, -1].topk(10).indices.tolist())
            & set(last_native.topk(10).indices.tolist())
        ),
        "native_top1_tie_count_last_position": int(
            (last_native == last_native[top_native]).sum()
        ),
        **_hidden_states_layout(adapter, input_ids, capture.residuals, final),
    }


def _hidden_states_layout(
    adapter: GlimmerAdapter,
    input_ids: torch.Tensor,
    residuals: dict[int, torch.Tensor],
    final: torch.Tensor,
) -> dict[str, Any]:
    """Compare hooked residuals with ``output_hidden_states``.

    In transformers 5.19 the last entry of ``hidden_states`` is replaced by
    the final-normed state, so it must not be fed to ``head`` again.
    """
    with torch.no_grad():
        out = adapter.model(
            input_ids=input_ids, output_hidden_states=True, use_cache=False
        )
        states = out.hidden_states
        normed_final = adapter.final_norm(final)
    inner = max(
        (_max_abs(states[i + 1], residuals[i]) for i in range(adapter.num_layers - 1)),
        default=0.0,
    )
    return {
        "hidden_states_count": len(states),
        "hidden_states_inner_vs_hooked_max_abs": inner,
        "hidden_states_last_vs_final_norm_max_abs": _max_abs(states[-1], normed_final),
        "hidden_states_last_vs_prenorm_final_max_abs": _max_abs(states[-1], final),
    }


def resume_parity(
    adapter: GlimmerAdapter, input_ids: torch.Tensor, layers: Sequence[int]
) -> dict[str, float]:
    native = adapter.native_logits(input_ids)
    capture = adapter.forward_capture(input_ids, layers=layers, compute_logits=False)
    result: dict[str, float] = {}
    with torch.no_grad():
        for layer in layers:
            _, logits = adapter.resume_from(layer, capture.residuals[layer])
            assert logits is not None
            result[str(layer)] = _max_abs(logits, native)
    return result


SKETCH_SIZE: int = 16


def _sketch_basis(d_model: int, seed: int) -> torch.Tensor:
    """Seeded orthonormal-ish rows used to compare gradients across runs."""
    generator = torch.Generator().manual_seed(seed + 1)
    basis = torch.randn(SKETCH_SIZE, d_model, generator=generator, dtype=torch.float64)
    return basis / basis.norm(dim=1, keepdim=True)


def compare_vjp_sketches(
    receipt: Mapping[str, Any], reference: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Cosine and norm ratio of matching gradient sketches in two receipts."""

    def index(r: Mapping[str, Any]) -> dict[tuple[Any, ...], Any]:
        rows = {}
        for check in r.get("gradient_checks", []):
            for name in ("residual_projection", "native_logit"):
                key = (
                    check["layer"],
                    check["source_position"],
                    check["target_position"],
                    check["logit_token_id"],
                    check["seed"],
                    name,
                )
                rows[key] = check[name]
        return rows

    mine, theirs = index(receipt), index(reference)
    result = []
    for key in sorted(set(mine) & set(theirs), key=str):
        a = torch.tensor(mine[key]["vjp_sketch"], dtype=torch.float64)
        b = torch.tensor(theirs[key]["vjp_sketch"], dtype=torch.float64)
        result.append(
            {
                "layer": key[0],
                "source_position": key[1],
                "target_position": key[2],
                "scalar": key[5],
                "sketch_cosine": float(a @ b / (a.norm() * b.norm())),
                "sketch_norm_ratio": float(a.norm() / b.norm()),
            }
        )
    return result


def _unit(shape: Sequence[int], generator: torch.Generator) -> torch.Tensor:
    vector = torch.randn(*shape, generator=generator, dtype=torch.float64)
    return vector / vector.norm()


def gradient_checks(
    adapter: GlimmerAdapter,
    input_ids: torch.Tensor,
    layer: int,
    source_position: int,
    target_position: int,
    eps_relative: Sequence[float],
    seed: int = 0,
) -> dict[str, Any]:
    """VJP vs finite differences for two scalars of the residual at ``layer``.

    The predeclared step is the median entry of ``eps_relative``; the best
    step over the sweep is reported alongside it.

    ``residual_projection`` = <u, final_residual[target]> (the quantity a
    Jacobian lens differentiates); ``native_logit`` = softcapped logit of the
    native top-1 token at ``target`` (exercises the full head).
    """
    generator = torch.Generator().manual_seed(seed)
    probe = _unit([adapter.d_model], generator).to(adapter.device)
    direction = _unit([adapter.d_model], generator).to(adapter.device)
    capture = adapter.forward_capture(input_ids, layers=[layer], requires_grad=True)
    assert capture.logits is not None
    token = int(capture.logits[0, target_position].float().argmax())

    def residual_projection(
        final: torch.Tensor, logits: torch.Tensor | None
    ) -> torch.Tensor:
        return (final[0, target_position].double() * probe).sum()

    def native_logit(final: torch.Tensor, logits: torch.Tensor | None) -> torch.Tensor:
        assert logits is not None
        return logits[0, target_position, token].double()

    scalars: dict[str, Scalar] = {
        "residual_projection": residual_projection,
        "native_logit": native_logit,
    }
    result: dict[str, Any] = {
        "layer": layer,
        "source_position": source_position,
        "target_position": target_position,
        "logit_token_id": token,
        "seed": seed,
    }
    for name, scalar in scalars.items():
        value = scalar(capture.final_residual, capture.logits)
        (grad,) = torch.autograd.grad(
            value, [capture.residuals[layer]], retain_graph=True
        )
        result[name] = _check_one_scalar(
            adapter,
            capture.residuals[layer].detach(),
            grad,
            scalar,
            layer,
            source_position,
            target_position,
            direction,
            eps_relative,
            seed,
        )
    return result


def _check_one_scalar(
    adapter: GlimmerAdapter,
    hidden: torch.Tensor,
    grad: torch.Tensor,
    scalar: Scalar,
    layer: int,
    source_position: int,
    target_position: int,
    direction: torch.Tensor,
    eps_relative: Sequence[float],
    seed: int,
) -> dict[str, Any]:
    future = grad[0, target_position + 1 :]
    leaf = hidden.clone().requires_grad_(True)
    final, logits = adapter.resume_from(layer, leaf)
    (leaf_grad,) = torch.autograd.grad(scalar(final, logits), [leaf])
    g_src = grad[0, source_position].double()
    norm_h = float(hidden[0, source_position].double().norm())
    sweeps = []
    for eps in eps_relative:
        delta = direction * (eps * norm_h)
        plus, d_plus = _perturb(hidden, source_position, delta)
        minus, d_minus = _perturb(hidden, source_position, -delta)
        with torch.no_grad():
            f_plus = float(scalar(*adapter.resume_from(layer, plus)))
            f_minus = float(scalar(*adapter.resume_from(layer, minus)))
        predicted = float(g_src @ (d_plus - d_minus))
        finite = f_plus - f_minus
        sweeps.append(
            {
                "eps_relative": eps,
                "finite_difference": finite,
                "vjp_prediction": predicted,
                "relative_error": abs(finite - predicted) / max(abs(predicted), 1e-30),
            }
        )
    sketch = _sketch_basis(g_src.numel(), seed) @ g_src.cpu()
    return {
        "grad_norm_at_source": float(g_src.norm()),
        "vjp_sketch": [float(x) for x in sketch],
        "future_positions_grad_max_abs": float(future.abs().max())
        if future.numel()
        else 0.0,
        "full_graph_vs_cut_graph_max_abs": _max_abs(grad, leaf_grad),
        "cut_graph_grad_norm": float(leaf_grad.float().norm()),
        "best_relative_error": min(s["relative_error"] for s in sweeps),
        "predeclared_eps_relative": sweeps[len(sweeps) // 2]["eps_relative"],
        "relative_error_at_predeclared_eps": sweeps[len(sweeps) // 2]["relative_error"],
        "sweep": sweeps,
    }


def _perturb(
    hidden: torch.Tensor, position: int, delta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add ``delta`` at one position; return the tensor and the realized step."""
    perturbed = hidden.clone()
    row = (hidden[0, position].double() + delta).to(hidden.dtype)
    perturbed[0, position] = row
    return perturbed, row.double() - hidden[0, position].double()


def compute_profile(
    adapter: GlimmerAdapter,
    seq_len: int,
    batch_sizes: Sequence[int],
    token_id: int | None = None,
) -> list[dict[str, Any]]:
    """Time a forward and an all-layer VJP for a replicated batch."""
    token_id = adapter.vocab_size // 2 if token_id is None else token_id
    rows = []
    for batch in batch_sizes:
        ids = torch.full((batch, seq_len), token_id, device=adapter.device)
        row: dict[str, Any] = {"batch": batch, "seq_len": seq_len}
        try:
            row.update(_profile_once(adapter, ids))
        except torch.cuda.OutOfMemoryError:
            row["status"] = "out_of_memory"
        finally:
            if adapter.device.type == "cuda":
                torch.cuda.empty_cache()
        rows.append(row)
    return rows


def _profile_once(adapter: GlimmerAdapter, ids: torch.Tensor) -> dict[str, Any]:
    cuda = adapter.device.type == "cuda"
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.no_grad():
        adapter.forward_capture(ids, layers=(), compute_logits=False)
    if cuda:
        torch.cuda.synchronize()
    forward_s = time.perf_counter() - start
    start = time.perf_counter()
    capture = adapter.forward_capture(
        ids,
        layers=range(adapter.num_layers - 1),
        requires_grad=True,
        compute_logits=False,
    )
    cotangent = torch.randn_like(capture.final_residual)
    grads = torch.autograd.grad(
        capture.final_residual, list(capture.residuals.values()), grad_outputs=cotangent
    )
    if cuda:
        torch.cuda.synchronize()
    backward_s = time.perf_counter() - start
    finite = all(bool(torch.isfinite(g).all()) for g in grads)
    return {
        "status": "ok",
        "forward_no_grad_s": forward_s,
        "forward_plus_all_layer_vjp_s": backward_s,
        "vjp_grads_finite": finite,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30
        if cuda
        else math.nan,
    }
