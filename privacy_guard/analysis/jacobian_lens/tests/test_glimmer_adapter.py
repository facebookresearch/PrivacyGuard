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

import argparse
import hashlib
import os
import tempfile
import unittest
from collections.abc import Sequence

import torch
import transformers
from privacy_guard.analysis.jacobian_lens.glimmer_adapter import (
    build_identity,
    cache_argument_name,
    GlimmerAdapter,
    sha256_file,
)
from privacy_guard.analysis.jacobian_lens.model_numerics import (
    compare_vjp_sketches,
    compute_profile,
    gradient_checks,
    head_parity,
    resume_parity,
)
from transformers.masking_utils import create_causal_mask

try:
    from privacy_guard.analysis.jacobian_lens import glimmer_loading, glimmer_numerics
except ImportError:  # transformers < 5.15 has no Muse Glimmer
    glimmer_loading = None
    glimmer_numerics = None


SEQ_LEN = 9


class _FakeTokenizer:
    def __call__(self, text: str, **kwargs: object) -> dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor([[1] + [3 + ord(c) % 90 for c in text]])}

    def decode(self, token_ids: Sequence[int]) -> str:
        return " ".join(str(i) for i in token_ids)


def _tiny_adapter() -> GlimmerAdapter:
    """Five-layer float64 Muse Glimmer with sliding window shorter than input."""
    torch.manual_seed(0)
    config = transformers.MuseGlimmerConfig(
        text_config={
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 5,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "sliding_window": 3,
            "final_logit_softcapping": 2.0,
            "output_multiplier": 1.5,
        },
        vision_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "pos_emb_height": 4,
            "pos_emb_width": 4,
            "max_position_embeddings": 16,
        },
        image_token_id=120,
        video_token_id=121,
        out_hidden_size=64,
        projector_hidden_size=32,
    )
    model = transformers.MuseGlimmerForConditionalGeneration(config).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.2 * torch.randn_like(parameter))
    return GlimmerAdapter(model, _FakeTokenizer())


def _ids() -> torch.Tensor:
    return torch.randint(
        3, 110, (1, SEQ_LEN), generator=torch.Generator().manual_seed(1)
    )


@unittest.skipIf(glimmer_loading is None, "transformers without muse_glimmer")
class GlimmerAdapterTest(unittest.TestCase):
    def test_head_and_resume_match_native_forward(self) -> None:
        adapter = _tiny_adapter()
        ids = _ids()
        with self.subTest("head_parity"):
            parity = head_parity(adapter, ids)
            self.assertTrue(parity["adapter_vs_native_bitwise_equal"])
            self.assertLess(parity["formula_fp32_vs_native_max_abs"], 1e-5)
            self.assertLessEqual(parity["native_logit_max_abs"], 2.0)
            self.assertGreater(parity["raw_unembed_max_abs"] * 1.5, 2.0)
        with self.subTest("hidden_states_layout"):
            self.assertEqual(parity["hidden_states_count"], adapter.num_layers + 1)
            self.assertEqual(parity["hidden_states_inner_vs_hooked_max_abs"], 0.0)
            self.assertEqual(parity["hidden_states_last_vs_final_norm_max_abs"], 0.0)
            self.assertGreater(
                parity["hidden_states_last_vs_prenorm_final_max_abs"], 0.0
            )
        with self.subTest("resume_parity"):
            layers = list(range(-1, adapter.num_layers))
            for layer, diff in resume_parity(adapter, ids, layers).items():
                self.assertEqual(diff, 0.0, msg=f"layer {layer}")
        with self.subTest("lens_model_protocol"):
            self.assertEqual(len(adapter.layers), adapter.n_layers)
            final = adapter(ids.repeat(2, 1))
            self.assertEqual(tuple(final.shape), (2, SEQ_LEN, adapter.d_model))
            self.assertTrue(
                torch.equal(adapter.unembed(final[:1]), adapter.native_logits(ids))
            )
            self.assertEqual(adapter.encode("ab").shape[0], 1)
        with self.subTest("cache_argument_name"):
            self.assertIn(
                cache_argument_name(create_causal_mask), ("past_key_values", "cache")
            )

            def renamed(config: object, cache: object = None) -> None:
                return None

            self.assertEqual(cache_argument_name(renamed), "cache")
            self.assertIsNone(cache_argument_name(lambda config: None))
        with self.subTest("rejects_multimodal_tokens"):
            with self.assertRaises(ValueError):
                adapter.embed(torch.tensor([[5, 120]]))

    def test_gradients_match_finite_differences(self) -> None:
        adapter = _tiny_adapter()
        ids = _ids()
        # Muse Glimmer RMSNorms compute in float32 even in a float64 model, so
        # finite differences need steps well above float32 resolution; a wrong
        # gradient misses by O(1), a correct one by the float32 noise floor.
        for layer in (0, 2, 3):
            with self.subTest(layer=layer):
                result = gradient_checks(
                    adapter,
                    ids,
                    layer,
                    2,
                    SEQ_LEN - 2,
                    eps_relative=[1e-4, 1e-3, 1e-2],
                    causal_target_position=3,
                )
                with self.subTest("causality"):
                    causality = result["causality"]
                    self.assertEqual(causality["later_positions_tested"], SEQ_LEN - 4)
                    self.assertEqual(causality["later_positions_grad_max_abs"], 0.0)
                    self.assertEqual(causality["earlier_positions_tested"], 3)
                    self.assertGreater(causality["earlier_positions_grad_norm"], 0.0)
                    self.assertGreater(causality["target_position_grad_norm"], 0.0)
                with self.subTest("sketch_self_comparison"):
                    receipt = {"gradient_checks": [result]}
                    rows = compare_vjp_sketches(receipt, receipt)
                    self.assertEqual(len(rows), 2)
                    for row in rows:
                        self.assertAlmostEqual(row["sketch_cosine"], 1.0, places=12)
                for name in ("residual_projection", "native_logit"):
                    check = result[name]
                    self.assertEqual(len(check["vjp_sketch"]), 16)
                    self.assertLess(check["best_relative_error"], 1e-2, msg=str(check))
                    self.assertEqual(check["future_positions_tested"], 1)
                    self.assertEqual(check["future_positions_grad_max_abs"], 0.0)
                    self.assertLess(check["full_graph_vs_cut_graph_max_abs"], 1e-12)
                    self.assertGreater(check["grad_norm_at_source"], 0.0)

    def test_identity_and_profile(self) -> None:
        adapter = _tiny_adapter()
        with tempfile.TemporaryDirectory() as tmp:
            for name, payload in (
                ("model-00001-of-00001.safetensors", b"weights"),
                ("tokenizer.json", b"{}"),
                ("config.json", b"{}"),
            ):
                with open(os.path.join(tmp, name), "wb") as handle:
                    handle.write(payload)
            identity = build_identity(adapter.model, tmp, "repo/name", "abc123")
            with self.subTest("hashes"):
                self.assertEqual(
                    identity.safetensors_sha256_index,
                    {
                        "model-00001-of-00001.safetensors": hashlib.sha256(
                            b"weights"
                        ).hexdigest()
                    },
                )
                self.assertEqual(
                    sha256_file(os.path.join(tmp, "config.json")),
                    hashlib.sha256(b"{}").hexdigest(),
                )
        with self.subTest("fields"):
            fields = identity.to_dict()
            self.assertEqual(
                fields["model_class"], "MuseGlimmerForConditionalGeneration"
            )
            self.assertEqual(fields["num_layers"], 5)
            self.assertEqual(fields["dtype"], "float64")
            self.assertTrue(fields["lm_head_separate_from_embedding"])
        with self.subTest("text_only_float32_cast"):
            assert glimmer_loading is not None
            ids = _ids()
            before = adapter.native_logits(ids)
            dropped = glimmer_loading._drop_vision_modules(adapter.model)
            self.assertEqual(len(dropped), 3)
            glimmer_loading._cast_parameters_in_place(adapter.model, torch.float32)
            self.assertEqual(adapter.dtype, torch.float32)
            after = adapter.native_logits(ids)
            self.assertLess(float((after.double() - before).abs().max()), 1e-4)
            self.assertEqual(
                glimmer_loading.checked_weight_names(5)[-1],
                "model.language_model.layers.4.post_feedforward_layernorm.weight",
            )
        with self.subTest("last_token_target_reports_untested_future"):
            result = gradient_checks(
                adapter, _ids(), 1, 2, SEQ_LEN - 1, eps_relative=[1e-3]
            )
            check = result["residual_projection"]
            self.assertEqual(check["future_positions_tested"], 0)
            self.assertIsNone(check["future_positions_grad_max_abs"])
            with self.assertRaises(ValueError):
                gradient_checks(
                    adapter,
                    _ids(),
                    1,
                    2,
                    SEQ_LEN - 1,
                    eps_relative=[1e-3],
                    causal_target_position=SEQ_LEN - 1,
                )
        with self.subTest("recorded_arguments"):
            assert glimmer_numerics is not None
            args = argparse.Namespace(
                checkpoint_dir="/Users/Jane Doe/ckpt",
                output="-vo/srv/out.json",
                expected_sha256_json=None,
                reference_receipt="~/ref.json",
                device="cpu",
                grad_layers="12,25",
            )
            self.assertEqual(
                glimmer_numerics.recorded_arguments(args),
                {
                    "checkpoint_dir": "<path>",
                    "device": "cpu",
                    "expected_sha256_json": None,
                    "grad_layers": "12,25",
                    "output": "<path>",
                    "reference_receipt": "<path>",
                },
            )
        with self.subTest("profile"):
            rows = compute_profile(adapter, seq_len=6, batch_sizes=[1, 3])
            self.assertEqual([row["status"] for row in rows], ["ok", "ok"])
            self.assertTrue(all(row["vjp_grads_finite"] for row in rows))
