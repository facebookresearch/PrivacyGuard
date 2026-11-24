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
import json
import os
import tempfile
import unittest
from collections.abc import Sequence
from typing import cast

import torch
from privacy_guard.analysis.jacobian_lens import glimmer_preview_pipeline as pipeline
from privacy_guard.analysis.jacobian_lens.glimmer_adapter import (
    GlimmerAdapter,
    Tokenizer,
)
from privacy_guard.analysis.jacobian_lens.lens_fitting import JacobianLens
from privacy_guard.analysis.jacobian_lens.preview_schema import JLensPreviewReceipt
from privacy_guard.attacks.jacobian_lens.jacobian_lens_attack import (
    ConceptEdit,
    JacobianLensAttack,
)

try:
    from transformers import MuseGlimmerConfig, MuseGlimmerForConditionalGeneration
except ImportError:  # transformers < 5.15 has no Muse Glimmer
    MuseGlimmerConfig = None
    MuseGlimmerForConditionalGeneration = None


class _CharTokenizer:
    """Character-level fixture tokenizer (BOS id 1)."""

    bos_token_id = 1

    def __call__(
        self, text: str, add_special_tokens: bool = True
    ) -> dict[str, list[int]]:
        ids = [ord(c) for c in text]
        return {"input_ids": [1, *ids] if add_special_tokens else ids}

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(chr(i) for i in ids)


def _tiny_glue() -> pipeline.Glue:
    if MuseGlimmerConfig is None or MuseGlimmerForConditionalGeneration is None:
        raise unittest.SkipTest("transformers without muse_glimmer")
    torch.manual_seed(0)
    config = MuseGlimmerConfig(
        text_config={
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 4,
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
        image_token_id=126,
        video_token_id=127,
        out_hidden_size=64,
        projector_hidden_size=32,
    )
    model = MuseGlimmerForConditionalGeneration(config).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.2 * torch.randn_like(parameter))
    tokenizer = cast(Tokenizer, _CharTokenizer())
    return pipeline.Glue(GlimmerAdapter(model, tokenizer), bos_id=1)


IDENTITY = {
    "repo_id": "fixture/tiny-muse-glimmer",
    "revision_sha": "0" * 40,
    "safetensors_sha256_index": {"model.safetensors": "1" * 64},
    "tokenizer_sha256": "2" * 64,
    "transformers_version": "fixture",
    "torch_version": torch.__version__,
    "dtype": "float64",
}


@unittest.skipIf(MuseGlimmerConfig is None, "transformers without muse_glimmer")
class GlimmerPreviewPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.glue = _tiny_glue()
        self.adapter = self.glue.adapter
        self.tmp = tempfile.mkdtemp()
        self.args = argparse.Namespace(
            command="run",
            top_k=5,
            max_new_tokens=2,
            lens="",
            fit_manifest="",
            code_rev="fixture",
            seed=0,
            private_dir=self.tmp,
        )
        self.tracked, self.tracked_ids, self.answers, _ = pipeline.tracked_tokens(
            self.glue
        )

    def _attack(
        self, lens: JacobianLens | None, edits: Sequence[ConceptEdit] = ()
    ) -> JacobianLensAttack:
        record = pipeline.provenance(
            IDENTITY, self.args, lens, run_kind="cpu_fixture", model_role="fixture"
        )
        return pipeline.prompt_attack(
            self.adapter,
            self.glue,
            "blog_no_period",
            lens,
            self.args,
            self.tracked,
            record,
            edits,
        )

    def _receipt(self, lens: JacobianLens | None) -> JLensPreviewReceipt:
        return self._attack(lens).run_attack().receipt

    def test_receipts_with_and_without_lens(self) -> None:
        for token in self.tracked:
            with self.subTest(tracked=token["text"]):
                self.assertEqual(self.glue.decode([token["token_id"]]), token["text"])
                self.assertIn(token["text"], pipeline.TRACKED[token["concept"]])
        bare = self._receipt(None)
        self.assertEqual(bare.prompt.token_ids[0], 1)
        self.assertTrue(
            all(c.status == "unavailable" for c in bare.cells if c.method == "jlens")
        )
        self.assertEqual(bare.display["head_parity"]["max_abs_logit_diff"], 0.0)
        eye = {k: torch.eye(32) for k in range(3)}
        self.args.lens = os.path.join(self.tmp, "eye.pt")
        JacobianLens(eye, n_prompts=1, d_model=32).save(self.args.lens)
        with_lens = self._receipt(JacobianLens.load(self.args.lens))
        by_key = {(c.method, c.layer, c.position): c for c in with_lens.cells}
        for (method, layer, position), cell in by_key.items():
            if method == "jlens":
                with self.subTest(layer=layer, position=position):
                    self.assertEqual(
                        cell.top, by_key[("logit_lens", layer, position)].top
                    )
        texts = ["the quick brown fox jumps over a lazy dog", "a" * 40]
        quality = pipeline.heldout_quality(
            self.adapter, self.glue, JacobianLens.load(self.args.lens), texts, 64
        )
        self.assertEqual(quality["n_positions"], sum(len(t) + 1 - 17 for t in texts))
        for layer, row in quality["top1_agreement"].items():
            with self.subTest(heldout_layer=layer):
                self.assertEqual(row["jlens"], row["logit_lens"])

    def test_causal_arms_fill_a_valid_receipt(self) -> None:
        means, contrast = pipeline.build_means(self.adapter, self.glue)
        self.assertEqual(contrast["skipped"], [])
        sha = contrast["templates_sha256"]
        followup = ("blog_no_period", "spider", "dog", "8", "4")
        edits = [
            pipeline.predeclared_edit(
                pipeline.directions_for(means, "spider", "ant", self.args),
                self.answers,
                sha,
                band=(1, 2),
                downstream=(3,),
                single_layers=(1,),
            ),
            pipeline.followup_edit(
                pipeline.directions_for(means, "spider", "dog", self.args),
                followup,
                self.answers,
                sha,
                band=(1, 2),
                downstream=(3,),
            ),
        ]
        attack = self._attack(None, edits)
        receipt = attack.run_attack().receipt
        primary, post = attack.edit_results
        by_name = {a["name"]: a for a in primary.arms}
        self.assertEqual(
            by_name["noop_zero_edit"]["max_abs_logit_diff_vs_baseline"], 0.0
        )
        self.assertTrue(primary.assessment and primary.assessment["hooks_effective"])
        self.assertEqual(
            [r.name for r in receipt.interventions if r.primary], ["clamp_band_s1"]
        )
        self.assertIn(
            "clamp_band_s2_spider_to_dog_post", [r.name for r in receipt.interventions]
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.tmp, "directions_spider_to_ant.pt"))
        )
        follow = pipeline.followup_record(
            self.adapter, "blog_no_period", post, None, band=(1, 2)
        )
        self.assertEqual(follow["concept_to"], "dog")
        self.assertIn("clamp_band_s2_spider_to_dog_post", follow["summary"])
        (per_layer,) = follow["answer_leakage_cosine"].values()
        self.assertEqual(sorted(per_layer), [1, 2])
        self.assertLessEqual(abs(per_layer[1]["edit"]), 1.0)
        restore = follow["summary"]["restore_all_positions_band_s2_spider_to_dog_post"]
        base = follow["summary"]["baseline_spider_to_dog_post"]
        self.assertEqual(restore["p_from"], base["p_from"])

    def test_merge_and_public_command(self) -> None:
        lens_paths, manifest_paths = [], []
        prompts = [["a"], ["b", "c", "d"]]
        for shard, (value, n) in enumerate(((1.0, 1), (4.0, 3))):
            lens = JacobianLens({0: torch.full((2, 2), value)}, n_prompts=n, d_model=2)
            lens_paths.append(os.path.join(self.tmp, f"lens{shard}.pt"))
            manifest = {
                "model_identity": IDENTITY,
                "environment": {"gpu": "fixture"},
                "corpus": {
                    "corpus_sha256": "c" * 64,
                    "prompt_sha256": ["a", "b", "c", "d"],
                },
                "fit": {
                    "n_prompts_used": n,
                    "prompts": [{"prompt_sha256": h} for h in prompts[shard]],
                    "extra": {
                        "lens_file_sha256": lens.save(lens_paths[-1]),
                        "shard": shard,
                        "num_shards": 2,
                        "wall_seconds": 1.0,
                    },
                },
            }
            manifest_paths.append(os.path.join(self.tmp, f"manifest{shard}.json"))
            with open(manifest_paths[-1], "w") as handle:
                json.dump(manifest, handle)
        args = pipeline.parse_args(
            [
                "merge",
                "--checkpoint-dir",
                self.tmp,
                "--out-dir",
                self.tmp,
                "--private-dir",
                self.tmp,
                "--shard-lenses",
                *lens_paths,
                "--shard-manifests",
                *manifest_paths,
            ]
        )
        pipeline.cmd_merge(args)
        merged = JacobianLens.load(os.path.join(self.tmp, "lens_merged.pt"))
        self.assertEqual(merged.n_prompts, 4)
        torch.testing.assert_close(merged.jacobians[0], torch.full((2, 2), 3.25))
        with self.subTest("shards from another checkpoint are refused"):
            with open(manifest_paths[1]) as handle:
                other = json.load(handle)
            other["model_identity"] = {**IDENTITY, "revision_sha": "f" * 40}
            other_path = os.path.join(self.tmp, "manifest_other.json")
            with open(other_path, "w") as handle:
                json.dump(other, handle)
            args.shard_manifests = [manifest_paths[0], other_path]
            with self.assertRaises(ValueError):
                pipeline.cmd_merge(args)
        with self.subTest("duplicate shard is refused"):
            args.shard_manifests = [manifest_paths[1], manifest_paths[1]]
            args.shard_lenses = [lens_paths[1], lens_paths[1]]
            with self.assertRaises(ValueError):
                pipeline.cmd_merge(args)
        with self.subTest("receipt lens must be the fit manifest's lens"):
            self.args.lens, self.args.fit_manifest = lens_paths[0], manifest_paths[1]
            with self.assertRaises(ValueError):
                pipeline.lens_record(self.args, JacobianLens.load(lens_paths[0]))
        args.top_k = 0
        command = pipeline.public_command(args)
        self.assertIn("--top-k 0", command)
        self.assertNotIn(self.tmp, command)
        self.assertIn(
            "--shard-lenses PRIVATE/LENS_SHARD.pt:0 PRIVATE/LENS_SHARD.pt:1", command
        )
