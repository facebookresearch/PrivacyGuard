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

import os
import tempfile
import unittest

import torch
from privacy_guard.analysis.jacobian_lens.calibration_corpus import (
    select_prompts,
    separation_report,
    sha256_text,
    tokenize_prompts,
)
from privacy_guard.analysis.jacobian_lens.lens_fitting import (
    fit,
    jacobian_for_prompt,
    JacobianLens,
    PromptTooShortError,
)
from privacy_guard.analysis.jacobian_lens.tests.jlens_tiny_model import TinyLensModel


def _explicit_estimator(
    model: TinyLensModel, ids: torch.Tensor, layer: int, skip_first: int
) -> torch.Tensor:
    """Materialize d z[t', i] / d x[t, j] and reduce: sum valid t', mean valid t."""
    with torch.no_grad():
        x = model.embed(ids)
        for block in model.layers[: layer + 1]:
            x = block(x)
    jac = torch.autograd.functional.jacobian(lambda h: model.run_from(layer, h), x[0])
    valid = list(range(skip_first, ids.shape[1] - 1))
    sub = jac[valid][:, :, valid]  # [t', i, t, j]
    return sub.sum(0).mean(1)


class LensFittingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = TinyLensModel()
        gen = torch.Generator().manual_seed(1)
        self.prompts = [
            torch.randint(0, 23, (1, n), generator=gen) for n in (7, 9, 2, 8)
        ]
        self.hashes = [sha256_text(str(p.tolist())) for p in self.prompts]

    def test_estimator_matches_explicit_jacobian_and_finite_difference(self) -> None:
        ids = self.prompts[1]
        for dim_batch in (1, 2, 5):
            with self.subTest(dim_batch=dim_batch):
                got, n_valid = jacobian_for_prompt(
                    self.model,
                    ids,
                    [0, 2],
                    target_layer=3,
                    dim_batch=dim_batch,
                    skip_first=2,
                )
                self.assertEqual(n_valid, 6)
                for layer in (0, 2):
                    want = _explicit_estimator(self.model, ids, layer, 2).float()
                    torch.testing.assert_close(got[layer], want, atol=1e-6, rtol=1e-6)
        with self.subTest("central finite difference along a fixed direction"):
            got, _ = jacobian_for_prompt(
                self.model, ids, [1], dim_batch=5, skip_first=2
            )
            with torch.no_grad():
                x = self.model.embed(ids)
                for block in self.model.layers[:2]:
                    x = block(x)
                v = torch.randn(
                    5, dtype=torch.float64, generator=torch.Generator().manual_seed(2)
                )
                valid, eps = list(range(2, ids.shape[1] - 1)), 1e-5
                fd = torch.zeros(5, dtype=torch.float64)
                for t in valid:
                    plus, minus = x.clone(), x.clone()
                    plus[0, t] += eps * v
                    minus[0, t] -= eps * v
                    diff = self.model.run_from(1, plus) - self.model.run_from(1, minus)
                    fd += diff[0, valid].sum(0) / (2 * eps)
                fd /= len(valid)
            torch.testing.assert_close(got[1].double() @ v, fd, atol=1e-6, rtol=1e-5)

    def test_fit_resume_merge_and_save_roundtrip(self) -> None:
        full, manifest = fit(
            self.model,
            self.prompts,
            prompt_hashes=self.hashes,
            dim_batch=2,
            skip_first=2,
        )
        self.assertEqual(manifest.n_prompts_used, 3)
        self.assertEqual([r.status for r in manifest.prompts].count("skipped"), 1)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "ckpt.pt")
            snap = os.path.join(tmp, "after_prompt0.pt")

            def stop_after_two(record: object) -> None:
                if len(manifest_seen) == 1:
                    raise KeyboardInterrupt
                manifest_seen.append(record)

            manifest_seen: list[object] = []
            with self.assertRaises(KeyboardInterrupt):
                fit(
                    self.model,
                    self.prompts,
                    prompt_hashes=self.hashes,
                    dim_batch=2,
                    skip_first=2,
                    checkpoint_path=ckpt,
                    on_prompt=stop_after_two,
                    first_prompt_snapshot=snap,
                )
            first = torch.load(snap, weights_only=True)
            prompt0, _ = jacobian_for_prompt(
                self.model, self.prompts[0], [0, 1, 2], dim_batch=2, skip_first=2
            )
            self.assertEqual((first["n_done"], first["next_idx"]), (1, 1))
            torch.testing.assert_close(first["jacobian_sum"][1], prompt0[1])
            resumed, resumed_manifest = fit(
                self.model,
                self.prompts,
                prompt_hashes=self.hashes,
                dim_batch=2,
                skip_first=2,
                checkpoint_path=ckpt,
            )
            self.assertEqual(len(resumed_manifest.prompts), 4)
            for layer in full.source_layers:
                torch.testing.assert_close(
                    resumed.jacobians[layer], full.jacobians[layer]
                )
            part_a, _ = fit(
                self.model,
                self.prompts[:2],
                prompt_hashes=self.hashes[:2],
                dim_batch=2,
                skip_first=2,
            )
            part_b, _ = fit(
                self.model,
                self.prompts[2:],
                prompt_hashes=self.hashes[2:],
                dim_batch=2,
                skip_first=2,
            )
            merged = JacobianLens.merge([part_a, part_b])
            for layer in full.source_layers:
                torch.testing.assert_close(
                    merged.jacobians[layer], full.jacobians[layer]
                )
            path = os.path.join(tmp, "lens.pt")
            digest = full.save(path, dtype=torch.float32)
            self.assertEqual(len(digest), 64)
            loaded = JacobianLens.load(path)
            self.assertEqual(loaded.source_layers, [0, 1, 2])
            self.assertEqual((loaded.target_layer, loaded.skip_first), (3, 2))
            torch.testing.assert_close(loaded.jacobians[1], full.jacobians[1])
            other_mask = JacobianLens(
                part_b.jacobians, n_prompts=1, d_model=part_b.d_model, target_layer=3
            )
            with self.assertRaises(ValueError):
                JacobianLens.merge([part_a, other_mask])

    def test_rejections_and_corpus_rules(self) -> None:
        with self.subTest("source must precede target"):
            with self.assertRaises(ValueError):
                jacobian_for_prompt(self.model, self.prompts[1], [3], target_layer=3)
        with self.subTest("too-short prompt"):
            with self.assertRaises(PromptTooShortError):
                jacobian_for_prompt(self.model, self.prompts[2], [0], skip_first=2)
        with self.subTest("other errors are not recorded as skipped prompts"):
            with self.assertRaises(ValueError):
                fit(self.model, [self.prompts[1].repeat(2, 1)], prompt_hashes=["x"])
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "ckpt.pt")
            fit(
                self.model,
                self.prompts[:1],
                prompt_hashes=self.hashes[:1],
                dim_batch=5,
                skip_first=2,
                checkpoint_path=ckpt,
            )
            with self.subTest("checkpoint settings mismatch"):
                with self.assertRaises(ValueError):
                    fit(
                        self.model,
                        self.prompts,
                        prompt_hashes=self.hashes,
                        skip_first=1,
                        checkpoint_path=ckpt,
                    )
            with self.subTest("checkpoint prompt sequence mismatch"):
                with self.assertRaises(ValueError):
                    fit(
                        self.model,
                        self.prompts[1:],
                        prompt_hashes=self.hashes[1:],
                        skip_first=2,
                        checkpoint_path=ckpt,
                    )
        with self.subTest("corpus selection, truncation and separation"):
            records = ["short", "x" * 700, "", "y " * 400, "z" * 650]
            self.assertEqual(select_prompts(records, 2), ["x" * 700, "y " * 400])
            with self.assertRaises(ValueError):
                select_prompts(records, 4)
            ids, lengths, truncated = tokenize_prompts(
                lambda s: torch.zeros(1, len(s) // 100, dtype=torch.long),
                records[1:2],
                max_seq_len=5,
            )
            self.assertEqual((lengths, truncated, ids[0].shape[1]), ([5], [True], 5))
            report = separation_report(["The spider is small."], ["legs on the spider"])
            self.assertEqual(report["eval_prompts_contained"], 0)
            with self.assertRaises(ValueError):
                separation_report(["so The  Spider is small"], ["the spider is"])
            with self.assertRaises(ValueError):
                separation_report(
                    ["... spiders have eight legs . ..."], ["Spiders have eight legs."]
                )
            self.assertEqual(
                separation_report(["a giant"], ["ant"])["eval_prompts_contained"], 0
            )
