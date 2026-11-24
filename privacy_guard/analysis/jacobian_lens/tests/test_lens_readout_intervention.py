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

import unittest
from collections.abc import Sequence

import torch
from privacy_guard.analysis.jacobian_lens.lens_fitting import JacobianLens
from privacy_guard.analysis.jacobian_lens.lens_intervention import (
    _edit_for,
    Arm,
    assess,
    build_contrast_examples,
    concept_means,
    greedy_answer,
    layer_directions,
    predeclared_arms,
    run_arms,
)
from privacy_guard.analysis.jacobian_lens.lens_readout import (
    capture_residuals,
    full_vocab_ranks,
    greedy_continuation,
    head_parity,
    rank_scores,
    readout_cells,
    resolve_tracked,
)
from privacy_guard.analysis.jacobian_lens.preview_schema import CellReadout
from privacy_guard.analysis.jacobian_lens.tests.jlens_tiny_model import (
    encode_words,
    TinyLensModel,
)


def _decode(ids: Sequence[int]) -> str:
    return " ".join(f"t{i}" for i in ids)


def _encode_tensor(text: str) -> torch.Tensor:
    return torch.tensor([encode_words(text)])


class LensReadoutTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = TinyLensModel()
        self.ids = _encode_tensor("the number of legs on the animal that spins webs is")

    def test_full_vocab_ranks_and_ties(self) -> None:
        scores = torch.tensor([0.5, 2.0, 2.0, -1.0, 2.0])
        self.assertEqual(full_vocab_ranks(scores, [1, 2, 4, 0, 3]), [1, 2, 3, 4, 5])
        top, tracked = rank_scores(scores, [3, 3, 0], 2, _decode)
        self.assertEqual([t["token_id"] for t in top], [1, 2])
        self.assertEqual([t["ties"] for t in top], [2, 2])
        self.assertEqual(
            [(t["token_id"], t["rank"]) for t in tracked], [(3, 5), (0, 4)]
        )
        with self.assertRaises(ValueError):
            full_vocab_ranks(torch.tensor([0.0, float("nan")]), [0])
        tracked_tokens = resolve_tracked(
            lambda s: [len(w) for w in s.split()], {"spider": [" spider", "big spider"]}
        )
        self.assertEqual([t.single_token for t in tracked_tokens], [True, False])

    def test_cells_identity_lens_equals_logit_lens(self) -> None:
        layers = [0, 1, 3]
        residuals = capture_residuals(self.model, self.ids, layers)
        eye = {k: torch.eye(5) for k in (0, 2)}
        lens = JacobianLens(eye, n_prompts=1, d_model=5)
        cells = readout_cells(
            self.model,
            lens,
            residuals,
            positions=[1, 5],
            tracked_ids=[4, 7],
            top_k=3,
            decode=_decode,
        )
        by_key = {(c["method"], c["layer"], c["position"]): c for c in cells}
        self.assertEqual(len(cells), 12)
        self.assertEqual(by_key[("jlens", 1, 5)]["status"], "unavailable")
        for layer in (0, 3):
            for pos in (1, 5):
                with self.subTest(layer=layer, pos=pos):
                    self.assertEqual(
                        by_key[("jlens", layer, pos)]["top"],
                        by_key[("logit_lens", layer, pos)]["top"],
                    )
        final = by_key[("logit_lens", 3, 5)]["top"][0]["token_id"]
        native = self.model.native_logits(self.ids)[0, 5].argmax().item()
        self.assertEqual(final, native)
        early_target = JacobianLens(eye, n_prompts=1, d_model=5, target_layer=2)
        with self.assertRaises(ValueError):
            readout_cells(
                self.model,
                early_target,
                residuals,
                positions=[1],
                tracked_ids=[4],
                top_k=1,
                decode=_decode,
            )

    def test_greedy_and_head_parity(self) -> None:
        new, first = greedy_continuation(
            self.model.native_logits, self.ids, max_new_tokens=3
        )
        self.assertEqual(len(new), 3)
        self.assertEqual(new[0], int(first.argmax()))
        parity = head_parity(self.model, self.model.native_logits, self.ids)
        self.assertEqual(parity["max_abs_logit_diff"], 0.0)
        self.assertEqual(parity["top1_agreement"], self.ids.shape[1])


class LensInterventionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = TinyLensModel(d_model=6, n_layers=5)
        self.ids = _encode_tensor("the number of legs on the animal that spins webs is")
        concepts = ["spider", "ant", "dog", "cat", "violin", "trumpet"]
        examples, self.skipped = build_contrast_examples(encode_words, concepts)
        self.means = concept_means(self.model, _encode_tensor, examples, [1, 2, 3])
        self.dirs = layer_directions(
            self.means,
            concept_from="spider",
            concept_to="ant",
            pool=["spider", "ant", "dog", "cat"],
            unrelated=("violin", "trumpet"),
        )
        self.clean = capture_residuals(self.model, self.ids, [1, 2, 3])

    def test_clamp_geometry_and_matched_norm(self) -> None:
        self.assertEqual(self.skipped, [])
        d = self.dirs[2]
        moved = d.mu_from[None] + d.clamp_delta(d.mu_from[None], 1.0)
        torch.testing.assert_close(moved[0], d.mu_to, atol=1e-5, rtol=1e-5)
        twice = moved + d.clamp_delta(moved, 1.0)
        torch.testing.assert_close(twice, moved)
        span = torch.stack([d.mu_from - d.pool_mean, d.mu_to - d.pool_mean], 1)
        self.assertLess(float((span.T @ d.unrelated).abs().max()), 1e-5)
        h = self.clean[2][None].float() + 0.3 * (d.mu_from - d.pool_mean)
        arm = Arm("a", "intervention", "contrast_mean_clamp", (2,))
        realized: dict[int, torch.Tensor] = {}
        main = _edit_for(arm, 2, d, 12, self.clean, realized, None)(h) - h
        torch.testing.assert_close(realized[2], main[0, 1:].norm(dim=-1))
        ctrl_arm = Arm(
            "c", "matched_norm_unrelated", "contrast_mean_clamp", (2,), control="random"
        )
        shifted = h + 0.5 * d.random
        ctrl = (
            _edit_for(ctrl_arm, 2, d, 12, self.clean, {}, realized)(shifted) - shifted
        )
        torch.testing.assert_close(main.norm(dim=-1), ctrl.norm(dim=-1))
        self.assertEqual(float(main[0, 0].abs().max()), 0.0)

    def test_noop_and_restoration_reproduce_baseline(self) -> None:
        arms = predeclared_arms([1, 2], [2], [1.0])
        results = run_arms(
            self.model,
            self.model.native_logits,
            self.ids,
            arms,
            self.dirs,
            self.clean,
            readout_layers=[1, 2, 3],
            tracked_ids=[3, 4],
            answer_forms={"8": [3, 5], "6": [4]},
            decode=_decode,
            stop_ids=[0],
            cell_layers=[3],
        )
        by_name = {r["name"]: r for r in results}
        for name in ("noop_zero_edit", "restore_all_positions_band_s1"):
            with self.subTest(name=name):
                self.assertEqual(by_name[name]["max_abs_logit_diff_vs_baseline"], 0.0)
        self.assertEqual(
            by_name["noop_zero_edit"]["greedy_ids"], by_name["baseline"]["greedy_ids"]
        )
        prim = by_name["clamp_band_s1"]
        self.assertTrue(prim["primary"])
        self.assertGreater(prim["max_abs_logit_diff_vs_baseline"], 0.0)
        self.assertEqual(prim["positions"], list(range(1, self.ids.shape[1])))
        for name in ("unrelated_matched_norm_band_s1", "random_matched_norm_band_s1"):
            with self.subTest(matched=name):
                self.assertAlmostEqual(
                    by_name[name]["delta_norm"], prim["delta_norm"], places=5
                )
        rows = {r["layer"]: r for r in prim["representation_final_position"]}
        self.assertEqual(sorted(rows), [1, 2, 3])
        self.assertIn("pre_edit_coef_from", rows[1])
        self.assertFalse(rows[3]["edited"])
        cells = [CellReadout.from_dict(c) for c in prim["readout_cells"]]
        self.assertEqual(
            [(c.method, c.layer, c.position) for c in cells],
            [("logit_lens", 3, self.ids.shape[1] - 1)],
        )
        self.assertNotIn("_first_logits", by_name["baseline"])
        verdict = assess(results, "8", "6", [3])
        self.assertEqual(verdict["primary_arm"], "clamp_band_s1")
        self.assertTrue(verdict["hooks_effective"])

    def test_predeclared_arms_and_assessment(self) -> None:
        arms = predeclared_arms(range(13, 39), [13, 26], [0.5, 1.0])
        kinds = [a.kind for a in arms]
        self.assertEqual(len(arms), 2 + 3 * 2 + 2 + 2)
        self.assertEqual(kinds.count("matched_norm_unrelated"), 4)
        self.assertEqual(kinds.count("restoration"), 2)
        self.assertTrue(all(a.method_label != "linear_probe_swap" for a in arms))
        self.assertEqual([a.name for a in arms if a.primary], ["clamp_band_s1"])
        self.assertEqual(
            {a.norm_source for a in arms if a.control},
            {"clamp_band_s0.5", "clamp_band_s1"},
        )
        with self.assertRaises(ValueError):
            predeclared_arms([1], [], [0.5])
        for text, want in (
            (" eight. Spiders", "8"),
            ("6. The", "6"),
            (" A. 6 B. 8", None),
        ):
            with self.subTest(greedy=text):
                self.assertEqual(greedy_answer(text), want)

        def result(
            name: str,
            kind: str,
            ranks: tuple[int, int],
            coef: tuple[float, float],
            primary: bool = False,
            after_space: tuple[int, int] | None = None,
        ) -> dict[str, object]:
            def groups(r: tuple[int, int]) -> dict[str, dict[str, int]]:
                return {"8": {"best_rank": r[0]}, "6": {"best_rank": r[1]}}

            return {
                "name": name,
                "kind": kind,
                "primary": primary,
                "max_abs_logit_diff_vs_baseline": 0.0 if kind == "baseline" else 1.0,
                "answers": groups(ranks),
                "answers_after_whitespace": groups(after_space)
                if after_space
                else None,
                "representation_final_position": [
                    {"layer": 5, "coef_from": coef[0], "coef_to": coef[1]}
                ],
            }

        base = result("baseline", "baseline", (1, 9), (1.0, 0.0))
        clamp = result("clamp", "intervention", (2, 1), (0.1, 0.9), primary=True)
        cases = {
            "success": ([base, clamp, result("c", "noop", (1, 7), (1.0, 0.0))], True),
            "control flips": (
                [base, clamp, result("c", "noop", (2, 1), (1.0, 0.0))],
                False,
            ),
            "control loses the answer": (
                [base, clamp, result("c", "noop", (3, 40), (1.0, 0.0))],
                False,
            ),
            "answer after whitespace": (
                [
                    result(
                        "baseline", "baseline", (2, 9), (1.0, 0.0), after_space=(1, 5)
                    ),
                    clamp,
                    result("c", "noop", (1, 7), (1.0, 0.0)),
                ],
                True,
            ),
            "flip read after whitespace": (
                [
                    result(
                        "baseline", "baseline", (2, 9), (1.0, 0.0), after_space=(1, 5)
                    ),
                    result(
                        "clamp",
                        "intervention",
                        (2, 3),
                        (0.1, 0.9),
                        primary=True,
                        after_space=(2, 1),
                    ),
                    result("c", "noop", (1, 7), (1.0, 0.0)),
                ],
                True,
            ),
            "baseline has no answer": (
                [result("baseline", "baseline", (33, 74), (1.0, 0.0)), clamp],
                False,
            ),
            "no representation move": (
                [
                    base,
                    result("clamp", "intervention", (2, 1), (0.9, 0.1), primary=True),
                ],
                False,
            ),
        }
        for label, (results, want) in cases.items():
            with self.subTest(assess=label):
                self.assertEqual(assess(results, "8", "6", [5])["success"], want)
        self.assertEqual(
            assess(cases["success"][0], "8", "6", [5])["flipped_arms"], ["clamp"]
        )
