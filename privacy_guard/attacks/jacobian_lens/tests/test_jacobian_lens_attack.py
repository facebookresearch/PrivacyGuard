# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import json
import os
import tempfile
import unittest
from collections.abc import Sequence

import torch
from privacy_guard.analysis.jacobian_lens.lens_fitting import JacobianLens
from privacy_guard.analysis.jacobian_lens.lens_intervention import (
    Arm,
    build_contrast_examples,
    concept_means,
    followup_arms,
    layer_directions,
    predeclared_arms,
)
from privacy_guard.analysis.jacobian_lens.preview_analysis_node import (
    JacobianLensPreviewNode,
)
from privacy_guard.analysis.jacobian_lens.preview_schema import JLensPreviewReceipt
from privacy_guard.analysis.jacobian_lens.tests.jlens_tiny_model import (
    encode_words,
    TinyLensModel,
)
from privacy_guard.attacks.jacobian_lens.jacobian_lens_attack import (
    ConceptEdit,
    JacobianLensAttack,
    LensPrompt,
    ReceiptProvenance,
)

PROMPT = "the number of legs on the animal that spins webs is"
CONCEPTS = ["spider", "ant", "dog", "cat", "violin", "trumpet"]


def _decode(ids: Sequence[int]) -> str:
    return " ".join(f"t{i}" for i in ids)


def _encode_tensor(text: str) -> torch.Tensor:
    return torch.tensor([encode_words(text)])


class JacobianLensAttackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = TinyLensModel(d_model=6, n_layers=5)
        layers = list(range(self.model.n_layers))
        examples, _ = build_contrast_examples(encode_words, CONCEPTS)
        means = concept_means(self.model, _encode_tensor, examples, layers)
        self.directions = layer_directions(
            means,
            concept_from="spider",
            concept_to="ant",
            pool=CONCEPTS[:4],
            unrelated=("violin", "trumpet"),
        )
        words = {w: encode_words(w)[1] for w in ("spider", "ant", "8", "6")}
        self.assertEqual(len(set(words.values())), 4)
        self.tracked = [
            {"concept": w, "token_id": i, "text": _decode([i])}
            for w, i in words.items()
        ]
        self.answers = {"8": [words["8"]], "6": [words["6"]]}
        self.provenance = ReceiptProvenance(
            model_identity={
                "repo_id": "fixture/tiny-lens",
                "revision_sha": "0" * 40,
                "safetensors_sha256_index": {"model.safetensors": "1" * 64},
                "tokenizer_sha256": "2" * 64,
                "transformers_version": "fixture",
                "torch_version": torch.__version__,
                "dtype": "float64",
            },
            code={"privacy_guard_rev": "fixture", "run_command": "fixture"},
            score_definition="fixture soft-capped head",
            run_kind="cpu_fixture",
            model_role="fixture",
        )
        self.lens_provenance = dataclasses.replace(
            self.provenance,
            lens={
                "lens_id": "eye",
                "sha256": "3" * 64,
                "source_layers": [0, 1, 2, 3],
                "target_site": "output of the final block",
                "estimator": "identity fixture",
            },
        )
        self.tmp = tempfile.mkdtemp()

    def _attack(
        self,
        lens: JacobianLens | None = None,
        concept_edits: Sequence[ConceptEdit] = (),
        output_file: str | None = None,
    ) -> JacobianLensAttack:
        return JacobianLensAttack(
            self.model,
            prompt=LensPrompt("spider_legs", PROMPT, "fixture prompt"),
            encode_ids=encode_words,
            decode=_decode,
            tracked_tokens=self.tracked,
            provenance=self.provenance if lens is None else self.lens_provenance,
            native_logits=self.model.native_logits,
            lens=lens,
            bos_id=0,
            top_k=5,
            max_new_tokens=3,
            stop_ids=[0],
            concept_edits=concept_edits,
            output_file=output_file,
        )

    def _edit(self, arms: Sequence[Arm], to: str = "ant") -> ConceptEdit:
        return ConceptEdit(
            concept_from="spider",
            concept_to=to,
            from_answer="8",
            to_answer="6",
            directions=self.directions,
            arms=arms,
            answer_forms=self.answers,
            contrast_set_sha256="0" * 64,
            downstream_layers=[4],
        )

    def test_run_attack_feeds_the_preview_node(self) -> None:
        lens = JacobianLens({k: torch.eye(6) for k in range(4)}, n_prompts=1, d_model=6)
        path = os.path.join(self.tmp, "receipt.json")
        followup = self._edit(followup_arms([1, 2], "spider_to_dog_post"), to="dog")
        attack = self._attack(
            lens=lens,
            output_file=path,
            concept_edits=[
                self._edit(predeclared_arms([1, 2], [2], [1.0, 2.0])),
                followup,
            ],
        )
        analysis_input = attack.run_attack()
        receipt = analysis_input.receipt
        with self.subTest("receipt is validated, hashed and written"):
            with open(path) as handle:
                written = JLensPreviewReceipt.from_dict(json.load(handle))
            self.assertEqual(written.receipt_sha256, receipt.receipt_sha256)
            self.assertEqual(list(receipt.prompt.token_ids), encode_words(PROMPT))
            self.assertTrue(all(c.status == "ok" for c in receipt.cells))
            self.assertEqual(receipt.display["head_parity"]["max_abs_logit_diff"], 0.0)
        with self.subTest("predeclared edit carries its verdict"):
            result, post = attack.edit_results
            self.assertIsNotNone(result.assessment)
            self.assertIsNone(post.assessment)
            (primary,) = [r for r in receipt.interventions if r.primary]
            self.assertIn("SUCCESS_RULE (predeclared primary)", primary.notes or "")
            self.assertEqual(
                len(receipt.interventions), len(result.arms) + len(post.arms) - 1
            )
        output = JacobianLensPreviewNode(analysis_input).run_analysis()
        with self.subTest("preview node runs on the attack output"):
            self.assertEqual(output.receipt_sha256, receipt.receipt_sha256)
            self.assertEqual(output.model_repo_id, "fixture/tiny-lens")
            self.assertEqual(output.lens_id, "eye")
            self.assertFalse(analysis_input.tracked_ranks.empty)

    def test_without_lens_and_with_follow_up_edits(self) -> None:
        followup = self._edit(followup_arms([1, 2], "spider_to_dog_post"), to="dog")
        attack = self._attack(concept_edits=[followup])
        receipt = attack.run_attack().receipt
        self.assertTrue(
            all(c.status == "unavailable" for c in receipt.cells if c.method == "jlens")
        )
        self.assertEqual(list(receipt.interventions), [])
        (result,) = attack.edit_results
        self.assertIsNone(result.assessment)
        self.assertIn("clamp_band_s2_spider_to_dog_post", result.flips)
        with self.assertRaises(ValueError):
            JacobianLensAttack(
                self.model,
                prompt=LensPrompt("p", PROMPT),
                encode_ids=encode_words,
                decode=_decode,
                tracked_tokens=self.tracked,
                provenance=self.provenance,
                lens=JacobianLens({0: torch.eye(6)}, n_prompts=1, d_model=6),
            )
        with self.assertRaises(ValueError):
            self._attack(
                concept_edits=[
                    self._edit(predeclared_arms([1], [], [1.0])),
                    self._edit(predeclared_arms([2], [], [1.0])),
                ]
            )
