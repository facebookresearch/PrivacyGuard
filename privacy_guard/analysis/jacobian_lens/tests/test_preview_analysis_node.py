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

from privacy_guard.analysis.jacobian_lens.preview_analysis_input import (
    JacobianLensPreviewInput,
    TRACKED_RANK_COLUMNS,
)
from privacy_guard.analysis.jacobian_lens.preview_analysis_node import (
    JacobianLensPreviewNode,
)
from privacy_guard.analysis.jacobian_lens.preview_fixture import (
    fixture_receipt_dict,
    make_fixture_receipt,
    NUM_LAYERS,
    PROMPT_IDS,
    TRACKED,
)
from privacy_guard.analysis.jacobian_lens.preview_schema import TokenScore


class PreviewAnalysisNodeTest(unittest.TestCase):
    def test_summary_outputs(self) -> None:
        node = JacobianLensPreviewNode(JacobianLensPreviewInput(fixture_receipt_dict()))
        out = node.compute_outputs()
        last = len(PROMPT_IDS) - 1

        self.assertEqual(out["run_kind"], "cpu_fixture")
        self.assertEqual(out["receipt_sha256"], make_fixture_receipt().receipt_sha256)
        self.assertEqual(out["num_cells"]["jlens"], {"ok": 18, "unavailable": 6})
        self.assertEqual(out["num_cells"]["logit_lens"], {"ok": 24, "unavailable": 0})
        spider = next(
            c
            for c in out["rank_curves"]
            if c["method"] == "jlens"
            and c["concept"] == "spider"
            and c["position"] == last
        )
        self.assertEqual(spider["layers"], list(range(NUM_LAYERS)))
        self.assertIsNone(spider["best_rank"][0])
        self.assertEqual(spider["best_rank"][1:3], [1, 1])
        first = next(
            f
            for f in out["first_layer_within_top"]
            if f["method"] == "jlens"
            and f["concept"] == "spider"
            and f["position"] == last
        )
        self.assertEqual(first["layer"], 1)
        self.assertEqual(out["actual_next_token"], {"token_id": 7, "text": " 8"})

        margins = {m["name"]: m for m in out["answer_margins"]}
        self.assertLess(margins["actual_output"]["margin"], 0)
        self.assertGreater(margins["spider_to_ant"]["margin"], 0)
        self.assertNotIn("restoration", margins)
        readout = out["intervention_readouts"][0]
        self.assertEqual(readout["name"], "spider_to_ant")
        changes = {c["concept"]: c for c in readout["rank_changes"]}
        self.assertEqual(changes["ant"]["patched_rank"], 1)
        self.assertGreater(changes["ant"]["baseline_rank"], 1)
        self.assertTrue(any("not a Glimmer result" in label for label in out["labels"]))
        self.assertTrue(any("candidate sites" in label for label in out["labels"]))

    def test_head_consistency(self) -> None:
        node = JacobianLensPreviewNode(JacobianLensPreviewInput(make_fixture_receipt()))
        consistent = node.run_analysis().head_consistency
        self.assertTrue(consistent["checked"])
        self.assertTrue(consistent["consistent"])
        self.assertEqual(consistent["max_abs_score_diff"], 0.0)

        data = fixture_receipt_dict()
        data["receipt_sha256"] = None
        actual = data["actual_output"]["next_token_top"]
        actual[0]["score"] += 0.5
        shifted = JacobianLensPreviewNode(
            JacobianLensPreviewInput(data), head_score_tolerance=0.1
        ).run_analysis()
        self.assertFalse(shifted.head_consistency["consistent"])
        self.assertAlmostEqual(shifted.head_consistency["max_abs_score_diff"], 0.5)

        data["cells"] = [c for c in data["cells"] if c["method"] == "jlens"]
        missing = JacobianLensPreviewNode(JacobianLensPreviewInput(data)).run_analysis()
        self.assertFalse(missing.head_consistency["checked"])
        with self.assertRaises(ValueError):
            JacobianLensPreviewNode(JacobianLensPreviewInput(data), top_threshold=0)

    def test_tracked_rank_table(self) -> None:
        analysis_input = JacobianLensPreviewInput(make_fixture_receipt())
        table = analysis_input.tracked_ranks
        self.assertEqual(list(table.columns), list(TRACKED_RANK_COLUMNS))
        receipt_rows = table[table["source"] == "receipt"]
        self.assertEqual(len(receipt_rows), (18 + 24) * len(TRACKED))
        self.assertEqual(
            set(table["source"]), {"receipt", "intervention:spider_to_ant"}
        )
        self.assertTrue((table["rank"] >= 1).all())
        self.assertTrue(analysis_input.df_test_user.empty)
        self.assertIsInstance(analysis_input.receipt.cells[1].top[0], TokenScore)
