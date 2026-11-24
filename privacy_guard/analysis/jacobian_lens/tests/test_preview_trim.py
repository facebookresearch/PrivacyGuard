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

from privacy_guard.analysis.jacobian_lens.preview_fixture import make_fixture_receipt
from privacy_guard.analysis.jacobian_lens.preview_schema import JLensPreviewReceipt
from privacy_guard.analysis.jacobian_lens.preview_trim import trim_receipt


class PreviewTrimTest(unittest.TestCase):
    def test_trim_keeps_ranks_and_provenance(self) -> None:
        source = make_fixture_receipt()
        trimmed = trim_receipt(source, top_k=3, score_decimals=2, keep_readout_runs=())

        with self.subTest("re-validates and records its source"):
            self.assertEqual(JLensPreviewReceipt.from_dict(trimmed.to_dict()), trimmed)
            info = trimmed.display["trimmed"]
            self.assertEqual(info["source_receipt_sha256"], source.content_sha256())
            self.assertEqual(info["readout_cells_dropped_for_runs"], ["spider_to_ant"])
            self.assertEqual(trimmed.top_k, 3)

        with self.subTest("top lists shortened, tracked ranks kept"):
            for before, after in zip(source.cells, trimmed.cells):
                self.assertEqual(
                    [t.rank for t in after.top], [t.rank for t in before.top][:3]
                )
                self.assertEqual(
                    [(t.token_id, t.rank) for t in after.tracked],
                    [(t.token_id, t.rank) for t in before.tracked],
                )
                for t in after.tracked:
                    self.assertEqual(t.score, round(t.score, 2))
            self.assertEqual(
                [r.readout_cells for r in trimmed.interventions], [(), (), ()]
            )

        with self.subTest("bad arguments"):
            with self.assertRaises(ValueError):
                trim_receipt(source, top_k=source.top_k + 1)
            with self.assertRaises(ValueError):
                trim_receipt(source, top_k=2, score_decimals=-1)
            kept = trim_receipt(source, top_k=2)
            self.assertEqual(
                len(kept.interventions[0].readout_cells),
                len(source.interventions[0].readout_cells),
            )
