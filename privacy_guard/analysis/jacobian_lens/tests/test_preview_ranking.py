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

import torch
from privacy_guard.analysis.jacobian_lens.preview_ranking import (
    cell_from_scores,
    rank_scores,
    UnavailableScoresError,
)


def _decode(token_id: int) -> str:
    return f"<{token_id}>"


class PreviewRankingTest(unittest.TestCase):
    def test_matches_brute_force_with_ties(self) -> None:
        generator = torch.Generator().manual_seed(0)
        for trial in range(5):
            with self.subTest(trial=trial):
                scores = torch.randint(0, 6, (50,), generator=generator).float()
                tracked = [0, 7, 49]
                top, tracked_entries = rank_scores(scores, tracked, 10, _decode)
                order = sorted(range(50), key=lambda i: (-float(scores[i]), i))
                expected_rank = {token_id: r + 1 for r, token_id in enumerate(order)}
                self.assertEqual([t.token_id for t in top], order[:10])
                self.assertEqual([t.rank for t in top], list(range(1, 11)))
                for entry in tracked_entries:
                    self.assertEqual(entry.rank, expected_rank[entry.token_id])
                    self.assertEqual(entry.text, f"<{entry.token_id}>")
                    self.assertEqual(
                        entry.ties, int((scores == scores[entry.token_id]).sum()) - 1
                    )

    def test_non_finite_scores_are_unavailable(self) -> None:
        for bad in (float("nan"), float("inf")):
            with self.subTest(bad=bad):
                scores = torch.zeros(8)
                scores[3] = bad
                with self.assertRaises(UnavailableScoresError):
                    rank_scores(scores, [1], 3, _decode)
                cell = cell_from_scores("jlens", 0, 0, scores, [1], 3, _decode)
                self.assertEqual(cell.status, "unavailable")
                self.assertEqual(cell.top, ())
                self.assertIn("NaN or inf", str(cell.unavailable_reason))

        cell = cell_from_scores("logit_lens", 2, 1, torch.arange(8.0), [0], 3, _decode)
        self.assertEqual(cell.status, "ok")
        self.assertEqual([t.token_id for t in cell.top], [7, 6, 5])
        self.assertEqual(cell.tracked[0].rank, 8)

    def test_rejects_bad_arguments(self) -> None:
        cases = {
            "2-D scores": (torch.zeros(2, 4), [0], 2),
            "zero top_k": (torch.zeros(4), [0], 0),
            "tracked out of vocabulary": (torch.zeros(4), [4], 2),
        }
        for name, (scores, tracked, top_k) in cases.items():
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    rank_scores(scores, tracked, top_k, _decode)
