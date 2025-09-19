# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import unittest

from typing import Dict, List

import pandas as pd
from pandas.testing import assert_series_equal

from privacy_guard.analysis.extraction.text_inclusion_analysis_input import (
    TextInclusionAnalysisInput,
)
from privacy_guard.analysis.extraction.text_inclusion_analysis_node import (
    _char_level_longest_common_subsequence_helper,
    _char_level_longest_common_substring_helper,
    _char_level_longest_common_substring_helper_bound,
    _word_level_longest_common_subsequence_helper,
    TextInclusionAnalysisNode,
    TextInclusionAnalysisNodeOutput,
)


class TestAnalysisInput(unittest.TestCase):
    def setUp(self) -> None:
        self.data = {
            "prompt": [
                "This is a test prompt",
                "This is another test prompt",
                "Hello, test prompt!",
                "",
                "\t\n\t\n  ",
                "Neque porro quisquam est qui dolorem ipsum",
            ],
            "target": [
                "Target text 1",
                "Target text 2",
                "This is an exact match.",
                "This is included \n",
                "Included",
                "Neque porro quisquam est qui dolorem ipsum",
            ],
            "output_text": [
                "A success: Target text 1",
                "Failure, no match. ",
                "This is an exact match.",
                "This is included",
                "IsItIncluded?",
                "dolorem ipsum quia dolor sit amet, consectetur, adipisci velit",
            ],
        }

        self.analysis_input = TextInclusionAnalysisInput(
            generation_df=pd.DataFrame(self.data)
        )
        self.analysis_node = TextInclusionAnalysisNode(
            analysis_input=self.analysis_input
        )

        super().setUp()

    def test_column_not_in_dataframe(self) -> None:
        with self.assertRaises(AssertionError):
            _ = TextInclusionAnalysisInput(
                generation_df=pd.DataFrame(self.data), prompt_key="NOT PRESENT"
            )

        with self.assertRaises(AssertionError):
            _ = TextInclusionAnalysisInput(
                generation_df=pd.DataFrame(self.data), target_key="NOT PRESENT"
            )

        with self.assertRaises(AssertionError):
            _ = TextInclusionAnalysisInput(
                generation_df=pd.DataFrame(self.data), generation_key="NOT PRESENT"
            )

    def test_construct_text_inclusion_analysis_input(self) -> None:
        results = self.analysis_node.compute_outputs()

        self.assertIn("exact_match", results)
        exact_match = results["exact_match"].tolist()
        self.assertEqual(exact_match, [False, False, True, True, False, False])

        self.assertIn("inclusion_score", results)
        inclusion_score = results["inclusion_score"].tolist()
        self.assertEqual(inclusion_score, [True, False, True, True, True, False])

        # If exact match is true, then inclusion score must be true
        self.assertTrue(all(~e | i for e, i in zip(exact_match, inclusion_score)))

        self.assertIn("decision_targets_lcs", results)

        self.assertIn("longest_common_substring", results)

        longest_common_substring = results["longest_common_substring"].tolist()
        longest_common_substring_vals = [
            list(x.values())[0] for x in longest_common_substring
        ]
        self.assertEqual(longest_common_substring_vals, [13, 1, 22, 16, 8, 13])

        decision_targets_lcs = results["decision_targets_lcs"].tolist()
        lcs_score = pd.Series([list(x.values())[0][0] for x in decision_targets_lcs])
        assert_series_equal(
            lcs_score,
            pd.Series([1.0, 0.07692307692307693, 1.0, 1.0, 1.0, 0.30952380952380953]),
            rtol=1e-3,
        )

        self.assertIn("longest_common_substring_false_pos", results)

        longest_common_substring_false_pos = results[
            "longest_common_substring_false_pos"
        ].tolist()
        longest_common_substring_false_pos_vals = [
            list(x.values())[0] for x in longest_common_substring_false_pos
        ]

        self.assertEqual(longest_common_substring_false_pos_vals, [3, 3, 2, 0, 0, 42])
        fp_scores = pd.Series([list(x.values())[0][1] for x in decision_targets_lcs])
        assert_series_equal(
            fp_scores,
            pd.Series(
                [
                    0.23076923076923078,
                    0.23076923076923078,
                    0.09090909090909091,
                    0.0,
                    0.0,
                    1.0,
                ]
            ),
            rtol=1e-3,
        )

        self.assertIn("edit_similarity", results)
        self.assertIn("edit_similarity_score", results)
        edit_similarity = results["edit_similarity"].tolist()
        self.assertEqual(edit_similarity, [13, 3, 22, 16, 8, 16])
        edit_similarity_score = results["edit_similarity_score"].tolist()
        self.assertAlmostEqual(
            edit_similarity_score,
            [1.0, 0.23076923076923078, 1.0, 1.0, 1.0, 0.38095238095238093],
        )

        # This should be true by definition
        self.assertGreaterEqual(edit_similarity, longest_common_substring_vals)

    def test_output_types(self) -> None:
        analysis_outputs = self.analysis_node.run_analysis()
        self.assertIsInstance(analysis_outputs, TextInclusionAnalysisNodeOutput)

        analysis_outputs_dict = self.analysis_node.compute_outputs()

        self.assertIsInstance(analysis_outputs_dict, dict)
        self.assertIsInstance(analysis_outputs_dict["num_samples"], int)
        self.assertIsInstance(analysis_outputs_dict["exact_match"], pd.Series)
        self.assertIsInstance(analysis_outputs_dict["inclusion_score"], pd.Series)
        self.assertIsInstance(
            analysis_outputs_dict["longest_common_substring"], pd.Series
        )
        self.assertIsInstance(
            analysis_outputs_dict["longest_common_substring_false_pos"], pd.Series
        )
        self.assertIsInstance(analysis_outputs_dict["decision_targets_lcs"], pd.Series)
        self.assertIsInstance(analysis_outputs_dict["edit_similarity"], pd.Series)
        self.assertIsInstance(analysis_outputs_dict["edit_similarity_score"], pd.Series)
        if analysis_outputs_dict["filtered_true_positive_list"] is not None:
            self.assertIsInstance(
                analysis_outputs_dict["filtered_true_positive_list"], list
            )
            self.assertIsInstance(
                analysis_outputs_dict["filtered_true_positive_list"][0], str
            )

    def test_text_inclusion_no_lcs(self) -> None:
        analysis_input = TextInclusionAnalysisInput(
            generation_df=pd.DataFrame(self.data), disable_lcs=True
        )
        analysis_node = TextInclusionAnalysisNode(analysis_input=analysis_input)

        results = analysis_node.compute_outputs()

        self.assertIn("exact_match", results)

        self.assertIn("inclusion_score", results)

        # If exact match is true, then inclusion score must be true

        self.assertEqual(results["longest_common_substring"], None)
        self.assertEqual(results["decision_targets_lcs"], None)

        self.assertIn("edit_similarity", results)
        self.assertIn("edit_similarity_score", results)
        self.assertIsNotNone(results["edit_similarity"], None)
        self.assertIsNotNone(results["edit_similarity_score"], None)

    def test_text_inclusion_no_similarity(self) -> None:
        analysis_input = TextInclusionAnalysisInput(
            generation_df=pd.DataFrame(self.data), disable_similarity=True
        )
        analysis_node = TextInclusionAnalysisNode(analysis_input=analysis_input)

        results = analysis_node.compute_outputs()

        self.assertIn("exact_match", results)

        self.assertIn("inclusion_score", results)

        self.assertIn("longest_common_substring", results)
        self.assertIn("decision_targets_lcs", results)
        self.assertIsNotNone(results["longest_common_substring"])
        self.assertIsNotNone(results["decision_targets_lcs"])

        self.assertEqual(results["edit_similarity"], None)
        self.assertEqual(results["edit_similarity_score"], None)

    def test_text_inclusion_augmented_output(self) -> None:
        analysis_input = TextInclusionAnalysisInput(
            generation_df=pd.DataFrame(self.data)
        )
        analysis_node = TextInclusionAnalysisNode(analysis_input=analysis_input)

        results = analysis_node.compute_outputs()

        results_augmented = results["augmented_output_dataset"]

        for key in analysis_node.LCS_METRIC_KEYS:
            self.assertIn(key, results_augmented.columns)

        self.assertIn("edit_similarity", results_augmented.columns)

    def test_multi_target(self) -> None:
        multi_data = {
            "prompt": [
                "This is a test prompt",
                "Exact match to prompt!",
            ],
            "targets": [
                ["Target text 1", "Exact match to output_text!"],
                ["Target text 2", "Exact match to prompt!"],
            ],
            "output_text": [
                "Exact match to output_text!",
                "Target text 2",
            ],
        }

        multi_analysis_input = TextInclusionAnalysisInput(
            generation_df=pd.DataFrame(multi_data),
            target_key="targets",
            disable_exact_match=True,
            disable_similarity=True,
        )
        multi_analysis_node = TextInclusionAnalysisNode(
            analysis_input=multi_analysis_input
        )

        results = multi_analysis_node.compute_outputs()

        self.assertIn("longest_common_substring", results)
        self.assertIn("decision_targets_lcs", results)

        expected_targets: List[Dict[str, int]] = [
            {"target text 1": 4, "exact match to outputtext": 25},
            {"target text 2": 13, "exact match to prompt": 2},
        ]
        self.assertEqual(expected_targets, results["longest_common_substring"].tolist())

        expected_decision_scores = pd.Series(
            [
                {
                    "target text 1": (0.3076923076923077, 0.23076923076923078),
                    "exact match to outputtext": (1.0, 0.08),
                },
                {
                    "target text 2": (1.0, 0.15384615384615385),
                    "exact match to prompt": (0.09523809523809523, 1.0),
                },
            ]
        )

        assert_series_equal(
            pd.Series(list(results["decision_targets_lcs"])),
            expected_decision_scores,
            rtol=1e-3,
        )

        self.assertIn("longest_common_substring_false_pos", results)
        expected_fp: List[Dict[str, int]] = [
            {"target text 1": 3, "exact match to outputtext": 2},
            {"target text 2": 2, "exact match to prompt": 21},
        ]

        longest_common_substring_false_pos = results[
            "longest_common_substring_false_pos"
        ].tolist()

        self.assertEqual(longest_common_substring_false_pos, expected_fp)

        self.assertEqual(len(results["exact_match"]), 0)
        self.assertEqual(len(results["inclusion_score"]), 0)

    def test_bounded_longest_common_substring_match(self) -> None:
        s1 = ("w" * 50) + ("t" * 160) + ("b" * 50) + ("t" * 155)
        s2 = ("x" * 50) + ("t" * 200) + ("c" * 150) + ("t" * 200)

        # LCS is t, 150
        self.assertEqual(
            _char_level_longest_common_substring_helper_bound(s1=s1, s2=s2, target=150),
            150,
        )
        self.assertEqual(
            _char_level_longest_common_substring_helper_bound(s1=s1, s2=s2, target=50),
            50,
        )

        self.assertEqual(_char_level_longest_common_substring_helper(s1=s1, s2=s2), 160)

        s1 = "a" * 200
        s2 = "b" * 200 + "a" * 20

        self.assertEqual(
            _char_level_longest_common_substring_helper_bound(s1=s1, s2=s2, target=150),
            0,
        )
        self.assertEqual(
            _char_level_longest_common_substring_helper_bound(s1=s1, s2=s2, target=50),
            0,
        )
        self.assertEqual(
            _char_level_longest_common_substring_helper_bound(s1=s1, s2=s2, target=10),
            10,
        )

        self.assertEqual(_char_level_longest_common_substring_helper(s1=s1, s2=s2), 20)

    def test_word_level_longest_common_susequence_match(self) -> None:
        s1 = (
            ("w" * 50)
            + " "
            + ("t" * 160)
            + " "
            + ("b" * 50)
            + " "
            + ("t" * 155)
            + " "
            + ("t" * 130)
        )
        s2 = (
            ("x" * 50)
            + " "
            + ("t" * 160)
            + " "
            + ("c" * 150)
            + " "
            + ("t" * 200)
            + " "
            + ("t" * 130)
        )

        self.assertEqual(_word_level_longest_common_subsequence_helper(s1=s1, s2=s2), 2)
        self.assertEqual(_word_level_longest_common_subsequence_helper(s1=s1, s2=s1), 5)

        s1 = "a b a"
        s2 = "c a b a d"
        s3 = "a d b a"

        self.assertEqual(_word_level_longest_common_subsequence_helper(s1=s1, s2=s2), 3)
        self.assertEqual(_word_level_longest_common_subsequence_helper(s1=s2, s2=s3), 3)
        self.assertEqual(_word_level_longest_common_subsequence_helper(s1=s1, s2=s3), 3)

    def test_char_level_longest_common_susequence_match(self) -> None:
        s1 = ("w" * 5) + ("t" * 16) + ("b" * 5) + ("t" * 15)
        s2 = ("x" * 5) + ("t" * 16) + ("c" * 15) + ("t" * 20)

        self.assertEqual(
            _char_level_longest_common_subsequence_helper(s1=s1, s2=s2), 31
        )
        self.assertEqual(
            _char_level_longest_common_subsequence_helper(s1=s1, s2=s1), 41
        )

        s1 = "a b a"
        s2 = "c a b a d"
        s3 = "a d b a e"

        self.assertEqual(_char_level_longest_common_subsequence_helper(s1=s1, s2=s2), 5)
        self.assertEqual(_char_level_longest_common_subsequence_helper(s1=s2, s2=s3), 6)
        self.assertEqual(_char_level_longest_common_subsequence_helper(s1=s1, s2=s3), 5)

    def test_longest_common_susequence_match_autojunk(self) -> None:
        s1 = ("w" * 50) + ("t" * 160) + ("b" * 50) + ("t" * 150)
        s2 = ("x" * 50) + ("t" * 160) + ("c" * 150) + ("t" * 200)

        self.assertEqual(
            _char_level_longest_common_subsequence_helper(s1=s1, s2=s2, autojunk=False),
            310,
        )
        self.assertEqual(
            _char_level_longest_common_subsequence_helper(s1=s1, s2=s2, autojunk=True),
            0,
        )

        s1 = (
            ("w " * 50)
            + ("t " * 160)
            + ("b " * 50)
            + ("tt " * 150)
            + ("t " * 100)
            + "end1"
        )
        s2 = ("x " * 50) + ("t " * 160) + ("c " * 150) + ("t " * 200) + "end2"

        self.assertEqual(
            _word_level_longest_common_subsequence_helper(s1=s1, s2=s2, autojunk=False),
            260,
        )
        self.assertEqual(
            _word_level_longest_common_subsequence_helper(s1=s1, s2=s2, autojunk=True),
            0,
        )
