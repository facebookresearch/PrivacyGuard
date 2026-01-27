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
    _char_level_longest_common_substring_with_matched_text,
    _clean_text,
    _clean_text_remove_consecutive_whitespace,
    _normalize_by_target_len,
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
            generation_df=pd.DataFrame(self.data), disable_longest_common_substring=True
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

        self.assertIsNone(results["char_level_longest_common_subsequence"])
        self.assertIsNotNone(results["word_level_longest_common_subsequence"])

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

        self.assertIsNone(results["char_level_longest_common_subsequence"])
        self.assertIsNotNone(results["word_level_longest_common_subsequence"])

    def test_text_inclusion_with_char_level_longest_common_subsequence(self) -> None:
        analysis_input = TextInclusionAnalysisInput(
            generation_df=pd.DataFrame(self.data),
            disable_char_level_longest_common_subsequence=False,
            disable_word_level_longest_common_subsequence=False,
        )
        analysis_node = TextInclusionAnalysisNode(analysis_input=analysis_input)

        results = analysis_node.compute_outputs()

        self.assertIn("exact_match", results)

        self.assertIn("inclusion_score", results)

        self.assertIn("longest_common_substring", results)
        self.assertIn("decision_targets_lcs", results)
        self.assertIsNotNone(results["longest_common_substring"])
        self.assertIsNotNone(results["decision_targets_lcs"])

        self.assertIsNotNone(results["edit_similarity"])
        self.assertIsNotNone(results["edit_similarity_score"])

        self.assertIsNotNone(results["char_level_longest_common_subsequence"])
        self.assertIsNotNone(results["word_level_longest_common_subsequence"])

        for char_lcs, word_lcs in zip(
            results["char_level_longest_common_subsequence"],
            results["word_level_longest_common_subsequence"],
        ):
            self.assertGreaterEqual(char_lcs, word_lcs[0])

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
            disable_word_level_longest_common_subsequence=True,
            disable_char_level_longest_common_subsequence=True,
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

    def test_bounded_longest_common_substring_with_matched_text(self) -> None:
        s1 = 'Yesterday I woke up and thought " I will go to\nthe park"'
        s2 = "He said to me that he will go to the gym"

        max_len, max_substring = _char_level_longest_common_substring_with_matched_text(
            _clean_text(s1), _clean_text(s2)
        )
        self.assertEqual(max_substring, " will go to the ")
        self.assertEqual(len(" will go to the "), max_len)

    def test_format_lcs_result(self) -> None:
        analysis_outputs = self.analysis_node.run_analysis()
        self.assertIsInstance(analysis_outputs, TextInclusionAnalysisNodeOutput)

        lcs_result_formatted = analysis_outputs.lcs_result_formatted(
            display_lcs_match=True
        )

        result = dict(lcs_result_formatted.iloc[-1])

        self.assertIn("lcs", result.keys())
        self.assertIn("% target extracted", result.keys())
        self.assertIn("prompt", result.keys())
        self.assertIn("output_text", result.keys())
        self.assertIn("target", result.keys())
        self.assertIn("lcs_match", result.keys())

        self.assertEqual(result["lcs_match"], "dolorem ipsum")

    def test_format_lcs_result_no_analysis_input(self) -> None:
        outputs = TextInclusionAnalysisNodeOutput(
            num_samples=0,
            exact_match=pd.Series(),
            inclusion_score=pd.Series(),
            longest_common_substring=None,
            longest_common_substring_false_pos=None,
            decision_targets_lcs=None,
            decision_targets_lcs_len=None,
            edit_similarity=None,
            edit_similarity_score=None,
            filtered_true_positive_list=None,
            augmented_output_dataset=pd.DataFrame(),
            word_level_longest_common_subsequence=None,
            char_level_longest_common_subsequence=None,
            analysis_input=None,
        )
        with self.assertRaisesRegex(ValueError, "No lcs results to display"):
            outputs.lcs_result_formatted(display_lcs_match=True)

        outputs.longest_common_substring = pd.Series()
        with self.assertRaisesRegex(ValueError, "No analysis input"):
            outputs.lcs_result_formatted(display_lcs_match=True)

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

        self.assertEqual(
            _word_level_longest_common_subsequence_helper(s1=s1, s2=s2)[0], 2
        )
        self.assertEqual(
            _word_level_longest_common_subsequence_helper(s1=s1, s2=s1)[0], 5
        )

        s1 = "a b a"
        s2 = "c a b a d"
        s3 = "a d b a"

        self.assertEqual(
            _word_level_longest_common_subsequence_helper(s1=s1, s2=s2), (3, "a b a")
        )
        self.assertEqual(
            _word_level_longest_common_subsequence_helper(s1=s2, s2=s3), (3, "a b a")
        )
        self.assertEqual(
            _word_level_longest_common_subsequence_helper(s1=s1, s2=s3), (3, "a b a")
        )

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
            _word_level_longest_common_subsequence_helper(s1=s1, s2=s2, autojunk=False)[
                0
            ],
            260,
        )
        self.assertEqual(
            _word_level_longest_common_subsequence_helper(s1=s1, s2=s2, autojunk=True)[
                0
            ],
            0,
        )

    def test_clean_text_remove_consecutive_whitespace(self) -> None:
        # Test that consecutive whitespace is properly removed
        text_with_consecutive_spaces = "hello  world"
        result = _clean_text_remove_consecutive_whitespace(text_with_consecutive_spaces)
        self.assertEqual(result, "hello world")

        # Test multiple consecutive spaces
        text_with_many_spaces = "hello    world   test"
        result = _clean_text_remove_consecutive_whitespace(text_with_many_spaces)
        self.assertEqual(result, "hello world test")

        # Test tabs and newlines are also collapsed
        text_with_mixed_whitespace = "hello\t\t\nworld"
        result = _clean_text_remove_consecutive_whitespace(text_with_mixed_whitespace)
        self.assertEqual(result, "hello world")

        # Test leading/trailing whitespace is stripped
        text_with_leading_trailing = "  hello world  "
        result = _clean_text_remove_consecutive_whitespace(text_with_leading_trailing)
        self.assertEqual(result, "hello world")

    def test_normalize_by_target_len_respects_clean_method(self) -> None:
        # Test that _normalize_by_target_len uses the provided clean_text_method
        targets = pd.Series(["hello  world", "test  text"])  # consecutive spaces
        scores = pd.Series([10.0, 8.0])

        # With _clean_text, consecutive spaces are preserved
        # "hello  world" -> "hello  world" (12 chars after cleaning)
        result_basic = _normalize_by_target_len(scores, targets, _clean_text)
        # len("hello  world") = 12
        self.assertAlmostEqual(result_basic.iloc[0], 10.0 / 12.0)

        # With _clean_text_remove_consecutive_whitespace, consecutive spaces are removed
        # "hello  world" -> "hello world" (11 chars after cleaning)
        result_no_consecutive = _normalize_by_target_len(
            scores, targets, _clean_text_remove_consecutive_whitespace
        )
        # len("hello world") = 11
        self.assertAlmostEqual(result_no_consecutive.iloc[0], 10.0 / 11.0)

        # Verify the two methods produce different results
        self.assertNotAlmostEqual(result_basic.iloc[0], result_no_consecutive.iloc[0])

    def test_analysis_with_remove_consecutive_whitespace(self) -> None:
        # Test that TextInclusionAnalysisNode respects remove_consecutive_whitespace config
        data = {
            "prompt": ["test  prompt"],  # consecutive spaces
            "target": ["target  text"],  # consecutive spaces
            "output_text": ["target text"],  # no consecutive spaces
        }

        # Without remove_consecutive_whitespace
        analysis_input_basic = TextInclusionAnalysisInput(
            generation_df=pd.DataFrame(data),
            remove_consecutive_whitespace=False,
            disable_longest_common_substring=True,
            disable_word_level_longest_common_subsequence=True,
            disable_char_level_longest_common_subsequence=True,
        )
        analysis_node_basic = TextInclusionAnalysisNode(
            analysis_input=analysis_input_basic
        )
        results_basic = analysis_node_basic.compute_outputs()

        # With remove_consecutive_whitespace
        analysis_input_cleaned = TextInclusionAnalysisInput(
            generation_df=pd.DataFrame(data),
            remove_consecutive_whitespace=True,
            disable_longest_common_substring=True,
            disable_word_level_longest_common_subsequence=True,
            disable_char_level_longest_common_subsequence=True,
        )
        analysis_node_cleaned = TextInclusionAnalysisNode(
            analysis_input=analysis_input_cleaned
        )
        results_cleaned = analysis_node_cleaned.compute_outputs()

        # edit_similarity_score should be different because target length differs
        # when consecutive whitespace is removed vs preserved
        self.assertNotAlmostEqual(
            results_basic["edit_similarity_score"].iloc[0],
            results_cleaned["edit_similarity_score"].iloc[0],
        )

    def test_format_single_word_level_lcs_result(self) -> None:
        """Test format_single_word_level_lcs_result returns correct dictionary structure."""
        analysis_outputs = self.analysis_node.run_analysis()
        self.assertIsInstance(analysis_outputs, TextInclusionAnalysisNodeOutput)

        # Get the augmented row data
        augmented_row = analysis_outputs.augmented_output_dataset.iloc[-1].to_dict()

        # Call format_single_word_level_lcs_result directly
        result = analysis_outputs.format_single_word_level_lcs_result(
            num_matched_words=3,
            matched_string="dolorem ipsum quia",
            augmented_row=augmented_row,
            analysis_input=self.analysis_input,
        )

        # Verify the result dictionary has the expected keys
        self.assertIn("Count of matched words", result.keys())
        self.assertIn("Length of matched words", result.keys())
        self.assertIn("Matched consecutive sequence", result.keys())
        self.assertIn("% target extracted", result.keys())
        self.assertIn("prompt", result.keys())
        self.assertIn("output_text", result.keys())
        self.assertIn("target", result.keys())

        # Verify the values are correct
        self.assertEqual(result["Count of matched words"], 3)
        self.assertEqual(result["Length of matched words"], len("dolorem ipsum quia"))
        self.assertEqual(result["Matched consecutive sequence"], "dolorem ipsum quia")

    def test_format_single_word_level_lcs_result_empty_target(self) -> None:
        """Test format_single_word_level_lcs_result handles empty target correctly."""
        analysis_outputs = self.analysis_node.run_analysis()

        # Create an augmented row with an empty target
        augmented_row = {
            "prompt": "test prompt",
            "target": "",
            "output_text": "test output",
        }

        result = analysis_outputs.format_single_word_level_lcs_result(
            num_matched_words=0,
            matched_string="",
            augmented_row=augmented_row,
            analysis_input=self.analysis_input,
        )

        # Verify % target extracted is N/A for empty target
        self.assertEqual(result["% target extracted"], "N/A")

    def test_word_level_lcs_result_formatted(self) -> None:
        """Test word_level_lcs_result_formatted returns correct DataFrame."""
        analysis_outputs = self.analysis_node.run_analysis()
        self.assertIsInstance(analysis_outputs, TextInclusionAnalysisNodeOutput)

        # Ensure word-level LCS is computed
        self.assertIsNotNone(analysis_outputs.word_level_longest_common_subsequence)

        # Call word_level_lcs_result_formatted
        word_level_formatted = analysis_outputs.word_level_lcs_result_formatted()

        # Verify it returns a DataFrame
        self.assertIsInstance(word_level_formatted, pd.DataFrame)

        # Verify the DataFrame has the expected columns
        self.assertIn("Count of matched words", word_level_formatted.columns)
        self.assertIn("Length of matched words", word_level_formatted.columns)
        self.assertIn("Matched consecutive sequence", word_level_formatted.columns)
        self.assertIn("% target extracted", word_level_formatted.columns)
        self.assertIn("prompt", word_level_formatted.columns)
        self.assertIn("target", word_level_formatted.columns)
        self.assertIn("output_text", word_level_formatted.columns)

        # Verify the DataFrame has the same number of rows as the input data
        self.assertEqual(len(word_level_formatted), len(self.data["prompt"]))

    def test_word_level_lcs_result_formatted_no_lcs_results(self) -> None:
        """Test word_level_lcs_result_formatted raises error when no LCS results."""
        outputs = TextInclusionAnalysisNodeOutput(
            num_samples=0,
            exact_match=pd.Series(),
            inclusion_score=pd.Series(),
            longest_common_substring=None,
            longest_common_substring_false_pos=None,
            decision_targets_lcs=None,
            decision_targets_lcs_len=None,
            edit_similarity=None,
            edit_similarity_score=None,
            filtered_true_positive_list=None,
            augmented_output_dataset=pd.DataFrame(),
            word_level_longest_common_subsequence=None,
            char_level_longest_common_subsequence=None,
            analysis_input=None,
        )
        with self.assertRaisesRegex(ValueError, "No lcs results to display"):
            outputs.word_level_lcs_result_formatted()

    def test_word_level_lcs_result_formatted_no_analysis_input(self) -> None:
        """Test word_level_lcs_result_formatted raises error when no analysis input."""
        outputs = TextInclusionAnalysisNodeOutput(
            num_samples=0,
            exact_match=pd.Series(),
            inclusion_score=pd.Series(),
            longest_common_substring=None,
            longest_common_substring_false_pos=None,
            decision_targets_lcs=None,
            decision_targets_lcs_len=None,
            edit_similarity=None,
            edit_similarity_score=None,
            filtered_true_positive_list=None,
            augmented_output_dataset=pd.DataFrame(),
            word_level_longest_common_subsequence=pd.Series([(1, "test")]),
            char_level_longest_common_subsequence=None,
            analysis_input=None,
        )
        with self.assertRaisesRegex(ValueError, "No analysis input"):
            outputs.word_level_lcs_result_formatted()

    def test_word_level_lcs_result_formatted_german(self) -> None:
        """Test word_level_lcs_result_formatted with German text containing non-consecutive matches."""
        # Target and output differ only in filler words (HIER vs DORT)
        # This tests that non-consecutive matching works correctly
        german_data = {
            "prompt": [
                "Erzähle mir eine Geschichte über einen Hund im Wald",
            ],
            "target": [
                "Der kleine Hund läuft HIER durch den großen Wald HIER und findet HIER einen roten Ball HIER unter dem alten Baum HIER neben dem kleinen Bach",
            ],
            "output_text": [
                "Der kleine Hund läuft DORT durch den großen Wald DORT und findet DORT einen roten Ball DORT unter dem alten Baum DORT neben dem kleinen Bach",
            ],
        }

        german_analysis_input = TextInclusionAnalysisInput(
            generation_df=pd.DataFrame(german_data)
        )
        german_analysis_node = TextInclusionAnalysisNode(
            analysis_input=german_analysis_input
        )

        analysis_outputs = german_analysis_node.run_analysis()

        # Ensure word-level LCS is computed
        self.assertIsNotNone(analysis_outputs.word_level_longest_common_subsequence)

        # Call word_level_lcs_result_formatted
        word_level_formatted = analysis_outputs.word_level_lcs_result_formatted()

        # Verify it returns a DataFrame with correct structure
        self.assertIsInstance(word_level_formatted, pd.DataFrame)
        self.assertEqual(len(word_level_formatted), 1)

        first_row = word_level_formatted.iloc[0]

        # Target has 26 words, 5 are "HIER" which don't match "DORT" in output
        # So we expect 21 matched words across multiple non-consecutive blocks:
        # Block 1: "der kleine hund läuft" (4 words)
        # Block 2: "durch den großen wald" (4 words)
        # Block 3: "und findet" (2 words)
        # Block 4: "einen roten ball" (3 words)
        # Block 5: "unter dem alten baum" (4 words)
        # Block 6: "neben dem kleinen bach" (4 words)
        # Total: 4 + 4 + 2 + 3 + 4 + 4 = 21 words
        self.assertEqual(first_row["Count of matched words"], 21)

        # The matched string should be all words except HIER (after cleaning: lowercase, no punctuation)
        expected_matched_string = (
            "der kleine hund läuft durch den großen wald und findet "
            "einen roten ball unter dem alten baum neben dem kleinen bach"
        )
        self.assertEqual(
            first_row["Matched consecutive sequence"], expected_matched_string
        )

    def test_word_level_lcs_result_formatted_spanish(self) -> None:
        """Test word_level_lcs_result_formatted with Spanish text containing non-consecutive matches."""
        # Target and output differ only in filler words (AQUI vs ALLI)
        # This tests that non-consecutive matching works correctly
        spanish_data = {
            "prompt": [
                "Cuéntame una historia sobre un perro en el bosque",
            ],
            "target": [
                "El pequeño perro corre AQUI por el gran bosque AQUI y encuentra AQUI una pelota roja AQUI bajo el viejo árbol AQUI junto al pequeño río",
            ],
            "output_text": [
                "El pequeño perro corre ALLI por el gran bosque ALLI y encuentra ALLI una pelota roja ALLI bajo el viejo árbol ALLI junto al pequeño río",
            ],
        }

        spanish_analysis_input = TextInclusionAnalysisInput(
            generation_df=pd.DataFrame(spanish_data)
        )
        spanish_analysis_node = TextInclusionAnalysisNode(
            analysis_input=spanish_analysis_input
        )

        analysis_outputs = spanish_analysis_node.run_analysis()

        # Ensure word-level LCS is computed
        self.assertIsNotNone(analysis_outputs.word_level_longest_common_subsequence)

        # Call word_level_lcs_result_formatted
        word_level_formatted = analysis_outputs.word_level_lcs_result_formatted()

        # Verify it returns a DataFrame with correct structure
        self.assertIsInstance(word_level_formatted, pd.DataFrame)
        self.assertEqual(len(word_level_formatted), 1)

        first_row = word_level_formatted.iloc[0]

        # Target has 26 words, 5 are "AQUI" which don't match "ALLI" in output
        # So we expect 21 matched words across multiple non-consecutive blocks:
        # Block 1: "el pequeño perro corre" (4 words)
        # Block 2: "por el gran bosque" (4 words)
        # Block 3: "y encuentra" (2 words)
        # Block 4: "una pelota roja" (3 words)
        # Block 5: "bajo el viejo árbol" (4 words)
        # Block 6: "junto al pequeño río" (4 words)
        # Total: 4 + 4 + 2 + 3 + 4 + 4 = 21 words
        self.assertEqual(first_row["Count of matched words"], 21)

        # The matched string should be all words except AQUI (after cleaning: lowercase, no punctuation)
        expected_matched_string = (
            "el pequeño perro corre por el gran bosque y encuentra "
            "una pelota roja bajo el viejo árbol junto al pequeño río"
        )
        self.assertEqual(
            first_row["Matched consecutive sequence"], expected_matched_string
        )
