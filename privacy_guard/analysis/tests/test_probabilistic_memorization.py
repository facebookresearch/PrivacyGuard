# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
import unittest

import pandas as pd

from privacy_guard.analysis.extraction.probabilistic_memorization_analysis_input import (
    ProbabilisticMemorizationAnalysisInput,
)
from privacy_guard.analysis.extraction.probabilistic_memorization_analysis_node import (
    _check_above_probability_threshold,
    _compute_model_probability,
    _compute_n_probabilities_dict,
    ProbabilisticMemorizationAnalysisNode,
    ProbabilisticMemorizationAnalysisNodeOutput,
)


class TestProbabilisticMemorizationAnalysis(unittest.TestCase):
    def setUp(self) -> None:
        self.data = {
            "prediction_logprobs": [
                [
                    [-0.1, -0.2, -0.05]
                ],  # Single sublist: Sum: -0.35, exp(-0.35) ≈ 0.705 (above threshold)
                [
                    [-0.5, -0.3, -0.9]
                ],  # Single sublist: Sum: -1.7, exp(-1.7) ≈ 0.183 (below threshold)
                [[-0.2, -0.3]],  # Sum: -0.5, exp(-0.5) ≈ 0.607 (above threshold)
                [[-1.2, -0.8]],  # Sum: -2.0, exp(-2.0) ≈ 0.135 (below threshold)
                [],  # Empty, should return 0.0 (below threshold)
            ]
        }

        self.prob_threshold = 0.5
        self.n_values = [2, 5, 10]

        self.analysis_input = ProbabilisticMemorizationAnalysisInput(
            generation_df=pd.DataFrame(self.data),
            prob_threshold=self.prob_threshold,
            n_values=self.n_values,
        )
        self.analysis_node = ProbabilisticMemorizationAnalysisNode(
            analysis_input=self.analysis_input
        )

        super().setUp()

    def test_required_columns_validation(self) -> None:
        """Test that missing required columns raise an error."""
        invalid_data = {"invalid_column": [1, 2, 3]}
        with self.assertRaises(ValueError):
            ProbabilisticMemorizationAnalysisInput(
                generation_df=pd.DataFrame(invalid_data),
                prob_threshold=0.5,
            )

    def test_compute_model_probability(self) -> None:
        # Test with single nested list (valid format)
        row1 = pd.Series({"prediction_logprobs": [[-0.1, -0.2, -0.3]]})
        expected1 = math.exp(-0.6)  # Sum is -0.6
        self.assertAlmostEqual(
            _compute_model_probability(row1, "prediction_logprobs"), expected1, places=5
        )

        # Test with empty list
        row2 = pd.Series({"prediction_logprobs": []})
        self.assertEqual(_compute_model_probability(row2, "prediction_logprobs"), 0.0)

        # Test with single nested list containing one element
        row3 = pd.Series({"prediction_logprobs": [[0.0]]})
        self.assertEqual(_compute_model_probability(row3, "prediction_logprobs"), 1.0)

        # Test with 1D list (valid format)
        row4 = pd.Series({"prediction_logprobs": [-0.5, -0.3, -0.2]})
        expected4 = math.exp(-1.0)  # Sum is -1.0
        self.assertAlmostEqual(
            _compute_model_probability(row4, "prediction_logprobs"), expected4, places=5
        )

    def test_compute_model_probability_error_cases(self) -> None:
        # Test with multiple nested lists (should raise ValueError)
        row_invalid = pd.Series({"prediction_logprobs": [[-0.1, -0.2], [-0.3, -0.4]]})
        with self.assertRaises(ValueError) as context:
            _compute_model_probability(row_invalid, "prediction_logprobs")
        self.assertIn("Invalid format for prediction_logprobs", str(context.exception))

    def test_compute_n_probabilities_dict(self) -> None:
        model_prob = 0.8
        n_values = [2, 5, 10]
        result = _compute_n_probabilities_dict(model_prob, n_values)

        # Expected calculations: p = 1 - (1 - model_prob)**n
        expected = {
            2: 1.0 - ((1.0 - 0.8) ** 2),  # 1 - 0.04 = 0.96
            5: 1.0 - ((1.0 - 0.8) ** 5),  # 1 - 0.00032 ≈ 0.99968
            10: 1.0 - ((1.0 - 0.8) ** 10),  # 1 - 1.024e-7 ≈ 0.9999999
        }

        self.assertEqual(set(result.keys()), set(expected.keys()))
        for n in n_values:
            self.assertAlmostEqual(result[n], expected[n], places=5)

    def test_check_above_probability_threshold(self) -> None:
        self.assertTrue(_check_above_probability_threshold(1.5, 1.0))
        self.assertFalse(_check_above_probability_threshold(0.5, 1.0))
        self.assertFalse(
            _check_above_probability_threshold(1.0, 1.0)
        )  # Equal is not above

    def test_run_analysis_basic(self) -> None:
        results = self.analysis_node.compute_outputs()

        # Check that expected keys are present
        self.assertIn("num_samples", results)
        self.assertIn("model_probability", results)
        self.assertIn("above_probability_threshold", results)
        self.assertIn("n_probabilities", results)
        self.assertIn("augmented_output_dataset", results)

        # Check number of samples
        self.assertEqual(results["num_samples"], 5)

        # Check model probabilities are computed correctly
        model_probs = results["model_probability"]
        expected_model_probs = [
            math.exp(-0.35),  # ≈ 0.705
            math.exp(-1.7),  # ≈ 0.183
            math.exp(-0.5),  # ≈ 0.607
            math.exp(-2.0),  # ≈ 0.135
            0.0,  # empty list
        ]

        for i, expected in enumerate(expected_model_probs):
            self.assertAlmostEqual(model_probs.iloc[i], expected, places=3)

        # Check threshold comparisons (threshold = 0.5)
        above_probability_threshold = results["above_probability_threshold"]
        expected_above_probability_threshold = [True, False, True, False, False]
        self.assertEqual(
            above_probability_threshold.tolist(), expected_above_probability_threshold
        )

        # Check n_probabilities structure
        n_probabilities = results["n_probabilities"]
        self.assertIsNotNone(n_probabilities)
        for _i, n_prob_dict in enumerate(n_probabilities):
            self.assertIsInstance(n_prob_dict, dict)
            self.assertEqual(set(n_prob_dict.keys()), {2, 5, 10})
            # Check that values are between 0 and 1
            for _n, prob in n_prob_dict.items():
                self.assertGreaterEqual(prob, 0.0)
                self.assertLessEqual(prob, 1.0)

    def test_output_types(self) -> None:
        analysis_outputs = self.analysis_node.run_analysis()
        self.assertIsInstance(
            analysis_outputs, ProbabilisticMemorizationAnalysisNodeOutput
        )

        analysis_outputs_dict = self.analysis_node.compute_outputs()

        self.assertIsInstance(analysis_outputs_dict, dict)
        self.assertIsInstance(analysis_outputs_dict["num_samples"], int)
        self.assertIsInstance(analysis_outputs_dict["model_probability"], pd.Series)
        self.assertIsInstance(
            analysis_outputs_dict["above_probability_threshold"], pd.Series
        )
        self.assertIsInstance(analysis_outputs_dict["n_probabilities"], pd.Series)
        self.assertIsInstance(
            analysis_outputs_dict["augmented_output_dataset"], pd.DataFrame
        )

    def test_no_n_values(self) -> None:
        analysis_input = ProbabilisticMemorizationAnalysisInput(
            generation_df=pd.DataFrame(self.data),
            prob_threshold=self.prob_threshold,
        )
        analysis_node = ProbabilisticMemorizationAnalysisNode(
            analysis_input=analysis_input
        )

        results = analysis_node.compute_outputs()

        # Check that basic outputs are still present
        self.assertIn("model_probability", results)
        self.assertIn("above_probability_threshold", results)

        # Check that n_probabilities is None when no n_values provided
        self.assertIsNone(results["n_probabilities"])

        # Check that n_probabilities column is not in augmented dataset
        augmented_df = results["augmented_output_dataset"]
        self.assertNotIn("n_probabilities", augmented_df.columns)

    def test_augmented_output_dataset(self) -> None:
        """Test that the augmented dataset contains all expected columns."""
        results = self.analysis_node.compute_outputs()
        augmented_df = results["augmented_output_dataset"]

        # Check that original columns are preserved
        self.assertIn("prediction_logprobs", augmented_df.columns)

        # Check that computed columns are added
        self.assertIn("model_probability", augmented_df.columns)
        self.assertIn("above_probability_threshold", augmented_df.columns)
        self.assertIn("n_probabilities", augmented_df.columns)

        # Check that the number of rows is preserved
        self.assertEqual(len(augmented_df), len(self.data["prediction_logprobs"]))

    def test_custom_logprobs_column(self) -> None:
        """Test that custom logprobs column names work correctly."""
        # Create data with custom column name
        custom_data = {
            "custom_logprobs": [
                [[-0.1, -0.2, -0.05]],  # Sum: -0.35, exp(-0.35) ≈ 0.705
                [[-0.5, -0.3, -0.9]],  # Sum: -1.7, exp(-1.7) ≈ 0.183
            ]
        }

        analysis_input = ProbabilisticMemorizationAnalysisInput(
            generation_df=pd.DataFrame(custom_data),
            prob_threshold=0.5,
            logprobs_column="custom_logprobs",
        )

        analysis_node = ProbabilisticMemorizationAnalysisNode(
            analysis_input=analysis_input
        )

        results = analysis_node.compute_outputs()

        # Check that original column is preserved
        augmented_df = results["augmented_output_dataset"]
        self.assertIn("custom_logprobs", augmented_df.columns)

        # Check that model probabilities are computed correctly
        model_probs = results["model_probability"]
        expected_model_probs = [math.exp(-0.35), math.exp(-1.7)]

        for i, expected in enumerate(expected_model_probs):
            self.assertAlmostEqual(model_probs.iloc[i], expected, places=3)

        # Check threshold comparisons
        above_probability_threshold = results["above_probability_threshold"]
        expected_above_probability_threshold = [True, False]  # 0.705 > 0.5, 0.183 < 0.5
        self.assertEqual(
            above_probability_threshold.tolist(), expected_above_probability_threshold
        )
