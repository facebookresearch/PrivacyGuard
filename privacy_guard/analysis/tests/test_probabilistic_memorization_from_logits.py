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
import warnings

import pandas as pd
import torch

from privacy_guard.analysis.extraction.probabilistic_memorization_analysis_from_logits_input import (
    ProbabilisticMemorizationAnalysisFromLogitsInput,
)
from privacy_guard.analysis.extraction.probabilistic_memorization_analysis_from_logits_node import (
    _check_above_probability_threshold,
    _compute_model_probability_from_logits,
    _compute_n_probabilities_dict,
    apply_sampling_params,
    ProbabilisticMemorizationAnalysisFromLogitsNode,
    ProbabilisticMemorizationAnalysisFromLogitsNodeOutput,
)


class TestProbabilisticMemorizationFromLogitsAnalysis(unittest.TestCase):
    def setUp(self) -> None:
        vocab_size = 50

        logits1 = torch.randn(3, vocab_size)
        target_tokens1 = [5, 10, 15]
        logits1[0, 5] = 5.0
        logits1[1, 10] = 5.0
        logits1[2, 15] = 5.0

        logits2 = torch.randn(2, vocab_size)
        target_tokens2 = [20, 25]
        logits2[0, 20] = -5.0
        logits2[1, 25] = -5.0

        logits3 = torch.randn(4, vocab_size)
        target_tokens3 = [1, 2, 3, 4]
        logits3[0, 1] = 1.0
        logits3[1, 2] = 1.0
        logits3[2, 3] = 1.0
        logits3[3, 4] = 1.0

        logits4 = torch.randn(1, vocab_size)
        target_tokens4 = [30]
        logits4[0, 30] = 2.0

        self.data = {
            "prediction_logits": [
                logits1,
                logits2.tolist(),
                logits3,
                logits4.tolist(),
            ],
            "target_tokens": [
                target_tokens1,
                torch.tensor(target_tokens2),
                target_tokens3,
                torch.tensor(target_tokens4),
            ],
        }

        self.prob_threshold = 0.5
        self.n_values = [2, 5, 10]

        self.analysis_input = ProbabilisticMemorizationAnalysisFromLogitsInput(
            generation_df=pd.DataFrame(self.data),
            prob_threshold=self.prob_threshold,
            n_values=self.n_values,
        )
        self.analysis_node = ProbabilisticMemorizationAnalysisFromLogitsNode(
            analysis_input=self.analysis_input
        )

        super().setUp()

    def test_required_columns_validation(self) -> None:
        invalid_data1 = {"prediction_logits": [[1, 2, 3]]}
        with self.assertRaises(ValueError) as context:
            ProbabilisticMemorizationAnalysisFromLogitsInput(
                generation_df=pd.DataFrame(invalid_data1),
                prob_threshold=0.5,
            )
        self.assertIn("Missing required columns", str(context.exception))
        self.assertIn("target_tokens", str(context.exception))

        invalid_data2 = {"target_tokens": [[1, 2, 3]]}
        with self.assertRaises(ValueError) as context:
            ProbabilisticMemorizationAnalysisFromLogitsInput(
                generation_df=pd.DataFrame(invalid_data2),
                prob_threshold=0.5,
            )
        self.assertIn("Missing required columns", str(context.exception))
        self.assertIn("prediction_logits", str(context.exception))

    def test_apply_sampling_params(self) -> None:
        logits = torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])

        scaled_logits = apply_sampling_params(logits, temperature=2.0)
        torch.testing.assert_close(scaled_logits, logits / 2.0)

        scaled_logits = apply_sampling_params(logits, temperature=0.5)
        torch.testing.assert_close(scaled_logits, logits / 0.5)

        logits_topk = torch.tensor([[3.0, 1.0, 2.0, 0.5]])
        filtered_logits = apply_sampling_params(logits_topk, top_k=2)

        self.assertEqual(filtered_logits[0, 1], float("-inf"))
        self.assertEqual(filtered_logits[0, 3], float("-inf"))
        self.assertEqual(filtered_logits[0, 0], 3.0)
        self.assertEqual(filtered_logits[0, 2], 2.0)

        logits_topp = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
        filtered_logits_topp = apply_sampling_params(logits_topp, top_p=0.8)
        self.assertFalse(torch.all(torch.isfinite(filtered_logits_topp)))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            apply_sampling_params(logits_topk, top_k=2, top_p=0.8)
            self.assertEqual(len(w), 1)
            self.assertIn("Both top_k and top_p", str(w[0].message))

    def test_compute_model_probability_from_logits(self) -> None:
        df = pd.DataFrame(self.data)
        row_high_prob = df.iloc[0]

        prob = _compute_model_probability_from_logits(
            row_high_prob, "prediction_logits", "target_tokens"
        )

        self.assertGreater(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

        row_low_prob = df.iloc[1]
        prob_low = _compute_model_probability_from_logits(
            row_low_prob, "prediction_logits", "target_tokens"
        )

        self.assertGreater(prob, prob_low)
        self.assertGreater(prob_low, 0.0)
        self.assertLessEqual(prob_low, 1.0)

        row_none_logits = pd.Series(
            {"prediction_logits": None, "target_tokens": [1, 2]}
        )
        with self.assertRaises(ValueError) as context:
            _compute_model_probability_from_logits(
                row_none_logits, "prediction_logits", "target_tokens"
            )
        self.assertIn("Logits column", str(context.exception))
        self.assertIn("contains None value", str(context.exception))

        row_none_targets = pd.Series(
            {
                "prediction_logits": self.data["prediction_logits"][0],
                "target_tokens": None,
            }
        )
        with self.assertRaises(ValueError) as context:
            _compute_model_probability_from_logits(
                row_none_targets, "prediction_logits", "target_tokens"
            )
        self.assertIn("Target tokens column", str(context.exception))
        self.assertIn("contains None value", str(context.exception))

    def test_compute_model_probability_with_temperature(self) -> None:
        logits = torch.tensor([[2.0, 1.0, 0.0]])
        target_tokens = [0]

        row = pd.Series(
            {"prediction_logits": logits.tolist(), "target_tokens": target_tokens}
        )

        prob_high_temp = _compute_model_probability_from_logits(
            row, "prediction_logits", "target_tokens", temperature=2.0
        )
        prob_low_temp = _compute_model_probability_from_logits(
            row, "prediction_logits", "target_tokens", temperature=0.5
        )
        prob_default = _compute_model_probability_from_logits(
            row, "prediction_logits", "target_tokens"
        )

        self.assertGreater(prob_low_temp, prob_high_temp)
        for prob in [prob_high_temp, prob_low_temp, prob_default]:
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

    def test_compute_n_probabilities_dict(self) -> None:
        """Test n-based probability calculations."""
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
        """Test threshold checking function."""
        self.assertTrue(_check_above_probability_threshold(0.8, 0.5))
        self.assertFalse(_check_above_probability_threshold(0.3, 0.5))
        self.assertFalse(
            _check_above_probability_threshold(0.5, 0.5)
        )  # Equal is not above

    def test_run_analysis_basic(self) -> None:
        """Test the complete analysis pipeline."""
        results = self.analysis_node.compute_outputs()

        # Check that expected keys are present
        self.assertIn("num_samples", results)
        self.assertIn("model_probability", results)
        self.assertIn("above_probability_threshold", results)
        self.assertIn("n_probabilities", results)
        self.assertIn("augmented_output_dataset", results)

        # Check number of samples
        self.assertEqual(results["num_samples"], 4)

        # Check model probabilities are computed
        model_probs = results["model_probability"]
        self.assertEqual(len(model_probs), 4)

        # All probabilities should be valid (between 0 and 1)
        for prob in model_probs:
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

        # Check threshold comparisons
        above_probability_threshold = results["above_probability_threshold"]
        self.assertEqual(len(above_probability_threshold), 4)

        # Results should be boolean
        for result in above_probability_threshold:
            self.assertIsInstance(result, (bool, bool))

        # Check n_probabilities structure
        n_probabilities = results["n_probabilities"]
        self.assertIsNotNone(n_probabilities)
        for n_prob_dict in n_probabilities:
            if n_prob_dict:  # Skip empty cases
                self.assertIsInstance(n_prob_dict, dict)
                self.assertEqual(set(n_prob_dict.keys()), {2, 5, 10})
                # Check that values are between 0 and 1
                for prob in n_prob_dict.values():
                    self.assertGreaterEqual(prob, 0.0)
                    self.assertLessEqual(prob, 1.0)

    def test_output_types(self) -> None:
        """Test that output types are correct."""
        analysis_outputs = self.analysis_node.run_analysis()
        self.assertIsInstance(
            analysis_outputs, ProbabilisticMemorizationAnalysisFromLogitsNodeOutput
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
        """Test analysis when n_values is not provided."""
        analysis_input = ProbabilisticMemorizationAnalysisFromLogitsInput(
            generation_df=pd.DataFrame(self.data),
            prob_threshold=self.prob_threshold,
        )
        analysis_node = ProbabilisticMemorizationAnalysisFromLogitsNode(
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
        self.assertIn("prediction_logits", augmented_df.columns)
        self.assertIn("target_tokens", augmented_df.columns)

        # Check that computed columns are added
        self.assertIn("model_probability", augmented_df.columns)
        self.assertIn("above_probability_threshold", augmented_df.columns)
        self.assertIn("n_probabilities", augmented_df.columns)

        # Check that the number of rows is preserved
        self.assertEqual(len(augmented_df), len(self.data["prediction_logits"]))

    def test_custom_column_names(self) -> None:
        """Test that custom column names work correctly."""
        # Create data with custom column names
        custom_data = {
            "custom_logits": [
                torch.randn(2, 10).tolist(),
                torch.randn(3, 10).tolist(),
            ],
            "custom_targets": [
                [1, 5],
                [2, 7, 9],
            ],
        }

        analysis_input = ProbabilisticMemorizationAnalysisFromLogitsInput(
            generation_df=pd.DataFrame(custom_data),
            prob_threshold=0.5,
            logits_column="custom_logits",
            target_tokens_column="custom_targets",
        )

        analysis_node = ProbabilisticMemorizationAnalysisFromLogitsNode(
            analysis_input=analysis_input
        )

        results = analysis_node.compute_outputs()

        # Check that original columns are preserved
        augmented_df = results["augmented_output_dataset"]
        self.assertIn("custom_logits", augmented_df.columns)
        self.assertIn("custom_targets", augmented_df.columns)

        # Check that model probabilities are computed
        model_probs = results["model_probability"]
        self.assertEqual(len(model_probs), 2)

        # All probabilities should be valid
        for prob in model_probs:
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

    def test_dimension_mismatch_handling(self) -> None:
        """Test handling of dimension mismatches between logits and target tokens."""
        # Create data with mismatched dimensions
        mismatch_data = {
            "prediction_logits": [
                torch.randn(3, 10).tolist(),  # 3 tokens
            ],
            "target_tokens": [
                [1, 5],  # Only 2 target tokens (mismatch with 3 logit tokens)
            ],
        }

        analysis_input = ProbabilisticMemorizationAnalysisFromLogitsInput(
            generation_df=pd.DataFrame(mismatch_data),
            prob_threshold=0.5,
        )

        analysis_node = ProbabilisticMemorizationAnalysisFromLogitsNode(
            analysis_input=analysis_input
        )

        # Should raise an error due to dimension mismatch validation
        with self.assertRaises(ValueError) as context:
            analysis_node.run_analysis()
        self.assertIn("Sequence length mismatch", str(context.exception))

    def test_invalid_logits_dimensions(self) -> None:
        """Test handling of invalid logits dimensions."""
        # Create data with invalid 1D logits (should be 2D)
        invalid_data = {
            "prediction_logits": [
                [1, 2, 3],  # 1D list instead of 2D
            ],
            "target_tokens": [
                [0],  # Single target token
            ],
        }

        analysis_input = ProbabilisticMemorizationAnalysisFromLogitsInput(
            generation_df=pd.DataFrame(invalid_data),
            prob_threshold=0.5,
        )

        analysis_node = ProbabilisticMemorizationAnalysisFromLogitsNode(
            analysis_input=analysis_input
        )

        # Should raise an error due to invalid tensor dimensions
        with self.assertRaises(ValueError) as context:
            analysis_node.run_analysis()
        self.assertIn("Expected 2D logits tensor", str(context.exception))

    def test_invalid_target_tokens_dimensions(self) -> None:
        """Test handling of invalid target tokens dimensions."""
        # Create a direct test by providing a malformed target_tokens that will create 2D tensor
        logits = torch.randn(2, 10)  # Valid 2D logits

        # This creates invalid data where target_tokens becomes a 2D tensor when loaded
        invalid_target_data = [
            [1, 2]
        ]  # This will become a 2D tensor [1, 2] when converted

        row = pd.Series(
            {
                "prediction_logits": logits.tolist(),
                "target_tokens": invalid_target_data,
            }
        )

        # Should raise an error due to invalid tensor dimensions
        with self.assertRaises(ValueError) as context:
            _compute_model_probability_from_logits(
                row, "prediction_logits", "target_tokens"
            )
        # The error could be about target tokens dimensions or sequence mismatch
        error_msg = str(context.exception)
        self.assertTrue(
            "Expected 1D target tokens tensor" in error_msg
            or "Sequence length mismatch" in error_msg
        )
