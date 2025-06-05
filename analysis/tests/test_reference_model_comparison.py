# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import unittest

import pandas as pd

from privacy_guard.analysis.reference_model_comparison_input import (
    ReferenceModelComparisonInput,
)
from privacy_guard.analysis.reference_model_comparison_node import (
    ReferenceModelComparisonNode,
    ReferenceModelComparisonNodeOutput,
)


class TestReferenceModelComparison(unittest.TestCase):
    def setUp(self) -> None:
        # Create test data for target and reference dataframes
        self.target_data = {
            "id": [1, 2, 3, 4],
            "text": ["Sample 1", "Sample 2", "Sample 3", "Sample 4"],
            "decision_prompt": [True, True, False, False],
        }

        self.reference_data = {
            "id": [1, 2, 3, 4],
            "text": ["Sample 1", "Sample 2", "Sample 3", "Sample 4"],
            "decision_prompt": [True, False, True, False],
        }

        self.target_df = pd.DataFrame(self.target_data)
        self.reference_df = pd.DataFrame(self.reference_data)

        self.analysis_input = ReferenceModelComparisonInput(
            target_df=self.target_df,
            reference_df=self.reference_df,
            result_key="decision_prompt",
        )
        self.analysis_node = ReferenceModelComparisonNode(
            analysis_input=self.analysis_input
        )

        super().setUp()

    def test_construct_reference_model_comparison_input(self) -> None:
        """Test that the input is constructed correctly."""
        self.assertEqual(self.analysis_input.target_df.equals(self.target_df), True)
        self.assertEqual(
            self.analysis_input.reference_df.equals(self.reference_df), True
        )
        self.assertEqual(self.analysis_input.result_key, "decision_prompt")

    def test_run_analysis(self) -> None:
        """Test that the analysis runs correctly and produces the expected output."""
        results = self.analysis_node.compute_outputs()

        # Check that the expected keys are in the results
        self.assertIn("num_samples", results)
        self.assertEqual(results["num_samples"], 4)

        self.assertIn("tgt_pos_ref_pos", results)
        self.assertIn("tgt_pos_ref_neg", results)
        self.assertIn("tgt_neg_ref_pos", results)
        self.assertIn("tgt_neg_ref_neg", results)
        self.assertIn("augmented_output_dataset", results)

        # Check that the comparison columns have the expected values
        expected_tgt_pos_ref_pos = pd.Series([True, False, False, False])
        expected_tgt_pos_ref_neg = pd.Series([False, True, False, False])
        expected_tgt_neg_ref_pos = pd.Series([False, False, True, False])
        expected_tgt_neg_ref_neg = pd.Series([False, False, False, True])

        self.assertEqual(
            results["tgt_pos_ref_pos"].tolist(), expected_tgt_pos_ref_pos.tolist()
        )
        self.assertEqual(
            results["tgt_pos_ref_neg"].tolist(),
            expected_tgt_pos_ref_neg.tolist(),
        )
        self.assertEqual(
            results["tgt_neg_ref_pos"].tolist(),
            expected_tgt_neg_ref_pos.tolist(),
        )
        self.assertEqual(
            results["tgt_neg_ref_neg"].tolist(), expected_tgt_neg_ref_neg.tolist()
        )

    def test_output_types(self) -> None:
        """Test that the output has the expected types."""
        analysis_outputs = self.analysis_node.run_analysis()
        self.assertIsInstance(analysis_outputs, ReferenceModelComparisonNodeOutput)

        analysis_outputs_dict = self.analysis_node.compute_outputs()

        self.assertIsInstance(analysis_outputs_dict, dict)
        self.assertIsInstance(analysis_outputs_dict["num_samples"], int)
        self.assertIsInstance(analysis_outputs_dict["tgt_pos_ref_pos"], pd.Series)
        self.assertIsInstance(analysis_outputs_dict["tgt_pos_ref_neg"], pd.Series)
        self.assertIsInstance(analysis_outputs_dict["tgt_neg_ref_pos"], pd.Series)
        self.assertIsInstance(analysis_outputs_dict["tgt_neg_ref_neg"], pd.Series)
        self.assertIsInstance(
            analysis_outputs_dict["augmented_output_dataset"], pd.DataFrame
        )

    def test_different_result_key(self) -> None:
        """Test that the analysis works with a different result_key."""
        # Create test data with a different result_key
        target_data = {
            "id": [1, 2, 3, 4],
            "text": ["Sample 1", "Sample 2", "Sample 3", "Sample 4"],
            "custom_key": [True, True, False, False],
        }

        reference_data = {
            "id": [1, 2, 3, 4],
            "text": ["Sample 1", "Sample 2", "Sample 3", "Sample 4"],
            "custom_key": [True, False, True, False],
        }

        target_df = pd.DataFrame(target_data)
        reference_df = pd.DataFrame(reference_data)

        analysis_input = ReferenceModelComparisonInput(
            target_df=target_df,
            reference_df=reference_df,
            result_key="custom_key",
        )
        analysis_node = ReferenceModelComparisonNode(analysis_input=analysis_input)

        results = analysis_node.compute_outputs()

        # Check that the comparison columns have the expected values
        expected_tgt_pos_ref_pos = pd.Series([True, False, False, False])
        expected_tgt_pos_ref_neg = pd.Series([False, True, False, False])
        expected_tgt_neg_ref_pos = pd.Series([False, False, True, False])
        expected_tgt_neg_ref_neg = pd.Series([False, False, False, True])

        self.assertEqual(
            results["tgt_pos_ref_pos"].tolist(), expected_tgt_pos_ref_pos.tolist()
        )
        self.assertEqual(
            results["tgt_pos_ref_neg"].tolist(),
            expected_tgt_pos_ref_neg.tolist(),
        )
        self.assertEqual(
            results["tgt_neg_ref_pos"].tolist(),
            expected_tgt_neg_ref_pos.tolist(),
        )
        self.assertEqual(
            results["tgt_neg_ref_neg"].tolist(), expected_tgt_neg_ref_neg.tolist()
        )
