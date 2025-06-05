# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
import tempfile
import unittest

import pandas as pd

from privacy_guard.analysis.scripts.reference_model_comparison import (
    dump_augmented_df,
    run_comparison_analysis,
)


class TestReferenceModelComparisonScript(unittest.TestCase):
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

        # Create temporary files for input and output
        self.temp_target_file = tempfile.NamedTemporaryFile(
            prefix="test_target", suffix=".jsonl", mode="w"
        )
        self.temp_target_file_name = self.temp_target_file.name
        self.temp_target_file.write(
            pd.DataFrame(self.target_data).to_json(orient="records", lines=True)
        )
        self.temp_target_file.flush()
        self.temp_target_file.seek(0)

        self.temp_reference_file = tempfile.NamedTemporaryFile(
            prefix="test_reference", suffix=".jsonl", mode="w"
        )
        self.temp_reference_file_name = self.temp_reference_file.name
        self.temp_reference_file.write(
            pd.DataFrame(self.reference_data).to_json(orient="records", lines=True)
        )
        self.temp_reference_file.flush()
        self.temp_reference_file.seek(0)

        self.temp_output_file = tempfile.NamedTemporaryFile(
            prefix="test_output", suffix=".jsonl", mode="w"
        )
        self.temp_output_file_name = self.temp_output_file.name

        super().setUp()

    def test_run_comparison_analysis(self) -> None:
        """Test that the comparison analysis runs correctly."""
        result_df = run_comparison_analysis(
            target_df_path=self.temp_target_file_name,
            reference_df_path=self.temp_reference_file_name,
            output_path=self.temp_output_file_name,
            result_key="decision_prompt",
        )

        # Check that the result dataframe has the expected columns
        self.assertIn("tgt_pos_ref_pos", result_df.columns)
        self.assertIn("tgt_pos_ref_neg", result_df.columns)
        self.assertIn("tgt_neg_ref_pos", result_df.columns)
        self.assertIn("tgt_neg_ref_neg", result_df.columns)

        # Check that the comparison columns have the expected values
        self.assertEqual(
            result_df["tgt_pos_ref_pos"].tolist(), [True, False, False, False]
        )
        self.assertEqual(
            result_df["tgt_pos_ref_neg"].tolist(), [False, True, False, False]
        )
        self.assertEqual(
            result_df["tgt_neg_ref_pos"].tolist(), [False, False, True, False]
        )
        self.assertEqual(
            result_df["tgt_neg_ref_neg"].tolist(), [False, False, False, True]
        )

    def test_run_comparison_analysis_custom_key(self) -> None:
        """Test that the comparison analysis works with a custom result_key."""
        # Create test data with a custom result_key
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

        # Create temporary files for input and output
        temp_target_file = tempfile.NamedTemporaryFile(
            prefix="test_target_custom", suffix=".jsonl", mode="w"
        )
        temp_target_file_name = temp_target_file.name
        temp_target_file.write(
            pd.DataFrame(target_data).to_json(orient="records", lines=True)
        )
        temp_target_file.flush()
        temp_target_file.seek(0)

        temp_reference_file = tempfile.NamedTemporaryFile(
            prefix="test_reference_custom", suffix=".jsonl", mode="w"
        )
        temp_reference_file_name = temp_reference_file.name
        temp_reference_file.write(
            pd.DataFrame(reference_data).to_json(orient="records", lines=True)
        )
        temp_reference_file.flush()
        temp_reference_file.seek(0)

        temp_output_file = tempfile.NamedTemporaryFile(
            prefix="test_output_custom", suffix=".jsonl", mode="w"
        )
        temp_output_file_name = temp_output_file.name

        result_df = run_comparison_analysis(
            target_df_path=temp_target_file_name,
            reference_df_path=temp_reference_file_name,
            output_path=temp_output_file_name,
            result_key="custom_key",
        )

        # Check that the result dataframe has the expected columns
        self.assertIn("tgt_pos_ref_pos", result_df.columns)
        self.assertIn("tgt_pos_ref_neg", result_df.columns)
        self.assertIn("tgt_neg_ref_pos", result_df.columns)
        self.assertIn("tgt_neg_ref_neg", result_df.columns)

        # Check that the comparison columns have the expected values
        self.assertEqual(
            result_df["tgt_pos_ref_pos"].tolist(), [True, False, False, False]
        )
        self.assertEqual(
            result_df["tgt_pos_ref_neg"].tolist(), [False, True, False, False]
        )
        self.assertEqual(
            result_df["tgt_neg_ref_pos"].tolist(), [False, False, True, False]
        )
        self.assertEqual(
            result_df["tgt_neg_ref_neg"].tolist(), [False, False, False, True]
        )

    def test_dump_augmented_df(self) -> None:
        """Test that the dump_augmented_df function works correctly."""
        # Create a test dataframe
        test_df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "text": ["Sample 1", "Sample 2", "Sample 3", "Sample 4"],
                "decision_prompt": [True, True, False, False],
                "tgt_pos_ref_pos": [True, False, False, False],
                "tgt_pos_ref_neg": [False, True, False, False],
                "tgt_neg_ref_pos": [False, False, True, False],
                "tgt_neg_ref_neg": [False, False, False, True],
            }
        )

        # Dump the dataframe to a file
        dump_augmented_df(df=test_df, jsonl_output_path=self.temp_output_file_name)

        # Read the file back and check that it matches the original dataframe
        read_df = pd.read_json(self.temp_output_file_name, lines=True)

        # Check that the dataframes have the same columns
        self.assertEqual(set(test_df.columns), set(read_df.columns))

        # Check that the dataframes have the same values
        for col in test_df.columns:
            self.assertEqual(test_df[col].tolist(), read_df[col].tolist())
