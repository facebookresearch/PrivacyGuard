# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import os
import tempfile

from typing import Callable

from unittest.mock import Mock, patch

import pandas as pd
from later.unittest import TestCase
from privacy_guard.analysis.text_inclusion_analysis_input import (
    TextInclusionAnalysisInput,
)

from privacy_guard.attacks.text_inclusion_attack import (
    TextInclusionAttack,
    TextInclusionAttackBatch,
)


class TestTextInclusionAttack(TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_exactly_one_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one of"):
            _ = TextInclusionAttack(llm_generation_file="test", data=pd.DataFrame())
        with self.assertRaisesRegex(ValueError, "exactly one of"):
            _ = TextInclusionAttack(llm_generation_file=None, data=None)

        _ = TextInclusionAttack(llm_generation_file=None, data=pd.DataFrame())

    def test_sft_mode_assertion_not_user(self) -> None:
        sft_data_no_user = {
            "prompt": [
                {
                    "type": "SampleSFT",
                    "dialog": [{"body": "This is a test prompt", "source": "user"}],
                },
                {
                    "type": "SampleSFT",
                    "dialog": [
                        {"body": "This is another test prompt", "source": "agent"}
                    ],
                },
            ],
            "targets": [
                ["Target text 1"],
                ["Target text 2", "Another target"],
            ],
            "prediction": [
                "A success: Target text 1",
                "Failure, no match. ",
            ],
        }
        attack_node = TextInclusionAttack(data=pd.DataFrame(sft_data_no_user))

        with self.assertRaises(ValueError):
            _ = attack_node.preprocess_data()

    def test_sft_mode_assertion_multi_turn(self) -> None:
        sft_data_multi_turn = {
            "prompt": [
                {
                    "type": "SampleSFT",
                    "dialog": [
                        {"body": "This is a test prompt", "source": "user"},
                        {"body": "This a second turn.", "source": "agent"},
                    ],
                },
                {
                    "type": "SampleSFT",
                    "dialog": [
                        {"body": "This is another test prompt", "source": "user"}
                    ],
                },
            ],
            "targets": [
                ["Target text 1"],
                ["Target text 2", "Another target"],
            ],
            "prediction": [
                "A success: Target text 1",
                "Failure, no match. ",
            ],
        }

        attack_node = TextInclusionAttack(data=pd.DataFrame(sft_data_multi_turn))

        with self.assertRaises(NotImplementedError):
            _ = attack_node.preprocess_data()


class TestTextInclusionAttackBatch(TestCase):
    def setUp(self) -> None:
        self.temp_dir_file = tempfile.TemporaryDirectory()
        self.temp_dir = self.temp_dir_file.name
        self.temp_dir_subdir = os.path.join(self.temp_dir, "subdir")
        os.mkdir(self.temp_dir_subdir)

        self.text_inclusion_attack_batch: TextInclusionAttackBatch = (
            TextInclusionAttackBatch(
                dump_dir_str=self.temp_dir,
                num_rows=15,
            )
        )
        # Create 3 JSONL files in the temporary directory
        file1_path = f"{self.temp_dir}/text_inclusion_test_10.jsonl"
        file2_path = f"{self.temp_dir}/text_inclusion_prod_5K.jsonl"
        file3_path = f"{self.temp_dir_subdir}/text_inclusion_prod_subdir.jsonl"

        # Write some sample data to the files
        with open(file1_path, "w") as f1:
            f1.write('{"id": 1, "text": "Sample text 1"}\n')
            f1.write('{"id": 2, "text": "Sample text 2"}\n')

        with open(file2_path, "w") as f2:
            f2.write('{"id": 3, "text": "Sample text 3"}\n')
            f2.write('{"id": 4, "text": "Sample text 4"}\n')

        with open(file3_path, "w") as f2:
            f2.write('{"id": 3, "text": "Sample text 5"}\n')
            f2.write('{"id": 4, "text": "Sample text 6"}\n')

        super().setUp()

    def get_mock_run_attack_equality(
        self,
        comparison_attack_batch: TextInclusionAttackBatch,
    ) -> Callable[[TextInclusionAttack], TextInclusionAnalysisInput]:
        def mock_run_attack(
            attack: TextInclusionAttack,
        ) -> TextInclusionAnalysisInput:
            self.assertEqual(attack.bound_lcs, comparison_attack_batch.bound_lcs)
            self.assertEqual(attack.num_rows, comparison_attack_batch.num_rows)
            return TextInclusionAnalysisInput(generation_df=pd.DataFrame())

        return mock_run_attack

    def test_load_results_from_mount(self) -> None:
        """
        Loads multiple results from the temp dir.
        Ensures resulting input has length equal to
        directory.
        """

        with patch.object(
            TextInclusionAttack,
            "run_attack",
            self.get_mock_run_attack_equality(
                comparison_attack_batch=self.text_inclusion_attack_batch
            ),
        ):
            result_batch = self.text_inclusion_attack_batch.load_results_from_mnt()

        # Verify that the result has 3 items (one for each file)
        self.assertEqual(len(result_batch.input_batch), 3)
        print(f"Input batch keys: {result_batch.input_batch.keys()}")
        self.assertTrue(
            "text_inclusion_test_10.jsonl" in result_batch.input_batch.keys()
        )
        self.assertTrue(
            "text_inclusion_prod_5K.jsonl" in result_batch.input_batch.keys()
        )
        # replaces / with . in key
        self.assertTrue(
            "subdir.text_inclusion_prod_subdir.jsonl" in result_batch.input_batch.keys()
        )

    @patch.object(TextInclusionAttack, "run_attack")
    def test_load_results_from_mount_with_filter(self, mock_run_attack: Mock) -> None:
        """
        Only loads result which matches the result_name_filter.
        """

        text_inclusion_attack_batch_filter: TextInclusionAttackBatch = (
            TextInclusionAttackBatch(
                dump_dir_str=self.temp_dir,
                result_name_filter="prod",
                bound_lcs=True,
            )
        )

        mock_run_attack.return_value = TextInclusionAnalysisInput(
            generation_df=pd.DataFrame()
        )

        with patch.object(TextInclusionAttack, "run_attack", mock_run_attack):
            result_batch = text_inclusion_attack_batch_filter.load_results_from_mnt()

        mock_run_attack.assert_called()
        # Verify that the result has 2 items (one for each file)
        self.assertEqual(len(result_batch.input_batch), 2)
        self.assertFalse(
            "text_inclusion_test_10.jsonl" in result_batch.input_batch.keys()
        )
        self.assertTrue(
            "text_inclusion_prod_5K.jsonl" in result_batch.input_batch.keys()
        )
        self.assertTrue(
            "subdir.text_inclusion_prod_subdir.jsonl" in result_batch.input_batch.keys()
        )

    @patch.object(TextInclusionAttack, "run_attack")
    def test_load_results_from_mount_with_filter_no_match(
        self, mock_run_attack: Mock
    ) -> None:
        """
        Throws error when no files match the result name fitler.
        """

        text_inclusion_attack_batch_no_match: TextInclusionAttackBatch = (
            TextInclusionAttackBatch(
                dump_dir_str=self.temp_dir,
                result_name_filter="none_match_this_filter",
                bound_lcs=True,
            )
        )

        with self.assertRaisesRegex(ValueError, "No analysis results found in"):
            _ = text_inclusion_attack_batch_no_match.load_results_from_mnt()

        mock_run_attack.assert_not_called()

    def test_attack_inputs_propogated_to_analysis_input(self) -> None:
        """
        Ensures that attack arguments like "bound_lcs",
        "num_rows", etc are forwarded to analysis inputs.
        """

        with patch.object(
            TextInclusionAttack,
            "run_attack",
            self.get_mock_run_attack_equality(
                comparison_attack_batch=self.text_inclusion_attack_batch
            ),
        ):
            _ = self.text_inclusion_attack_batch.load_results_from_mnt()

    def test_attack_inputs_propogated_to_analysis_input_test_only(self) -> None:
        """
        Ensures that attack arguments like "bound_lcs",
        "num_rows", etc are forwarded to analysis inputs when filter is used.
        """
        attack_batch_test_only: TextInclusionAttackBatch = TextInclusionAttackBatch(
            dump_dir_str=self.temp_dir,
            result_name_filter="test",
            bound_lcs=True,
        )

        with patch.object(
            TextInclusionAttack,
            "run_attack",
            self.get_mock_run_attack_equality(
                comparison_attack_batch=attack_batch_test_only
            ),
        ):
            _ = attack_batch_test_only.load_results_from_mnt()
