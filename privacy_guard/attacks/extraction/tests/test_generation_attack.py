# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-strict
import os
import tempfile
import unittest

from unittest.mock import MagicMock

import transformers
from packaging.version import Version
from privacy_guard.attacks.extraction.generation_attack import GenerationAttack


class TestGenerationAttack(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test data and mocks."""
        self.input_file = tempfile.NamedTemporaryFile(suffix=".jsonl")
        self.input_file_name = self.input_file.name

        with open(self.input_file_name, "w") as f:
            f.write('{"id": 1, "target": "Sample text 1", "prompt": "prompt 1"}\n')
            f.write('{"id": 2, "target": "Sample text 2", "prompt": "prompt 2"}\n')

        self.output_file = tempfile.NamedTemporaryFile(suffix=".jsonl")
        self.output_file_name = self.output_file.name

        # Mock predictor
        self.mock_predictor = MagicMock()
        self.mock_predictor.generate.return_value = [
            "generated text 1",
            "generated text 2",
        ]

    def test_generation_attack_no_output_file(self) -> None:
        """Test generation attack without output file."""
        generation_attack = GenerationAttack(
            input_file=self.input_file_name,
            output_file=None,
            predictor=self.mock_predictor,
            temperature=1,
            top_k=40,
        )

        result = generation_attack.run_attack()

        # Verify predictor was called correctly
        self.mock_predictor.generate.assert_called_once_with(
            prompts=["prompt 1", "prompt 2"], temperature=1, top_k=40
        )

        # Verify result structure
        self.assertIsNotNone(result)
        self.assertEqual(len(result.generation_df), 2)
        self.assertIn("prediction", result.generation_df.columns)
        self.assertEqual(
            result.generation_df["prediction"].tolist(),
            ["generated text 1", "generated text 2"],
        )

    def test_generation_attack_with_output_file(self) -> None:
        """Test generation attack with output file."""
        generation_attack = GenerationAttack(
            input_file=self.input_file_name,
            output_file=self.output_file_name,
            predictor=self.mock_predictor,
            temperature=1,
            top_k=40,
        )

        result = generation_attack.run_attack()

        # Verify predictor was called correctly
        self.mock_predictor.generate.assert_called_once_with(
            prompts=["prompt 1", "prompt 2"], temperature=1, top_k=40
        )

        # Verify output file was created
        self.assertTrue(os.path.getsize(self.output_file_name) > 0)

        # Verify result structure
        self.assertIsNotNone(result)
        self.assertEqual(len(result.generation_df), 2)
        self.assertIn("prediction", result.generation_df.columns)

    def test_generation_attack_custom_columns(self) -> None:
        """Test generation attack with custom column names."""
        # Create input file with custom column name
        custom_input_file = tempfile.NamedTemporaryFile(suffix=".jsonl")
        with open(custom_input_file.name, "w") as f:
            f.write(
                '{"id": 1, "custom_prompt": "test prompt 1", "custom_target": "target text 1"}\n'
            )
            f.write(
                '{"id": 2, "custom_prompt": "test prompt 2", "custom_target": "target text 2"}\n'
            )

        generation_attack = GenerationAttack(
            input_file=custom_input_file.name,
            output_file=None,
            predictor=self.mock_predictor,
            input_column="custom_prompt",
            target_column="custom_target",
            output_column="custom_output",
        )

        result = generation_attack.run_attack()

        # Verify predictor was called with correct prompts
        self.mock_predictor.generate.assert_called_once_with(
            prompts=["test prompt 1", "test prompt 2"]
        )

        # Verify custom column names
        self.assertIn("custom_output", result.generation_df.columns)
        self.assertEqual(result.target_key, "custom_target")
        self.assertEqual(result.prompt_key, "custom_prompt")
        self.assertEqual(result.generation_key, "custom_output")
        custom_input_file.close()

    def test_generation_attack_missing_column(self) -> None:
        """Test generation attack with missing required column."""
        # Create input file without prompt column
        bad_input_file = tempfile.NamedTemporaryFile(suffix=".jsonl")
        with open(bad_input_file.name, "w") as f:
            f.write('{"id": 1, "other_column": "value 1"}\n')

        with self.assertRaises(ValueError) as context:
            GenerationAttack(
                input_file=bad_input_file.name,
                output_file=None,
                predictor=self.mock_predictor,
            )

        self.assertIn("Missing required columns", str(context.exception))
        bad_input_file.close()

    def test_transformers_version_in_generation_attack(self) -> None:
        """Verify that transformers version is greater than or equal to 4.55.0"""
        current_version = transformers.__version__
        required_version = "4.55.0"

        self.assertGreaterEqual(
            Version(current_version),
            Version(required_version),
            f"Transformers version {current_version} must be greater than or equal to 4.55.0",
        )

    def tearDown(self) -> None:
        """Clean up temporary files."""
        self.input_file.close()
        self.output_file.close()
