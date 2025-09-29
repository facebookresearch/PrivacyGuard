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
import tempfile
import unittest
from typing import Any
from unittest.mock import MagicMock

import torch
from privacy_guard.attacks.extraction.logprobs_attack import LogprobsAttack
from privacy_guard.attacks.extraction.predictors.base_predictor import BasePredictor


class MockPredictor(BasePredictor):
    """Mock predictor for testing logprobs attack."""

    def __init__(self, vocab_size: int = 50257) -> None:
        self.vocab_size = vocab_size

    def generate(self, prompts: list[str], **generation_kwargs: Any) -> list[str]:
        return [f"Generated text for: {prompt}" for prompt in prompts]

    def get_logits(
        self, prompts: list[str], targets: list[str], batch_size: int = 1
    ) -> list[torch.Tensor]:
        logits_list = []
        for target in targets:
            target_length = len(target.split())
            logits = torch.randn(target_length, self.vocab_size)
            logits_list.append(logits)
        return logits_list

    def get_logprobs(
        self, prompts: list[str], targets: list[str], **generation_kwargs: Any
    ) -> list[torch.Tensor]:
        logprobs_list = []
        for target in targets:
            target_length = len(target.split())
            logprobs = torch.randn(target_length)
            logprobs_list.append(logprobs)
        return logprobs_list


class TestLogprobsAttack(unittest.TestCase):
    def setUp(self) -> None:
        self.vocab_size = 50257
        self.input_file = tempfile.NamedTemporaryFile(suffix=".jsonl")
        self.input_file_name = self.input_file.name

        with open(self.input_file_name, "w") as f:
            f.write('{"id": 1, "target": "Sample text one", "prompt": "prompt 1"}\n')
            f.write('{"id": 2, "target": "Sample text two", "prompt": "prompt 2"}\n')

        self.output_file = tempfile.NamedTemporaryFile(suffix=".jsonl")
        self.output_file_name = self.output_file.name

        # Create mock predictor
        self.mock_predictor = MockPredictor(vocab_size=self.vocab_size)

    def test_logprobs_attack_initialization(self) -> None:
        logprobs_attack = LogprobsAttack(
            input_file=self.input_file_name,
            output_file=None,
            predictor=self.mock_predictor,
            batch_size=2,
            temperature=0.8,
            top_k=40,
        )

        self.assertEqual(logprobs_attack.input_file, self.input_file_name)
        self.assertEqual(logprobs_attack.output_file, None)
        self.assertEqual(logprobs_attack.batch_size, 2)
        self.assertEqual(logprobs_attack.generation_kwargs["temperature"], 0.8)
        self.assertEqual(logprobs_attack.generation_kwargs["top_k"], 40)
        self.assertEqual(len(logprobs_attack.input_df), 2)

    def test_logprobs_attack_run(self) -> None:
        logprobs_attack = LogprobsAttack(
            input_file=self.input_file_name,
            output_file=None,
            predictor=self.mock_predictor,
            batch_size=2,
            temperature=0.7,
            top_k=30,
            prob_threshold=0.3,
        )

        result = logprobs_attack.run_attack()

        # Check that the analysis input object is created correctly
        self.assertIsNotNone(result)
        self.assertEqual(result.prob_threshold, 0.3)

        # Check that logprobs were added to the dataframe
        df = result.generation_df
        self.assertIn("prediction_logprobs", df.columns)
        self.assertEqual(len(df), 2)

        # Check that the default logprobs column name was passed to the analysis input
        self.assertEqual(result.logprobs_column, "prediction_logprobs")

        # Check logprobs list shapes (now stored as nested lists)
        logprobs_list = df["prediction_logprobs"].tolist()
        self.assertEqual(len(logprobs_list), 2)

        # First target: "Sample text one" -> 3 tokens -> shape (3,)
        self.assertIsInstance(logprobs_list[0], list)
        self.assertEqual(len(logprobs_list[0]), 3)  # 3 tokens

        # Second target: "Sample text two" -> 3 tokens -> shape (3,)
        self.assertIsInstance(logprobs_list[1], list)
        self.assertEqual(len(logprobs_list[1]), 3)  # 3 tokens

    def test_logprobs_attack_custom_columns(self) -> None:
        # Create input file with custom column names
        custom_input_file = tempfile.NamedTemporaryFile(suffix=".jsonl")
        with open(custom_input_file.name, "w") as f:
            f.write(
                '{"id": 1, "custom_target": "Hello world", "custom_prompt": "Say hello"}\n'
            )

        logprobs_attack = LogprobsAttack(
            input_file=custom_input_file.name,
            output_file=None,
            predictor=self.mock_predictor,
            prompt_column="custom_prompt",
            target_column="custom_target",
            output_column="custom_logprobs",
        )

        result = logprobs_attack.run_attack()

        # Check that custom columns were used
        df = result.generation_df
        self.assertIn("custom_logprobs", df.columns)
        self.assertEqual(len(df), 1)

        # Check that the custom logprobs column name was passed to the analysis input
        self.assertEqual(result.logprobs_column, "custom_logprobs")

        # Check logprobs list shape for "Hello world" -> 2 tokens
        logprobs_list = df["custom_logprobs"].tolist()
        self.assertIsInstance(logprobs_list[0], list)
        self.assertEqual(len(logprobs_list[0]), 2)  # 2 tokens

        custom_input_file.close()

    def test_logprobs_attack_batch_size_passed_to_predictor(self) -> None:
        # Create a mock predictor to track calls
        mock_predictor = MagicMock(spec=BasePredictor)
        mock_predictor.get_logprobs.return_value = [
            torch.randn(3),  # "Sample text one"
            torch.randn(3),  # "Sample text two"
        ]

        logprobs_attack = LogprobsAttack(
            input_file=self.input_file_name,
            output_file=None,
            predictor=mock_predictor,
            batch_size=4,
        )

        logprobs_attack.run_attack()

        # Verify get_logprobs was called with correct batch_size
        mock_predictor.get_logprobs.assert_called_once()
        call_args = mock_predictor.get_logprobs.call_args
        self.assertEqual(call_args.kwargs["batch_size"], 4)

        # Verify prompts and targets were passed correctly
        self.assertEqual(call_args.kwargs["prompts"], ["prompt 1", "prompt 2"])
        self.assertEqual(
            call_args.kwargs["targets"], ["Sample text one", "Sample text two"]
        )

    def test_logprobs_attack_output_file_creation(self) -> None:
        # Create a mock predictor that returns tensors (which will be converted to lists by LogprobsAttack)
        mock_predictor = MagicMock(spec=BasePredictor)

        # Return tensors that will be converted to lists by the LogprobsAttack code
        mock_logprobs_as_tensors = [
            torch.tensor([0.1, 0.2]),  # 2 token probabilities
            torch.tensor([0.3, 0.4, 0.5]),  # 3 token probabilities
        ]
        mock_predictor.get_logprobs.return_value = mock_logprobs_as_tensors

        logprobs_attack = LogprobsAttack(
            input_file=self.input_file_name,
            output_file=self.output_file_name,
            predictor=mock_predictor,
            batch_size=2,
            temperature=0.8,
            top_k=25,
            prob_threshold=0.4,
        )

        result = logprobs_attack.run_attack()

        # Check that the analysis input object is created correctly
        self.assertIsNotNone(result)
        self.assertEqual(result.prob_threshold, 0.4)

        # Check that output file was created and has content
        with open(self.output_file_name, "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)  # Should have 2 rows of data

        # Verify the data structure in the result dataframe (tensors converted to lists)
        df = result.generation_df
        self.assertIn("prediction_logprobs", df.columns)
        logprobs_list = df["prediction_logprobs"].tolist()

        # Verify that the tensors were properly converted to lists
        self.assertEqual(len(logprobs_list), 2)
        expected_lists = [tensor.tolist() for tensor in mock_logprobs_as_tensors]
        self.assertEqual(logprobs_list[0], expected_lists[0])  # First sample
        self.assertEqual(logprobs_list[1], expected_lists[1])  # Second sample

        # Verify get_logprobs was called correctly
        mock_predictor.get_logprobs.assert_called_once()


if __name__ == "__main__":
    unittest.main()
