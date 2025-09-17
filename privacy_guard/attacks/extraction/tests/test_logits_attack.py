# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
import tempfile
import unittest
from typing import Any
from unittest.mock import MagicMock

import torch
from privacy_guard.attacks.extraction.logits_attack import LogitsAttack
from privacy_guard.attacks.extraction.predictors.base_predictor import BasePredictor


class MockPredictor(BasePredictor):
    """Mock predictor for testing logits attack."""

    def __init__(self, vocab_size: int = 50257) -> None:
        self.vocab_size = vocab_size

    def generate(self, prompts: list[str], **generation_kwargs: Any) -> list[str]:
        return [f"Generated text for: {prompt}" for prompt in prompts]

    def get_logits(
        self, prompts: list[str], targets: list[str], batch_size: int = 1
    ) -> list[torch.Tensor]:
        logits_list = []
        for target in targets:
            # Simulate target tokenization - assume each character is a token for simplicity
            target_length = len(
                target.split()
            )  # Use word count as proxy for token count
            # Create random logits tensor with shape (target_length, vocab_size)
            logits = torch.randn(target_length, self.vocab_size)
            logits_list.append(logits)
        return logits_list

    def get_logprobs(
        self, prompts: list[str], targets: list[str], **generation_kwargs: Any
    ) -> list[torch.Tensor]:
        logprobs_list = []
        for target in targets:
            target_length = len(target.split())
            # Create random log probabilities with shape (target_length,)
            logprobs = torch.randn(target_length)
            logprobs_list.append(logprobs)
        return logprobs_list


class TestLogitsAttack(unittest.TestCase):
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

    def test_logits_attack_initialization(self) -> None:
        logits_attack = LogitsAttack(
            input_file=self.input_file_name,
            output_file=None,
            predictor=self.mock_predictor,
            batch_size=2,
            temp=0.8,
            top_k=40,
        )

        self.assertEqual(logits_attack.input_file, self.input_file_name)
        self.assertEqual(logits_attack.output_file, None)
        self.assertEqual(logits_attack.batch_size, 2)
        self.assertEqual(logits_attack.generation_kwargs["temp"], 0.8)
        self.assertEqual(logits_attack.generation_kwargs["top_k"], 40)
        self.assertEqual(len(logits_attack.input_df), 2)

    def test_logits_attack_run(self) -> None:
        logits_attack = LogitsAttack(
            input_file=self.input_file_name,
            output_file=None,
            predictor=self.mock_predictor,
            batch_size=2,
            temp=0.7,
            top_k=30,
            prob_threshold=0.3,
        )

        result = logits_attack.run_attack()

        # Check that the analysis input object is created correctly
        self.assertIsNotNone(result)
        self.assertEqual(result.generation_kwargs["temp"], 0.7)
        self.assertEqual(result.generation_kwargs["top_k"], 30)
        self.assertEqual(result.prob_threshold, 0.3)

        # Check that logits were added to the dataframe
        df = result.generation_df
        self.assertIn("prediction_logits", df.columns)
        self.assertEqual(len(df), 2)

        # Check that the default logits column name was passed to the analysis input
        self.assertEqual(result.logits_column, "prediction_logits")

        # Check logits list shapes (now stored as nested lists)
        logits_list = df["prediction_logits"].tolist()
        self.assertEqual(len(logits_list), 2)

        # First target: "Sample text one" -> 3 tokens -> shape (3, vocab_size)
        self.assertIsInstance(logits_list[0], list)
        self.assertEqual(len(logits_list[0]), 3)  # 3 tokens
        self.assertEqual(
            len(logits_list[0][0]), self.vocab_size
        )  # vocab_size logits per token

        # Second target: "Sample text two" -> 3 tokens -> shape (3, vocab_size)
        self.assertIsInstance(logits_list[1], list)
        self.assertEqual(len(logits_list[1]), 3)  # 3 tokens
        self.assertEqual(
            len(logits_list[1][0]), self.vocab_size
        )  # vocab_size logits per token

    def test_logits_attack_custom_columns(self) -> None:
        # Create input file with custom column names
        custom_input_file = tempfile.NamedTemporaryFile(suffix=".jsonl")
        with open(custom_input_file.name, "w") as f:
            f.write(
                '{"id": 1, "custom_target": "Hello world", "custom_prompt": "Say hello"}\n'
            )

        logits_attack = LogitsAttack(
            input_file=custom_input_file.name,
            output_file=None,
            predictor=self.mock_predictor,
            prompt_column="custom_prompt",
            target_column="custom_target",
            output_column="custom_logits",
        )

        result = logits_attack.run_attack()

        # Check that custom columns were used
        df = result.generation_df
        self.assertIn("custom_logits", df.columns)
        self.assertEqual(len(df), 1)

        # Check that the custom logits column name was passed to the analysis input
        self.assertEqual(result.logits_column, "custom_logits")

        # Check logits list shape for "Hello world" -> 2 tokens
        logits_list = df["custom_logits"].tolist()
        self.assertIsInstance(logits_list[0], list)
        self.assertEqual(len(logits_list[0]), 2)  # 2 tokens
        self.assertEqual(
            len(logits_list[0][0]), self.vocab_size
        )  # vocab_size logits per token

        custom_input_file.close()

    def test_logits_attack_batch_size_passed_to_predictor(self) -> None:
        # Create a mock predictor to track calls
        mock_predictor = MagicMock(spec=BasePredictor)
        mock_predictor.get_logits.return_value = [
            torch.randn(3, self.vocab_size),  # "Sample text one"
            torch.randn(3, self.vocab_size),  # "Sample text two"
        ]

        logits_attack = LogitsAttack(
            input_file=self.input_file_name,
            output_file=None,
            predictor=mock_predictor,
            batch_size=4,
        )

        logits_attack.run_attack()

        # Verify get_logits was called with correct batch_size
        mock_predictor.get_logits.assert_called_once()
        call_args = mock_predictor.get_logits.call_args
        self.assertEqual(call_args.kwargs["batch_size"], 4)

        # Verify prompts and targets were passed correctly
        self.assertEqual(call_args.kwargs["prompts"], ["prompt 1", "prompt 2"])
        self.assertEqual(
            call_args.kwargs["targets"], ["Sample text one", "Sample text two"]
        )

    def test_logits_attack_output_file_creation(self) -> None:
        # Create a mock predictor that returns tensors (which will be converted to lists by LogitsAttack)
        mock_predictor = MagicMock(spec=BasePredictor)

        # Return tensors that will be converted to lists by the LogitsAttack code
        mock_logits_as_tensors = [
            torch.tensor(
                [[0.1] * self.vocab_size] * 2
            ),  # 2 tokens, each with vocab_size logits
            torch.tensor(
                [[0.2] * self.vocab_size] * 3
            ),  # 3 tokens, each with vocab_size logits
        ]
        mock_predictor.get_logits.return_value = mock_logits_as_tensors

        logits_attack = LogitsAttack(
            input_file=self.input_file_name,
            output_file=self.output_file_name,
            predictor=mock_predictor,
            batch_size=2,
            temp=0.8,
            top_k=25,
            prob_threshold=0.4,
        )

        result = logits_attack.run_attack()

        # Check that the analysis input object is created correctly
        self.assertIsNotNone(result)
        self.assertEqual(result.generation_kwargs["temp"], 0.8)
        self.assertEqual(result.generation_kwargs["top_k"], 25)
        self.assertEqual(result.prob_threshold, 0.4)

        # Check that output file was created and has content
        with open(self.output_file_name, "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)  # Should have 2 rows of data

        # Verify the data structure in the result dataframe (tensors converted to lists)
        df = result.generation_df
        self.assertIn("prediction_logits", df.columns)
        logits_list = df["prediction_logits"].tolist()

        # Verify that the tensors were properly converted to lists
        self.assertEqual(len(logits_list), 2)
        expected_lists = [tensor.tolist() for tensor in mock_logits_as_tensors]
        self.assertEqual(logits_list[0], expected_lists[0])  # First sample
        self.assertEqual(logits_list[1], expected_lists[1])  # Second sample

        # Verify that the default logits column name was passed to the analysis input
        self.assertEqual(result.logits_column, "prediction_logits")

        # Verify get_logits was called correctly
        mock_predictor.get_logits.assert_called_once()


if __name__ == "__main__":
    unittest.main()
