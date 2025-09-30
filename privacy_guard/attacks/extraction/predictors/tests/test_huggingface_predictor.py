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
import unittest
from unittest.mock import MagicMock, patch

import torch
from privacy_guard.attacks.extraction.predictors.huggingface_predictor import (
    HuggingFacePredictor,
)


class TestHuggingFacePredictor(unittest.TestCase):
    def setUp(self) -> None:
        self.model_name = "test-model"
        self.device = "cpu"
        self.vocab_size = 50257

        # Create simple mocks for model and tokenizer
        self.mock_model = MagicMock(
            spec=["generate", "config"]
        )  # Only allow these attributes
        self.mock_model.config.vocab_size = self.vocab_size
        self.mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.pad_token = None
        self.mock_tokenizer.eos_token = "<|endoftext|>"
        self.mock_tokenizer.pad_token_id = 0
        self.mock_tokenizer.batch_decode.return_value = ["Generated text"]

    @patch(
        "privacy_guard.attacks.extraction.predictors.huggingface_predictor.load_model_and_tokenizer"
    )
    def test_init(self, mock_load_model_and_tokenizer: MagicMock) -> None:
        """Test predictor initialization."""
        mock_load_model_and_tokenizer.return_value = (
            self.mock_model,
            self.mock_tokenizer,
        )

        predictor = HuggingFacePredictor(self.model_name, self.device)

        self.assertEqual(predictor.model_name, self.model_name)
        self.assertEqual(predictor.device, self.device)
        mock_load_model_and_tokenizer.assert_called_once_with(
            self.model_name, self.device, model_kwargs={}, tokenizer_kwargs={}
        )

    @patch(
        "privacy_guard.attacks.extraction.predictors.huggingface_predictor.load_model_and_tokenizer"
    )
    def test_generate(self, mock_load_model_and_tokenizer: MagicMock) -> None:
        """Test generate functionality."""
        mock_load_model_and_tokenizer.return_value = (
            self.mock_model,
            self.mock_tokenizer,
        )

        # Mock tokenizer responses
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        self.mock_tokenizer.return_value = mock_inputs
        self.mock_tokenizer.batch_decode.return_value = ["Generated text"]

        predictor = HuggingFacePredictor(self.model_name, self.device)

        # Mock the tqdm within the generate method - patch the specific import
        with patch(
            "privacy_guard.attacks.extraction.predictors.huggingface_predictor.tqdm"
        ) as mock_tqdm:
            mock_tqdm.side_effect = lambda x, **kwargs: x
            result = predictor.generate(["Test prompt"])

        self.assertEqual(result, ["Generated text"])
        self.mock_model.generate.assert_called_once()

    @patch(
        "privacy_guard.attacks.extraction.predictors.huggingface_predictor.load_model_and_tokenizer"
    )
    def test_get_logits(self, mock_load_model_and_tokenizer: MagicMock) -> None:
        """Test get_logits functionality."""
        mock_load_model_and_tokenizer.return_value = (
            self.mock_model,
            self.mock_tokenizer,
        )

        # Mock model outputs
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.randn(1, 5, self.vocab_size)
        self.mock_model.return_value = mock_outputs

        # Create mock input tensors that behave properly with tensor operations
        mock_full_sequence_ids = torch.tensor([[1, 2, 3, 4, 5]])
        mock_prompt_ids = torch.tensor([[1, 2, 3]])
        mock_target_ids = torch.tensor([[4, 5]])

        # Mock tokenizer responses
        self.mock_tokenizer.side_effect = [
            # Full sequence tokenization
            MagicMock(
                input_ids=mock_full_sequence_ids,
                to=lambda device: MagicMock(
                    input_ids=mock_full_sequence_ids,
                    attention_mask=torch.tensor([[1, 1, 1, 1, 1]]),
                ),
            ),
            # Prompt tokenization
            MagicMock(
                input_ids=mock_prompt_ids,
                to=lambda device: MagicMock(
                    input_ids=mock_prompt_ids,
                    attention_mask=torch.tensor([[1, 1, 1]]),
                ),
            ),
            # Target tokenization
            MagicMock(
                input_ids=mock_target_ids,
                shape=[1, 2],
                to=lambda device: MagicMock(input_ids=mock_target_ids),
            ),
        ]

        predictor = HuggingFacePredictor(self.model_name, self.device)

        # Mock the tqdm within the get_logits method
        with patch(
            "privacy_guard.attacks.extraction.predictors.huggingface_predictor.tqdm"
        ) as mock_tqdm:
            mock_tqdm.side_effect = lambda x, **kwargs: x
            result = predictor.get_logits(["Hello"], [" world"])

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], torch.Tensor)
        # Shape checking: target has 2 tokens, so logits should be (2, vocab_size)
        self.assertEqual(result[0].shape, (2, self.vocab_size))

    @patch(
        "privacy_guard.attacks.extraction.predictors.huggingface_predictor.load_model_and_tokenizer"
    )
    def test_get_logprobs(self, mock_load_model_and_tokenizer: MagicMock) -> None:
        """Test get_logprobs functionality."""
        mock_load_model_and_tokenizer.return_value = (
            self.mock_model,
            self.mock_tokenizer,
        )

        # Mock model outputs
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.randn(1, 5, self.vocab_size)
        self.mock_model.return_value = mock_outputs

        # Create mock input tensors that behave properly with tensor operations
        mock_full_sequence_ids = torch.tensor([[1, 2, 3, 4, 5]])
        mock_prompt_ids = torch.tensor([[1, 2, 3]])
        mock_target_ids = torch.tensor([[4, 5]])

        # Mock tokenizer responses for get_logits call (used internally by get_logprobs)
        self.mock_tokenizer.side_effect = [
            # Full sequence tokenization (called by get_logits)
            MagicMock(
                input_ids=mock_full_sequence_ids,
                to=lambda device: MagicMock(
                    input_ids=mock_full_sequence_ids,
                    attention_mask=torch.tensor([[1, 1, 1, 1, 1]]),
                ),
            ),
            # Prompt tokenization (called by get_logits)
            MagicMock(
                input_ids=mock_prompt_ids,
                to=lambda device: MagicMock(
                    input_ids=mock_prompt_ids,
                    attention_mask=torch.tensor([[1, 1, 1]]),
                ),
            ),
            # Target tokenization (called by get_logits)
            MagicMock(
                input_ids=mock_target_ids,
                shape=[1, 2],
                to=lambda device: MagicMock(input_ids=mock_target_ids),
            ),
            # Target tokenization (called by get_logprobs directly)
            MagicMock(
                input_ids=mock_target_ids,
                to=lambda device: MagicMock(input_ids=mock_target_ids),
            ),
        ]

        predictor = HuggingFacePredictor(self.model_name, self.device)

        # Mock the tqdm within the get_logprobs method
        with patch(
            "privacy_guard.attacks.extraction.predictors.huggingface_predictor.tqdm"
        ) as mock_tqdm:
            mock_tqdm.side_effect = lambda x, **kwargs: x
            result = predictor.get_logprobs(["Hello"], [" world"])

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], torch.Tensor)
        # Shape checking: target has 2 tokens, so logprobs should be (2,)
        self.assertEqual(result[0].shape, (2,))


if __name__ == "__main__":
    unittest.main()
