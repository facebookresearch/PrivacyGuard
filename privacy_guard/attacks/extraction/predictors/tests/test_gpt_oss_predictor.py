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
from privacy_guard.attacks.extraction.predictors.gpt_oss_predictor import (
    GPTOSSPredictor,
)


class TestGPTOSSPredictor(unittest.TestCase):
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

        with patch.object(
            GPTOSSPredictor, "accelerate_available_workaround", return_value=True
        ), patch(
            "privacy_guard.attacks.extraction.predictors.huggingface_predictor.load_model_and_tokenizer",
            return_value=(
                self.mock_model,
                self.mock_tokenizer,
            ),
        ):
            self.predictor = GPTOSSPredictor(self.model_name, self.device)

    def test_init(self) -> None:
        """Test predictor initialization."""
        self.assertEqual(self.predictor.model_name, self.model_name)
        self.assertEqual(self.predictor.device, self.device)

    def test_generate(self) -> None:
        """Test generate functionality."""

        # Mock tokenizer responses
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        self.mock_tokenizer.return_value = mock_inputs
        self.mock_tokenizer.batch_decode.return_value = ["Generated text"]

        # Mock the tqdm within the generate method - patch the specific import
        with patch(
            "privacy_guard.attacks.extraction.predictors.huggingface_predictor.tqdm"
        ) as mock_tqdm:
            mock_tqdm.side_effect = lambda x, **kwargs: x
            result = self.predictor.generate(["Test prompt"])

        self.assertEqual(result, ["Generated text"])
        self.mock_model.generate.assert_called_once()

    def test_generate_with_kwargs(self) -> None:
        """Test generate functionality specifying add_generation_prompt
        and reasoning_effort"""

        # Mock tokenizer responses
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        self.mock_tokenizer.return_value = mock_inputs
        self.mock_tokenizer.batch_decode.return_value = ["Generated text"]

        # Mock the tqdm within the generate method - patch the specific import
        with patch(
            "privacy_guard.attacks.extraction.predictors.huggingface_predictor.tqdm"
        ) as mock_tqdm:
            mock_tqdm.side_effect = lambda x, **kwargs: x
            result = self.predictor.generate(
                ["Test prompt"],
                add_generation_prompt=True,
                reasoning_effort="medium",
            )

        self.assertEqual(result, ["Generated text"])
        self.mock_model.generate.assert_called_once()

    @patch(
        "privacy_guard.attacks.extraction.predictors.gpt_oss_predictor.is_accelerate_available"
    )
    def test_accelerate_available_workaround_when_initially_true(
        self, mock_is_accelerate_available: MagicMock
    ) -> None:
        """Test accelerate_available_workaround when is_accelerate_available is True initially."""

        # Setup: mock is_accelerate_available to return True
        mock_is_accelerate_available.return_value = True

        # Execute: call the workaround method
        # accelerate_available_workaround is called in __init__
        result = self.predictor.accelerate_available_workaround()

        # Assert: method returns True and only checks is_accelerate_available
        self.assertTrue(result)
        mock_is_accelerate_available.assert_called_once()

    @patch(
        "privacy_guard.attacks.extraction.predictors.gpt_oss_predictor._is_package_available"
    )
    @patch(
        "privacy_guard.attacks.extraction.predictors.gpt_oss_predictor.is_accelerate_available"
    )
    def test_accelerate_available_workaround_when_package_available(
        self,
        mock_is_accelerate_available: MagicMock,
        mock_is_package_available: MagicMock,
    ) -> None:
        """Test when is_accelerate_available is initially false but _is_package_available returns true."""

        # Setup: mock is_accelerate_available to return False initially, then True after workaround
        mock_is_accelerate_available.side_effect = [False, True]

        # Setup: mock _is_package_available to return True and a version string
        mock_is_package_available.return_value = (True, "0.21.0")

        # Execute: call the workaround method
        result = self.predictor.accelerate_available_workaround()

        # Assert: method returns True after setting the accelerate availability
        self.assertTrue(result)
        self.assertEqual(mock_is_accelerate_available.call_count, 2)
        mock_is_package_available.assert_called_once()
        # mock_import_utils._is_package_available.assert_called_once_with(
        #    "accelerate", return_version=True
        # )

    @patch(
        "privacy_guard.attacks.extraction.predictors.gpt_oss_predictor._is_package_available"
    )
    @patch(
        "privacy_guard.attacks.extraction.predictors.gpt_oss_predictor.is_accelerate_available"
    )
    def test_accelerate_available_workaround_when_both_false(
        self,
        mock_is_accelerate_available: MagicMock,
        mock_is_package_available: MagicMock,
    ) -> None:
        """Test when both is_accelerate_available and _is_package_available are false."""

        # Setup: mock is_accelerate_available to return False
        mock_is_accelerate_available.return_value = False

        # Setup: mock _is_package_available to return False
        mock_is_package_available.return_value = (False, "N/A")

        # Execute: call the workaround method
        result = self.predictor.accelerate_available_workaround()

        # Assert: method returns False
        self.assertFalse(result)
        mock_is_accelerate_available.assert_called_once()
        mock_is_package_available.assert_called_once()
        # mock_import_utils._is_package_available.assert_called_once_with(
        #    "accelerate", return_version=True
        # )

    def test_init_fails_when_accelerate_not_available(
        self,
    ) -> None:
        """Test that instantiating GPTOSSPredictor when accelerate is not available
        raises exception."""
        with self.assertRaises(ImportError):
            with patch.object(
                GPTOSSPredictor, "accelerate_available_workaround", return_value=False
            ):
                _ = GPTOSSPredictor(self.model_name, self.device)


if __name__ == "__main__":
    unittest.main()
