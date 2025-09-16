# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
import os
import tempfile
import unittest

from unittest.mock import ANY, MagicMock, patch

import pandas as pd
from privacy_guard.attacks.extraction.generation_attack import GenerationAttack


class TestGenerationAttack(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test data and mocks."""
        self.generation_kwargs = {"temperature": 1, "top_k": 40}
        self.input_file = tempfile.NamedTemporaryFile(suffix=".jsonl")
        self.input_file_name = self.input_file.name

        with open(self.input_file_name, "w") as f:
            f.write('{"id": 1, "target": "Sample text 1", "prompt": "prompt 1"}\n')
            f.write('{"id": 2, "target": "Sample text 2", "prompt": "prompt 1"}\n')

        self.output_file = tempfile.NamedTemporaryFile(suffix=".jsonl")
        self.output_file_name = self.output_file.name

        # Mock model outputs
        self.mock_model_outputs = pd.DataFrame(
            [
                {
                    "id": 1,
                    "target": "Sample text 1",
                    "prompt": "prompt 1",
                    "prediction": "pred 1",
                },
                {
                    "id": 2,
                    "target": "Sample text 2",
                    "prompt": "prompt 1",
                    "prediction": "pred 2",
                },
            ]
        )

        # Create simple mocks for model and tokenizer
        # These are only used as parameters and not actually called in our tests
        # since we're mocking process_with_llm
        self.mock_model = MagicMock(device="cpu")
        self.mock_tokenizer = MagicMock()

    @patch("privacy_guard.attacks.extraction.generation_attack.process_dataframe")
    @patch(
        "privacy_guard.attacks.extraction.generation_attack.load_model_and_tokenizer"
    )
    def test_process_dataframe(
        self,
        mock_load_model_and_tokenizer: MagicMock,
        mock_process_dataframe: MagicMock,
    ) -> None:
        mock_process_dataframe.return_value = self.mock_model_outputs
        mock_load_model_and_tokenizer.side_effect
        mock_load_model_and_tokenizer.return_value = (
            self.mock_model,
            self.mock_tokenizer,
        )

        generation_attack = GenerationAttack(
            task="pretrain",
            input_file=self.input_file_name,
            output_file=None,
            model_path="does/not/exist",
            **self.generation_kwargs,  # pyre-ignore Incompatible parameter type [6]
        )

        _ = generation_attack.run_attack()

        mock_process_dataframe.assert_called_once_with(
            df=ANY,
            input_column="prompt",
            output_column="prediction",
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            task="pretrain",
            batch_size=generation_attack.batch_size,
            max_new_tokens=generation_attack.max_new_tokens,
            **self.generation_kwargs,
        )

    @patch("privacy_guard.attacks.extraction.generation_attack.process_dataframe")
    @patch(
        "privacy_guard.attacks.extraction.generation_attack.load_model_and_tokenizer"
    )
    def test_process_dataframe_output_path(
        self,
        mock_load_model_and_tokenizer: MagicMock,
        mock_process_dataframe: MagicMock,
    ) -> None:
        mock_process_dataframe.return_value = self.mock_model_outputs
        mock_load_model_and_tokenizer.side_effect
        mock_load_model_and_tokenizer.return_value = (
            self.mock_model,
            self.mock_tokenizer,
        )

        generation_attack = GenerationAttack(
            task="instruct",
            input_file=self.input_file_name,
            output_file=self.output_file_name,
            model_path="does/not/exist",
        )

        _ = generation_attack.run_attack()

        mock_process_dataframe.assert_called_once_with(
            df=ANY,
            input_column="prompt",
            output_column="prediction",
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            task="instruct",
            batch_size=generation_attack.batch_size,
            max_new_tokens=generation_attack.max_new_tokens,
        )

        self.assertTrue(os.path.getsize(self.output_file_name) > 0)
