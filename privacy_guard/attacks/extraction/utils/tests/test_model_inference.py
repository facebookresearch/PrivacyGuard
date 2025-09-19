# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from privacy_guard.attacks.extraction.utils.model_inference import process_dataframe


class TestAudienceExpansionPGEModelInference(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test data and mocks."""
        # Sample test data
        self.test_texts = [
            "This is test text 1. ",
            "This is test text 2. ",
            "This is test text 3. ",
        ]
        self.test_targets = [
            "This is target text 1",
            "This is target text 2",
            "This is target text 3",
        ]

        # Sample DataFrame
        self.df = pd.DataFrame(
            {
                "id": ["1", "2", "3"],
                "text": self.test_texts,
                "prompt": self.test_texts,
                "target": self.test_targets,
                "prompt_list": ["[text1]", "[text2]", "[text3]"],
            }
        )

        # Mock model outputs
        self.mock_model_outputs = [
            "This is test text 1. This is target text 1",
            "This is test text 2. This is target text 2",
            "This is test text 3. This is target text 3",
        ]

        # Create simple mocks for model and tokenizer
        # These are only used as parameters and not actually called in our tests
        # since we're mocking process_with_llm
        self.mock_model = MagicMock(device="cpu")
        self.mock_tokenizer = MagicMock()

    @patch("privacy_guard.attacks.extraction.utils.model_inference.process_with_llm")
    def test_process_dataframe(self, mock_process_with_llm: MagicMock) -> None:
        mock_process_with_llm.return_value = self.mock_model_outputs

        result_df = process_dataframe(
            df=self.df,
            input_column="text",
            output_column="output",
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            task="instruct",
            batch_size=2,
            max_new_tokens=100,
            device="cpu",
        )

        self.assertEqual(len(result_df), len(self.df))
        self.assertIn("raw_model_output", result_df.columns)
        self.assertIn("output", result_df.columns)
        self.assertEqual(
            result_df["raw_model_output"].tolist(), self.mock_model_outputs
        )
        self.assertEqual(result_df["output"].tolist(), self.test_targets)

        mock_process_with_llm.assert_called_once_with(
            self.df["text"].tolist(),
            self.mock_model,
            self.mock_tokenizer,
            task="instruct",
            batch_size=2,
            max_new_tokens=100,
            device="cpu",
        )

    @patch("privacy_guard.attacks.extraction.utils.model_inference.process_with_llm")
    def test_process_dataframe_with_prompt_list(
        self, mock_process_with_llm: MagicMock
    ) -> None:
        mock_process_with_llm.return_value = self.mock_model_outputs

        df_without_sentence = self.df.drop(columns=["prompt"])

        result_df = process_dataframe(
            df=df_without_sentence,
            input_column="prompt_list",
            output_column="output",
            task="instruct",
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            batch_size=2,
            max_new_tokens=100,
        )

        self.assertIn("prompt_list", result_df.columns)

        self.assertEqual(len(result_df), len(df_without_sentence))
        self.assertIn("raw_model_output", result_df.columns)
        self.assertIn("output", result_df.columns)
