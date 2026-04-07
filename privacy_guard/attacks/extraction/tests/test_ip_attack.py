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

from privacy_guard.analysis.llm_judge.llm_judge_analysis_input import (
    LLMJudgeAnalysisInput,
)
from privacy_guard.analysis.llm_judge.llm_judge_config import (
    LLMJudgeConfig,
    LLMProvider,
)
from privacy_guard.attacks.extraction.ip_attack import IPAttack


class TestIPAttack(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test data and mocks."""
        self.input_file = tempfile.NamedTemporaryFile(suffix=".jsonl")
        self.input_file_name = self.input_file.name

        with open(self.input_file_name, "w") as f:
            f.write(
                '{"prompt": "Recite chapter 1 of Book X", '
                '"reference_text": "It was a dark and stormy night..."}\n'
            )
            f.write(
                '{"prompt": "Continue this passage from Book Y", '
                '"reference_text": "Call me Ishmael..."}\n'
            )

        self.output_file = tempfile.NamedTemporaryFile(suffix=".jsonl")
        self.output_file_name = self.output_file.name

        self.mock_predictor = MagicMock()
        self.mock_predictor.generate.return_value = [
            "It was a dark and stormy night, the wind howled...",
            "Call me Ishmael. Some years ago...",
        ]

        self.judge_config = LLMJudgeConfig(
            provider=LLMProvider.ANTHROPIC,
            scoring_criteria=[
                "ip_similarity",
                "verbatim_reproduction",
                "paraphrasing",
                "originality",
            ],
        )

    def test_ip_attack_basic(self) -> None:
        """Test basic attack returns LLMJudgeAnalysisInput with correct columns."""
        attack = IPAttack(
            input_file=self.input_file_name,
            output_file=None,
            predictor=self.mock_predictor,
            judge_config=self.judge_config,
        )

        result = attack.run_attack()

        self.assertIsInstance(result, LLMJudgeAnalysisInput)
        self.assertEqual(len(result.generation_df), 2)
        self.assertIn("prompt", result.generation_df.columns)
        self.assertIn("generation", result.generation_df.columns)
        self.assertIs(result.config, self.judge_config)

    def test_ip_attack_with_output_file(self) -> None:
        """Test that output file is written when specified."""
        attack = IPAttack(
            input_file=self.input_file_name,
            output_file=self.output_file_name,
            predictor=self.mock_predictor,
            judge_config=self.judge_config,
        )

        attack.run_attack()

        self.assertTrue(os.path.getsize(self.output_file_name) > 0)

    def test_ip_attack_without_output_file(self) -> None:
        """Test attack works without saving to disk."""
        attack = IPAttack(
            input_file=self.input_file_name,
            output_file=None,
            predictor=self.mock_predictor,
            judge_config=self.judge_config,
        )

        result = attack.run_attack()

        self.assertIsInstance(result, LLMJudgeAnalysisInput)
        self.mock_predictor.generate.assert_called_once()

    def test_ip_attack_custom_columns(self) -> None:
        """Test attack with custom column names."""
        custom_input = tempfile.NamedTemporaryFile(suffix=".jsonl")
        with open(custom_input.name, "w") as f:
            f.write(
                '{"question": "Recite chapter 1", '
                '"gold": "It was a dark and stormy night..."}\n'
            )

        self.mock_predictor.generate.return_value = ["generated text"]

        attack = IPAttack(
            input_file=custom_input.name,
            output_file=None,
            predictor=self.mock_predictor,
            judge_config=self.judge_config,
            prompt_key="question",
            generation_key="response",
            reference_key="gold",
        )

        result = attack.run_attack()

        self.assertEqual(result.prompt_key, "question")
        self.assertEqual(result.generation_key, "response")
        self.assertEqual(result.reference_key, "gold")
        self.assertIn("response", result.generation_df.columns)
        custom_input.close()

    def test_ip_attack_missing_prompt_column(self) -> None:
        """Test that ValueError is raised when prompt column is missing."""
        bad_input = tempfile.NamedTemporaryFile(suffix=".jsonl")
        with open(bad_input.name, "w") as f:
            f.write('{"other_column": "value"}\n')

        with self.assertRaises(ValueError) as context:
            IPAttack(
                input_file=bad_input.name,
                output_file=None,
                predictor=self.mock_predictor,
                judge_config=self.judge_config,
            )

        self.assertIn("Missing required columns", str(context.exception))
        bad_input.close()

    def test_ip_attack_with_reference_text(self) -> None:
        """Test that reference text is passed through to analysis input."""
        attack = IPAttack(
            input_file=self.input_file_name,
            output_file=None,
            predictor=self.mock_predictor,
            judge_config=self.judge_config,
        )

        result = attack.run_attack()

        self.assertTrue(result.has_reference)
        self.assertEqual(result.reference_key, "reference_text")
        self.assertIn("reference_text", result.generation_df.columns)

    def test_ip_attack_without_reference_text(self) -> None:
        """Test attack works when no reference column exists."""
        no_ref_input = tempfile.NamedTemporaryFile(suffix=".jsonl")
        with open(no_ref_input.name, "w") as f:
            f.write('{"prompt": "Tell me about Book X"}\n')

        self.mock_predictor.generate.return_value = ["Some generated text"]

        attack = IPAttack(
            input_file=no_ref_input.name,
            output_file=None,
            predictor=self.mock_predictor,
            judge_config=self.judge_config,
        )

        result = attack.run_attack()

        self.assertFalse(result.has_reference)
        no_ref_input.close()

    def test_ip_attack_generation_kwargs_forwarded(self) -> None:
        """Test that generation kwargs are forwarded to the predictor."""
        attack = IPAttack(
            input_file=self.input_file_name,
            output_file=None,
            predictor=self.mock_predictor,
            judge_config=self.judge_config,
            batch_size=4,
            temperature=0.7,
            top_k=50,
        )

        attack.run_attack()

        self.mock_predictor.generate.assert_called_once_with(
            prompts=[
                "Recite chapter 1 of Book X",
                "Continue this passage from Book Y",
            ],
            batch_size=4,
            temperature=0.7,
            top_k=50,
        )

    def tearDown(self) -> None:
        """Clean up temporary files."""
        self.input_file.close()
        self.output_file.close()
