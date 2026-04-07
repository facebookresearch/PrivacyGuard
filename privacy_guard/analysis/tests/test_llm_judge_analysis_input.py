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

import pandas as pd
from privacy_guard.analysis.llm_judge.llm_judge_analysis_input import (
    LLMJudgeAnalysisInput,
)
from privacy_guard.analysis.llm_judge.llm_judge_config import LLMJudgeConfig


class TestLLMJudgeAnalysisInput(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame(
            {
                "prompt": ["What is AI?", "Explain ML"],
                "generation": ["AI is...", "ML is..."],
                "reference_text": [
                    "Artificial intelligence is...",
                    "Machine learning is...",
                ],
            }
        )
        self.config = LLMJudgeConfig()
        super().setUp()

    def test_init_with_valid_data(self) -> None:
        analysis_input = LLMJudgeAnalysisInput(
            generation_df=self.df, config=self.config
        )
        self.assertEqual(analysis_input.prompt_key, "prompt")
        self.assertEqual(analysis_input.generation_key, "generation")
        self.assertEqual(analysis_input.reference_key, "reference_text")
        self.assertIs(analysis_input.config, self.config)

    def test_generation_df_property(self) -> None:
        analysis_input = LLMJudgeAnalysisInput(
            generation_df=self.df, config=self.config
        )
        pd.testing.assert_frame_equal(analysis_input.generation_df, self.df)

    def test_has_reference_true(self) -> None:
        analysis_input = LLMJudgeAnalysisInput(
            generation_df=self.df, config=self.config
        )
        self.assertTrue(analysis_input.has_reference)

    def test_has_reference_false_when_key_is_none(self) -> None:
        analysis_input = LLMJudgeAnalysisInput(
            generation_df=self.df, config=self.config, reference_key=None
        )
        self.assertFalse(analysis_input.has_reference)
        self.assertIsNone(analysis_input.reference_key)

    def test_missing_reference_column_warns_and_sets_none(self) -> None:
        df_no_ref = self.df[["prompt", "generation"]].copy()
        with self.assertLogs(
            "privacy_guard.analysis.llm_judge.llm_judge_analysis_input",
            level="WARNING",
        ) as cm:
            analysis_input = LLMJudgeAnalysisInput(
                generation_df=df_no_ref, config=self.config
            )
        self.assertTrue(any("reference_text" in msg for msg in cm.output))
        self.assertIsNone(analysis_input.reference_key)
        self.assertFalse(analysis_input.has_reference)

    def test_missing_prompt_key_raises(self) -> None:
        df_bad = pd.DataFrame({"question": ["What is AI?"], "generation": ["AI is..."]})
        with self.assertRaises(AssertionError):
            LLMJudgeAnalysisInput(generation_df=df_bad, config=self.config)

    def test_missing_generation_key_raises(self) -> None:
        df_bad = pd.DataFrame({"prompt": ["What is AI?"], "response": ["AI is..."]})
        with self.assertRaises(AssertionError):
            LLMJudgeAnalysisInput(generation_df=df_bad, config=self.config)

    def test_custom_column_names(self) -> None:
        df_custom = pd.DataFrame(
            {
                "question": ["What is AI?"],
                "response": ["AI is..."],
                "gold": ["Artificial intelligence is..."],
            }
        )
        analysis_input = LLMJudgeAnalysisInput(
            generation_df=df_custom,
            config=self.config,
            prompt_key="question",
            generation_key="response",
            reference_key="gold",
        )
        self.assertEqual(analysis_input.prompt_key, "question")
        self.assertEqual(analysis_input.generation_key, "response")
        self.assertEqual(analysis_input.reference_key, "gold")
        self.assertTrue(analysis_input.has_reference)

    def test_stores_dataframe_via_base_class(self) -> None:
        analysis_input = LLMJudgeAnalysisInput(
            generation_df=self.df, config=self.config
        )
        pd.testing.assert_frame_equal(analysis_input.df_train_user, self.df)
        self.assertTrue(analysis_input.df_test_user.empty)
