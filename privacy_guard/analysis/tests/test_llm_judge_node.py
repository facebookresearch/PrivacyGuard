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


import math
import unittest

import pandas as pd
from privacy_guard.analysis.code_similarity.code_similarity_analysis_input import (
    LLMJudgeAnalysisInput,
)
from privacy_guard.analysis.code_similarity.llm_judge_node import (
    LLMJudgeNode,
    LLMJudgeNodeOutput,
)
from privacy_guard.analysis.code_similarity.llm_judge_prompts import SONG_2024

CODE = "def add(a, b):\n    return a + b\n"


def _run(
    responses: list[str],
    languages: list[str] | None = None,
) -> LLMJudgeNodeOutput:
    data = {
        "target_code_string": [CODE] * len(responses),
        "model_generated_code_string": [CODE] * len(responses),
        "judge_raw_response": responses,
    }
    if languages is not None:
        data["language"] = languages
    node = LLMJudgeNode(
        analysis_input=LLMJudgeAnalysisInput(pd.DataFrame(data)),
        prompt_spec=SONG_2024,
    )
    return node.run_analysis()


class TestLLMJudgeNode(unittest.TestCase):
    def test_scores_and_average(self) -> None:
        output = _run(["[[1.000]]", "[[0.000]]", "[[0.500]]"])
        self.assertIsInstance(output, LLMJudgeNodeOutput)
        self.assertEqual(output.num_samples, 3)
        self.assertEqual(output.num_unparsed, 0)
        self.assertAlmostEqual(output.avg_judge_score, 0.5)
        self.assertAlmostEqual(
            output.per_sample_judge_score["judge_score"].iloc[0], 1.0
        )

    def test_unparsed_are_nan_and_excluded_from_average(self) -> None:
        """A failed parse is missing data, not a zero score."""
        output = _run(["[[1.000]]", "sorry, I cannot help with that"])
        self.assertEqual(output.num_unparsed, 1)
        self.assertAlmostEqual(output.avg_judge_score, 1.0)
        self.assertTrue(
            math.isnan(output.per_sample_judge_score["judge_score"].iloc[1])
        )

    def test_average_is_nan_when_nothing_parses(self) -> None:
        output = _run(["garbage", ""])
        self.assertEqual(output.num_unparsed, 2)
        self.assertTrue(math.isnan(output.avg_judge_score))

    def test_avg_judge_score_by_language(self) -> None:
        output = _run(
            ["[[1.000]]", "[[0.500]]", "[[0.200]]"],
            languages=["python", "python", "cpp"],
        )
        by_lang = output.avg_judge_score_by_language
        assert by_lang is not None
        self.assertAlmostEqual(by_lang["python"], 0.75)
        self.assertAlmostEqual(by_lang["cpp"], 0.2)

    def test_no_language_column(self) -> None:
        self.assertIsNone(_run(["[[0.5]]"]).avg_judge_score_by_language)

    def test_missing_required_column_raises(self) -> None:
        with self.assertRaises(ValueError):
            LLMJudgeAnalysisInput(pd.DataFrame({"target_code_string": [CODE]}))

    def test_compute_outputs_returns_dict(self) -> None:
        node = LLMJudgeNode(
            analysis_input=LLMJudgeAnalysisInput(
                pd.DataFrame(
                    {
                        "target_code_string": [CODE],
                        "model_generated_code_string": [CODE],
                        "judge_raw_response": ["[[0.5]]"],
                    }
                )
            ),
            prompt_spec=SONG_2024,
        )
        outputs = node.compute_outputs()
        self.assertAlmostEqual(outputs["avg_judge_score"], 0.5)
        self.assertEqual(outputs["num_samples"], 1)
