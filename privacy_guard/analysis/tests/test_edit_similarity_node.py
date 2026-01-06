# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-strict

import unittest

import pandas as pd

from privacy_guard.analysis.extraction.edit_similarity_node import (
    EditSimilarityNode,
    EditSimilarityNodeOutput,
)

from privacy_guard.analysis.extraction.text_inclusion_analysis_input import (
    TextInclusionAnalysisInput,
)

from privacy_guard.analysis.extraction.text_inclusion_analysis_node import (
    TextInclusionAnalysisNode,
)

from tqdm import tqdm

tqdm.pandas()


class TestEditSimilarityNode(unittest.TestCase):
    def setUp(self) -> None:
        self.data = {
            "prompt": [
                "This is a test prompt",
                "This is another test prompt",
                "Hello, test prompt!",
                "",
                "\t\n\t\n  ",
                "Neque porro quisquam est qui dolorem ipsum",
            ],
            "target": [
                "Target text 1",
                "Target text 2",
                "This is an exact match.",
                "This is included \n",
                "Included",
                "Neque porro quisquam est qui dolorem ipsum",
            ],
            "output_text": [
                "A success: Target text 1",
                "Failure, no match. ",
                "This is an exact match.",
                "This is included",
                "IsItIncluded?",
                "dolorem ipsum quia dolor sit amet, consectetur, adipisci velit",
            ],
        }

        self.analysis_input = TextInclusionAnalysisInput(
            generation_df=pd.DataFrame(self.data)
        )
        self.analysis_node = TextInclusionAnalysisNode(
            analysis_input=self.analysis_input
        )

        super().setUp()

    def test_valid_keys_in_analysis_input(self) -> None:
        self.assertEqual(self.analysis_input.prompt_key, "prompt")
        self.assertEqual(self.analysis_input.target_key, "target")
        self.assertEqual(self.analysis_input.generation_key, "output_text")

    def test_edit_similarity_node_output_creation(self) -> None:
        generation_df = self.analysis_input.generation_df
        output = EditSimilarityNodeOutput(
            num_samples=len(generation_df),
            edit_similarity=None,
            edit_similarity_score=None,
            augmented_output_dataset=generation_df,
        )
        self.assertEqual(output.augmented_output_dataset.equals(generation_df), True)

    def test_text_inclusion_edit_similarity(self) -> None:
        analysis_input = TextInclusionAnalysisInput(
            generation_df=pd.DataFrame(self.data)
        )
        analysis_node = EditSimilarityNode(analysis_input=analysis_input)
        results = analysis_node.compute_outputs()
        self.assertIn("edit_similarity", results)
        self.assertIn("edit_similarity_score", results)
        self.assertEqual(results["edit_similarity"].tolist(), [13, 3, 22, 16, 8, 16])
