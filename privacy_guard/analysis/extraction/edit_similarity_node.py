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

from dataclasses import dataclass
from typing import cast, Optional

import pandas as pd
import textdistance
from privacy_guard.analysis.base_analysis_node import BaseAnalysisNode
from privacy_guard.analysis.base_analysis_output import BaseAnalysisOutput
from privacy_guard.analysis.extraction.text_inclusion_analysis_input import (
    TextInclusionAnalysisInput,
)
from privacy_guard.analysis.extraction.text_inclusion_analysis_node import (
    _clean_text,
    _normalize_by_target_len,
)
from tqdm import tqdm

tqdm.pandas()


@dataclass
class EditSimilarityNodeOutput(BaseAnalysisOutput):
    """A dataclass to encapsulate the outputs of TextInclusionAnalysisNode."""

    num_samples: int
    edit_similarity: Optional[pd.Series]
    edit_similarity_score: Optional[pd.Series]
    augmented_output_dataset: pd.DataFrame


class EditSimilarityNode(BaseAnalysisNode):
    """EditSimilarityNode class for PrivacyGuard.

    Takes in a single dataframe containing prompt, target, and generation columns, and computes different edit similarity score.

    Additionally supports filtering true positives, in situations
    where the target is errantly included in the prompt text.

    NOTE: currently supported for single target only.

    Args:
        text_inclusion_analysis_input: AnalysisInputObject containing the
            prompt, target, and output_text columns.
    """

    def __init__(self, analysis_input: TextInclusionAnalysisInput) -> None:
        """
        args:
            user_aggregation: specifies user aggregation strategy
        """
        self.prompt_key: str = analysis_input.prompt_key
        self.generation_key: str = analysis_input.generation_key
        self.target_key: str = analysis_input.target_key
        self.generation_df: pd.DataFrame = analysis_input.generation_df
        super().__init__(analysis_input=analysis_input)

    def _compute_edit_similarity(
        self, row: pd.Series, s1_column: str | None = None, s2_column: str | None = None
    ) -> int:
        """Compute edit similarity between target and generation text. Texts are cleaned first.
        Currently not supported for multi target mode.

        Args:
            row (pd.Series): A row of a DataFrame containing the s1 and s2 columns.

        Returns:
            int: Edit similarity between the two strings.
        """
        s1 = _clean_text(row[s1_column or self.target_key])
        s2 = _clean_text(row[s2_column or self.generation_key])
        levenshtein = textdistance.levenshtein.similarity(s1, s2)
        return levenshtein

    def run_analysis(self) -> EditSimilarityNodeOutput:
        analysis_input: TextInclusionAnalysisInput = cast(
            TextInclusionAnalysisInput, self.analysis_input
        )
        generation_df = analysis_input.generation_df

        outputs = EditSimilarityNodeOutput(
            num_samples=len(generation_df),
            edit_similarity=None,
            edit_similarity_score=None,
            augmented_output_dataset=generation_df,
        )

        generation_df["edit_similarity"] = generation_df.progress_apply(
            self._compute_edit_similarity, axis=1
        )
        generation_df["edit_similarity_score"] = _normalize_by_target_len(
            generation_df["edit_similarity"], generation_df[self.target_key]
        )

        outputs.edit_similarity = generation_df["edit_similarity"]
        outputs.edit_similarity_score = generation_df["edit_similarity_score"]

        return outputs
