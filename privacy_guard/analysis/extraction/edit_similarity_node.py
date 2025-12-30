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
from typing import Optional

import pandas as pd
from privacy_guard.analysis.base_analysis_node import BaseAnalysisNode
from privacy_guard.analysis.base_analysis_output import BaseAnalysisOutput
from privacy_guard.analysis.extraction.text_inclusion_analysis_input import (
    TextInclusionAnalysisInput,
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
        super().__init__(analysis_input=analysis_input)

    def run_analysis(self) -> EditSimilarityNodeOutput:
        return EditSimilarityNodeOutput(
            num_samples=0,
            edit_similarity=None,
            edit_similarity_score=None,
            augmented_output_dataset=pd.DataFrame(),
        )
