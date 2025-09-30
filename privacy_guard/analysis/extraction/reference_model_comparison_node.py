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
from typing import cast

import pandas as pd
from privacy_guard.analysis.base_analysis_node import BaseAnalysisNode
from privacy_guard.analysis.base_analysis_output import BaseAnalysisOutput
from privacy_guard.analysis.extraction.reference_model_comparison_input import (
    ReferenceModelComparisonInput,
)

from tqdm import tqdm

tqdm.pandas()


@dataclass
class ReferenceModelComparisonNodeOutput(BaseAnalysisOutput):
    """A dataclass to encapsulate the outputs of ReferenceModelComparisonNode."""

    num_samples: int
    tgt_pos_ref_pos: pd.Series
    tgt_pos_ref_neg: pd.Series
    tgt_neg_ref_pos: pd.Series
    tgt_neg_ref_neg: pd.Series
    augmented_output_dataset: pd.DataFrame


class ReferenceModelComparisonNode(BaseAnalysisNode):
    """ReferenceModelComparisonNode class for PrivacyGuard.

    Takes in two dataframes (target and reference) and compares them based on a specified result_key.
    Adds four columns to the target dataframe:
    1. tgt_pos_ref_pos: Both target and reference are positive
    2. tgt_pos_ref_neg: Target is positive, reference is negative
    3. tgt_neg_ref_pos: Target is negative, reference is positive
    4. tgt_neg_ref_neg: Both target and reference are negative

    Args:
        reference_model_comparison_input: AnalysisInputObject containing the
            target and reference dataframes and the result_key to compare.
    """

    def __init__(self, analysis_input: ReferenceModelComparisonInput) -> None:
        """
        args:
            analysis_input: ReferenceModelComparisonInput containing target_df, reference_df, and result_key
        """
        self.target_df: pd.DataFrame = analysis_input.target_df
        self.reference_df: pd.DataFrame = analysis_input.reference_df
        self.result_key: str = analysis_input.result_key

        super().__init__(analysis_input=analysis_input)

    def run_analysis(self) -> ReferenceModelComparisonNodeOutput:
        """
        Compares target_df and reference_df based on result_key and adds comparison columns to target_df.
        """
        analysis_input: ReferenceModelComparisonInput = cast(
            ReferenceModelComparisonInput, self.analysis_input
        )

        target_df = analysis_input.target_df
        reference_df = analysis_input.reference_df
        result_key = analysis_input.result_key

        # Ensure both dataframes have the same length structure for comparison
        if len(target_df) != len(reference_df):
            raise ValueError(
                f"Target and reference dataframes have different lengths: {len(target_df)} vs {len(reference_df)}"
            )

        # Create a combined dataframe for comparison
        combined_df = pd.DataFrame(
            {"target": target_df[result_key], "reference": reference_df[result_key]}
        )

        # Calculate the four comparison columns
        target_df["tgt_pos_ref_pos"] = combined_df.apply(
            lambda row: row["target"] and row["reference"], axis=1
        )

        target_df["tgt_pos_ref_neg"] = combined_df.apply(
            lambda row: row["target"] and not row["reference"], axis=1
        )

        target_df["tgt_neg_ref_pos"] = combined_df.apply(
            lambda row: not row["target"] and row["reference"], axis=1
        )

        target_df["tgt_neg_ref_neg"] = combined_df.apply(
            lambda row: not row["target"] and not row["reference"], axis=1
        )

        outputs = ReferenceModelComparisonNodeOutput(
            num_samples=len(target_df),
            tgt_pos_ref_pos=target_df["tgt_pos_ref_pos"],
            tgt_pos_ref_neg=target_df["tgt_pos_ref_neg"],
            tgt_neg_ref_pos=target_df["tgt_neg_ref_pos"],
            tgt_neg_ref_neg=target_df["tgt_neg_ref_neg"],
            augmented_output_dataset=target_df,
        )

        return outputs
