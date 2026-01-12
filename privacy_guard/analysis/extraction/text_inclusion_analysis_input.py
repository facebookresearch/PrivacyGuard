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
import logging
from dataclasses import dataclass
from typing import Dict

import pandas as pd
from privacy_guard.analysis.base_analysis_input import BaseAnalysisInput


logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class LCSBoundConfig:
    """A dataclass to specify target LCS lengths for TextInclusionAnalysisNode.
    This is required when computing LCS is computationally expensive."""

    lcs_len_target: int
    fp_len_target: int


class TextInclusionAnalysisInput(BaseAnalysisInput):
    """
    Takes in a single dataframe of prompt+generation data.

    args:
        generation_df: dataframe containing prompt, target, and output_text columns
    """

    def __init__(
        self,
        generation_df: pd.DataFrame,
        prompt_key: str = "prompt",
        target_key: str = "target",
        generation_key: str = "output_text",
        disable_exact_match: bool = False,
        disable_longest_common_substring: bool = False,
        disable_similarity: bool = False,
        lcs_bound_config: LCSBoundConfig | None = None,
        disable_word_level_longest_common_subsequence: bool = False,
        disable_char_level_longest_common_subsequence: bool = True,
    ) -> None:
        columns = generation_df.columns.tolist()
        assert prompt_key in columns, (
            f"Prompt key '{prompt_key}' not found in dataframe columns {columns}"
        )
        assert target_key in columns, (
            f"Target key '{target_key}' not found in dataframe columns {columns}"
        )
        assert generation_key in columns, (
            f"Generation key '{generation_key}' not found in dataframe columns {columns}"
        )

        self.prompt_key = prompt_key
        self.target_key = target_key
        self.generation_key = generation_key

        self.disable_exact_match = disable_exact_match
        self.disable_longest_common_substring = disable_longest_common_substring
        self.disable_similarity = disable_similarity
        self.lcs_bound_config = lcs_bound_config

        self.disable_word_level_longest_common_subsequence = (
            disable_word_level_longest_common_subsequence
        )
        self.disable_char_level_longest_common_subsequence = (
            disable_char_level_longest_common_subsequence
        )

        super().__init__(df_train_user=generation_df, df_test_user=pd.DataFrame())

    @property
    def generation_df(self) -> pd.DataFrame:
        """
        Property accessor for generation_df
        """
        return self._df_train_user

    def update_lcs_bound_config(self, lcs_len_target: int, fp_len_target: int) -> None:
        """
        Update the lcs bound config
        """
        self.lcs_bound_config = LCSBoundConfig(
            lcs_len_target=lcs_len_target, fp_len_target=fp_len_target
        )


class TextInclusionAnalysisInputBatch(BaseAnalysisInput):
    def __init__(self, input_batch: Dict[str, TextInclusionAnalysisInput]) -> None:
        self.input_batch = input_batch

    @property
    def generation_df(self) -> pd.DataFrame:
        """
        Property accessor for generation_df
        """
        # TODO: return a representative dataframe.
        return pd.DataFrame()
