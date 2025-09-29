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
from typing import List, Optional

import pandas as pd
from privacy_guard.analysis.base_analysis_input import BaseAnalysisInput


logger: logging.Logger = logging.getLogger(__name__)


class ProbabilisticMemorizationAnalysisInput(BaseAnalysisInput):
    """
    Takes in a single dataframe of generation data with prediction logprobs.

    args:
        generation_df: dataframe containing logprobs column
        prob_threshold: threshold for comparing model probabilities
        n_values: optional list of n values for computing corresponding probabilities of model outputting the target in n attempts. Refer to https://arxiv.org/abs/2410.19482 for details.
        logprobs_column: name of the column containing logprobs (default: "prediction_logprobs")
    """

    def __init__(
        self,
        generation_df: pd.DataFrame,
        prob_threshold: float,
        n_values: Optional[List[int]] = None,
        logprobs_column: str = "prediction_logprobs",
    ) -> None:
        self.prob_threshold: float = prob_threshold
        self.n_values: List[int] = n_values or []
        self.logprobs_column: str = logprobs_column

        # Validate required columns
        required_columns = {logprobs_column}
        missing_columns = required_columns - set(generation_df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        super().__init__(df_train_user=generation_df, df_test_user=pd.DataFrame())

    @property
    def generation_df(self) -> pd.DataFrame:
        """
        Property accessor for generation_df
        """
        return self._df_train_user
