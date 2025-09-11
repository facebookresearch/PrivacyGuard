# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

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
        generation_df: dataframe containing prediction_logprobs column (2D list)
        prob_threshold: threshold for comparing model probabilities
        n_values: optional list of n values for computing corresponding probabilities of model outputting the target in n attempts. Refer to https://arxiv.org/abs/2410.19482 for details.
    """

    REQUIRED_COLUMNS = {
        "prediction_logprobs",
    }

    def __init__(
        self,
        generation_df: pd.DataFrame,
        prob_threshold: float,
        n_values: Optional[List[int]] = None,
    ) -> None:
        self.prob_threshold: float = prob_threshold
        self.n_values: List[int] = n_values or []

        # Validate required columns
        missing_columns = self.REQUIRED_COLUMNS - set(generation_df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        super().__init__(df_train_user=generation_df, df_test_user=pd.DataFrame())

    @property
    def generation_df(self) -> pd.DataFrame:
        """
        Property accessor for generation_df
        """
        return self._df_train_user
