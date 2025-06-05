# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
import logging

import pandas as pd
from privacy_guard.analysis.base_analysis_input import BaseAnalysisInput


logger: logging.Logger = logging.getLogger(__name__)


class ReferenceModelComparisonInput(BaseAnalysisInput):
    """
    Takes in two dataframes for comparison: target and reference.

    args:
        target_df: dataframe containing the target model's outputs
        reference_df: dataframe containing the reference model's outputs
        result_key: column name to use for comparison (default: "decision_prompt")
    """

    def __init__(
        self,
        target_df: pd.DataFrame,
        reference_df: pd.DataFrame,
        result_key: str = "decision_prompt",
    ) -> None:
        self.target_df = target_df
        self.reference_df = reference_df
        self.result_key = result_key
        super().__init__(df_train_user=target_df, df_test_user=reference_df)

    @property
    def target_dataframe(self) -> pd.DataFrame:
        """
        Property accessor for target_df
        """
        return self._df_train_user

    @property
    def reference_dataframe(self) -> pd.DataFrame:
        """
        Property accessor for reference_df
        """
        return self._df_test_user
