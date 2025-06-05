# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import pandas as pd


class BaseAnalysisInput:
    """
    Takes in two dataframes of training and testing data produced from
    membership inference attacks.
    The input is already scored and is ready for metric computation (analysis)

    args:
        df_train_user: train dataframe, returned from attack
        df_test_user: test dataframe, returned from attack
    """

    def __init__(
        self,
        df_train_user: pd.DataFrame,
        df_test_user: pd.DataFrame,
    ) -> None:
        self._df_train_user = df_train_user
        self._df_test_user = df_test_user

    @property
    def df_train_user(self) -> pd.DataFrame:
        """
        Property accessor for df_train_user
        This is the aggregated result of self.df_train_merge
        """
        return self._df_train_user

    @property
    def df_test_user(self) -> pd.DataFrame:
        """
        Property accessor for df_test_user
        This is the aggregated result of self.df_test_merge
        """
        return self._df_test_user
