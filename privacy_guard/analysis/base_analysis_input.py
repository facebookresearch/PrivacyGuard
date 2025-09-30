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
