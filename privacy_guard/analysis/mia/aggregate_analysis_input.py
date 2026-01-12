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
from enum import Enum
from typing import Set

import pandas as pd
from privacy_guard.analysis.base_analysis_input import BaseAnalysisInput
from scipy.special import logsumexp
from scipy.stats import gmean

logger: logging.Logger = logging.getLogger(__name__)


class AggregationType(Enum):
    MIN = "min"
    MAX = "max"
    AVG = "avg"
    LOGSUMEXP = "logsumexp"
    GMEAN = "gmean"
    NONE = "none"
    ABS_MAX = "abs_max"


class AggregateAnalysisInput(BaseAnalysisInput):
    """
    Takes in two dataframes of training and testing data produced from
    membership inference attacks.
    - Verifies the presence of required columns
    - Precomputes and stores processed versions of the dataframes

    args:
        row_aggregation: specifies user aggregation strategy
        df_train_merge: train dataframe, returned from attack
        df_test_merge: test dataframe, returned from attack
        user_id_key: key of dataframes containing the user ids, used for grouping and aggregating results.
    """

    REQUIRED_COLUMNS = {
        "score",
    }

    def __init__(
        self,
        row_aggregation: AggregationType,
        df_train_merge: pd.DataFrame,
        df_test_merge: pd.DataFrame,
        user_id_key: str,
    ) -> None:
        self._df_train_merge = df_train_merge
        self._df_test_merge = df_test_merge
        self._row_aggregation = row_aggregation
        self._user_id_key = user_id_key

        self.required_columns: Set[str] = self.REQUIRED_COLUMNS | {user_id_key}

        self.check_required_columns(df_train_merge)
        self.check_required_columns(df_test_merge)

        df_train_user, df_test_user = self.aggregate_users()
        super().__init__(df_train_user=df_train_user, df_test_user=df_test_user)

    @property
    def df_train_merge(self) -> pd.DataFrame:
        """
        Property accessor for df_test_user
        """
        return self._df_train_merge

    @property
    def df_test_merge(self) -> pd.DataFrame:
        """
        Property accessor for df_test_user
        """
        return self._df_test_merge

    @property
    def row_aggregation(self) -> AggregationType:
        """
        Property accessor for row aggregation strategy
        """
        return self._row_aggregation

    def check_required_columns(self, df: pd.DataFrame) -> None:
        """
        Checks that the dataframe 'df' has the required columns,
        specified in self.required_columns
        """
        columns = set(df.columns)
        missing_columns = self.required_columns - columns
        if missing_columns:
            raise ValueError(
                f"dataframe must contain required columns {list(missing_columns)}. Its columns are {list(columns)}"
            )

    def aggregate_users(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aggregates self.df_train_merge and self.df_test_merge
        using the strategy specified in self.row_aggregation,
        aggregating on the column user_id_key.
        Returns:
            a tuple of the aggregated dataframes
            (df_train_user, df_test_user)
        """
        # row_aggregation
        row_aggregation = self.row_aggregation
        df_train_merge = self.df_train_merge
        df_test_merge = self.df_test_merge

        if row_aggregation == AggregationType.NONE:
            # if user aggregation is none, we don't aggregate users
            return (df_train_merge, df_test_merge)
        elif row_aggregation == AggregationType.MAX:
            df_train_user = df_train_merge.groupby(
                [self._user_id_key], sort=False
            ).max()
            df_test_user = df_test_merge.groupby([self._user_id_key], sort=False).max()
            return (df_train_user, df_test_user)
        elif row_aggregation == AggregationType.MIN:
            df_train_user = df_train_merge.groupby(
                [self._user_id_key], sort=False
            ).min()
            df_test_user = df_test_merge.groupby([self._user_id_key], sort=False).min()
            return (df_train_user, df_test_user)
        elif row_aggregation == AggregationType.AVG:
            df_train_user = df_train_merge.groupby(
                [self._user_id_key], sort=False
            ).mean()
            df_test_user = df_test_merge.groupby([self._user_id_key], sort=False).mean()
            return (df_train_user, df_test_user)
        elif row_aggregation == AggregationType.GMEAN:
            if df_train_merge["score"].min() < 0 or df_test_merge["score"].min() < 0:
                logger.warn(
                    "Gmean (geometric mean) aggregation assumes non-negative values. There are negative per-impression scores in the data which will be clipped to 1e-3. This can destoy the meaning of gmean. We recommend using a different type of user score aggregation."
                )
            df_train_user = pd.DataFrame(
                df_train_merge[["score", self._user_id_key]]
                .groupby(self._user_id_key, sort=False)
                .score.apply(lambda x: gmean(x.clip(lower=1e-3))),
                columns=["score"],
            )

            df_test_user = pd.DataFrame(
                df_test_merge[["score", self._user_id_key]]
                .groupby([self._user_id_key], sort=False)
                .score.apply(lambda x: gmean(x.clip(lower=1e-3))),
                columns=["score"],
            )
            return (df_train_user, df_test_user)
        # user aggregation is logsumexp
        else:
            df_train_user = pd.DataFrame(
                df_train_merge[["score", self._user_id_key]]
                .groupby([self._user_id_key], sort=False)
                .score.apply(logsumexp),
                columns=["score"],
            )
            df_test_user = pd.DataFrame(
                df_test_merge[["score", self._user_id_key]]
                .groupby([self._user_id_key], sort=False)
                .score.apply(logsumexp),
                columns=["score"],
            )
            return (df_train_user, df_test_user)
