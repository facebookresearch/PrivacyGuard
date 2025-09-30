# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-strict
from typing import Tuple, Union

import pandas as pd
from pandas import Series
from privacy_guard.analysis.base_analysis_input import BaseAnalysisInput

from privacy_guard.analysis.mia.aggregate_analysis_input import (
    AggregateAnalysisInput,
    AggregationType,
)
from privacy_guard.attacks.base_attack import BaseAttack

from scipy.stats import norm


class LiraAttack(BaseAttack):
    """
    This is an implementation of an MIA attack

    In the LiRA attack, there is a target model (orig) that contains the users in the
    hold_out_train set.
    There also is a set of N reference (shadow) models that do not
    contain these hold_out_train users (in the offline attack) and a further set of N reference shadow models that do contain these hold_out_train users (in the online attack).

    For each user, scores in the shadow tables are aggregated and then combined with the ones in the orig table to generate a final score
    """

    def __init__(
        self,
        df_train_merge: pd.DataFrame,
        df_test_merge: pd.DataFrame,
        row_aggregation: AggregationType,
        user_id_key: str = "user_id",
        use_fixed_variance: bool = False,
        std_dev_type: str = "global",
        online_attack: bool = False,
        offline_shadows_evals_in: bool = False,
    ) -> None:
        """
        args:
            df_train_merge: training data dataframe
            df_test_merge: test data dataframe
                has columns "score_orig" from the orig table
                also has "score_mean" and "score_std" from the shadow tables

            row_aggregation: specifies user aggregation strategy
            user_id_key: key corresponding to user id, used for aggregating scores.

            used_fixed_variance: whether to use fixed variance or not,
                normalizing using the orig scores of the attack.

            std_dev_type: specifies the type of standard deviation to be used in the attack calculations.

            online_attack: indicates whether the attack is an online attack. It defaults to False, indicating that the attack is offline.

            offline_shadows_evals_in: specifies whether the offline shadow evaluations are included in the attack.
            It defaults to False, indicating that offline shadow evaluations are not included unless specified otherwise.

        Returns:
            AnalysisInput has:
                start_eval_ds: in sql query, where to start the date range
                end_eval_ds: in sql query, where to end the date range
                start_eval_ts_hour: int = 0
                end_eval_ts_hour: int = 23
                users_intersect_eval_window: number of distinct user timestamps allowed in window
                apply_hard_cut: whether to apply hard cut
                eval_output_table_settings: dict of configuration for output table
        """
        self.df_train_merge = df_train_merge
        self.df_test_merge = df_test_merge

        self.row_aggregation: AggregationType = row_aggregation
        self.user_id_key = user_id_key
        self.std_dev_type = std_dev_type
        self.online_attack = online_attack
        self.offline_shadows_evals_in = offline_shadows_evals_in
        self.use_fixed_variance = use_fixed_variance

    def _get_std_dev(self) -> Tuple[Union[float, Series], Union[float, Series]]:
        """
        Get the std dev for the in and out scores.

        Returns:
            std_in: std dev for the in scores
            std_out: std dev for the out scores
        """
        std_in, std_out = 0.0, 0.0
        match self.std_dev_type:
            case "global":
                std_in = std_out = (
                    pd.concat(
                        [
                            self.df_train_merge.score_orig,
                            self.df_test_merge.score_orig,
                        ]
                    )
                ).std()
            case "shadows_in":
                if self.online_attack:
                    std_in = std_out = pd.concat(
                        [
                            self.df_train_merge.score_std_in,
                            self.df_test_merge.score_std_in,
                        ]
                    ).mean()
                else:
                    # offline case where std dev is computed on the hold out test set
                    std_in = std_out = pd.concat(
                        [
                            self.df_train_merge.score_std,
                            self.df_test_merge.score_std,
                        ]
                    ).mean()
            case "shadows_out":
                if self.online_attack:
                    std_in = std_out = pd.concat(
                        [
                            self.df_train_merge.score_std_out,
                            self.df_test_merge.score_std_out,
                        ]
                    ).mean()
                else:
                    # offline case where std dev is computed on the hold out test set
                    std_in = std_out = pd.concat(
                        [
                            self.df_train_merge.score_std,
                            self.df_test_merge.score_std,
                        ]
                    ).mean()
            case "mix":
                if not self.online_attack:
                    raise ValueError(
                        "mix std dev type is only supported for online attacks"
                    )
                std_in = pd.concat(
                    [
                        self.df_train_merge.score_std_in,
                        self.df_test_merge.score_std_in,
                    ]
                ).mean()

                std_out = pd.concat(
                    [
                        self.df_train_merge.score_std_out,
                        self.df_test_merge.score_std_out,
                    ]
                ).mean()
            case _:
                raise ValueError(f"{self.std_dev_type} is not a valid std_dev type.")
        return std_in, std_out

    def run_attack(self) -> BaseAnalysisInput:
        """
        Run lira attack on the shadows and original models.

        Returns:
            AggregateAnalysisInput: input for analysis with train and testing datasets
        """

        std_in, std_out = self._get_std_dev()

        if self.online_attack:
            self.df_train_merge["score"] = norm.logpdf(
                self.df_train_merge.score_orig,
                self.df_train_merge.score_mean_in,
                std_in if self.use_fixed_variance else self.df_train_merge.score_std_in,
            ) - norm.logpdf(
                self.df_train_merge.score_orig,
                self.df_train_merge.score_mean_out,
                std_out
                if self.use_fixed_variance
                else self.df_train_merge.score_std_out,
            )

            self.df_test_merge["score"] = norm.logpdf(
                self.df_test_merge.score_orig,
                self.df_test_merge.score_mean_in,
                std_in if self.use_fixed_variance else self.df_test_merge.score_std_in,
            ) - norm.logpdf(
                self.df_test_merge.score_orig,
                self.df_test_merge.score_mean_out,
                std_out
                if self.use_fixed_variance
                else self.df_test_merge.score_std_out,
            )

        else:
            self.df_train_merge["score"] = norm.logpdf(
                self.df_train_merge.score_orig,
                self.df_train_merge.score_mean,
                std_in,
            )
            self.df_test_merge["score"] = norm.logpdf(
                self.df_test_merge.score_orig, self.df_test_merge.score_mean, std_out
            )

        if not (self.online_attack or self.offline_shadows_evals_in):
            # this corresponds to the case of offline shadows evals on the hold out test set
            self.df_train_merge["score"] = -self.df_train_merge["score"]
            self.df_test_merge["score"] = -self.df_test_merge["score"]

        analysis_input = AggregateAnalysisInput(
            row_aggregation=self.row_aggregation,
            df_train_merge=self.df_train_merge,
            df_test_merge=self.df_test_merge,
            user_id_key=self.user_id_key,
        )

        return analysis_input
