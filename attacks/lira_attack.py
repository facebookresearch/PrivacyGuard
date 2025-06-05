# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
import pandas as pd

from privacy_guard.analysis.aggregate_analysis_input import (
    AggregateAnalysisInput,
    AggregationType,
)
from privacy_guard.analysis.base_analysis_input import BaseAnalysisInput
from privacy_guard.attacks.base_attack import BaseAttack

from scipy.stats import norm


class LiraAttack(BaseAttack):
    """
    This is an implementation of an MIA attack

    In the LiRA attack, there is a target model (orig) that contains the users in the
    hold_out_train set.
    There also is a set of N reference (shadow) models that do not
    contain these hold_out_train users.
    For each user, scores in the shadow tables are aggregated and then combined with the ones in the orig table to generate a final score
    """

    def __init__(
        self,
        df_train_merge: pd.DataFrame,
        df_test_merge: pd.DataFrame,
        row_aggregation: AggregationType,
        use_fixed_variance: bool,
    ) -> None:
        """
        args:
            df_train_merge: training data dataframe
            df_test_merge: test data dataframe
                has columns "score_orig" from the orig table
                also has "score_mean" and "score_std" from the shadow tables

            row_aggregation: specifies user aggregation strategy

            used_fixed_variance: whether to use fixed variance or not,
                normalizing using the orig scores of the attack.


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
        self.use_fixed_variance = use_fixed_variance

    def run_attack(self) -> BaseAnalysisInput:
        """
        Run lira attack on the shadows and original models.

        Returns:
            AggregateAnalysisInput: input for analysis with train and testing datasets
        """

        if self.use_fixed_variance:
            fixed_std = pd.concat(
                [self.df_train_merge["score_orig"], self.df_test_merge["score_orig"]]
            ).std()

            self.df_train_merge["score"] = norm.logpdf(
                self.df_train_merge.score_orig,
                self.df_train_merge.score_mean,
                fixed_std,
            )

            self.df_test_merge["score"] = norm.logpdf(
                self.df_test_merge.score_orig, self.df_test_merge.score_mean, fixed_std
            )
        else:
            self.df_train_merge["score"] = norm.logpdf(
                self.df_train_merge.score_orig,
                self.df_train_merge.score_mean,
                self.df_train_merge.score_std,
            )

            self.df_test_merge["score"] = norm.logpdf(
                self.df_test_merge.score_orig,
                self.df_test_merge.score_mean,
                self.df_test_merge.score_std,
            )

        self.df_train_merge["score"] = -self.df_train_merge["score"]
        self.df_test_merge["score"] = -self.df_test_merge["score"]

        analysis_input = AggregateAnalysisInput(
            row_aggregation=self.row_aggregation,
            df_train_merge=self.df_train_merge,
            df_test_merge=self.df_test_merge,
        )

        return analysis_input
