# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import logging

import numpy as np
import pandas as pd
from privacy_guard.analysis.base_analysis_input import BaseAnalysisInput

from privacy_guard.analysis.mia.analysis_node import AnalysisNode

logger: logging.Logger = logging.getLogger(__name__)


class BalancedAnalysisNode(AnalysisNode):
    """
    BalancedAnalysisNode class that extends AnalysisNode with
    additional functionality: it adds a balance() method that balances the number of train and test samples.

    args:
        analysis_input: AnalysisInput object containing the training and testing dataframes
        delta: delta parameter from (epsilon, delta)-differential privacy (close to 0)
        n_users_for_eval: number of users to use for computing the metrics
        num_bootstrap_resampling_times: length of array used to generate metric arrays
        use_upper_bound: boolean for whether to compute epsilon at the upper-bound of CI
        cap_eps: boolean for whether to cap large epsilon values to log(size of scores)
        show_progress: boolean for whether to show tqdm progress bar
        with_timer: boolean for whether to show timer for analysis node
    """

    def __init__(
        self,
        analysis_input: BaseAnalysisInput,
        delta: float,
        n_users_for_eval: int,
        use_upper_bound: bool = True,
        num_bootstrap_resampling_times: int = 1000,
        cap_eps: bool = True,
        show_progress: bool = False,
        with_timer: bool = False,
    ) -> None:
        """
        Initialize the BalancedAnalysisNode with the same parameters as AnalysisNode.
        """
        df_train_user_bal, df_test_user_bal = self._balance(
            df_train_user=analysis_input.df_train_user,
            df_test_user=analysis_input.df_test_user,
        )

        balanced_analysis_input = BaseAnalysisInput(
            df_train_user=df_train_user_bal, df_test_user=df_test_user_bal
        )

        super().__init__(
            analysis_input=balanced_analysis_input,
            delta=delta,
            n_users_for_eval=n_users_for_eval,
            use_upper_bound=use_upper_bound,
            num_bootstrap_resampling_times=num_bootstrap_resampling_times,
            cap_eps=cap_eps,
            show_progress=show_progress,
            with_timer=with_timer,
        )

    @staticmethod
    def _upsample(scores: pd.Series, sample_count_diff: int) -> pd.Series:
        """
        Upsamples scores by first shuffling it and concatenating it
        as many times as necessary to add sample_count_diff samples.

        Args:
            scores: Series of scores to upsample
            sample_count_diff: Number of additional samples to add

        Returns:
            Upsampled series of scores
        """
        n = len(scores)
        # Create a permutation of indices
        perm = np.random.permutation(n)
        shuffled_scores = scores.iloc[perm].reset_index(drop=True)

        # Calculate how many chunks we need
        n_chunks = sample_count_diff // n + 2

        # Create a list of dataframes to concatenate
        chunks = [shuffled_scores] * n_chunks

        # Concatenate and slice to get the exact number of samples needed
        result = pd.concat(chunks, ignore_index=True).iloc[: n + sample_count_diff]

        return result

    @staticmethod
    def _balance_smaller(
        smaller_df: pd.DataFrame, sample_count_diff: int
    ) -> pd.DataFrame:
        smaller_df_scores = smaller_df["score"]
        upsampled_scores = BalancedAnalysisNode._upsample(
            smaller_df_scores, sample_count_diff
        )

        # Create new dataframe with upsampled scores
        new_indices = range(len(smaller_df), len(smaller_df) + sample_count_diff)
        new_rows = pd.DataFrame(
            {"score": upsampled_scores.iloc[-sample_count_diff:].values},
            index=new_indices,
        )

        # Add any other columns that might be in the original dataframe
        for col in smaller_df.columns:
            if col != "score":
                # For other columns, just copy the values from the original dataframe
                # This is a simplification and might need to be adjusted based on the actual data
                # TODO: This fills the columns with the first row in the original dataframe, which creates an out-of-distribution datasets for either train or test
                # This is fine for now as we only use the "score" column in the epsilon analysis, but would be an issue if we used other columns
                new_rows[col] = smaller_df[col].iloc[0]

        smaller_df = pd.concat([smaller_df, new_rows])

        return smaller_df

    @staticmethod
    def _balance(
        df_train_user: pd.DataFrame, df_test_user: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Balances the number of train and test samples by up/downsampling the smaller one.

        Args:
            df_train_user: DataFrame containing the training data
            df_test_user: DataFrame containing the test data

        Returns:
            Balanced training and test dataframes
        """

        n_train = len(df_train_user)
        n_test = len(df_test_user)

        # Balance the datasets by upsampling the smaller one
        sample_count_diff = n_train - n_test
        if sample_count_diff > 0:
            df_test_user = BalancedAnalysisNode._balance_smaller(
                df_test_user, sample_count_diff
            )
        elif sample_count_diff < 0:
            df_train_user = BalancedAnalysisNode._balance_smaller(
                df_train_user, -sample_count_diff
            )

        return df_train_user, df_test_user
