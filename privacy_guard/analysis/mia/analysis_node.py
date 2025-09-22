# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from numpy.typing import NDArray
from privacy_guard.analysis.base_analysis_input import BaseAnalysisInput
from privacy_guard.analysis.base_analysis_node import BaseAnalysisNode
from privacy_guard.analysis.base_analysis_output import BaseAnalysisOutput
from privacy_guard.analysis.mia.mia_results import MIAResults
from tqdm import tqdm


logger: logging.Logger = logging.getLogger(__name__)

TimerStats = dict[str, float]


@dataclass
class AnalysisNodeOutput(BaseAnalysisOutput):
    """
    A dataclass to encapsulate the outputs of AnalsyisNode.
    Attributes:
        eps (float): Epsilon value at the TPR=1% UB or LB threshold depending on attack settings.
        eps_lb (float): Lower bound of epsilon.
        eps_fpr_max_ub (float): Maximum upper bound of epsilon for FPR.
        eps_fpr_lb (List[float]): List of lower bound epsilon values for various FPRs.
        eps_fpr_ub (List[float]): List of upper bound epsilon values for various FPRs.
        eps_tpr_lb (List[float]): List of lower bound epsilon values for various TPRs.
        eps_tpr_ub (List[float]): List of upper bound epsilon values for various TPRs.
        eps_max_lb (List[float]): List of lower bound epsilon values when taking max epsilon over TPR and FPR thresholds (and TNR,FNR is use_tnr_fnr set to true).
        eps_max_ub (List[float]): List of upper bound epsilon values when taking max epsilon over TPR and FPR thresholds (and TNR,FNR is use_tnr_fnr set to true).
        eps_cp (float): Empirical epsilon computed using the Clopper-Pearson CI method.
        accuracy (float): Mean accuracy of the attack.
        accuracy_ci (List[float]): Confidence interval for the accuracy, represented as [lower_bound, upper_bound].
        auc (float): Mean area under the curve (AUC) of the attack.
        auc_ci (List[float]): Confidence interval for the AUC, represented as [lower_bound, upper_bound].
        data_size (dict[str, int]): Size of the training, test dataset and bootstrap sample size.
    """

    # Empirical epsilons
    eps: float
    eps_lb: float
    eps_fpr_max_ub: float
    eps_fpr_lb: List[float]
    eps_fpr_ub: List[float]
    eps_tpr_lb: List[float]
    eps_tpr_ub: List[float]
    eps_max_lb: List[float]
    eps_max_ub: List[float]
    eps_cp: float
    # Accuracy and AUC
    accuracy: float
    accuracy_ci: List[float]
    auc: float
    auc_ci: List[float]
    # Dataset sizes
    data_size: dict[str, int]


class AnalysisNode(BaseAnalysisNode):
    """
    AnalysisNode class for PrivacyGuard, which computes a general set of output metrics
    required to evaluate the performance of a privacy attack.

    Calculates privacy eval metrics by computing epsilons at the upper-bound of the
    95% confidence interval, using a score threshold such that adversary has ~1% TPR.

    args:
        analysis_input: AnalysisInput object containing the training (members) and testing (non-members) dataframes
        delta: delta parameter in (epsilon, delta)-differential privacy, close to 0
        n_users_for_eval: number of users to use for computing epsilon with Clopper-Pearson method
        num_bootstrap_resampling_times: number of times to resample the training and testing data for computing bootstrap confidence intervals
        use_upper_bound: boolean for whether to compute epsilon at the upper-bound of CI
        cap_eps: boolean for whether to cap large epsilon values to log(size of scores)
        use_fnr_tnr: boolean for whether to use FNR and TNR in addition to FPR and TPR error thresholds in eps_max_array computation.
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
        use_fnr_tnr: bool = False,
        show_progress: bool = False,
        with_timer: bool = False,
    ) -> None:
        self._delta = delta
        self._n_users_for_eval = n_users_for_eval
        self._num_bootstrap_resampling_times = num_bootstrap_resampling_times
        self._show_progress = show_progress
        self._with_timer = with_timer
        self._cap_eps = cap_eps
        self._use_fnr_tnr = use_fnr_tnr

        self._use_upper_bound = use_upper_bound

        self._timer_stats: TimerStats = {}

        if self._n_users_for_eval < 0:
            raise ValueError(
                'Input to AnalysisNode "n_users_for_eval" must be nonnegative'
            )

        super().__init__(analysis_input=analysis_input)

    def _calculate_one_off_eps(self) -> float:
        df_train_user = self.analysis_input.df_train_user
        df_test_user = self.analysis_input.df_test_user
        score_train = df_train_user["score"]
        score_test = df_test_user["score"]
        num_users_for_eps_cp_eval = min(
            self._n_users_for_eval,
            score_train.shape[0] // 2,
            score_test.shape[0] // 2,
        )

        assert num_users_for_eps_cp_eval > 0 and num_users_for_eps_cp_eval < min(
            score_train.shape[0], score_test.shape[0]
        )

        loss_train = torch.from_numpy(
            score_train[:num_users_for_eps_cp_eval].to_numpy()
        )
        loss_test = torch.from_numpy(score_test[:num_users_for_eps_cp_eval].to_numpy())

        results = MIAResults(loss_train, loss_test)
        # compute one-off accuracy & AUC & CI for epsilon
        _, _, eps_cp = results.compute_acc_auc_ci_epsilon(self._delta)

        return eps_cp

    def _compute_ci(
        self, array: NDArray[float], axis: int = 0
    ) -> tuple[NDArray, NDArray]:
        """Compute confidence intervals (used for eps, auc, accuracy)"""
        # Sort along the specified axis
        sorted_array = np.sort(array, axis=axis)

        lower_idx = max(int(0.025 * self._num_bootstrap_resampling_times) - 1, 0)
        upper_idx = int(0.975 * self._num_bootstrap_resampling_times)

        # Index into the sorted array at the percentile positions
        lower_bound = np.take(sorted_array, lower_idx, axis=axis)
        upper_bound = np.take(sorted_array, upper_idx, axis=axis)

        # Ensure return is arrays
        if np.isscalar(lower_bound):
            lower_bound = np.array([lower_bound])
            upper_bound = np.array([upper_bound])

        return lower_bound, upper_bound

    def _compute_bootstrap_sample_indexes(
        self,
        num_users: int,
        sample_size: int,
    ) -> list[int]:
        """
        Compute bootstrap indexes by random sampling with replacement for the given sample size from a range [0, num_users)

        Args:
            num_users (int): number of users for indexes 0..num_users-1
            sample_size (int): number of samples among the user indexes
        Returns:
            A list of indexes (with duplicates)
        """
        return np.random.randint(0, num_users, sample_size)

    def run_analysis(self) -> BaseAnalysisOutput:
        """
        Computes analysis outputs based on the input dataframes.
        Overrides "BaseAnalysisNode::run_analysis"

        First, makes loss_train and loss_test and computes one off metrics like psilon confidence intervals.

        Then, uses "make_metrics_array" to build lists of
        metrics, each computed from random subsets of
        loss_train and loss_test. The length of these lists
        is determined by self._num_bootstrap_resampling_times

        These metrics are combined into the output of this analysis,
        and returned from the call.

        Returns:
            AnalysisNodeOutput dataclass with fields:
            "eps": epsilon at TPR=1% UB threshold if use_upper_bound is True, else epsilon at TPR=1% LB threshold
            "eps_fpr_max_lb", "eps_fpr_lb", "eps_fpr_ub": epsilon at various false positive rates
            "eps_tpr_lb", "eps_tpr_ub": epsilon at various true positive rates
            "eps_max_lb", "eps_max_ub": max of epsilon at various true positive rates and false positive rates
            "eps_cp": epsilon calculated via Clopper-Pearson confidence interval
            "accuracy", "accuracy_ci": accuracy values
            "auc", "auc_ci": area under ROC curve values
            "data_size": dictionary with keys "train_size", "test_size", "bootstrap_size"
        """
        df_train_user = self.analysis_input.df_train_user
        df_test_user = self.analysis_input.df_test_user
        score_train = df_train_user["score"]
        score_test = df_test_user["score"]

        logger.info(
            f"Train/Test unique users: {score_train.shape[0]}/{score_test.shape[0]}"
        )

        train_size, test_size = score_train.shape[0], score_test.shape[0]

        eps_cp = self._calculate_one_off_eps()
        logger.info(f"Epsilon CP: {eps_cp}")

        train_size, test_size = score_train.shape[0], score_test.shape[0]
        bootstrap_sample_size = min(train_size, test_size)

        with self.timer("make_metrics_array"):
            metrics_array = self._make_metrics_array()

        accuracy = np.array([run[0] for run in metrics_array])
        auc = np.array([run[1] for run in metrics_array])
        eps_fpr = np.array([run[2] for run in metrics_array])
        eps_tpr = np.array([run[3] for run in metrics_array])
        eps_max = np.array([run[4] for run in metrics_array])

        # get CI bounds with 95% confidence
        accuracy_lb, accuracy_ub = self._compute_ci(accuracy)
        auc_lb, auc_ub = self._compute_ci(auc)
        eps_fpr_lb, eps_fpr_ub = self._compute_ci(eps_fpr)
        eps_tpr_lb, eps_tpr_ub = self._compute_ci(eps_tpr)
        eps_max_lb, eps_max_ub = self._compute_ci(eps_max)

        accuracy_mean = accuracy.mean()
        auc_mean = auc.mean()

        eps_tpr_boundary = eps_tpr_ub if self._use_upper_bound else eps_tpr_lb

        outputs = AnalysisNodeOutput(
            eps=eps_tpr_boundary[0],  # epsilon at TPR=1% UB threshold
            eps_lb=eps_tpr_lb[0],
            eps_fpr_max_ub=np.nanmax(eps_fpr_ub),
            eps_fpr_lb=list(eps_fpr_lb),
            eps_fpr_ub=list(eps_fpr_ub),
            eps_tpr_lb=list(eps_tpr_lb),
            eps_tpr_ub=list(eps_tpr_ub),
            eps_max_lb=list(eps_max_lb),
            eps_max_ub=list(eps_max_ub),
            eps_cp=eps_cp,
            accuracy=accuracy_mean,
            accuracy_ci=[accuracy_lb[0], accuracy_ub[0]],
            auc=auc_mean,
            auc_ci=[auc_lb[0], auc_ub[0]],
            data_size={
                "train_size": train_size,
                "test_size": test_size,
                "bootstrap_size": bootstrap_sample_size,
            },
        )

        if self._with_timer:
            logger.info(f"Timer stats: {self.get_timer_stats()}")

        return outputs

    def _make_metrics_array(
        self,
    ) -> list[
        tuple[
            np.float64, np.float64, list[np.float64], list[np.float64], list[np.float64]
        ]
    ]:
        """
        Make list of tuples metrics at error thresholds, each of which contains the
        accuracy, AUC, and epsilon values for a given number of samples
        The tuples are randomly generated from permutations of subsets of loss_train
        and loss_test.

        Args:
            N: Length of sublist of loss_train and loss_test to pass into MIAResults
        Returns:
            List[Tuple] with elements
                (accuracy,
                auc_value,
                eps_fpr_array,
                eps_tpr_array,
                eps_max_array)
        """
        score_train = self.analysis_input.df_train_user["score"]
        score_test = self.analysis_input.df_test_user["score"]

        # error thresholds set equally spaced at 1% intervals
        loss_train = torch.from_numpy(score_train.to_numpy())
        loss_test = torch.from_numpy(score_test.to_numpy())
        train_size, test_size = score_train.shape[0], score_test.shape[0]

        bootstrap_sample_size = min(train_size, test_size)

        error_thresholds = np.linspace(0.01, 1, 100)

        metrics_array = [
            MIAResults(
                loss_train[
                    self._compute_bootstrap_sample_indexes(
                        train_size, bootstrap_sample_size
                    )
                ],
                loss_test[
                    self._compute_bootstrap_sample_indexes(
                        test_size, bootstrap_sample_size
                    )
                ],
            ).compute_metrics_at_error_threshold(
                self._delta,
                error_threshold=error_thresholds,
                cap_eps=self._cap_eps,
                use_fnr_tnr=self._use_fnr_tnr,
                verbose=False,
            )
            for _ in tqdm(
                range(self._num_bootstrap_resampling_times),
                disable=not self._show_progress,
            )
        ]
        return metrics_array

    @contextmanager
    def timer(self, name: str) -> Generator[None, None, None]:
        """
        Context manager for timing analysis node
        """
        if self._with_timer:
            start = time.time()
            yield
            end = time.time()
            self._timer_stats[name] = end - start
        else:
            yield

    def get_timer_stats(self) -> TimerStats:
        """
        Get timer stats
        """
        return self._timer_stats
