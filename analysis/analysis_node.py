# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from privacy_guard.analysis.base_analysis_input import BaseAnalysisInput
from privacy_guard.analysis.base_analysis_node import BaseAnalysisNode
from privacy_guard.analysis.base_analysis_output import BaseAnalysisOutput
from privacy_guard.analysis.mia_results import MIAResults
from tqdm import tqdm


logger: logging.Logger = logging.getLogger(__name__)

TimerStats = dict[str, float]


@dataclass
class AnalysisNodeOutput(BaseAnalysisOutput):
    """
    A dataclass to encapsulate the outputs of AnalsyisNode.
    Attributes:
        eps (float): Epsilon value at the TPR=1% UB or LB threshold depending on attack settings.
        eps_geo_split (float): Epsilon value at the TPR=1% UB threshold for geographical split.
        eps_lb (float): Lower bound of epsilon.
        eps_fpr_max_ub (float): Maximum upper bound of epsilon for FPR.
        eps_fpr_lb (List[float]): List of lower bound epsilon values for various FPRs.
        eps_fpr_ub (List[float]): List of upper bound epsilon values for various FPRs.
        eps_tpr_lb (List[float]): List of lower bound epsilon values for various TPRs.
        eps_tpr_ub (List[float]): List of upper bound epsilon values for various TPRs.
        eps_ci (float): Empirical epsilon computed using the Clopper-Pearson CI method.
        accuracy (float): Mean accuracy of the attack.
        accuracy_ci (List[float]): Confidence interval for the accuracy, represented as [lower_bound, upper_bound].
        auc (float): Mean area under the curve (AUC) of the attack.
        auc_ci (List[float]): Confidence interval for the AUC, represented as [lower_bound, upper_bound].
        train_size (int): Size of the training holdout set.
        test_size (int): Size of the testing holdout set.
        bootstrap_size (int): Size of the bootstrap sample used in the upper and lower bounds.
    """

    # Empirical epislons
    eps: float  # epsilon at TPR=1% UB threshold
    eps_geo_split: float  # epsilon at TPR=1% UB threshold
    eps_lb: float
    eps_fpr_max_ub: float
    eps_fpr_lb: List[float]
    eps_fpr_ub: List[float]
    eps_tpr_lb: List[float]
    eps_tpr_ub: List[float]
    eps_ci: float  # confidence interval
    # Accuracy and AUC
    accuracy: float
    accuracy_ci: List[float]  # confidence interval
    auc: float
    auc_ci: List[float]  # confidence interval
    # Dataset sizes
    data_size: dict[
        str, int
    ]  # size of the training, test dataset and bootstrap sample size


class AnalysisNode(BaseAnalysisNode):
    """
    AnalysisNode class for PrivacyGuard, which computes a general set of output metrics
    required to evaluate the performance of a privacy attack.

    Calculates privacy eval metrics by computing epsilons at the upper-bound of
    95% confidence interval with the top 1% TPR (True Postive Rates) threshold

    args:
        analysis_input: AnalysisInput object containing the training and testing dataframes
        delta: delta used in analysis calculations (close to 0)
        n_users_for_eval: number of users to use for computing the metrics
        num_bootstrap_resampling_times: length of array used to generate metric arrays
        use_upper_bound: boolean for whether to compute epsilon at the upper-bound of CI
            (true as default for geo-split; otherwise, use_lower_bound)
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
        self._delta = delta
        self._n_users_for_eval = n_users_for_eval
        self._num_bootstrap_resampling_times = num_bootstrap_resampling_times
        self._show_progress = show_progress
        self._with_timer = with_timer
        self._cap_eps = cap_eps

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
        num_users_for_eps_ci_eval = min(
            self._n_users_for_eval,
            score_train.shape[0] // 2,
            score_test.shape[0] // 2,
        )

        assert num_users_for_eps_ci_eval > 0 and num_users_for_eps_ci_eval < min(
            score_train.shape[0], score_test.shape[0]
        )

        loss_train = torch.from_numpy(
            score_train[:num_users_for_eps_ci_eval].to_numpy()
        )
        loss_test = torch.from_numpy(score_test[:num_users_for_eps_ci_eval].to_numpy())

        results = MIAResults(loss_train, loss_test)
        # compute one-off accuracy & AUC & CI for epsilon
        _, _, eps_ci = results.compute_acc_auc_ci_epsilon(self._delta)

        return eps_ci

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
            "eps_geo_split": epsilon at TPR=1% UB Threshold
            "eps_fpr_max_lb", "eps_fpr_lb", "eps_fpr_ub": epsilon at various false positive rate:
            "eps_tpr_lb", "eps_tpr_ub": epsilon at various true positive rates
            "eps_ci": epsilon calculate with Clopper-Pearson confidence interval
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

        # TODO: Change to a better name since CI implies a confidence interval output, but this is a float
        eps_ci = self._calculate_one_off_eps()

        logger.info(f"Epsilon CI: {eps_ci}")

        train_size, test_size = score_train.shape[0], score_test.shape[0]
        bootstrap_sample_size = min(train_size, test_size)

        with self.timer("make_metrics_array"):
            metrics_array = self._make_metrics_array()

        accuracy = np.array([run[0] for run in metrics_array])
        auc = np.array([run[1] for run in metrics_array])

        eps_fpr = np.array([run[2] for run in metrics_array])

        # get CI bounds with 95% confidence
        accuracy.sort()
        accuracy_mean = accuracy.mean()

        ci_lb_index = int(0.025 * self._num_bootstrap_resampling_times) - 1
        ci_ub_index = int(0.025 * self._num_bootstrap_resampling_times)
        accuracy_lb, accuracy_ub = accuracy[ci_lb_index], accuracy[-ci_ub_index]

        auc.sort()
        auc_mean = auc.mean()
        auc_lb, auc_ub = auc[ci_lb_index], auc[-ci_ub_index]

        eps_fpr.sort(0)
        eps_fpr_lb, eps_fpr_ub = eps_fpr[ci_lb_index, :], eps_fpr[-ci_ub_index, :]

        with self.timer("make_eps_tpr_array"):
            # compute eps at TPR thresholds (equally spaced at 1% intervals)
            eps_tpr_array = self._make_eps_tpr_array()
        # compute lb/ub of 95% CI for eps at TPR thresholds
        eps_tpr_array.sort(0)
        eps_tpr_lb, eps_tpr_ub = (
            eps_tpr_array[ci_lb_index, :],
            eps_tpr_array[-ci_ub_index, :],
        )

        eps_tpr_boundary = eps_tpr_ub if self._use_upper_bound else eps_tpr_lb

        outputs = AnalysisNodeOutput(
            eps=eps_tpr_boundary[0],  # epsilon at TPR=1% UB threshold
            eps_geo_split=eps_tpr_ub[0],  # epsilon at TPR=1% UB threshold
            eps_lb=eps_tpr_lb[0],
            eps_fpr_max_ub=np.nanmax(eps_fpr_ub),
            eps_fpr_lb=list(eps_fpr_lb),
            eps_fpr_ub=list(eps_fpr_ub),
            eps_tpr_lb=list(eps_tpr_lb),
            eps_tpr_ub=list(eps_tpr_ub),
            eps_ci=eps_ci,
            accuracy=accuracy_mean,
            accuracy_ci=[accuracy_lb, accuracy_ub],
            auc=auc_mean,
            auc_ci=[auc_lb, auc_ub],
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
    ) -> list[tuple[np.float64, np.float64, list[np.float64], list[np.float64]]]:
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
                verbose=False,
            )
            for _ in tqdm(
                range(self._num_bootstrap_resampling_times),
                disable=not self._show_progress,
            )
        ]
        return metrics_array

    def _make_eps_tpr_array(
        self,
        # pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
    ) -> np.ndarray:
        """
        Make list of epsilon at TPR threshold values
        Returns:
            np.ndarray of float values
        """
        score_train = self.analysis_input.df_train_user["score"]
        score_test = self.analysis_input.df_test_user["score"]

        # error thresholds set equally spaced at 1% intervals
        loss_train = torch.from_numpy(score_train.to_numpy())
        loss_test = torch.from_numpy(score_test.to_numpy())
        train_size, test_size = score_train.shape[0], score_test.shape[0]

        bootstrap_sample_size = min(train_size, test_size)

        error_thresholds = np.linspace(0.01, 1, 100)

        eps_tpr_array = np.array(
            [
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
                ).compute_eps_at_tpr_threshold(
                    self._delta,
                    tpr_threshold=error_thresholds,
                    cap_eps=self._cap_eps,
                    verbose=False,
                )
                for _ in tqdm(
                    range(self._num_bootstrap_resampling_times),
                    disable=not self._show_progress,
                )
            ]
        )
        return eps_tpr_array

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
