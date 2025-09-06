# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from privacy_guard.analysis.base_analysis_output import BaseAnalysisOutput
from privacy_guard.analysis.mia.analysis_node import AnalysisNode

from privacy_guard.analysis.mia.mia_results import MIAResults

# pyre-ignore:Undefined import [21]: Could not find a name `bootstrap` defined in module `scipy.stats`.
from scipy.stats import bootstrap
from tqdm import tqdm

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class FPRLowerBoundAnalysisNodeOutput(BaseAnalysisOutput):
    """
    A dataclass to encapsulate the outputs of ParallelAnalysisNode.
    Attributes:
        eps (float): Epsilon value at the TPR=1% UB threshold.
        eps_fpr_lb (List[float]): List of lower bound epsilon values at various FPRs.
        eps_fpr_ub (List[float]): List of upper bound epsilon values at various FPRs.
        eps_ci (float): Empirical epsilon computed using the Clopper-Pearson CI method.
        eps_mean (float): Mean epsilon value.
        eps_mean_lb (float): Lower bound of the mean epsilon value.
        eps_mean_ub (float): Upper bound of the mean epsilon value.
        accuracy (float): Mean accuracy of the attack.
        accuracy_lb (float): Lower bound of the accuracy.
        accuracy_ub (float): Upper bound of the accuracy.
        acu (float): Mean area under the curve (AUC) of the attack.
        auc_lb (float): Lower bound of the AUC.
        auc_ub (float): Upper bound of the AUC.
        data_size (dict[str, int]): Size of the training, test dataset and bootstrap sample size.
    """

    eps: float  # epsilon at TPR=1% UB threshold
    eps_fpr_lb: List[float]
    eps_fpr_ub: List[float]
    eps_ci: float
    eps_mean: float
    eps_mean_lb: float
    eps_mean_ub: float
    accuracy: float
    accuracy_lb: float
    accuracy_ub: float
    auc: float
    auc_lb: float
    auc_ub: float
    data_size: dict[
        str, int
    ]  # size of the training, test dataset and bootstrap sample size


# pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
def compute_metric_mean_with_ci(metric_array: np.ndarray) -> tuple[float, float, float]:
    # TODO: Identify descriptive values for mean, lb, ub when bootstrap fails

    metric_mean = metric_array.mean()
    metric_mean_lb, metric_mean_ub = 0, 0
    try:
        # pyre-ignore: Module `scipy.stats` has no attribute `bootstrap`.
        metric_mean_lb, metric_mean_ub = bootstrap(
            (metric_array,), statistic=np.mean, method="BCa"
        ).confidence_interval
    except Exception as ex:
        logger.info(f"An exception occurred when computing CI: {str(ex)}")
    return metric_mean, metric_mean_lb, metric_mean_ub


class FPRLowerBoundAnalysisNode(AnalysisNode):
    """
    FPRLowerBoundAnalysisNode class for PrivacyGuard, which computes a general set of output metrics
    required to evaluate the performance of a privacy attack.

    Calculate privacy eval metrics by computing epsilons at the lower-bound of 95% confidence interval
    with FPR (False Positive Rates) thresholds.

    args:
        analysis_input: AnalysisInput object containing the training and testing dataframes
        delta: delta used in analysis calculations (close to 0)
        n_users_for_eval: number of users to use for computing the metrics
        num_bootstrap_resampling_times: length of array used to generate metric arrays
        use_upper_bound: boolean for whether to compute epsilon at the upper-bound of CI
    """

    # lower bound index of 95% confidence interval (based on 1000 data points)
    LB_INDEX_OF_95_PCT_CI = 24

    # upper bound index of 95% confidence interval (based on 1000 data points)
    UB_INDEX_OF_95_PCT_CI = -25

    def run_analysis(self) -> BaseAnalysisOutput:
        """
        Computes analysis outputs based on the input dataframes.
        Overrides "BaseAnalysisNode::run_analysis"

        First, makes loss_train and loss_test and computes
        one off metrics like
        epsilon confidence intervals.

        Then, uses "make_acc_auc_epsilon_array" to build lists of
        metrics, each computed from random subsets of
        loss_train and loss_test. The length of these lists
        is determined by self._num_bootstrap_resampling_times

        These metrics are combined into the output of this analysis,
        and returned from the call.

        Returns:
            FPRLowerBoundAnalysisNodeOutput dataclass with fields:
            "eps": empirical epsilon calculated as highest epsilon upper bound from the FPR thresholds
            "eps_fpr_max_lb", "eps_fpr_lb", "eps_fpr_ub": epsilon for various false positive rate:
            "eps_tpr_lb", "eps_tpr_ub": epsilon for various true positive rates:
            "eps_ci": epsilon calculate with Clopper-Pearson confidence interval:
            "accuracy", "accuracy_lb", "accuracy_ub": accuracy values
            "auc", "auc_lb", "auc_ub": area under ROC curve values
            "data_size": dictionary with keys "train_size", "test_size", "bootstrap_size"
        """
        df_train_user = self.analysis_input.df_train_user
        df_test_user = self.analysis_input.df_test_user
        score_train = df_train_user["score"]
        score_test = df_test_user["score"]

        train_size, test_size = score_train.shape[0], score_test.shape[0]

        eps_ci = self._calculate_one_off_eps()  # inherited from AnalysisNode

        train_size, test_size = score_train.shape[0], score_test.shape[0]
        bootstrap_sample_size = min(train_size, test_size)

        with self.timer("make_acc_auc_epsilon_array"):
            metrics_array = self._make_acc_auc_epsilon_array()

        accuracy_array, auc_array, eps_array = (metrics_array[:, i] for i in range(3))
        auc_mean, auc_mean_lb, auc_mean_ub = compute_metric_mean_with_ci(
            metric_array=auc_array
        )
        eps_mean, eps_mean_lb, eps_mean_ub = compute_metric_mean_with_ci(
            metric_array=eps_array
        )
        accuracy_mean, accuracy_mean_lb, accuracy_mean_ub = compute_metric_mean_with_ci(
            metric_array=accuracy_array
        )

        # compute eps at TPR thresholds (equally spaced at 1% intervals)
        with self.timer("make_epsilon_at_error_thresholds_array"):
            metrics_array = self._make_epsilon_at_error_thresholds_array()
        # compute lb/ub of 95% CI for eps at TPR thresholds
        eps_fpr = np.array([run[0] for run in metrics_array])
        eps_fpr.sort(0)
        # get CI bounds with 95% confidence

        ci_lb_index = int(0.025 * self._num_bootstrap_resampling_times) - 1
        ci_ub_index = int(0.025 * self._num_bootstrap_resampling_times)

        eps_fpr_lb, eps_fpr_ub = (
            eps_fpr[ci_lb_index, :],
            eps_fpr[-ci_ub_index, :],
        )

        # TODO: Switch to fields auc_ci = [auc_lb, auc_ub] and accuracy_ci = [accuracy_lb, accuracy_ub] similar to AnalysisNodeOutput
        outputs = FPRLowerBoundAnalysisNodeOutput(
            eps=np.nanmax(eps_fpr_ub),
            eps_fpr_lb=list(eps_fpr_lb),
            eps_fpr_ub=list(eps_fpr_ub),
            eps_ci=eps_ci,
            eps_mean=eps_mean,
            eps_mean_lb=eps_mean_lb,
            eps_mean_ub=eps_mean_ub,
            accuracy=accuracy_mean,
            accuracy_lb=accuracy_mean_lb,
            accuracy_ub=accuracy_mean_ub,
            auc=auc_mean,
            auc_lb=auc_mean_lb,
            auc_ub=auc_mean_ub,
            data_size={
                "train_size": train_size,
                "test_size": test_size,
                "bootstrap_size": bootstrap_sample_size,
            },
        )

        if self._with_timer:
            logger.info(f"Timer stats: {self.get_timer_stats()}")

        return outputs

    def _make_acc_auc_epsilon_array(
        self,
        # pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
    ) -> np.ndarray:
        """
        Make list of tuples metrics at error thresholds, each of which contains the
        accuracy, AUC, and epsilon values for a given number of samples
        The tuples are randomly generated from permutations of subsets of loss_train
        and loss_test.

        Returns:
            List[Tuple] with elements
                (accuracy, AUC, empirical epsilon)
        """
        score_train = self.analysis_input.df_train_user["score"]
        score_test = self.analysis_input.df_test_user["score"]

        loss_train = torch.tensor(score_train.values)
        loss_test = torch.tensor(score_test.values)

        train_size, test_size = score_train.shape[0], score_test.shape[0]
        bootstrap_sample_size = min(train_size, test_size)

        metrics_array = np.array(
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
                ).compute_acc_auc_epsilon(delta=self._delta)
                for _ in tqdm(
                    range(self._num_bootstrap_resampling_times),
                    disable=not self._show_progress,
                )
            ]
        )
        return metrics_array

    def _make_epsilon_at_error_thresholds_array(
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

        loss_train = torch.tensor(score_train.values)
        loss_test = torch.tensor(score_test.values)
        # error thresholds set equally spaced at 1% intervals
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
                ).compute_epsilon_at_error_thresholds(
                    self._delta,
                    # pyre-fixme[6]: For 2nd argument expected `List[float]` but got
                    #  `ndarray[typing.Any, dtype[typing.Any]]`.
                    error_thresholds=error_thresholds,
                )
                for _ in tqdm(
                    range(self._num_bootstrap_resampling_times),
                    disable=not self._show_progress,
                )
            ]
        )
        return eps_tpr_array
