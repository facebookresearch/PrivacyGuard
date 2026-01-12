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
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from numpy.typing import NDArray
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
        eps_cp (float): Empirical epsilon computed using the Clopper-Pearson CI method.
        eps_mean (float): Mean epsilon value.
        eps_mean_lb (float): Lower bound of the mean epsilon value.
        eps_mean_ub (float): Upper bound of the mean epsilon value.
        accuracy (float): Mean accuracy of the attack.
        accuracy_ci (List[float]): Confidence interval for the accuracy, represented as [lower_bound, upper_bound].
        auc (float): Mean area under the curve (AUC) of the attack.
        auc_ci (List[float]): Confidence interval for the AUC, represented as [lower_bound, upper_bound].
        data_size (dict[str, int]): Size of the training, test dataset and bootstrap sample size.
    """

    eps: float  # epsilon at TPR=1% UB threshold
    eps_fpr_lb: List[float]
    eps_fpr_ub: List[float]
    eps_cp: float
    eps_mean: float
    eps_mean_lb: float
    eps_mean_ub: float
    accuracy: float
    accuracy_ci: List[float]
    auc: float
    auc_ci: List[float]
    data_size: dict[
        str, int
    ]  # size of the training, test dataset and bootstrap sample size


def compute_metric_mean_with_ci(
    metric_array: NDArray[float],
) -> tuple[float, float, float]:
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

    def run_analysis(self) -> BaseAnalysisOutput:
        """
        Computes analysis outputs based on the input dataframes.
        Overrides "BaseAnalysisNode::run_analysis"

        First, makes loss_train and loss_test and computes
        one off metrics like
        epsilon confidence intervals.

        Then, uses "_make_metrics_array" to build lists of
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
            "eps_cp": epsilon calculate with Clopper-Pearson confidence interval:
            "accuracy", "accuracy_ci": accuracy values
            "auc", "auc_ci": area under ROC curve values
            "data_size": dictionary with keys "train_size", "test_size", "bootstrap_size"
        """
        df_train_user = self.analysis_input.df_train_user
        df_test_user = self.analysis_input.df_test_user
        score_train = df_train_user["score"]
        score_test = df_test_user["score"]

        train_size, test_size = score_train.shape[0], score_test.shape[0]

        eps_cp = self._calculate_one_off_eps()  # inherited from AnalysisNode

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

        # compute eps at FPR thresholds (equally spaced at 1% intervals)
        with self.timer("make_epsilon_at_error_thresholds_array"):
            metrics_array = self._make_metrics_array()  # inherited from AnalysisNode
        # compute lb/ub of 95% CI for eps at FPR thresholds
        eps_fpr = np.array([run[2] for run in metrics_array])
        eps_fpr_lb, eps_fpr_ub = self._compute_ci(eps_fpr)

        outputs = FPRLowerBoundAnalysisNodeOutput(
            eps=np.nanmax(eps_fpr_ub),
            eps_fpr_lb=list(eps_fpr_lb),
            eps_fpr_ub=list(eps_fpr_ub),
            eps_cp=eps_cp,
            eps_mean=eps_mean,
            eps_mean_lb=eps_mean_lb,
            eps_mean_ub=eps_mean_ub,
            accuracy=accuracy_mean,
            accuracy_ci=[accuracy_mean_lb, accuracy_mean_ub],
            auc=auc_mean,
            auc_ci=[auc_mean_lb, auc_mean_ub],
            data_size={
                "train_size": train_size,
                "test_size": test_size,
                "bootstrap_size": bootstrap_sample_size,
            },
        )

        if self._with_timer:
            logger.info(f"Timer stats: {self.get_timer_stats()}")

        return outputs

    def _make_acc_auc_epsilon_array(self) -> NDArray[float]:
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
