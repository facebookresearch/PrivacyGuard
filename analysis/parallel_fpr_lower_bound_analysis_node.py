# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
import logging
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import numpy.typing as npt
import torch
from privacy_guard.analysis.base_analysis_input import BaseAnalysisInput
from privacy_guard.analysis.base_analysis_output import BaseAnalysisOutput
from privacy_guard.analysis.fpr_lower_bound_analysis_node import (
    FPRLowerBoundAnalysisNode,
    FPRLowerBoundAnalysisNodeOutput,
)

from privacy_guard.analysis.mia_results import MIAResults

# pyre-ignore:Undefined import [21]: Could not find a name `bootstrap` defined in module `scipy.stats`.
from scipy.stats import bootstrap

logger: logging.Logger = logging.getLogger(__name__)


def compute_metric_mean_with_ci(
    metric_array: npt.NDArray,
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


class ParallelFPRLowerBoundAnalysisNode(FPRLowerBoundAnalysisNode):
    """
    ParallelFPRLowerBoundAnalysisNode class for PrivacyGuard, which is an extension of FPRLowerBoundAnalysisNode to parallelize the epsilon computation bootstrapping process.
    """

    def __init__(
        self,
        analysis_input: BaseAnalysisInput,
        delta: float,
        n_users_for_eval: int,
        eps_computation_tasks_num: int,
        use_upper_bound: bool = True,
        num_bootstrap_resampling_times: int = 1000,
        with_timer: bool = False,
    ) -> None:
        super().__init__(
            analysis_input=analysis_input,
            delta=delta,
            n_users_for_eval=n_users_for_eval,
            use_upper_bound=use_upper_bound,
            num_bootstrap_resampling_times=num_bootstrap_resampling_times,
            with_timer=with_timer,
        )
        self._eps_computation_tasks_num = eps_computation_tasks_num

    def _compute_metrics_and_eps_fpr_array(
        self, args: tuple[str, str, float, int]
    ) -> tuple[list[npt.NDArray], list[npt.NDArray]]:
        """
        Make a tuple with two lists:
        -- A list of tuples metrics at error thresholds, each of which contains the
        accuracy, AUC, and epsilon values for a given number of samples.
        The tuples are randomly generated from permutations of subsets of loss_train
        and loss_test.
        -- A list of epsilons at error thresholds

        Args:
            A tuple of train/test filenames, delta, and chunk size (number of bootstrap resampling times)
        Returns:
            A tuple of a List[Tuple] with elements (accuracy, AUC, empirical epsilon)
            and a List[np.ndarray] of float values (epsilons at error thresholds)
        """

        train_filename, test_filename, delta, chunk_size = args
        loss_train: torch.Tensor = torch.load(train_filename, weights_only=True)
        loss_test: torch.Tensor = torch.load(test_filename, weights_only=True)
        train_size, test_size = loss_train.shape[0], loss_test.shape[0]
        bootstrap_sample_size = min(train_size, test_size)
        error_thresholds = np.linspace(0.01, 1, 100)
        metrics_results = []
        eps_fpr_results = []

        try:
            for _ in range(chunk_size):
                mia_results = MIAResults(
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
                )

                metrics_result = mia_results.compute_acc_auc_epsilon(delta=self._delta)
                eps_fpr_result = mia_results.compute_epsilon_at_error_thresholds(
                    self._delta, error_thresholds=error_thresholds
                )

                metrics_results.append(metrics_result)
                eps_fpr_results.append(eps_fpr_result)
        except Exception as e:
            logger.info(
                f"An exception occurred when computing acc/auc/epsilon metrics: {e}"
            )

        return metrics_results, eps_fpr_results

    def _parallel_compute_chunk_sizes(self, task_num: int) -> list[int]:
        """
        Compute chunk sizes for parallel computation given a task number

        Args:
            task_num (int): number of tasks for parallel computation
        Returns:
            A list of chunk sizes
        """

        base_chunk_size = self._num_bootstrap_resampling_times // task_num
        chunk_sizes = [base_chunk_size] * task_num
        if (remainder := self._num_bootstrap_resampling_times % task_num) > 0:
            for index in range(remainder):
                chunk_sizes[index] += 1

        return chunk_sizes

    def run_analysis(self) -> BaseAnalysisOutput:
        """
        Computes analysis outputs based on the input dataframes, using parallel computation.
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

        score_train = self.analysis_input.df_train_user["score"]
        score_test = self.analysis_input.df_test_user["score"]
        loss_train = torch.from_numpy(score_train.to_numpy())
        loss_test = torch.from_numpy(score_test.to_numpy())
        train_size, test_size = loss_train.shape[0], loss_test.shape[0]
        bootstrap_sample_size = min(train_size, test_size)
        logger.info(f"Train/Test unique users: {train_size}/{test_size}")

        eps_ci = self._calculate_one_off_eps()  # inherited from AnalysisNode

        metrics_array = []
        eps_fpr_array = []

        with self.timer("parallel_bootstrap"):
            with tempfile.TemporaryDirectory() as temp_dir:
                train_filename = os.path.join(temp_dir, "train_scores.pt")
                test_filename = os.path.join(temp_dir, "test_scores.pt")
                torch.save(loss_train, train_filename)
                torch.save(loss_test, test_filename)
                chunk_sizes = self._parallel_compute_chunk_sizes(
                    self._eps_computation_tasks_num
                )

                with ProcessPoolExecutor(
                    max_workers=self._eps_computation_tasks_num
                ) as pool:
                    results = pool.map(
                        self._compute_metrics_and_eps_fpr_array,
                        [
                            (
                                train_filename,
                                test_filename,
                                self._delta,
                                chunk_size,
                            )
                            for chunk_size in chunk_sizes
                        ],
                    )

                for metrics_result, eps_fpr_result in results:
                    metrics_array.extend(metrics_result)
                    eps_fpr_array.extend(eps_fpr_result)

        metrics_array = np.array(metrics_array)
        eps_fpr_array = np.array(eps_fpr_array)
        logger.info(
            f"epsilon at FPR thresholds: eps_tpr_array shape {eps_fpr_array.shape} - has NaNs: {np.isnan(eps_fpr_array).any()}"
        )

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

        # compute lb/ub of 95% CI for eps at TPR thresholds
        eps_fpr = np.array([run[0] for run in eps_fpr_array])
        eps_fpr.sort(0)
        # get CI bounds with 95% confidence
        eps_fpr_lb, eps_fpr_ub = (
            eps_fpr[self.LB_INDEX_OF_95_PCT_CI, :],
            eps_fpr[self.UB_INDEX_OF_95_PCT_CI, :],
        )

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
