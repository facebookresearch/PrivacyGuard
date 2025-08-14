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
from privacy_guard.analysis.mia.analysis_node import AnalysisNode, AnalysisNodeOutput
from privacy_guard.analysis.mia.mia_results import MIAResults


logger: logging.Logger = logging.getLogger(__name__)

TimerStats = dict[str, float]


class ParallelAnalysisNode(AnalysisNode):
    """
    ParallelAnalysisNode class for PrivacyGuard, which is an extension of AnalysisNode to parallelize the epsilon computation bootstrapping process.
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

    def _compute_metrics_and_eps_tpr_array(
        self, args: tuple[str, str, float, int]
    ) -> tuple[list[tuple[float, float, list[float], list[float]]], list[npt.NDArray]]:
        """
        Make a tuple with two lists:
        -- A list of tuples metrics at error thresholds, each of which contains the
        accuracy, AUC, and epsilon values for a given number of samples
        The tuples are randomly generated from permutations of subsets of loss_train
        and loss_test.
        -- A list of epsilons at error thresholds

        Args:
            A tuple of train/test filenames, delta, and chunk size (number of bootstrap resampling times)
        Returns:
            A tuple of a List[Tuple] with elements (accuracy, auc_value, eps_fpr_array, eps_max_array)
            and a List[np.ndarray] of float values (epsilons at error thresholds)
        """

        train_filename, test_filename, delta, chunk_size = args
        loss_train: torch.Tensor = torch.load(train_filename, weights_only=True)
        loss_test: torch.Tensor = torch.load(test_filename, weights_only=True)
        train_size, test_size = loss_train.shape[0], loss_test.shape[0]
        bootstrap_sample_size = min(train_size, test_size)
        error_thresholds = np.linspace(0.01, 1, 100)
        metrics_results = []
        eps_tpr_results = []

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

                metrics_result = mia_results.compute_metrics_at_error_threshold(
                    self._delta, error_threshold=error_thresholds, verbose=False
                )
                eps_tpr_result = mia_results.compute_eps_at_tpr_threshold(
                    self._delta, tpr_threshold=error_thresholds, verbose=False
                )

                metrics_results.append(metrics_result)
                eps_tpr_results.append(eps_tpr_result)
        except Exception as e:
            logger.info(
                f"An exception occurred when computing acc/auc/epsilon metrics: {e}"
            )

        return metrics_results, eps_tpr_results

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

    def run_analysis(self) -> AnalysisNodeOutput:
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
        score_train = self.analysis_input.df_train_user["score"]
        score_test = self.analysis_input.df_test_user["score"]
        loss_train = torch.from_numpy(score_train.to_numpy())
        loss_test = torch.from_numpy(score_test.to_numpy())
        train_size, test_size = loss_train.shape[0], loss_test.shape[0]
        bootstrap_sample_size = min(train_size, test_size)
        logger.info(f"Train/Test unique users: {train_size}/{test_size}")

        metrics_array = []
        eps_tpr_array = []

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
                        self._compute_metrics_and_eps_tpr_array,
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

                for metrics_result, eps_tpr_result in results:
                    metrics_array.extend(metrics_result)
                    eps_tpr_array.extend(eps_tpr_result)

        eps_tpr_array = np.array(eps_tpr_array)
        logger.info(
            f"epsilon at TPR thresholds: eps_tpr_array shape {eps_tpr_array.shape} - has NaNs: {np.isnan(eps_tpr_array).any()}"
        )

        accuracy = np.array([run[0] for run in metrics_array])
        auc = np.array([run[1] for run in metrics_array])

        eps_fpr = np.array([run[2] for run in metrics_array])

        # get CI bounds with 95% confidence
        accuracy.sort()
        accuracy_mean = accuracy.mean()
        accuracy_lb, accuracy_ub = accuracy[24], accuracy[-25]

        auc.sort()
        auc_mean = auc.mean()
        auc_lb, auc_ub = auc[24], auc[-25]

        eps_fpr.sort(0)
        eps_fpr_lb, eps_fpr_ub = eps_fpr[24, :], eps_fpr[-25, :]

        # compute lb/ub of 95% CI for eps at TPR thresholds
        eps_tpr_array.sort(0)
        eps_tpr_lb, eps_tpr_ub = eps_tpr_array[24, :], eps_tpr_array[-25, :]

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
            eps_ci=float(
                "nan"
            ),  # TODO: compute eps_ci properly, currently not computed in ParallelAnalysisNode
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
