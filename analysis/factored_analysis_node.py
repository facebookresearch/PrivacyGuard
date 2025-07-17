# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from privacy_guard.analysis.mia_results import MIAResults

logger: logging.Logger = logging.getLogger(__name__)


class FactoredAnalysisNode:
    @staticmethod
    def _compute_bootstrap_sample_indexes(
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

    @staticmethod
    def compute_partial_results(
        args: tuple[str, str, float, int],
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
        (train_filename, test_filename, delta, chunk_size) = args
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
                        FactoredAnalysisNode._compute_bootstrap_sample_indexes(
                            train_size, bootstrap_sample_size
                        )
                    ],
                    loss_test[
                        FactoredAnalysisNode._compute_bootstrap_sample_indexes(
                            test_size, bootstrap_sample_size
                        )
                    ],
                )

                metrics_result = mia_results.compute_metrics_at_error_threshold(
                    delta, error_threshold=error_thresholds, verbose=False
                )
                eps_tpr_result = mia_results.compute_eps_at_tpr_threshold(
                    delta, tpr_threshold=error_thresholds, verbose=False
                )

                metrics_results.append(metrics_result)
                eps_tpr_results.append(eps_tpr_result)
        except Exception as e:
            logger.info(
                f"An exception occurred when computing acc/auc/epsilon metrics: {e}"
            )

        return metrics_results, eps_tpr_results

    @staticmethod
    def merge_results(
        use_upper_bound: bool,
        metrics_array: list[tuple[float, float, list[float], list[float]]],
        eps_tpr_array_arg: list[npt.NDArray],
    ) -> dict[str, Any]:
        eps_tpr_array = np.array(eps_tpr_array_arg)
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

        eps_tpr_boundary = eps_tpr_ub if use_upper_bound else eps_tpr_lb

        # TODO: Consider returning a structured dataclass instead of a dict;
        # having the outputs be typed should go a long way towards keeping bugs
        # at bay
        outputs = {
            "eps": eps_tpr_boundary[0],  # epsilon at TPR=1% UB threshold
            "eps_geo_split": eps_tpr_ub[0],  # epsilon at TPR=1% UB threshold
            "eps_lb": eps_tpr_lb[0],
            "eps_fpr_max_ub": np.nanmax(eps_fpr_ub),
            "eps_fpr_lb": list(eps_fpr_lb),
            "eps_fpr_ub": list(eps_fpr_ub),
            "eps_tpr_lb": list(eps_tpr_lb),
            "eps_tpr_ub": list(eps_tpr_ub),
            "accuracy": accuracy_mean,
            "accuracy_ci": [accuracy_lb, accuracy_ub],
            "auc": auc_mean,
            "auc_ci": [auc_lb, auc_ub],
        }

        return outputs
