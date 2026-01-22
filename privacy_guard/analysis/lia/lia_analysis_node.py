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

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from privacy_guard.analysis.base_analysis_output import BaseAnalysisOutput
from privacy_guard.analysis.lia.lia_analysis_input import LIAAnalysisInput
from privacy_guard.analysis.mia.analysis_node import AnalysisNode
from privacy_guard.analysis.mia.mia_results import MIAResults
from tqdm import tqdm

logger: logging.Logger = logging.getLogger(__name__)

TimerStats = dict[str, float]


@dataclass
class LIAAnalysisOutput(BaseAnalysisOutput):
    """
    A dataclass to encapsulate the outputs of LIAAnalysisNode.
    """

    eps: float  # epsilon UB (highest across all error thresholds)
    eps_lb: float  # LB associated with UB epsilon
    # Accuracy and AUC
    accuracy: float
    accuracy_ci: List[float]  # confidence interval
    auc: float
    auc_ci: List[float]  # confidence interval
    error_rate_at_max_eps: float  # error rate at which max eps upper bound is achieved
    eps_max_bounds: Tuple[
        List[float], List[float]
    ]  # eps LB and UB at TPR and FPR thresholds (eps=max(eps_tpr, eps_fpr))
    eps_at_tpr_bounds: Tuple[
        List[float], List[float]
    ]  # eps LB and UB at TPR thresholds
    eps_at_fpr_bounds: Tuple[
        List[float], List[float]
    ]  # eps LB and UB at FPR thresholds
    # Dataset size
    data_size: int
    label_mean: float
    prediction_mean: float
    prediction_y1_generation_mean: float


class LIAAnalysisNode(AnalysisNode):
    def __init__(
        self,
        analysis_input: LIAAnalysisInput,
        delta: float,
        num_bootstrap_resampling_times: int = 10,
        cap_eps: bool = True,
        show_progress: bool = False,
        with_timer: bool = False,
        power: float = 0.0,
        use_fnr_and_tnr: bool = False,
    ) -> None:
        if power < 0:
            raise ValueError("Power used for score function must be non-negative")

        self._delta = delta
        self._num_bootstrap_resampling_times = num_bootstrap_resampling_times
        self._cap_eps = cap_eps
        self._show_progress = show_progress
        self._with_timer = with_timer
        self._analysis_input = analysis_input
        self._timer_stats: dict[str, float] = {}
        self._power = power
        self._use_fnr_and_tnr = use_fnr_and_tnr

    def compute_scores(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scores for a given game instantiation.

        Args:
            i (int): index of the game instantiation (i.e. the i-th game)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: scores for samples with training labels and reconstructed labels
        """

        true_bits = self._analysis_input.true_bits[i]

        received_labels = self._analysis_input.received_labels[i]
        y1_probs = self._analysis_input.predictions_y1_generation
        predictions = self._analysis_input.predictions

        prob_train = np.where(received_labels == 1, predictions, 1 - predictions)
        prob_reconstruct = np.where(received_labels == 1, y1_probs, 1 - y1_probs)
        prob_diff_label = np.where(received_labels == 1, 1 - y1_probs, y1_probs)

        scores = (
            np.log(prob_train + 1e-8) - np.log(prob_reconstruct + 1e-8)
        ) * prob_diff_label**self._power

        scores_train = torch.tensor(scores[true_bits == 0])
        scores_test = torch.tensor(scores[true_bits == 1])

        return scores_train, scores_test

    def run_analysis(self) -> BaseAnalysisOutput:
        """Run LIA analysis"""

        error_thresholds = np.linspace(0.01, 1, 100)
        num_resampling = self._analysis_input.y1.shape[0]
        num_samples = self._analysis_input.y1.shape[1]

        # run analysis for each game instance
        all_metrics = []
        with self.timer("compute all metrics"):
            for i in tqdm(range(num_resampling), disable=not self._show_progress):
                scores_train, scores_test = self.compute_scores(i)
                train_size, test_size = scores_train.shape[0], scores_test.shape[0]
                bootstrap_sample_size = min(train_size, test_size)
                for _ in range(self._num_bootstrap_resampling_times):
                    indices_train = AnalysisNode._compute_bootstrap_sample_indexes(
                        train_size, bootstrap_sample_size
                    )
                    indices_test = AnalysisNode._compute_bootstrap_sample_indexes(
                        test_size, bootstrap_sample_size
                    )
                    lia_results = MIAResults(
                        scores_train=scores_train[indices_train],
                        scores_test=scores_test[indices_test],
                    )

                    # metrics is a tuple: (accuracy, auc_value, eps_fpr_array, eps_tpr_array, eps_max_array)
                    metrics = lia_results.compute_metrics_at_error_threshold(
                        delta=self._delta,
                        error_threshold=error_thresholds,
                        cap_eps=self._cap_eps,
                        verbose=self._show_progress,
                        use_fnr_tnr=self._use_fnr_and_tnr,
                    )

                    all_metrics.append(metrics)

        all_accuracy_values = np.array([run[0] for run in all_metrics])
        all_auc_values = np.array([run[1] for run in all_metrics])
        all_eps_fpr_values = np.array([run[2] for run in all_metrics])
        all_eps_tpr_values = np.array([run[3] for run in all_metrics])
        all_eps_values = np.array([run[4] for run in all_metrics])

        # Compute upper bounds (95th percentile) for each error_threshold
        eps_lb_per_threshold, eps_ub_per_threshold = self._compute_ci(all_eps_values)
        # Find the maximum eps_ub across all error thresholds
        idx = np.argmax(eps_ub_per_threshold)

        error_rate_at_max_eps = error_thresholds[idx]

        eps_max_ub = eps_ub_per_threshold[idx]
        eps_lb_at_max_ub = eps_lb_per_threshold[idx]

        # Compute lb/ub for accuracy and auc
        accuracy_lb, accuracy_ub = self._compute_ci(np.array(all_accuracy_values))
        auc_lb, auc_ub = self._compute_ci(np.array(all_auc_values))

        # Compute lb/ub for eps computed using only TPR or only FPR thresholds
        eps_tpr_lb, eps_tpr_ub = self._compute_ci(np.array(all_eps_tpr_values))
        eps_fpr_lb, eps_fpr_ub = self._compute_ci(np.array(all_eps_fpr_values))

        return LIAAnalysisOutput(
            eps=float(eps_max_ub),
            eps_lb=float(eps_lb_at_max_ub),
            accuracy=np.mean(all_accuracy_values),
            accuracy_ci=[accuracy_lb[0], accuracy_ub[0]],
            auc=np.mean(all_auc_values),
            auc_ci=[auc_lb[0], auc_ub[0]],
            error_rate_at_max_eps=error_rate_at_max_eps,
            eps_max_bounds=(list(eps_lb_per_threshold), list(eps_ub_per_threshold)),
            eps_at_tpr_bounds=(list(eps_tpr_lb), list(eps_tpr_ub)),
            eps_at_fpr_bounds=(list(eps_fpr_lb), list(eps_fpr_ub)),
            data_size=num_samples,
            label_mean=np.mean(self._analysis_input.y0),
            prediction_mean=np.mean(self._analysis_input.predictions),
            prediction_y1_generation_mean=np.mean(
                self._analysis_input.predictions_y1_generation
            ),
        )
