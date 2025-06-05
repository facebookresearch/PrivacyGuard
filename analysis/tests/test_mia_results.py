# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict


import logging
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import torch
from privacy_guard.analysis.mia_results import MIAResults
from testfixtures import LogCapture
from windtunnel.lib.unittest_utils import test_log_assert


class TestMiaResults(unittest.TestCase):
    def setUp(self) -> None:
        self.scores_train = torch.rand(100)
        self.scores_test = torch.rand(100)

        self.mia_results = MIAResults(self.scores_train, self.scores_test)
        super().setUp()

    def test_mia_result_init(self) -> None:
        with self.assertRaises(AssertionError):
            _ = MIAResults(
                scores_train=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
                scores_test=torch.tensor([0.1, 0.2]),
            )
        with self.assertRaises(AssertionError):
            _ = MIAResults(
                scores_train=torch.tensor([0.1, 0.2]),
                scores_test=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
            )
        with self.assertRaises(AssertionError):
            _ = MIAResults(
                scores_train=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
                scores_test=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
            )

        _ = MIAResults(
            scores_train=torch.tensor([0.1, 0.2]), scores_test=torch.tensor([0.1, 0.2])
        )

    def test_get_scores_and_labels_ordered(self) -> None:
        (
            labels_ordered,
            scores_ordered,
        ) = self.mia_results._get_scores_and_labels_ordered()
        self.assertEqual(
            len(labels_ordered),
            len(scores_ordered),
            len(self.scores_train) + len(self.scores_test),
        )

    def test_get_tpr_fpr(self) -> None:
        _, _ = self.mia_results.get_tpr_fpr()

    def test_clopper_pearson_typical_passes(self) -> None:
        ci_low, ci_upp = MIAResults._clopper_pearson(count=2, trials=100, conf=0.5)
        self.assertIsInstance(ci_low, float)
        self.assertIsInstance(ci_upp, float)
        self.assertLessEqual(ci_low, ci_upp)

    def test_clopper_pearson_count_gt_trials_nan(self) -> None:
        ci_low, ci_upp = MIAResults._clopper_pearson(count=200, trials=100, conf=0.5)
        self.assertTrue(np.isnan(ci_low))
        self.assertTrue(np.isnan(ci_upp))

    def test_clopper_pearson_count_is_zero_edge_case(self) -> None:
        ci_low, ci_upp = MIAResults._clopper_pearson(count=0, trials=100, conf=0.5)

        self.assertIsInstance(ci_low, float)
        self.assertIsInstance(ci_upp, float)
        self.assertLessEqual(0, ci_upp)
        self.assertEqual(ci_low, 0)

    def test_clopper_pearson_count_eq_trials_edge_case(self) -> None:
        ci_low, ci_upp = MIAResults._clopper_pearson(count=100, trials=100, conf=0.5)

        self.assertIsInstance(ci_low, float)
        self.assertIsInstance(ci_upp, float)
        self.assertEqual(ci_upp, 1)
        self.assertLessEqual(ci_low, 1)

    @test_log_assert
    def test_compute_acc_auc_ci_epsilon(self, log_capture: LogCapture) -> None:
        logging.basicConfig(level=logging.DEBUG)

        accuracy, auc_value, emp_eps = self.mia_results.compute_acc_auc_ci_epsilon(
            delta=0.1
        )
        self.assertLessEqual(0, accuracy)
        self.assertLessEqual(accuracy, 1)

        self.assertLessEqual(0, auc_value)
        self.assertLessEqual(auc_value, 1)

        self.assertLessEqual(-1, emp_eps)
        self.assertLessEqual(emp_eps, 1)

        self.assertNotIn("TypeError", log_capture.actual())

    @test_log_assert
    def test_compute_acc_auc_epsilon(self) -> None:
        accuracy, auc_value, emp_eps = self.mia_results.compute_acc_auc_epsilon(
            delta=0.1
        )
        self.assertLessEqual(0, accuracy)
        self.assertLessEqual(accuracy, 1)

        self.assertLessEqual(0, auc_value)
        self.assertLessEqual(auc_value, 1)

        self.assertLessEqual(-1, emp_eps)
        self.assertLessEqual(emp_eps, 1)

    @test_log_assert
    def test_compute_metrics_at_error_threshold(self, log_capture: LogCapture) -> None:
        # pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
        error_thresholds: np.ndarray = np.linspace(0.01, 1, 100)

        (
            accuracy,
            auc_value,
            eps_fpr_array,
            eps_max_array,
        ) = self.mia_results.compute_metrics_at_error_threshold(
            delta=0.1, error_threshold=error_thresholds, verbose=True
        )

        self.assertLessEqual(0, accuracy)
        self.assertLessEqual(accuracy, 1)

        self.assertLessEqual(0, auc_value)
        self.assertLessEqual(auc_value, 1)

        self.assertEqual(len(eps_fpr_array), len(error_thresholds))
        self.assertEqual(len(eps_max_array), len(error_thresholds))

        self.assertNotIn("TypeError", log_capture.actual())

    @test_log_assert
    def test_compute_eps_at_tpr_threshold(self, log_capture: LogCapture) -> None:
        # pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
        error_thresholds: np.ndarray = np.linspace(0.01, 1, 100)

        eps_tpr_array = self.mia_results.compute_eps_at_tpr_threshold(
            delta=0.1, tpr_threshold=error_thresholds, verbose=True
        )

        self.assertEqual(len(eps_tpr_array), len(error_thresholds))

        self.assertNotIn("TypeError", log_capture.actual())

    def test_suppress_divide_and_invalid_warnings(self) -> None:
        threshold = np.array([0.1, 0.2, 0.3])
        tpr = torch.tensor([1.0, 1.0, 1.0, 0.5])
        fpr = torch.tensor([0.0, 0.0, 0.5, 1.0])
        delta = 0.1

        # test that these values raise warnings in eps calculation
        fnr = 1 - tpr
        tnr = 1 - fpr
        with warnings.catch_warnings(record=True) as w:
            eps1 = np.log(1 - fnr - delta) - np.log(fpr)
            eps2 = np.log(tnr - delta) - np.log(fnr)
            self.assertTrue(len(w) > 0)
            self.assertTrue(
                any("divide by zero" in str(warning.message) for warning in w)
            )
            self.assertTrue(
                any("invalid value" in str(warning.message) for warning in w)
            )

        # test that warnings are suppressed when compute_metrics_at_error_threshold is called
        with patch.object(MIAResults, "get_tpr_fpr") as mock_get_tpr_fpr:
            mock_get_tpr_fpr.return_value = (tpr, fpr)
            # Use the catch_warnings context manager to check that no warnings are raised
            with warnings.catch_warnings(record=True) as w:
                self.mia_results.compute_metrics_at_error_threshold(
                    delta, error_threshold=threshold
                )
                self.assertFalse(
                    any("divide by zero" in str(warning.message) for warning in w)
                )
                self.assertFalse(
                    any("invalid value" in str(warning.message) for warning in w)
                )

        # test that warnings are suppressed when compute_eps_at_tpr_threshold is called
        with patch.object(MIAResults, "get_tpr_fpr") as mock_get_tpr_fpr:
            mock_get_tpr_fpr.return_value = (tpr, fpr)
            # Use the catch_warnings context manager to check that no warnings are raised
            with warnings.catch_warnings(record=True) as w:
                self.mia_results.compute_eps_at_tpr_threshold(
                    delta=delta, tpr_threshold=threshold
                )
                self.assertFalse(
                    any("divide by zero" in str(warning.message) for warning in w)
                )
                self.assertFalse(
                    any("invalid value" in str(warning.message) for warning in w)
                )

        # test that warnings are suppressed when compute_acc_auc_epsilon is called
        with patch.object(MIAResults, "get_tpr_fpr") as mock_get_tpr_fpr:
            mock_get_tpr_fpr.return_value = (tpr, fpr)
            with warnings.catch_warnings(record=True) as w:
                self.mia_results.compute_acc_auc_epsilon(delta=delta)
                self.assertFalse(
                    any("divide by zero" in str(warning.message) for warning in w)
                )
                self.assertFalse(
                    any("invalid value" in str(warning.message) for warning in w)
                )

        # test that warnings are suppressed when compute_epsilon_at_error_thresholds is called
        with patch.object(MIAResults, "get_tpr_fpr") as mock_get_tpr_fpr:
            mock_get_tpr_fpr.return_value = (tpr, fpr)
            with warnings.catch_warnings(record=True) as w:
                self.mia_results.compute_epsilon_at_error_thresholds(
                    delta=delta, error_thresholds=list(threshold)
                )
                self.assertFalse(
                    any("divide by zero" in str(warning.message) for warning in w)
                )
                self.assertFalse(
                    any("invalid value" in str(warning.message) for warning in w)
                )
