# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict


import logging
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import torch
from privacy_guard.analysis.mia.mia_results import MIAResults


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

    def test_get_indices_of_error_at_thresholds(self) -> None:
        # Note that error_threshold also contains values outside of [min(error_rates), max(error_rates)]
        # We also include thresholds at exact values of error_rates
        error_thresholds = np.array([0.05, 0.1, 0.15, 0.25, 0.35, 0.45, 0.5, 0.55])

        # Test for tpr and fpr error type
        error_rates = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        indices = self.mia_results._get_indices_of_error_at_thresholds(
            error_rates, error_thresholds, "tpr"
        )
        self.assertEqual(indices.tolist(), [0, 0, 1, 2, 3, 4, 4, 4])

        indices = self.mia_results._get_indices_of_error_at_thresholds(
            error_rates, error_thresholds, "fpr"
        )
        self.assertEqual(indices.tolist(), [0, 0, 0, 1, 2, 3, 4, 4])

        # Test for fnr error type (fnr = 1 - tpr)
        error_rates = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
        indices = self.mia_results._get_indices_of_error_at_thresholds(
            error_rates, error_thresholds, "fnr"
        )
        self.assertEqual(indices.tolist(), [4, 4, 4, 3, 2, 1, 0, 0])

        # Test for tnr error type (tnr = 1 - fpr)
        indices = self.mia_results._get_indices_of_error_at_thresholds(
            error_rates, error_thresholds, "tnr"
        )
        self.assertEqual(indices.tolist(), [4, 4, 3, 2, 1, 0, 0, 0])

        # Test invalid error type
        with self.assertRaises(ValueError):
            self.mia_results._get_indices_of_error_at_thresholds(
                error_rates, error_thresholds, "invalid_type"
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

    def test_compute_acc_auc_ci_epsilon(self) -> None:
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

    def test_compute_metrics_at_error_threshold_use_fnr_tnr_false(self) -> None:
        """Test default behavior when use_fnr_tnr=False (default)"""
        error_thresholds: np.ndarray = np.linspace(0.01, 1, 100)

        (
            accuracy,
            auc_value,
            eps_fpr_array,
            eps_tpr_array,
            eps_max_array,
        ) = self.mia_results.compute_metrics_at_error_threshold(
            delta=0.1, error_threshold=error_thresholds, use_fnr_tnr=False, verbose=True
        )

        self.assertLessEqual(0, accuracy)
        self.assertLessEqual(accuracy, 1)
        self.assertLessEqual(0, auc_value)
        self.assertLessEqual(auc_value, 1)
        self.assertEqual(len(eps_fpr_array), len(error_thresholds))
        self.assertEqual(len(eps_tpr_array), len(error_thresholds))
        self.assertEqual(len(eps_max_array), len(error_thresholds))

        # eps_max should be the max of eps_fpr and eps_tpr when use_fnr_tnr=False
        for i in range(len(error_thresholds)):
            expected_max = np.fmax(eps_fpr_array[i], eps_tpr_array[i])
            # Handle NaN values: both should be NaN or both should be equal finite values
            if np.isnan(expected_max) and np.isnan(eps_max_array[i]):
                # Both are NaN, this is expected and acceptable
                continue
            else:
                self.assertAlmostEqual(eps_max_array[i], expected_max, places=10)

    def test_compute_metrics_at_error_threshold_use_fnr_tnr_true(self) -> None:
        """Test behavior when use_fnr_tnr=True"""
        error_thresholds: np.ndarray = np.linspace(0.01, 1, 100)

        (
            accuracy,
            auc_value,
            eps_fpr_array,
            eps_tpr_array,
            eps_max_array,
        ) = self.mia_results.compute_metrics_at_error_threshold(
            delta=0.1, error_threshold=error_thresholds, use_fnr_tnr=True, verbose=True
        )

        self.assertLessEqual(0, accuracy)
        self.assertLessEqual(accuracy, 1)
        self.assertLessEqual(0, auc_value)
        self.assertLessEqual(auc_value, 1)

        # When use_fnr_tnr=True, error thresholds >= 1.0 are filtered out
        filtered_thresholds = error_thresholds[error_thresholds < 1.0]
        self.assertEqual(len(eps_fpr_array), len(filtered_thresholds))
        self.assertEqual(len(eps_tpr_array), len(filtered_thresholds))
        self.assertEqual(len(eps_max_array), len(filtered_thresholds))

        # Arrays should be shorter than original when filtering occurs
        if len(filtered_thresholds) < len(error_thresholds):
            self.assertLess(len(eps_fpr_array), len(error_thresholds))
            self.assertLess(len(eps_tpr_array), len(error_thresholds))
            self.assertLess(len(eps_max_array), len(error_thresholds))

    def test_compute_metrics_at_error_threshold_filtering_behavior(self) -> None:
        """Test that use_fnr_tnr=True properly filters out thresholds >= 1.0"""
        # Create error thresholds that include values >= 1.0
        error_thresholds_with_1: np.ndarray = np.array([0.1, 0.5, 0.9, 1.0, 1.1])

        # Test with use_fnr_tnr=False (should use all thresholds)
        (
            _,
            _,
            eps_fpr_false,
            eps_tpr_false,
            eps_max_false,
        ) = self.mia_results.compute_metrics_at_error_threshold(
            delta=0.1, error_threshold=error_thresholds_with_1, use_fnr_tnr=False
        )

        # Test with use_fnr_tnr=True (should filter out >= 1.0)
        (
            _,
            _,
            eps_fpr_true,
            eps_tpr_true,
            eps_max_true,
        ) = self.mia_results.compute_metrics_at_error_threshold(
            delta=0.1, error_threshold=error_thresholds_with_1, use_fnr_tnr=True
        )

        # With use_fnr_tnr=False, should have 5 elements
        self.assertEqual(len(eps_fpr_false), 5)
        self.assertEqual(len(eps_tpr_false), 5)
        self.assertEqual(len(eps_max_false), 5)

        # With use_fnr_tnr=True, should have 3 elements (filtered out 1.0 and 1.1)
        self.assertEqual(len(eps_fpr_true), 3)
        self.assertEqual(len(eps_tpr_true), 3)
        self.assertEqual(len(eps_max_true), 3)

        # The first 3 elements should be the same for both cases (for thresholds < 1.0)
        for i in range(3):
            self.assertTrue(
                (np.isnan(eps_fpr_false[i]) and np.isnan(eps_fpr_true[i]))
                or np.isclose(eps_fpr_false[i], eps_fpr_true[i], atol=1e-10)
            )
            self.assertTrue(
                (np.isnan(eps_tpr_false[i]) and np.isnan(eps_tpr_true[i]))
                or np.isclose(eps_tpr_false[i], eps_tpr_true[i], atol=1e-10)
            )

    def test_compute_metrics_at_error_threshold_eps_max_calculation(self) -> None:
        """Test that eps_max includes FNR/TNR contributions when use_fnr_tnr=True"""
        error_thresholds: np.ndarray = np.array([0.1, 0.3, 0.5])

        # Get results with use_fnr_tnr=False
        (
            _,
            _,
            eps_fpr_false,
            eps_tpr_false,
            eps_max_false,
        ) = self.mia_results.compute_metrics_at_error_threshold(
            delta=0.1, error_threshold=error_thresholds, use_fnr_tnr=False
        )

        # Get results with use_fnr_tnr=True
        (
            _,
            _,
            eps_fpr_true,
            eps_tpr_true,
            eps_max_true,
        ) = self.mia_results.compute_metrics_at_error_threshold(
            delta=0.1, error_threshold=error_thresholds, use_fnr_tnr=True
        )

        # eps_fpr and eps_tpr should be the same in both cases
        for i in range(len(error_thresholds)):
            self.assertTrue(
                (np.isnan(eps_fpr_false[i]) and np.isnan(eps_fpr_true[i]))
                or np.isclose(eps_fpr_false[i], eps_fpr_true[i], atol=1e-10)
            )
            self.assertTrue(
                (np.isnan(eps_tpr_false[i]) and np.isnan(eps_tpr_true[i]))
                or np.isclose(eps_tpr_false[i], eps_tpr_true[i], atol=1e-10)
            )

        # With use_fnr_tnr=False: eps_max = max(eps_fpr, eps_tpr)
        for i in range(len(error_thresholds)):
            expected_max_false = np.fmax(eps_fpr_false[i], eps_tpr_false[i])
            self.assertTrue(
                (np.isnan(eps_max_false[i]) and np.isnan(expected_max_false))
                or np.isclose(
                    eps_max_false[i],
                    expected_max_false,
                    atol=1e-10,
                )
            )

        # With use_fnr_tnr=True: eps_max should be >= max(eps_fpr, eps_tpr)
        # since it includes additional FNR/TNR contributions
        for i in range(len(error_thresholds)):
            min_expected = np.fmax(eps_fpr_true[i], eps_tpr_true[i])
            # Handle NaN values: if both are NaN, that's acceptable
            # If only one is NaN, it's an error
            if np.isnan(min_expected):
                continue
            elif not np.isnan(min_expected) and np.isnan(eps_max_true[i]):
                self.fail(
                    f"eps_max should not be NaN when eps_fpr or eps_tpr is not NaN. eps_max: {eps_max_true[i]}, eps_fpr: {eps_fpr_true[i]}, eps_tpr: {eps_tpr_true[i]}"
                )
            else:
                self.assertGreaterEqual(eps_max_true[i], min_expected - 1e-10)

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

    def test_compute_metrics_at_error_threshold(self) -> None:
        error_thresholds: np.ndarray = np.linspace(0.01, 1, 100)

        (
            accuracy,
            auc_value,
            eps_fpr_array,
            eps_tpr_array,
            eps_max_array,
        ) = self.mia_results.compute_metrics_at_error_threshold(
            delta=0.1, error_threshold=error_thresholds, verbose=True
        )

        self.assertLessEqual(0, accuracy)
        self.assertLessEqual(accuracy, 1)

        self.assertLessEqual(0, auc_value)
        self.assertLessEqual(auc_value, 1)

        self.assertEqual(len(eps_fpr_array), len(error_thresholds))
        self.assertEqual(len(eps_tpr_array), len(error_thresholds))
        self.assertEqual(len(eps_max_array), len(error_thresholds))

    def test_compute_eps_at_tpr_threshold(self) -> None:
        error_thresholds: np.ndarray = np.linspace(0.01, 1, 100)

        eps_tpr_array = self.mia_results.compute_eps_at_tpr_threshold(
            delta=0.1, tpr_threshold=error_thresholds, verbose=True
        )

        self.assertEqual(len(eps_tpr_array), len(error_thresholds))

    def test_suppress_divide_and_invalid_warnings(self) -> None:
        threshold = np.array([0.1, 0.2, 0.3])
        tpr = np.array([1.0, 1.0, 1.0, 0.5])
        fpr = np.array([0.0, 0.0, 0.5, 1.0])
        delta = 0.1

        # test that these values raise warnings in eps calculation
        fnr = 1 - tpr
        tnr = 1 - fpr
        with warnings.catch_warnings(record=True) as w:
            # These operations should generate warnings for divide by zero and invalid values
            _ = np.log(1 - fnr - delta) - np.log(fpr)  # eps1
            _ = np.log(tnr - delta) - np.log(fnr)  # eps2
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
