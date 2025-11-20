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


import unittest

import numpy as np
import torch
from privacy_guard.analysis.lia.lia_analysis_input import LIAAnalysisInput
from privacy_guard.analysis.lia.lia_analysis_node import (
    LIAAnalysisNode,
    LIAAnalysisOutput,
)


class TestLIAAnalysisNode(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test data for LIA analysis node."""
        # Create synthetic test data
        self.num_samples = 1000
        self.num_resampling = 5

        # Generate base data
        np.random.seed(42)  # For reproducible tests
        self.predictions = np.random.uniform(0.1, 0.9, self.num_samples)
        self.y1_preds = np.random.uniform(0.1, 0.9, self.num_samples)

        # Create true_bits matrix (num_resampling x num_samples)
        self.true_bits = np.random.choice(
            [0, 1], size=(self.num_resampling, self.num_samples)
        )

        # Create y0 and y1 matrices (num_resampling x num_samples) with binary values
        # generate y0 and y1 according to the predictions
        self.y0 = np.where(np.random.rand(self.num_samples) < self.predictions, 1, 0)
        self.y1 = np.where(
            np.random.rand(self.num_resampling, self.num_samples) < self.y1_preds,
            1,
            0,
        )

        # Create received_labels matrix (num_resampling x num_samples)
        self.received_labels = np.where(self.true_bits == 1, self.y1, self.y0)

        self.analysis_input = LIAAnalysisInput(
            predictions=self.predictions,
            predictions_y1_generation=self.y1_preds,
            true_bits=self.true_bits,
            y0=self.y0,
            y1=self.y1,
            received_labels=self.received_labels,
        )

        self.analysis_node = LIAAnalysisNode(
            analysis_input=self.analysis_input,
            delta=1e-6,
            num_bootstrap_resampling_times=10,
        )

        # Create separable data for testing epsilon capping
        separable_predictions = np.array([0.3] * 100)
        separable_predictions_y1 = separable_predictions + 0.1
        separable_y0 = np.zeros(100)

        separable_true_bits = np.array([0] * 50 + [1] * 50)[None, :]
        separable_y1 = np.ones((1, 100))
        separable_received_labels = np.array([0] * 50 + [1] * 50)[None, :]

        self.separable_analysis_input = LIAAnalysisInput(
            predictions=separable_predictions,
            predictions_y1_generation=separable_predictions_y1,
            true_bits=separable_true_bits,
            y0=separable_y0,
            y1=separable_y1,
            received_labels=separable_received_labels,
        )

        super().setUp()

    def test_get_analysis_input(self) -> None:
        """Test that analysis input is correctly stored."""
        self.assertIsInstance(self.analysis_node._analysis_input, LIAAnalysisInput)
        np.testing.assert_array_equal(
            self.analysis_node._analysis_input.predictions, self.predictions
        )

    def test_progress_bar(self) -> None:
        """Test that the progress bar works as expected."""
        test_progress_analysis_node = LIAAnalysisNode(
            analysis_input=self.analysis_input,
            delta=0.000001,
            num_bootstrap_resampling_times=5,
            show_progress=True,
        )

        outputs = test_progress_analysis_node.compute_outputs()
        self.assertIsInstance(outputs, dict)

    def test_timer_enabled(self) -> None:
        """Test that the timer works as expected."""
        test_timer_analysis_node = LIAAnalysisNode(
            analysis_input=self.analysis_input,
            delta=0.000001,
            num_bootstrap_resampling_times=5,
            with_timer=True,
        )

        test_timer_analysis_node.compute_outputs()

        timer_stats = test_timer_analysis_node.get_timer_stats()
        self.assertIn("compute all metrics", timer_stats)

    def test_timer_disabled(self) -> None:
        """Test that the timer is disabled when with_timer=False."""
        test_timer_analysis_node = LIAAnalysisNode(
            analysis_input=self.analysis_input,
            delta=0.000001,
            num_bootstrap_resampling_times=5,
            with_timer=False,
        )

        test_timer_analysis_node.compute_outputs()

        timer_stats = test_timer_analysis_node.get_timer_stats()
        self.assertEqual(len(timer_stats), 0)

    def test_turn_cap_eps_on(self) -> None:
        """Test capping of computed epsilons."""
        analysis_node = LIAAnalysisNode(
            self.separable_analysis_input,
            delta=0.00001,
            num_bootstrap_resampling_times=5,
            cap_eps=True,
        )
        outputs = analysis_node.compute_outputs()

        # With capping enabled, epsilon should be bounded
        self.assertIsInstance(outputs["eps"], float)
        self.assertGreaterEqual(outputs["eps"], 0)
        # For separable data with 100 samples, max eps should be around log(100)
        self.assertLessEqual(outputs["eps"], np.log(100) + 1)  # allow some margin
        # eps_lb could differ from eps since different caps are applied based on
        # the number of train scores that get subsampled during bootstrap
        self.assertLessEqual(outputs["eps_lb"], outputs["eps"])
        self.assertEqual(outputs["auc"], 1.0)
        self.assertEqual(outputs["accuracy"], 1.0)

    def test_turn_cap_eps_off(self) -> None:
        """Test uncapped computed epsilons."""
        analysis_node = LIAAnalysisNode(
            self.separable_analysis_input,
            delta=0.00001,
            num_bootstrap_resampling_times=5,
            cap_eps=False,
        )
        outputs = analysis_node.compute_outputs()

        # With capping disabled, epsilon should be inf
        self.assertIsInstance(outputs["eps"], float)
        self.assertEqual(outputs["eps"], np.inf)
        self.assertEqual(outputs["eps_lb"], np.inf)
        self.assertEqual(outputs["auc"], 1.0)
        self.assertEqual(outputs["accuracy"], 1.0)

    def test_num_bootstrap_resampling(self) -> None:
        """Test that the number of bootstraps affects the analysis."""
        num_bootstrap_resampling_times = 8
        test_num_bootstraps_analysis_node = LIAAnalysisNode(
            analysis_input=self.analysis_input,
            delta=0.000001,
            num_bootstrap_resampling_times=num_bootstrap_resampling_times,
        )

        outputs = test_num_bootstraps_analysis_node.compute_outputs()

        # Verify that the analysis completes successfully
        self.assertIsInstance(outputs, dict)
        self.assertIsInstance(outputs["eps"], float)
        self.assertIsInstance(outputs["accuracy"], float)
        self.assertIsInstance(outputs["auc"], float)

    def test_outputs_structure(self) -> None:
        """Test that compute_outputs returns the correct output structure."""
        outputs = self.analysis_node.run_analysis()

        self.assertIsInstance(outputs, LIAAnalysisOutput)

        # Check that output is of correct type
        outputs = outputs.to_dict()

        # Check all required fields are present and of correct types
        self.assertIsInstance(outputs["eps"], float)
        self.assertIsInstance(outputs["eps_lb"], float)
        self.assertIsInstance(outputs["error_rate_at_max_eps"], float)
        self.assertIsInstance(outputs["eps_max_bounds"], tuple)
        self.assertEqual(len(outputs["eps_max_bounds"]), 2)
        self.assertIsInstance(outputs["eps_at_tpr_bounds"], tuple)
        self.assertEqual(len(outputs["eps_at_tpr_bounds"]), 2)
        self.assertIsInstance(outputs["eps_at_fpr_bounds"], tuple)
        self.assertEqual(len(outputs["eps_at_fpr_bounds"]), 2)

        self.assertIsInstance(outputs["accuracy"], float)
        self.assertIsInstance(outputs["accuracy_ci"], list)
        self.assertEqual(len(outputs["accuracy_ci"]), 2)

        self.assertIsInstance(outputs["auc"], float)
        self.assertIsInstance(outputs["auc_ci"], list)
        self.assertEqual(len(outputs["auc_ci"]), 2)

        self.assertIsInstance(outputs["data_size"], int)
        self.assertEqual(outputs["data_size"], self.num_samples)

        self.assertIsInstance(outputs["label_mean"], float)
        self.assertIsInstance(outputs["prediction_mean"], float)
        self.assertIsInstance(outputs["prediction_y1_generation_mean"], float)

    def test_compute_outputs_value_ranges(self) -> None:
        """Test that analysis outputs are within expected ranges."""
        outputs = self.analysis_node.compute_outputs()

        # Epsilon should be non-negative
        self.assertGreaterEqual(outputs["eps"], 0)
        self.assertLessEqual(outputs["eps_lb"], outputs["eps"])

        # Error rate should be between 0 and 1
        self.assertGreaterEqual(outputs["error_rate_at_max_eps"], 0)
        self.assertLessEqual(outputs["error_rate_at_max_eps"], 1)

        # Accuracy should be between 0 and 1
        self.assertGreaterEqual(outputs["accuracy"], 0)
        self.assertLessEqual(outputs["accuracy"], 1)

        # AUC should be between 0 and 1
        self.assertGreaterEqual(outputs["auc"], 0)
        self.assertLessEqual(outputs["auc"], 1)

        # Lower bounds should be less than upper bounds
        self.assertTrue(
            np.all(outputs["eps_max_bounds"][0] <= outputs["eps_max_bounds"][1])
        )
        self.assertTrue(
            np.all(outputs["eps_at_tpr_bounds"][0] <= outputs["eps_at_tpr_bounds"][1])
        )
        self.assertTrue(
            np.all(outputs["eps_at_fpr_bounds"][0] <= outputs["eps_at_fpr_bounds"][1])
        )

        # Confidence intervals should be ordered correctly
        self.assertLessEqual(outputs["accuracy_ci"][0], outputs["accuracy_ci"][1])
        self.assertLessEqual(outputs["auc_ci"][0], outputs["auc_ci"][1])

    def test_different_delta_values(self) -> None:
        """Test analysis with different delta values."""
        delta_values = [1e-6, 1e-5, 1e-4]

        for delta in delta_values:
            analysis_node = LIAAnalysisNode(
                analysis_input=self.analysis_input,
                delta=delta,
                num_bootstrap_resampling_times=2,
            )

            outputs = analysis_node.compute_outputs()

            # Analysis should complete successfully for all delta values
            self.assertIsInstance(outputs, dict)
            self.assertGreaterEqual(outputs["eps"], 0)

    def test_power_parameter(self) -> None:
        """Test that power parameter is properly handled."""
        power_values = [0.0, 1.0, 2.0, 4.0]

        for power in power_values:
            analysis_node = LIAAnalysisNode(
                analysis_input=self.analysis_input,
                delta=1e-6,
                num_bootstrap_resampling_times=2,
                power=power,
            )

            # Verify that the power parameter is stored correctly
            self.assertEqual(analysis_node._power, power)

            outputs = analysis_node.compute_outputs()

            # Analysis should complete successfully for all power values
            self.assertIsInstance(outputs, dict)
            self.assertGreaterEqual(outputs["eps"], 0)
            self.assertIsInstance(outputs["accuracy"], float)
            self.assertIsInstance(outputs["auc"], float)

    def test_use_fnr_and_tnr_parameter(self) -> None:
        """Test that use_fnr_and_tnr parameter is properly handled."""
        for use_fnr_and_tnr in [True, False]:
            analysis_node = LIAAnalysisNode(
                analysis_input=self.analysis_input,
                delta=1e-6,
                num_bootstrap_resampling_times=2,
                use_fnr_and_tnr=use_fnr_and_tnr,
            )

            # Verify that the use_fnr_and_tnr parameter is stored correctly
            self.assertEqual(analysis_node._use_fnr_and_tnr, use_fnr_and_tnr)

            outputs = analysis_node.compute_outputs()

            # Analysis should complete successfully for both values
            self.assertIsInstance(outputs, dict)
            self.assertGreaterEqual(outputs["eps"], 0)

    def test_power_and_use_fnr_and_tnr_combined(self) -> None:
        """Test that power and use_fnr_and_tnr parameters work together."""
        analysis_node = LIAAnalysisNode(
            analysis_input=self.analysis_input,
            delta=1e-6,
            num_bootstrap_resampling_times=2,
            power=1.5,
            use_fnr_and_tnr=True,
        )

        # Verify that both parameters are stored correctly
        self.assertEqual(analysis_node._power, 1.5)
        self.assertTrue(analysis_node._use_fnr_and_tnr)

        outputs = analysis_node.compute_outputs()

        # Analysis should complete successfully
        self.assertIsInstance(outputs, dict)
        self.assertGreaterEqual(outputs["eps"], 0)
        self.assertIsInstance(outputs["accuracy"], float)
        self.assertIsInstance(outputs["auc"], float)

    def test_default_parameter_values(self) -> None:
        """Test that default parameter values are set correctly."""
        analysis_node = LIAAnalysisNode(
            analysis_input=self.analysis_input,
            delta=1e-6,
            num_bootstrap_resampling_times=2,
        )

        # Verify default values
        self.assertEqual(analysis_node._power, 0.0)
        self.assertFalse(analysis_node._use_fnr_and_tnr)

        outputs = analysis_node.compute_outputs()
        self.assertIsInstance(outputs, dict)

    def test_negative_power_raises_error(self) -> None:
        """Test that negative power values raise ValueError."""
        negative_power_values = [-0.1, -1.0, -2.5]

        for power in negative_power_values:
            with self.assertRaises(ValueError) as context:
                LIAAnalysisNode(
                    analysis_input=self.analysis_input,
                    delta=1e-6,
                    num_bootstrap_resampling_times=2,
                    power=power,
                )

            # Verify the error message
            self.assertEqual(
                str(context.exception),
                "Power used for score function must be non-negative",
            )

    def test_compute_scores(self) -> None:
        """Test that compute_scores computes correct score tensors."""
        # Setup: Use the analysis_input from setUp for consistency
        # Execute: Compute scores for the first game instance (i=0)
        scores_train, scores_test = self.analysis_node.compute_scores(0)

        # Assert: Verify the computed scores match expected calculations
        # For training samples (true_bits == 0) and test samples (true_bits == 1)

        # Manually compute expected scores for verification
        received = self.received_labels[0]
        predictions = self.predictions
        predictions_y1 = self.y1_preds
        true_bits = self.true_bits[0]

        # For each sample, compute: (prob_train - prob_reconstruct) * prob_diff_label^power
        # prob_train = predictions if received_labels==1 else (1 - predictions)
        # prob_reconstruct = predictions_y1 if received_labels==1 else (1 - predictions_y1)
        # prob_diff_label = (1 - predictions_y1) if received_labels==1 else predictions_y1

        expected_scores = []
        for idx in range(self.num_samples):
            if received[idx] == 1:
                prob_train = predictions[idx]
                prob_reconstruct = predictions_y1[idx]
                prob_diff_label = 1 - predictions_y1[idx]
            else:
                prob_train = 1 - predictions[idx]
                prob_reconstruct = 1 - predictions_y1[idx]
                prob_diff_label = predictions_y1[idx]

            score = (prob_train - prob_reconstruct) * (
                prob_diff_label**0.0
            )  # power=0.0 (default)
            expected_scores.append(score)

        expected_scores = np.array(expected_scores)

        # Extract expected scores for train and test sets
        expected_scores_train = expected_scores[true_bits == 0]
        expected_scores_test = expected_scores[true_bits == 1]

        # Verify tensor shapes match expected counts
        train_count = np.sum(true_bits == 0)
        test_count = np.sum(true_bits == 1)
        self.assertEqual(
            scores_train.shape[0],
            train_count,
            f"Should have {train_count} training samples",
        )
        self.assertEqual(
            scores_test.shape[0], test_count, f"Should have {test_count} test samples"
        )

        # Verify tensor values match expected calculations
        np.testing.assert_allclose(
            scores_train.numpy(),
            expected_scores_train,
            rtol=1e-10,
            err_msg="Training scores should match expected calculations",
        )

        np.testing.assert_allclose(
            scores_test.numpy(),
            expected_scores_test,
            rtol=1e-10,
            err_msg="Test scores should match expected calculations",
        )

        # Verify tensors are torch.Tensor objects
        self.assertIsInstance(scores_train, torch.Tensor)
        self.assertIsInstance(scores_test, torch.Tensor)

    def test_miscalibration_statistics_computation(self) -> None:
        """Test that miscalibration statistics are computed correctly."""
        # Create test data with known values for easy verification
        num_samples = 100
        num_resampling = 2

        # Create specific test values
        predictions = np.array([0.2, 0.4, 0.6, 0.8] * 25)  # Mean = 0.5
        predictions_y1 = np.array([0.3, 0.5, 0.7, 0.9] * 25)  # Mean = 0.6
        y0 = np.array([0, 1, 0, 1] * 25)  # Mean = 0.5

        # Create matrices for resampling
        true_bits = np.random.choice([0, 1], size=(num_resampling, num_samples))
        y1 = np.random.choice([0, 1], size=(num_resampling, num_samples))
        received_labels = np.where(true_bits == 1, y1, y0)

        # Calculate expected means
        expected_label_mean = np.mean(y0)
        expected_prediction_mean = np.mean(predictions)
        expected_prediction_y1_mean = np.mean(predictions_y1)

        # Create analysis input
        test_analysis_input = LIAAnalysisInput(
            predictions=predictions,
            predictions_y1_generation=predictions_y1,
            true_bits=true_bits,
            y0=y0,
            y1=y1,
            received_labels=received_labels,
        )

        # Create analysis node with minimal bootstrap resampling for faster test
        analysis_node = LIAAnalysisNode(
            analysis_input=test_analysis_input,
            delta=1e-6,
            num_bootstrap_resampling_times=2,
        )

        # Run analysis and cast to LIAAnalysisOutput
        lia_outputs = analysis_node.compute_outputs()

        # Verify that the computed means match expected values
        self.assertAlmostEqual(
            lia_outputs["label_mean"],
            expected_label_mean,
            places=10,
            msg="label_mean should match the mean of y0",
        )

        self.assertAlmostEqual(
            lia_outputs["prediction_mean"],
            expected_prediction_mean,
            places=10,
            msg="prediction_mean should match the mean of predictions",
        )

        self.assertAlmostEqual(
            lia_outputs["prediction_y1_generation_mean"],
            expected_prediction_y1_mean,
            places=10,
            msg="prediction_y1_generation_mean should match the mean of predictions_y1_generation",
        )
