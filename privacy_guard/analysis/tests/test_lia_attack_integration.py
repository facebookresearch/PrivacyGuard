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
import pandas as pd
from privacy_guard.analysis.lia.lia_analysis_input import LIAAnalysisInput
from privacy_guard.analysis.lia.lia_analysis_node import LIAAnalysisNode
from privacy_guard.analysis.mia.aggregate_analysis_input import AggregationType
from privacy_guard.attacks.lia_attack import LIAAttack, LIAAttackInput


class TestLIAAttackIntegration(unittest.TestCase):
    """
    Test LIA attack and analysis on a model with randomized response.

    This test creates a synthetic dataset where the model applies randomized
    response with different gamma values, which should result in different
    privacy leakage metrics.
    """

    def setUp(self) -> None:
        """Set up common test configuration."""
        # Data Configuration
        self.num_users = 100000
        self.p = 0.05  # Probability of positive label

        # Configuration for attack
        self.delta = 1e-6  # Delta parameter for privacy analysis
        self.row_aggregation = AggregationType.NONE
        self.merge_columns = ["user_id", "label"]
        self.num_resampling_times = 10
        self.num_bootstrap_resampling_times = 10

    def _create_test_data(self, gamma: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create test data with specified gamma value.
        Args:
            gamma: Label flipping rate for randomized response
        Returns:
            Tuple of (df_hold_out_train, df_hold_out_train_calib)
        """
        # Generate synthetic data
        labels = np.random.choice([0, 1], size=self.num_users, p=[1 - self.p, self.p])

        rr_labels = np.where(
            np.random.rand(self.num_users) < gamma,
            1 - np.array(labels),
            labels,
        )

        # Calibration predictions match true distribution
        predictions_calib = np.full(self.num_users, self.p)

        # Model does randomized response (if gamma=0, then model memorizes training labels)
        predictions = rr_labels.copy()

        # Create dataframes
        df_hold_out_train = pd.DataFrame(
            {
                "user_id": np.arange(self.num_users),
                "label": labels,
                "predictions": predictions,
            }
        )

        df_hold_out_train_calib = pd.DataFrame(
            {
                "user_id": np.arange(self.num_users),
                "label": labels,
                "predictions": predictions_calib,
            }
        )

        return df_hold_out_train, df_hold_out_train_calib

    def _get_lia_attack_input(
        self,
        df_hold_out_train: pd.DataFrame,
        df_hold_out_train_calib: pd.DataFrame,
        gamma: float,
    ) -> dict[str, pd.DataFrame]:
        """
        Get attack input for LIA attack.

        Args:
            df_hold_out_train: Training data
            df_hold_out_train_calib: Calibration data

        Returns:
            Tuple of (df_train_merge, df_test_merge)
        """
        # Prepare attack input
        lia_attack_input = LIAAttackInput(
            df_hold_out_train,
            df_hold_out_train_calib,
            row_aggregation=self.row_aggregation,
            merge_columns=self.merge_columns,
        ).prepare_attack_input()

        # check properties of attack input
        attack_table = lia_attack_input["df_train_and_calib"]
        self.assertEqual(attack_table.shape[0], self.num_users)
        self.assertEqual(attack_table.shape[1], 5)

        # check that mean of labels is close to p
        self.assertAlmostEqual(
            np.mean(attack_table["label"].values), self.p, delta=0.01
        )

        # assert that all predictions_calib are equal to p
        self.assertTrue(np.allclose(attack_table["predictions_calib"].values, self.p))

        # check that mean of predictions is close to expected_mean
        expected_mean = self.p * (1 - gamma) + (1 - self.p) * gamma
        self.assertAlmostEqual(
            np.mean(attack_table["predictions"].values), expected_mean, delta=0.01
        )

        return lia_attack_input

    def _get_analysis_input(
        self,
        lia_attack_input: dict[str, pd.DataFrame],
        gamma: float,
    ) -> LIAAnalysisInput:
        """
        Get analysis input for LIA  analysis given attack input.
        """
        # Prepare analysis input
        attack = LIAAttack(
            lia_attack_input,
            row_aggregation=self.row_aggregation,
            num_resampling_times=self.num_resampling_times,
            y1_generation="calibration",
        )
        analysis_input = attack.run_attack()

        self.assertAlmostEqual(np.mean(analysis_input.y0), self.p, delta=0.01)
        self.assertAlmostEqual(np.mean(analysis_input.y1[0]), self.p, delta=0.01)

        # check that mean of predictions is close to expected_mean
        expected_mean = self.p * (1 - gamma) + (1 - self.p) * gamma
        self.assertAlmostEqual(
            np.mean(analysis_input.predictions), expected_mean, delta=0.01
        )

        # check that y1_generation is set correctly
        self.assertTrue(np.allclose(analysis_input.predictions_y1_generation, self.p))

        # check mean of true_bits is close to 0.5
        self.assertAlmostEqual(np.mean(analysis_input.true_bits[0]), 0.5, delta=0.01)

        # check received_bits is as expected
        expected_received_labels = np.where(
            analysis_input.true_bits[0] == 0, analysis_input.y0, analysis_input.y1[0]
        )
        self.assertTrue(
            np.allclose(analysis_input.received_labels[0], expected_received_labels)
        )

        return analysis_input

    def _run_lia_attack_and_analysis(
        self,
        gamma: float = 0.0,
    ) -> dict[str, float]:
        """
        Run LIA attack and analysis on the given data.

        Returns:
            Dictionary of metrics
        """
        df_hold_out_train, df_hold_out_train_calib = self._create_test_data(gamma)

        # Prepare attack input
        lia_attack_input = self._get_lia_attack_input(
            df_hold_out_train, df_hold_out_train_calib, gamma
        )

        # Prepare analysis input
        lia_analysis_input = self._get_analysis_input(lia_attack_input, gamma)

        # Run analysis
        lia_analysis_node = LIAAnalysisNode(
            lia_analysis_input,
            delta=self.delta,
            num_bootstrap_resampling_times=self.num_bootstrap_resampling_times,
        )

        outputs = lia_analysis_node.compute_outputs()

        return outputs

    def test_lia_attack_with_high_gamma(self) -> None:
        """
        Test LIA attack with high gamma (0.5).

        High gamma means high label flipping, so we expect no privacy leakage.
        """
        gamma = 0.5
        outputs = self._run_lia_attack_and_analysis(gamma)
        # For a model that outputs random labels, we expect low epsilon values
        # and accuracy/auc close to 0.5
        self.assertLessEqual(outputs["eps"], 0.3)
        self.assertLessEqual(outputs["eps_lb"], 0)
        self.assertAlmostEqual(outputs["accuracy"], 0.5, delta=0.02)
        self.assertAlmostEqual(outputs["auc"], 0.5, delta=0.02)

    def test_lia_attack_with_low_gamma(self) -> None:
        """
        Test LIA attack with both gamma values and compare results.

        We expect that lower gamma (less noise) results in higher epsilon values
        (more privacy leakage) compared to higher gamma (more noise).
        """
        # Test with low gamma
        gamma = 0.01
        outputs = self._run_lia_attack_and_analysis(gamma)
        # For a memorizing model, we expect high epsilon values
        # (indicating high privacy leakage)
        self.assertGreaterEqual(outputs["eps"], 2.0)
        self.assertGreaterEqual(outputs["eps_lb"], 2.0)
        self.assertGreaterEqual(outputs["accuracy"], 0.52)
        self.assertGreaterEqual(outputs["auc"], 0.52)

    def test_compare_attack_with_high_and_low_gamma(self) -> None:
        """
        Test LIA attack with both gamma values and compare results.

        We expect that lower gamma (less noise) results in higher epsilon values
        (more privacy leakage) compared to higher gamma (more noise).
        """
        # Test with low gamma
        low_gamma = 0.01
        outputs_low_gamma = self._run_lia_attack_and_analysis(low_gamma)

        # Test with high gamma
        high_gamma = 0.3
        outputs_high_gamma = self._run_lia_attack_and_analysis(high_gamma)

        # Compare results
        self.assertGreater(outputs_low_gamma["eps"], outputs_high_gamma["eps"])
        self.assertGreater(outputs_low_gamma["eps_lb"], outputs_high_gamma["eps_lb"])
        self.assertGreater(
            outputs_low_gamma["accuracy"], outputs_high_gamma["accuracy"]
        )
        self.assertGreater(outputs_low_gamma["auc"], outputs_high_gamma["auc"])
