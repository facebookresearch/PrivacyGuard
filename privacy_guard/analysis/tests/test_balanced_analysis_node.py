# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import numpy as np
import pandas as pd
from privacy_guard.analysis.base_analysis_node import BaseAnalysisInput
from privacy_guard.analysis.mia.balanced_analysis_node import BalancedAnalysisNode


class TestBalancedAnalysisNode(unittest.TestCase):
    """Test suite for BalancedAnalysisNode class."""

    def setUp(self) -> None:
        """Set up test data with three scenarios: train smaller, test smaller, and equal sizes."""
        # Create test dataframes for all three scenarios
        self.df_train_small = pd.DataFrame({"score": [0.1, 0.2, 0.3]})
        self.df_test_large = pd.DataFrame({"score": [0.4, 0.5, 0.6, 0.7, 0.8]})

        self.df_train_large = pd.DataFrame({"score": [0.1, 0.2, 0.3, 0.4, 0.5]})
        self.df_test_small = pd.DataFrame({"score": [0.6, 0.7, 0.8]})

        self.df_train_equal = pd.DataFrame({"score": [0.1, 0.2, 0.3, 0.4]})
        self.df_test_equal = pd.DataFrame({"score": [0.5, 0.6, 0.7, 0.8]})

        # Create analysis inputs for each scenario
        self.input_train_smaller = BaseAnalysisInput(
            df_train_user=self.df_train_small, df_test_user=self.df_test_large
        )
        self.input_test_smaller = BaseAnalysisInput(
            df_train_user=self.df_train_large, df_test_user=self.df_test_small
        )
        self.input_equal = BaseAnalysisInput(
            df_train_user=self.df_train_equal, df_test_user=self.df_test_equal
        )

        # Create analysis nodes for each scenario
        self.node_train_smaller = BalancedAnalysisNode(
            analysis_input=self.input_train_smaller, delta=0.000001, n_users_for_eval=10
        )
        self.node_test_smaller = BalancedAnalysisNode(
            analysis_input=self.input_test_smaller, delta=0.000001, n_users_for_eval=10
        )
        self.node_equal = BalancedAnalysisNode(
            analysis_input=self.input_equal, delta=0.000001, n_users_for_eval=10
        )

        # Create multi-column dataframe for additional tests
        self.df_multi_column = pd.DataFrame(
            {"score": [0.1, 0.2, 0.3], "user_id": [1, 2, 3], "feature": ["a", "b", "c"]}
        )

        super().setUp()

    def test_initialization_parameters(self) -> None:
        """Test that initialization parameters are correctly set."""
        # Basic parameters
        self.assertIsInstance(self.node_train_smaller, BalancedAnalysisNode)
        self.assertEqual(self.node_train_smaller._delta, 0.000001)
        self.assertEqual(self.node_train_smaller._n_users_for_eval, 10)

        # Optional parameters
        node = BalancedAnalysisNode(
            analysis_input=self.input_equal,
            delta=0.000001,
            n_users_for_eval=10,
            use_upper_bound=False,
            num_bootstrap_resampling_times=500,
            cap_eps=False,
            show_progress=True,
            with_timer=True,
        )
        self.assertFalse(node._use_upper_bound)
        self.assertEqual(node._num_bootstrap_resampling_times, 500)
        self.assertFalse(node._cap_eps)
        self.assertTrue(node._show_progress)
        self.assertTrue(node._with_timer)

    def test_automatic_balancing(self) -> None:
        """Test that datasets are automatically balanced during initialization for all scenarios."""
        # Scenario 1: Train smaller than test
        train_size = len(self.node_train_smaller.analysis_input.df_train_user)
        test_size = len(self.node_train_smaller.analysis_input.df_test_user)
        self.assertEqual(
            train_size,
            test_size,
            "Train and test should be balanced when train is smaller",
        )
        self.assertEqual(train_size, 5, "Should balance to the larger size (5)")

        # Scenario 2: Test smaller than train
        train_size = len(self.node_test_smaller.analysis_input.df_train_user)
        test_size = len(self.node_test_smaller.analysis_input.df_test_user)
        self.assertEqual(
            train_size,
            test_size,
            "Train and test should be balanced when test is smaller",
        )
        self.assertEqual(test_size, 5, "Should balance to the larger size (5)")

        # Scenario 3: Equal sizes
        train_size = len(self.node_equal.analysis_input.df_train_user)
        test_size = len(self.node_equal.analysis_input.df_test_user)
        self.assertEqual(train_size, test_size, "Equal sizes should remain balanced")
        self.assertEqual(train_size, 4, "Train set should remain at original size (4)")
        self.assertEqual(test_size, 4, "Test set should remain at original size (4)")

    def test_upsample(self) -> None:
        """Test the _upsample static method with various sample count differences."""
        scores = pd.Series([0.1, 0.2, 0.3])

        # Test upsampling by 2 elements
        upsampled = BalancedAnalysisNode._upsample(scores, 2)
        self.assertEqual(len(upsampled), 5, "Should add exactly 2 elements")
        self.assertTrue(
            all(elem in scores.values for elem in upsampled.values),
            "Upsampled should only contain elements from the original scores",
        )

        # Test upsampling by more elements than original
        upsampled = BalancedAnalysisNode._upsample(scores, 4)
        self.assertEqual(len(upsampled), 7, "Should add exactly 4 elements")
        self.assertTrue(
            all(elem in scores.values for elem in upsampled.values),
            "Upsampled should only contain elements from the original scores",
        )

        # Test upsampling by 0 elements
        upsampled = BalancedAnalysisNode._upsample(scores, 0)
        self.assertEqual(len(upsampled), 3, "Should not add any elements")
        self.assertTrue(
            all(elem in scores.values for elem in upsampled.values),
            "Upsampled should only contain elements from the original scores",
        )

    def test_balance_smaller(self) -> None:
        """Test the _balance_smaller static method with simple and multi-column dataframes."""
        # Test with simple dataframe
        balanced_df = BalancedAnalysisNode._balance_smaller(self.df_train_small, 2)
        self.assertEqual(len(balanced_df), 5, "Should add exactly 2 elements")
        np.testing.assert_array_equal(
            balanced_df["score"].values[:3],
            self.df_train_small["score"].values,
            "Original data should be preserved",
        )

        # Test with multi-column dataframe
        balanced_multi = BalancedAnalysisNode._balance_smaller(self.df_multi_column, 2)
        self.assertEqual(len(balanced_multi), 5, "Should add exactly 2 elements")
        self.assertListEqual(
            list(balanced_multi.columns),
            ["score", "user_id", "feature"],
            "All columns should be preserved",
        )
        np.testing.assert_array_equal(
            balanced_multi["score"].values[:3],
            self.df_multi_column["score"].values,
            "Original score data should be preserved",
        )
        np.testing.assert_array_equal(
            balanced_multi["user_id"].values[:3],
            self.df_multi_column["user_id"].values,
            "Original user_id data should be preserved",
        )
        np.testing.assert_array_equal(
            balanced_multi["feature"].values[:3],
            self.df_multi_column["feature"].values,
            "Original feature data should be preserved",
        )

    def test_balance_static_method(self) -> None:
        """Test the _balance static method with all three scenarios."""
        # Scenario 1: Train smaller than test
        balanced_train, balanced_test = BalancedAnalysisNode._balance(
            self.df_train_small, self.df_test_large
        )
        self.assertEqual(
            len(balanced_train),
            len(balanced_test),
            "Balanced datasets should have equal size",
        )
        self.assertEqual(len(balanced_train), 5, "Should balance to larger size (5)")
        np.testing.assert_array_equal(
            balanced_train["score"].values[:3],
            self.df_train_small["score"].values,
            "Original train data should be preserved",
        )
        np.testing.assert_array_equal(
            balanced_test["score"].values,
            self.df_test_large["score"].values,
            "Original test data should be preserved",
        )

        # Scenario 2: Test smaller than train
        balanced_train, balanced_test = BalancedAnalysisNode._balance(
            self.df_train_large, self.df_test_small
        )
        self.assertEqual(
            len(balanced_train),
            len(balanced_test),
            "Balanced datasets should have equal size",
        )
        self.assertEqual(len(balanced_train), 5, "Should balance to larger size (5)")
        np.testing.assert_array_equal(
            balanced_train["score"].values,
            self.df_train_large["score"].values,
            "Original train data should be preserved",
        )
        np.testing.assert_array_equal(
            balanced_test["score"].values[:3],
            self.df_test_small["score"].values,
            "Original test data should be preserved",
        )

        # Scenario 3: Equal sizes
        balanced_train, balanced_test = BalancedAnalysisNode._balance(
            self.df_train_equal, self.df_test_equal
        )
        self.assertEqual(
            len(balanced_train),
            len(balanced_test),
            "Equal datasets should remain equal size",
        )
        self.assertEqual(
            len(balanced_train), 4, "Equal datasets should remain at original size (4)"
        )
        np.testing.assert_array_equal(
            balanced_train["score"].values,
            self.df_train_equal["score"].values,
            "Equal train data should be unchanged",
        )
        np.testing.assert_array_equal(
            balanced_test["score"].values,
            self.df_test_equal["score"].values,
            "Equal test data should be unchanged",
        )

    def test_multi_column_balancing(self) -> None:
        """Test automatic balancing with dataframes that have multiple columns."""
        # Create multi-column dataframes with size difference
        df_train = pd.DataFrame(
            {"score": [0.1, 0.2, 0.3], "user_id": [1, 2, 3], "feature": ["a", "b", "c"]}
        )
        df_test = pd.DataFrame(
            {
                "score": [0.4, 0.5, 0.6, 0.7, 0.8],
                "user_id": [4, 5, 6, 7, 8],
                "feature": ["d", "e", "f", "g", "h"],
            }
        )

        # Test automatic balancing during initialization
        analysis_input = BaseAnalysisInput(df_train_user=df_train, df_test_user=df_test)
        node = BalancedAnalysisNode(
            analysis_input=analysis_input, delta=0.000001, n_users_for_eval=10
        )

        # Verify sizes are balanced
        train_size = len(node.analysis_input.df_train_user)
        test_size = len(node.analysis_input.df_test_user)
        self.assertEqual(train_size, test_size, "Train and test should be balanced")
        self.assertEqual(train_size, 5, "Should balance to the larger size (5)")

        # Verify all columns are preserved
        self.assertListEqual(
            list(node.analysis_input.df_train_user.columns),
            ["score", "user_id", "feature"],
            "All train columns should be preserved",
        )
        self.assertListEqual(
            list(node.analysis_input.df_test_user.columns),
            ["score", "user_id", "feature"],
            "All test columns should be preserved",
        )

    def test_inheritance(self) -> None:
        """Test that BalancedAnalysisNode properly inherits from AnalysisNode."""
        # Run compute_outputs to ensure it works properly
        outputs = self.node_equal.compute_outputs()

        # Check that the outputs have the expected keys
        expected_keys = [
            "eps",
            "eps_lb",
            "eps_fpr_max_ub",
            "eps_fpr_lb",
            "eps_fpr_ub",
            "eps_tpr_lb",
            "eps_tpr_ub",
            "eps_cp",
            "accuracy",
            "accuracy_ci",
            "auc",
            "auc_ci",
            "data_size",
        ]
        for key in expected_keys:
            self.assertIn(key, outputs, f"Output should contain {key}")
