# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal
from privacy_guard.analysis.mia.aggregate_analysis_input import (
    AggregateAnalysisInput,
    AggregationType,
)
from privacy_guard.attacks.rmia_attack import RmiaAttack


class TestRmiaAttack(unittest.TestCase):
    def setUp(self) -> None:
        # Create sample data for training
        self.df_train_merge = pd.DataFrame(
            {
                "user_id": [
                    123456,
                    123456,
                    789012,
                    345678,
                    901234,
                ],
                "score_orig": [0.8, 0.7, 0.6, 0.9, 0.5],
                "score_ref_0": [0.4, 0.3, 0.5, 0.4, 0.6],
                "score_ref_1": [0.5, 0.4, 0.4, 0.5, 0.3],
                "score_ref_2": [0.3, 0.5, 0.6, 0.3, 0.4],
                "member_ref_0": [True, False, True, False, True],
                "member_ref_1": [False, True, False, True, False],
                "member_ref_2": [True, True, False, False, True],
            }
        )

        # Create sample data for testing (same structure but different values)
        self.df_test_merge = pd.DataFrame(
            {
                "user_id": [
                    567890,
                    567890,
                    112233,
                    445566,
                    778899,
                ],
                "score_orig": [0.2, 0.3, 0.4, 0.1, 0.5],
                "score_ref_0": [0.6, 0.7, 0.5, 0.6, 0.4],
                "score_ref_1": [0.5, 0.6, 0.6, 0.5, 0.7],
                "score_ref_2": [0.7, 0.5, 0.4, 0.7, 0.6],
                "member_ref_0": [False, True, False, True, False],
                "member_ref_1": [True, False, True, False, True],
                "member_ref_2": [False, False, True, True, False],
            }
        )

        # Create population data for RMIA computation
        self.df_population = pd.DataFrame(
            {
                "user_id": [111222, 333444, 555666],
                "score_orig": [0.45, 0.55, 0.65],
                "score_ref_0": [0.4, 0.5, 0.6],
                "score_ref_1": [0.5, 0.4, 0.5],
                "score_ref_2": [0.6, 0.6, 0.4],
                "member_ref_0": [
                    False,
                    False,
                    False,
                ],  # Population data is always non-member
                "member_ref_1": [False, False, False],
                "member_ref_2": [False, False, False],
            }
        )

        # Initialize RMIA attack instance
        self.rmia_attack = RmiaAttack(
            df_train_merge=self.df_train_merge,
            df_test_merge=self.df_test_merge,
            df_population=self.df_population,
            row_aggregation=AggregationType.MAX,
            alpha_coefficient=0.3,
            num_reference_models=2,
            enable_auto_tuning=False,
        )

        # Initialize RMIA attack with auto-tuning enabled
        self.rmia_attack_auto_tune = RmiaAttack(
            df_train_merge=self.df_train_merge,
            df_test_merge=self.df_test_merge,
            df_population=self.df_population,
            row_aggregation=AggregationType.MAX,
            alpha_coefficient=0.3,
            num_reference_models=2,
            enable_auto_tuning=True,
        )

        super().setUp()

    def test_initialization_success(self) -> None:
        """Test successful initialization of RmiaAttack"""
        attack = RmiaAttack(
            df_train_merge=self.df_train_merge,
            df_test_merge=self.df_test_merge,
            df_population=self.df_population,
            row_aggregation=AggregationType.MAX,
            num_reference_models=2,
        )

        self.assertEqual(attack.alpha_coefficient, 0.3)
        self.assertEqual(attack.num_reference_models, 2)
        self.assertFalse(attack.enable_auto_tuning)
        self.assertEqual(attack.row_aggregation, AggregationType.MAX)

    def test_initialization_empty_dataframe_error(self) -> None:
        """Test that ValueError is raised when dataframes are empty"""
        empty_df = pd.DataFrame()

        with self.assertRaises(ValueError) as context:
            RmiaAttack(
                df_train_merge=empty_df,
                df_test_merge=self.df_test_merge,
                df_population=self.df_population,
                row_aggregation=AggregationType.MAX,
                num_reference_models=2,
            )
        self.assertIn("df_train_merge cannot be empty", str(context.exception))

    def test_initialization_missing_score_orig_error(self) -> None:
        """Test that ValueError is raised when score_orig column is missing"""
        df_missing_score = self.df_train_merge.drop(columns=["score_orig"])

        with self.assertRaises(ValueError) as context:
            RmiaAttack(
                df_train_merge=df_missing_score,
                df_test_merge=self.df_test_merge,
                df_population=self.df_population,
                row_aggregation=AggregationType.MAX,
                num_reference_models=2,
            )
        self.assertIn("must contain 'score_orig' column", str(context.exception))

    def test_initialization_missing_ref_score_columns_error(self) -> None:
        """Test that ValueError is raised when reference score columns are missing"""
        df_missing_ref_scores = self.df_train_merge.drop(
            columns=["score_ref_0", "score_ref_1", "score_ref_2"]
        )

        with self.assertRaises(ValueError) as context:
            RmiaAttack(
                df_train_merge=df_missing_ref_scores,
                df_test_merge=self.df_test_merge,
                df_population=self.df_population,
                row_aggregation=AggregationType.MAX,
                num_reference_models=2,
            )
        self.assertIn("No reference score columns found", str(context.exception))

    def test_initialization_missing_ref_member_columns_error(self) -> None:
        """Test that ValueError is raised when reference member columns are missing"""
        df_missing_ref_members = self.df_train_merge.drop(
            columns=["member_ref_0", "member_ref_1", "member_ref_2"]
        )

        with self.assertRaises(ValueError) as context:
            RmiaAttack(
                df_train_merge=df_missing_ref_members,
                df_test_merge=self.df_test_merge,
                df_population=self.df_population,
                row_aggregation=AggregationType.MAX,
                num_reference_models=2,
            )
        self.assertIn("No reference membership columns found", str(context.exception))

    def test_extract_model_data_success(self) -> None:
        """Test successful extraction of model data from dataframe"""
        target_scores, ref_scores, ref_memberships = (
            self.rmia_attack._extract_model_data(self.df_train_merge)
        )

        # Verify shapes
        self.assertEqual(target_scores.shape, (5,))
        self.assertEqual(ref_scores.shape, (5, 3))
        self.assertEqual(ref_memberships.shape, (5, 3))

        # Verify values
        expected_target_scores = np.array([0.8, 0.7, 0.6, 0.9, 0.5])
        assert_array_almost_equal(target_scores, expected_target_scores)

        # Verify membership is boolean
        self.assertTrue(ref_memberships.dtype == bool)

    def test_extract_model_data_missing_score_orig_error(self) -> None:
        """Test error when score_orig column is missing from dataframe"""
        df_missing_score = self.df_train_merge.drop(columns=["score_orig"])

        with self.assertRaises(ValueError) as context:
            self.rmia_attack._extract_model_data(df_missing_score)
        self.assertIn("Missing required 'score_orig' column", str(context.exception))

    def test_compute_ref_signal_averages_multiple_models(self) -> None:
        """Test computation of reference signal averages with multiple models"""
        ref_signals = np.array(
            [
                [0.4, 0.5, 0.3],
                [0.3, 0.4, 0.5],
                [0.5, 0.4, 0.6],
            ]
        )
        ref_memberships = np.array(
            [
                [True, False, True],
                [False, True, True],
                [True, False, False],
            ]
        )

        result = RmiaAttack.compute_ref_signal_averages(
            ref_signals=ref_signals,
            ref_memberships=ref_memberships,
            num_models=2,
            alpha=0.3,
        )

        # Verify shape
        self.assertEqual(result.shape, (3, 2))

        # Verify non-zero exclusion logic works
        self.assertTrue(np.all(result >= 0))

    def test_compute_ref_signal_averages_single_model(self) -> None:
        """Test computation of reference signal averages with single model"""
        ref_signals = np.array(
            [
                [0.4],
                [0.3],
                [0.5],
            ]
        )
        ref_memberships = np.array(
            [
                [True],
                [False],
                [True],
            ]
        )

        result = RmiaAttack.compute_ref_signal_averages(
            ref_signals=ref_signals,
            ref_memberships=ref_memberships,
            num_models=1,
            alpha=0.3,
        )

        # Verify shape
        self.assertEqual(result.shape, (3, 1))

    def test_compute_ref_signal_averages_alpha_zero(self) -> None:
        """Test computation with alpha=0 (fallback approximation)"""
        ref_signals = np.array(
            [
                [0.4],
                [0.8],
            ]
        )
        ref_memberships = np.array(
            [
                [True],
                [False],
            ]
        )

        result = RmiaAttack.compute_ref_signal_averages(
            ref_signals=ref_signals,
            ref_memberships=ref_memberships,
            num_models=1,
            alpha=0.0,
        )

        # Verify fallback formula is applied
        self.assertEqual(result.shape, (2, 1))

    def test_compute_membership_scores(self) -> None:
        """Test core membership score computation"""
        # Create simple test data
        target_scores = np.array([0.8, 0.2])
        ref_scores = np.array([[0.4, 0.5], [0.6, 0.5]])
        ref_memberships = np.array([[True, False], [False, True]])
        population_target_scores = np.array([0.5, 0.6])
        population_ref_scores = np.array([[0.4, 0.5], [0.5, 0.4]])

        scores = self.rmia_attack._compute_membership_scores(
            target_scores=target_scores,
            ref_scores=ref_scores,
            ref_memberships=ref_memberships,
            population_target_scores=population_target_scores,
            population_ref_scores=population_ref_scores,
            alpha=0.3,
            num_models=1,
        )

        # Verify output shape and type
        self.assertEqual(scores.shape, (2,))
        self.assertTrue(np.all(scores >= 0))
        self.assertTrue(np.all(scores <= 1))

    def test_auto_tune_alpha_coefficient(self) -> None:
        """Test automatic tuning of alpha coefficient using reference models"""
        # Create test data with multiple reference models (need at least 2 for auto-tuning)
        ref_scores = np.array([[0.4, 0.5, 0.3], [0.6, 0.5, 0.4], [0.3, 0.4, 0.6]])
        ref_memberships = np.array(
            [[True, False, True], [False, True, False], [True, False, True]]
        )
        population_ref_scores = np.array([[0.4, 0.5, 0.3], [0.5, 0.4, 0.5]])

        best_alpha = self.rmia_attack._auto_tune_alpha_coefficient(
            ref_scores=ref_scores,
            ref_memberships=ref_memberships,
            population_ref_scores=population_ref_scores,
            target_model_idx=0,
        )

        # Verify alpha is in valid range
        self.assertGreaterEqual(best_alpha, 0.0)
        self.assertLessEqual(best_alpha, 1.0)

    def test_auto_tune_alpha_coefficient_insufficient_models(self) -> None:
        """Test auto-tuning with insufficient reference models (< 2)"""
        # Create test data with only 1 reference model
        ref_scores = np.array([[0.4], [0.6], [0.3]])
        ref_memberships = np.array([[True], [False], [True]])
        population_ref_scores = np.array([[0.4], [0.5]])

        # Should return the original alpha coefficient when insufficient models
        best_alpha = self.rmia_attack._auto_tune_alpha_coefficient(
            ref_scores=ref_scores,
            ref_memberships=ref_memberships,
            population_ref_scores=population_ref_scores,
            target_model_idx=0,
        )

        # Should return the original alpha coefficient (0.3 from setUp)
        self.assertEqual(best_alpha, self.rmia_attack.alpha_coefficient)

    def test_run_attack_success(self) -> None:
        """Test successful execution of RMIA attack"""
        analysis_input = self.rmia_attack.run_attack()

        # Verify return type
        self.assertIsInstance(analysis_input, AggregateAnalysisInput)

        # Verify output dataframes have scores
        self.assertIn("score", analysis_input.df_train_merge.columns)
        self.assertIn("score", analysis_input.df_test_merge.columns)

        # Verify output lengths match input
        self.assertEqual(len(analysis_input.df_train_merge), len(self.df_train_merge))
        self.assertEqual(len(analysis_input.df_test_merge), len(self.df_test_merge))

        # Verify scores are numeric and in reasonable range
        train_scores = analysis_input.df_train_merge["score"].values
        test_scores = analysis_input.df_test_merge["score"].values

        self.assertTrue(np.all(train_scores >= 0))
        self.assertTrue(np.all(train_scores <= 1))
        self.assertTrue(np.all(test_scores >= 0))
        self.assertTrue(np.all(test_scores <= 1))

    def test_run_attack_with_auto_tuning(self) -> None:
        """Test RMIA attack execution with auto-tuning enabled"""
        analysis_input = self.rmia_attack_auto_tune.run_attack()

        # Verify successful execution
        self.assertIsInstance(analysis_input, AggregateAnalysisInput)

        # Verify output has scores
        self.assertIn("score", analysis_input.df_train_merge.columns)
        self.assertIn("score", analysis_input.df_test_merge.columns)

    def test_different_alpha_coefficients(self) -> None:
        """Test attack with different alpha coefficient values"""
        alpha_values = [0.0, 0.1, 0.5, 0.9, 1.0]

        for alpha in alpha_values:
            attack = RmiaAttack(
                df_train_merge=self.df_train_merge,
                df_test_merge=self.df_test_merge,
                df_population=self.df_population,
                row_aggregation=AggregationType.MAX,
                alpha_coefficient=alpha,
                num_reference_models=2,
            )

            analysis_input = attack.run_attack()

            # Verify successful execution for all alpha values
            self.assertIsInstance(analysis_input, AggregateAnalysisInput)
            self.assertIn("score", analysis_input.df_train_merge.columns)

    def test_different_num_reference_models(self) -> None:
        """Test attack with different numbers of reference models"""
        num_models_values = [None, 1, 2, 3, 10]

        for num_models in num_models_values:
            attack = RmiaAttack(
                df_train_merge=self.df_train_merge,
                df_test_merge=self.df_test_merge,
                df_population=self.df_population,
                row_aggregation=AggregationType.MAX,
                num_reference_models=num_models,
            )

            analysis_input = attack.run_attack()

            # Verify successful execution for all model counts
            self.assertIsInstance(analysis_input, AggregateAnalysisInput)
            self.assertIn("score", analysis_input.df_train_merge.columns)

    def test_score_determinism(self) -> None:
        """Test that running the same attack twice produces identical results"""
        # First run
        analysis_input_1 = self.rmia_attack.run_attack()
        scores_1_train = analysis_input_1.df_train_merge["score"].values
        scores_1_test = analysis_input_1.df_test_merge["score"].values

        # Second run with same parameters
        rmia_attack_2 = RmiaAttack(
            df_train_merge=self.df_train_merge,
            df_test_merge=self.df_test_merge,
            df_population=self.df_population,
            row_aggregation=AggregationType.MAX,
            alpha_coefficient=0.3,
            num_reference_models=2,
            enable_auto_tuning=False,
        )
        analysis_input_2 = rmia_attack_2.run_attack()
        scores_2_train = analysis_input_2.df_train_merge["score"].values
        scores_2_test = analysis_input_2.df_test_merge["score"].values

        # Verify identical results
        assert_array_almost_equal(scores_1_train, scores_2_train, decimal=10)
        assert_array_almost_equal(scores_1_test, scores_2_test, decimal=10)

    def test_class_constants(self) -> None:
        """Test that class constants are properly defined"""
        self.assertEqual(RmiaAttack.REF_SCORE_PREFIX, "score_ref_")
        self.assertEqual(RmiaAttack.REF_MEMBER_PREFIX, "member_ref_")

    def test_aggregation_type_preservation(self) -> None:
        """Test that aggregation type is preserved in output"""
        analysis_input = self.rmia_attack.run_attack()
        self.assertEqual(analysis_input.row_aggregation, AggregationType.MAX)

    def test_compute_ref_signal_averages_num_models_none(self) -> None:
        """Test compute_ref_signal_averages when num_models is None"""
        ref_signals = np.array(
            [
                [0.4, 0.5, 0.3],
                [0.3, 0.4, 0.5],
                [0.5, 0.4, 0.6],
            ]
        )
        ref_memberships = np.array(
            [
                [True, False, True],
                [False, True, True],
                [True, False, False],
            ]
        )

        # Call with num_models=None
        result = RmiaAttack.compute_ref_signal_averages(
            ref_signals=ref_signals,
            ref_memberships=ref_memberships,
            num_models=None,
            alpha=0.3,
        )

        # Verify that num_models was set to ref_signals.shape[1] (which is 3)
        self.assertEqual(result.shape, (3, 3))

    def test_extract_model_data_missing_ref_score_columns_error(self) -> None:
        """Test error when reference score columns are missing"""
        # Create dataframe without reference score columns
        df_no_ref_scores = pd.DataFrame(
            {
                "score_orig": [0.8, 0.7, 0.6],
                "member_ref_0": [True, False, True],
                "member_ref_1": [False, True, False],
            }
        )

        with self.assertRaises(ValueError) as context:
            self.rmia_attack._extract_model_data(df_no_ref_scores)

        self.assertIn("No reference score columns found", str(context.exception))
        self.assertIn("score_ref_", str(context.exception))

    def test_extract_model_data_missing_ref_member_columns_error(self) -> None:
        """Test error when reference member columns are missing"""
        # Create dataframe without reference member columns
        df_no_ref_members = pd.DataFrame(
            {
                "score_orig": [0.8, 0.7, 0.6],
                "score_ref_0": [0.4, 0.3, 0.5],
                "score_ref_1": [0.5, 0.4, 0.4],
            }
        )

        with self.assertRaises(ValueError) as context:
            self.rmia_attack._extract_model_data(df_no_ref_members)

        self.assertIn("No membership columns found", str(context.exception))
        self.assertIn("member_ref_", str(context.exception))
