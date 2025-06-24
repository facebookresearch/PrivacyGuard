# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict


import unittest

import pandas as pd
from privacy_guard.analysis.aggregate_analysis_input import (
    AggregateAnalysisInput,
    AggregationType,
)

from privacy_guard.attacks.lira_attack import LiraAttack


class TestLiraAttack(unittest.TestCase):
    def setUp(self) -> None:
        self.df_train_merge = {
            "separable_id": {
                "0": 100101280879201,
                "1": 100101280879201,
                "2": 100101280879201,
                "3": 100113514211201,
                "4": 100116367544001,
            },
            "ad_id": {
                "0": 120202291265480062,
                "1": 23861966757870570,
                "2": 120201337204280551,
                "3": 6371985258328,
                "4": 120200794973200645,
            },
            "timestamp": {
                "0": 1700972528,
                "1": 1700972594,
                "2": 1700972528,
                "3": 1700928130,
                "4": 1700933560,
            },
            "impression_signature": {
                "0": 39367.0,
                "1": 55814.0,
                "2": 304.0,
                "3": 8246.0,
                "4": 8514.0,
            },
            "score_orig": {"0": 0.2, "1": 0.4, "2": 0.5, "3": 0.67, "4": 0.99},
            "score_mean": {"0": 0.2, "1": 0.4, "2": 0.5, "3": 0.67, "4": 0.99},
            "score_std": {"0": 0.2, "1": 0.2, "2": 0.5, "3": 0.67, "4": 0.2},
            # Additional columns for online attack tests
            "score_mean_in": {"0": 0.25, "1": 0.45, "2": 0.55, "3": 0.7, "4": 0.95},
            "score_mean_out": {"0": 0.15, "1": 0.35, "2": 0.45, "3": 0.6, "4": 0.9},
            "score_std_in": {"0": 0.1, "1": 0.15, "2": 0.2, "3": 0.25, "4": 0.3},
            "score_std_out": {"0": 0.05, "1": 0.1, "2": 0.15, "3": 0.2, "4": 0.25},
        }
        self.df_train_merge = pd.DataFrame.from_dict(self.df_train_merge)

        self.lira_attack = LiraAttack(
            df_train_merge=self.df_train_merge,
            df_test_merge=self.df_train_merge,
            row_aggregation=AggregationType.MAX,
            use_fixed_variance=True,
        )

        self.lira_attack_no_fixed_variance = LiraAttack(
            df_train_merge=self.df_train_merge,
            df_test_merge=self.df_train_merge,
            row_aggregation=AggregationType.MAX,
            use_fixed_variance=False,
        )

        super().setUp()

    def test_run_attack_presto_mocked(self) -> None:
        analysis_input = self.lira_attack.run_attack()
        self.assertTrue(isinstance(analysis_input, AggregateAnalysisInput))

        self.assertTrue(analysis_input is not None)
        self.assertEqual(len(analysis_input.df_train_merge), 5)
        self.assertEqual(len(analysis_input.df_test_merge), 5)

    def test_run_attack_presto_mocked_no_fixed_variance(self) -> None:
        analysis_input = self.lira_attack_no_fixed_variance.run_attack()
        self.assertTrue(isinstance(analysis_input, AggregateAnalysisInput))

        self.assertTrue(analysis_input is not None)
        self.assertEqual(len(analysis_input.df_train_merge), 5)
        self.assertEqual(len(analysis_input.df_test_merge), 5)

    def test_get_std_dev_global(self) -> None:
        """Test _get_std_dev with std_dev_type='global'"""
        # Setup
        attack = LiraAttack(
            df_train_merge=self.df_train_merge,
            df_test_merge=self.df_train_merge,
            row_aggregation=AggregationType.MAX,
            std_dev_type="global",
        )

        # Execute
        std_in, std_out = attack._get_std_dev()

        # Verify
        # Calculate expected standard deviation of all score_orig values
        expected_std = pd.concat(
            [
                self.df_train_merge.score_orig,
                self.df_train_merge.score_orig,  # df_test_merge is same as df_train_merge in this test
            ]
        ).std()

        self.assertAlmostEqual(float(std_in), float(expected_std))
        self.assertAlmostEqual(float(std_out), float(expected_std))
        self.assertEqual(
            std_in, std_out
        )  # For global, std_in and std_out should be equal

    def test_get_std_dev_shadows_in_offline(self) -> None:
        """Test _get_std_dev with std_dev_type='shadows_in' and online_attack=False"""
        # Setup
        attack = LiraAttack(
            df_train_merge=self.df_train_merge,
            df_test_merge=self.df_train_merge,
            row_aggregation=AggregationType.MAX,
            std_dev_type="shadows_in",
            online_attack=False,
        )

        # Execute
        std_in, std_out = attack._get_std_dev()

        # Verify
        # Calculate expected mean of all score_std values
        expected_std = pd.concat(
            [
                self.df_train_merge.score_std,
                self.df_train_merge.score_std,  # df_test_merge is same as df_train_merge in this test
            ]
        ).mean()

        self.assertAlmostEqual(float(std_in), float(expected_std))
        self.assertAlmostEqual(float(std_out), float(expected_std))
        self.assertEqual(
            std_in, std_out
        )  # For shadows_in offline, std_in and std_out should be equal

    def test_get_std_dev_shadows_in_online(self) -> None:
        """Test _get_std_dev with std_dev_type='shadows_in' and online_attack=True"""
        # Setup
        attack = LiraAttack(
            df_train_merge=self.df_train_merge,
            df_test_merge=self.df_train_merge,
            row_aggregation=AggregationType.MAX,
            std_dev_type="shadows_in",
            online_attack=True,
        )

        # Execute
        std_in, std_out = attack._get_std_dev()

        # Verify
        # Calculate expected mean of all score_std_in values
        expected_std = pd.concat(
            [
                self.df_train_merge.score_std_in,
                self.df_train_merge.score_std_in,  # df_test_merge is same as df_train_merge in this test
            ]
        ).mean()

        self.assertAlmostEqual(float(std_in), float(expected_std))
        self.assertAlmostEqual(float(std_out), float(expected_std))
        self.assertEqual(
            std_in, std_out
        )  # For shadows_in online, std_in and std_out should be equal

    def test_get_std_dev_shadows_out_offline(self) -> None:
        """Test _get_std_dev with std_dev_type='shadows_out' and online_attack=False"""
        # Setup
        attack = LiraAttack(
            df_train_merge=self.df_train_merge,
            df_test_merge=self.df_train_merge,
            row_aggregation=AggregationType.MAX,
            std_dev_type="shadows_out",
            online_attack=False,
        )

        # Execute
        std_in, std_out = attack._get_std_dev()

        # Verify
        # Calculate expected mean of all score_std values
        expected_std = pd.concat(
            [
                self.df_train_merge.score_std,
                self.df_train_merge.score_std,  # df_test_merge is same as df_train_merge in this test
            ]
        ).mean()

        self.assertAlmostEqual(float(std_in), float(expected_std))
        self.assertAlmostEqual(float(std_out), float(expected_std))
        self.assertEqual(
            std_in, std_out
        )  # For shadows_out offline, std_in and std_out should be equal

    def test_get_std_dev_shadows_out_online(self) -> None:
        """Test _get_std_dev with std_dev_type='shadows_out' and online_attack=True"""
        # Setup
        attack = LiraAttack(
            df_train_merge=self.df_train_merge,
            df_test_merge=self.df_train_merge,
            row_aggregation=AggregationType.MAX,
            std_dev_type="shadows_out",
            online_attack=True,
        )

        # Execute
        std_in, std_out = attack._get_std_dev()

        # Verify
        # Calculate expected mean of all score_std_out values
        expected_std = pd.concat(
            [
                self.df_train_merge.score_std_out,
                self.df_train_merge.score_std_out,  # df_test_merge is same as df_train_merge in this test
            ]
        ).mean()

        self.assertAlmostEqual(float(std_in), float(expected_std))
        self.assertAlmostEqual(float(std_out), float(expected_std))
        self.assertEqual(
            std_in, std_out
        )  # For shadows_out online, std_in and std_out should be equal

    def test_get_std_dev_mix(self) -> None:
        """Test _get_std_dev with std_dev_type='mix' and online_attack=True"""
        # Setup
        attack = LiraAttack(
            df_train_merge=self.df_train_merge,
            df_test_merge=self.df_train_merge,
            row_aggregation=AggregationType.MAX,
            std_dev_type="mix",
            online_attack=True,
        )

        # Execute
        std_in, std_out = attack._get_std_dev()

        # Verify
        # Calculate expected mean of all score_std_in values for std_in
        expected_std_in = pd.concat(
            [
                self.df_train_merge.score_std_in,
                self.df_train_merge.score_std_in,  # df_test_merge is same as df_train_merge in this test
            ]
        ).mean()

        # Calculate expected mean of all score_std_out values for std_out
        expected_std_out = pd.concat(
            [
                self.df_train_merge.score_std_out,
                self.df_train_merge.score_std_out,  # df_test_merge is same as df_train_merge in this test
            ]
        ).mean()

        self.assertAlmostEqual(float(std_in), float(expected_std_in))
        self.assertAlmostEqual(float(std_out), float(expected_std_out))
        self.assertNotEqual(
            std_in, std_out
        )  # For mix, std_in and std_out should be different

    def test_get_std_dev_mix_offline_error(self) -> None:
        """Test _get_std_dev with std_dev_type='mix' and online_attack=False raises ValueError"""
        # Setup
        attack = LiraAttack(
            df_train_merge=self.df_train_merge,
            df_test_merge=self.df_train_merge,
            row_aggregation=AggregationType.MAX,
            std_dev_type="mix",
            online_attack=False,
        )

        # Execute and Verify
        with self.assertRaises(ValueError) as context:
            attack._get_std_dev()

        self.assertIn(
            "mix std dev type is only supported for online attacks",
            str(context.exception),
        )

    def test_get_std_dev_invalid_type(self) -> None:
        """Test _get_std_dev with invalid std_dev_type raises ValueError"""
        # Setup
        attack = LiraAttack(
            df_train_merge=self.df_train_merge,
            df_test_merge=self.df_train_merge,
            row_aggregation=AggregationType.MAX,
            std_dev_type="invalid_type",
        )

        # Execute and Verify
        with self.assertRaises(ValueError) as context:
            attack._get_std_dev()

        self.assertIn("is not a valid std_dev type", str(context.exception))
