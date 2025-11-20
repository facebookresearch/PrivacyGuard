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
from numpy.testing import assert_almost_equal, assert_array_equal
from privacy_guard.analysis.mia.aggregate_analysis_input import AggregationType
from privacy_guard.attacks.lia_attack import LIAAttack, LIAAttackInput


class TestLIAAttackInput(unittest.TestCase):
    def setUp(self) -> None:
        self.df_hold_out_train_json = """{"separable_id":{"0":100101280879201,"1":100101280879201,"2":100101280879201,"3":100113514211201,"4":100116367544001},"ad_id":{"0":120202291265480062,"1":23861966757870570,"2":120201337204280551,"3":6371985258328,"4":120200794973200645},"timestamp":{"0":1700972528,"1":1700972594,"2":1700972528,"3":1700928130,"4":1700933560},"impression_signature":{"0":39367.0,"1":55814.0,"2":304.0,"3":8246.0,"4":8514.0},"predictions":{"0":0.21985362,"1":0.10969869,"2":0.24854505,"3":0.0068224324,"4":0.004189688},"label":{"0":0.0,"1":1.0,"2":0.0,"3":1.0,"4":1.0}}"""
        self.df_hold_out_train = pd.read_json(self.df_hold_out_train_json)

        self.df_hold_out_train_calib_json = """{"separable_id":{"0":100101280879201,"1":100101280879201,"2":100101280879201,"3":100113514211201,"4":100116367544001},"ad_id":{"0":120202291265480062,"1":23861966757870570,"2":120201337204280551,"3":6371985258328,"4":120200794973200645},"timestamp":{"0":1700972528,"1":1700972594,"2":1700972528,"3":1700928130,"4":1700933560},"impression_signature":{"0":39367.0,"1":55814.0,"2":304.0,"3":8246.0,"4":8514.0},"predictions":{"0":0.19985362,"1":0.12969869,"2":0.22854505,"3":0.0078224324,"4":0.005189688}, "label":{"0":0.0,"1":1.0,"2":0.0,"3":1.0,"4":1.0}}"""
        self.df_hold_out_train_calib = pd.read_json(self.df_hold_out_train_calib_json)

        super().setUp()

    def test_lia_attack_input_initialization(self) -> None:
        """Test successful initialization of LIAAttackInput."""
        lia_input = LIAAttackInput(
            df_hold_out_train=self.df_hold_out_train,
            df_hold_out_train_calib=self.df_hold_out_train_calib,
            row_aggregation=AggregationType.MAX,
        )
        self.assertIsNotNone(lia_input)
        self.assertEqual(lia_input.row_aggregation, AggregationType.MAX)
        self.assertEqual(lia_input.merge_columns, LIAAttackInput.ADS_MERGE_COLUMNS)

    def test_custom_merge_columns(self) -> None:
        """Test initialization with custom merge columns."""
        custom_columns = ["separable_id", "ad_id", "timestamp"]
        lia_input = LIAAttackInput(
            df_hold_out_train=self.df_hold_out_train,
            df_hold_out_train_calib=self.df_hold_out_train_calib,
            row_aggregation=AggregationType.MAX,
            merge_columns=custom_columns,
        )
        self.assertEqual(lia_input.merge_columns, custom_columns)

    def test_input_validation_errors(self) -> None:
        """Test that appropriate errors are raised for invalid inputs."""
        # Test missing column error
        df_missing_column = self.df_hold_out_train.drop(
            columns=["impression_signature"]
        )
        with self.assertRaises(IndexError):
            LIAAttackInput(
                df_hold_out_train=df_missing_column,
                df_hold_out_train_calib=self.df_hold_out_train_calib,
                row_aggregation=AggregationType.MAX,
            )

        # Test empty dataframe errors
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            LIAAttackInput(
                df_hold_out_train=empty_df,
                df_hold_out_train_calib=self.df_hold_out_train_calib,
                row_aggregation=AggregationType.MAX,
            )

        with self.assertRaises(ValueError):
            LIAAttackInput(
                df_hold_out_train=self.df_hold_out_train,
                df_hold_out_train_calib=empty_df,
                row_aggregation=AggregationType.MAX,
            )

        # Test missing predictions column error
        df_no_predictions = self.df_hold_out_train.drop(columns=["predictions"])
        with self.assertRaises(ValueError):
            LIAAttackInput(
                df_hold_out_train=df_no_predictions,
                df_hold_out_train_calib=self.df_hold_out_train_calib,
                row_aggregation=AggregationType.MAX,
            )

    def test_aggregate_strategies(self) -> None:
        """Test all aggregation strategies."""
        # Create test dataframe with score column
        test_df = pd.DataFrame(
            {
                "separable_id": [1, 1, 2, 2],
                "score": [-0.5, 0.3, 0.8, -0.9],
                "other_col": ["a", "b", "c", "d"],
            }
        )

        # Test ABS_MAX strategy
        lia_input_abs_max = LIAAttackInput(
            df_hold_out_train=self.df_hold_out_train,
            df_hold_out_train_calib=self.df_hold_out_train_calib,
            row_aggregation=AggregationType.ABS_MAX,
        )
        result_abs_max = lia_input_abs_max.aggregate(test_df)
        # Should select rows with highest absolute score for each separable_id
        expected_indices_abs_max = [0, 3]  # -0.5 (abs=0.5) and -0.9 (abs=0.9)
        self.assertEqual(len(result_abs_max), 2)
        assert_array_equal(result_abs_max.index.values, expected_indices_abs_max)
        assert_almost_equal(result_abs_max["score"].values, [-0.5, -0.9])

        # Test MAX strategy
        lia_input_max = LIAAttackInput(
            df_hold_out_train=self.df_hold_out_train,
            df_hold_out_train_calib=self.df_hold_out_train_calib,
            row_aggregation=AggregationType.MAX,
        )
        result_max = lia_input_max.aggregate(test_df)
        expected_indices_max = [1, 2]  # 0.3 and 0.8
        self.assertEqual(len(result_max), 2)
        assert_array_equal(result_max.index.values, expected_indices_max)

        # Test MIN strategy
        lia_input_min = LIAAttackInput(
            df_hold_out_train=self.df_hold_out_train,
            df_hold_out_train_calib=self.df_hold_out_train_calib,
            row_aggregation=AggregationType.MIN,
        )
        result_min = lia_input_min.aggregate(test_df)
        expected_indices_min = [0, 3]  # -0.5 and -0.9
        self.assertEqual(len(result_min), 2)
        assert_array_equal(result_min.index.values, expected_indices_min)

        # Test NONE strategy
        lia_input_none = LIAAttackInput(
            df_hold_out_train=self.df_hold_out_train,
            df_hold_out_train_calib=self.df_hold_out_train_calib,
            row_aggregation=AggregationType.NONE,
        )
        result_none = lia_input_none.aggregate(test_df)
        # Should return the same dataframe
        pd.testing.assert_frame_equal(result_none, test_df)

    def test_prepare_attack_input(self) -> None:
        """Test preparation of attack input."""
        lia_input = LIAAttackInput(
            df_hold_out_train=self.df_hold_out_train,
            df_hold_out_train_calib=self.df_hold_out_train_calib,
            row_aggregation=AggregationType.MAX,
        )

        attack_input = lia_input.prepare_attack_input()

        self.assertIn("df_train_and_calib", attack_input)
        self.assertIn("df_aggregated", attack_input)

        # Check that merged dataframe has both predictions columns
        df_merged = attack_input["df_train_and_calib"]
        self.assertIn("predictions", df_merged.columns)
        self.assertIn("predictions_calib", df_merged.columns)
        self.assertIn("score", df_merged.columns)

        # Check that score is calculated correctly
        expected_scores = df_merged["predictions"] - df_merged["predictions_calib"]
        assert_almost_equal(df_merged["score"].values, expected_scores.values)


class TestLIAAttack(unittest.TestCase):
    def setUp(self) -> None:
        self.df_hold_out_train_json = """{"separable_id":{"0":100101280879201,"1":100101280879201,"2":100101280879201,"3":100113514211201,"4":100116367544001},"ad_id":{"0":120202291265480062,"1":23861966757870570,"2":120201337204280551,"3":6371985258328,"4":120200794973200645},"timestamp":{"0":1700972528,"1":1700972594,"2":1700972528,"3":1700928130,"4":1700933560},"impression_signature":{"0":39367.0,"1":55814.0,"2":304.0,"3":8246.0,"4":8514.0},"predictions":{"0":0.21985362,"1":0.10969869,"2":0.24854505,"3":0.0068224324,"4":0.004189688},"label":{"0":0.0,"1":1.0,"2":0.0,"3":1.0,"4":1.0}}"""
        self.df_hold_out_train = pd.read_json(self.df_hold_out_train_json)

        self.df_hold_out_train_calib_json = """{"separable_id":{"0":100101280879201,"1":100101280879201,"2":100101280879201,"3":100113514211201,"4":100116367544001},"ad_id":{"0":120202291265480062,"1":23861966757870570,"2":120201337204280551,"3":6371985258328,"4":120200794973200645},"timestamp":{"0":1700972528,"1":1700972594,"2":1700972528,"3":1700928130,"4":1700933560},"impression_signature":{"0":39367.0,"1":55814.0,"2":304.0,"3":8246.0,"4":8514.0},"predictions":{"0":0.19985362,"1":0.12969869,"2":0.22854505,"3":0.0078224324,"4":0.005189688}, "label":{"0":0.0,"1":1.0,"2":0.0,"3":1.0,"4":1.0}}"""
        self.df_hold_out_train_calib = pd.read_json(self.df_hold_out_train_calib_json)

        # Prepare attack input
        lia_input = LIAAttackInput(
            df_hold_out_train=self.df_hold_out_train,
            df_hold_out_train_calib=self.df_hold_out_train_calib,
            row_aggregation=AggregationType.MAX,
        )
        self.attack_input = lia_input.prepare_attack_input()

        super().setUp()

    def test_lia_attack_initialization(self) -> None:
        """Test successful initialization of LIAAttack."""
        lia_attack = LIAAttack(
            attack_input=self.attack_input,
            row_aggregation=AggregationType.MAX,
            y1_generation="calibration",
            num_resampling_times=10,
        )

        self.assertIsNotNone(lia_attack)
        self.assertEqual(lia_attack.row_aggregation, AggregationType.MAX)
        self.assertEqual(lia_attack.y1_generation, "calibration")
        self.assertEqual(lia_attack.num_resampling_times, 10)

    def test_get_y1_predictions_target(self) -> None:
        """Test y1 predictions generation using target strategy."""
        lia_attack = LIAAttack(
            attack_input=self.attack_input,
            row_aggregation=AggregationType.MAX,
            y1_generation="target",
        )

        df_attack = self.attack_input["df_aggregated"]
        predictions_y1 = lia_attack.get_y1_predictions(df_attack)

        expected_predictions = df_attack["predictions"].values
        assert_array_equal(predictions_y1, expected_predictions)

    def test_get_y1_predictions_calibration(self) -> None:
        """Test y1 predictions generation using calibration strategy."""
        lia_attack = LIAAttack(
            attack_input=self.attack_input,
            row_aggregation=AggregationType.MAX,
            y1_generation="calibration",
        )

        df_attack = self.attack_input["df_aggregated"]
        predictions_y1 = lia_attack.get_y1_predictions(df_attack)

        expected_predictions = df_attack["predictions_calib"].values
        assert_array_equal(predictions_y1, expected_predictions)

    def test_get_y1_predictions_reference(self) -> None:
        """Test y1 predictions generation using reference strategy."""
        # Add reference predictions to attack input
        df_with_reference = self.attack_input["df_aggregated"].copy()
        df_with_reference["predictions_reference"] = [0.15, 0.08, 0.20]

        attack_input_with_ref = {
            "df_train_and_calib": self.attack_input["df_train_and_calib"],
            "df_aggregated": df_with_reference,
        }

        lia_attack = LIAAttack(
            attack_input=attack_input_with_ref,
            row_aggregation=AggregationType.MAX,
            y1_generation="reference",
        )

        predictions_y1 = lia_attack.get_y1_predictions(df_with_reference)

        expected_predictions = df_with_reference["predictions_reference"].values
        assert_array_equal(predictions_y1, expected_predictions)

    def test_get_y1_predictions_combo(self) -> None:
        """Test y1 predictions generation using combo strategy."""
        lia_attack = LIAAttack(
            attack_input=self.attack_input,
            row_aggregation=AggregationType.MAX,
            y1_generation="0.7",  # 70% target, 30% calibration
        )

        df_attack = self.attack_input["df_aggregated"]
        predictions_y1 = lia_attack.get_y1_predictions(df_attack)

        expected_predictions = (
            0.7 * df_attack["predictions"].values
            + 0.3 * df_attack["predictions_calib"].values
        )
        assert_almost_equal(predictions_y1, expected_predictions)

    def test_get_y1_predictions_missing_columns(self) -> None:
        """Test that ValueError is raised when required prediction columns are missing."""
        # Test missing calibration predictions
        df_no_calib = self.attack_input["df_aggregated"].drop(
            columns=["predictions_calib"]
        )
        attack_input_no_calib = {
            "df_train_and_calib": self.attack_input["df_train_and_calib"],
            "df_aggregated": df_no_calib,
        }

        lia_attack_calib = LIAAttack(
            attack_input=attack_input_no_calib,
            row_aggregation=AggregationType.MAX,
            y1_generation="calibration",
        )

        with self.assertRaises(ValueError):
            lia_attack_calib.get_y1_predictions(df_no_calib)

        # Test missing reference predictions
        lia_attack_ref = LIAAttack(
            attack_input=self.attack_input,
            row_aggregation=AggregationType.MAX,
            y1_generation="reference",
        )

        df_attack = self.attack_input["df_aggregated"]
        with self.assertRaises(ValueError):
            lia_attack_ref.get_y1_predictions(df_attack)

    def test_y1_generation_statistical_properties(self) -> None:
        """Test that y1 generation produces correct statistical properties."""
        # Create a simple test case with constant calibration predictions
        test_df = pd.DataFrame(
            {
                "separable_id": [1, 2, 3, 4, 5],
                "predictions": [0.5, 0.5, 0.5, 0.5, 0.5],
                "predictions_calib": [0.3, 0.3, 0.3, 0.3, 0.3],
                "label": [0, 1, 0, 1, 0],
            }
        )

        attack_input_test = {
            "df_train_and_calib": test_df,
            "df_aggregated": test_df,
        }

        lia_attack = LIAAttack(
            attack_input=attack_input_test,
            row_aggregation=AggregationType.NONE,
            y1_generation="calibration",
            num_resampling_times=1000,
        )

        analysis_input = lia_attack.run_attack()

        # Check that mean of y1 is close to 0.3 (the calibration prediction value)
        y1_mean = np.mean(analysis_input.y1)
        self.assertAlmostEqual(y1_mean, 0.3, delta=0.05)

    def test_run_attack_with_aggregation(self) -> None:
        """Test running LIA attack with aggregation."""
        lia_attack = LIAAttack(
            attack_input=self.attack_input,
            row_aggregation=AggregationType.MAX,
            y1_generation="calibration",
            num_resampling_times=5,
        )

        analysis_input = lia_attack.run_attack()

        self.assertIsNotNone(analysis_input)
        self.assertEqual(
            analysis_input.predictions.shape[0], len(self.attack_input["df_aggregated"])
        )
        self.assertEqual(
            analysis_input.true_bits.shape, (5, len(self.attack_input["df_aggregated"]))
        )
        self.assertEqual(
            analysis_input.y1.shape, (5, len(self.attack_input["df_aggregated"]))
        )
        self.assertEqual(
            analysis_input.received_labels.shape,
            (5, len(self.attack_input["df_aggregated"])),
        )

    def test_run_attack_without_aggregation(self) -> None:
        """Test running LIA attack without aggregation."""
        # Prepare attack input without aggregation
        lia_input = LIAAttackInput(
            df_hold_out_train=self.df_hold_out_train,
            df_hold_out_train_calib=self.df_hold_out_train_calib,
            row_aggregation=AggregationType.NONE,
        )
        attack_input_no_agg = lia_input.prepare_attack_input()

        lia_attack = LIAAttack(
            attack_input=attack_input_no_agg,
            row_aggregation=AggregationType.NONE,
            y1_generation="calibration",
            num_resampling_times=3,
        )

        analysis_input = lia_attack.run_attack()

        self.assertIsNotNone(analysis_input)
        self.assertEqual(
            analysis_input.predictions.shape[0],
            len(attack_input_no_agg["df_train_and_calib"]),
        )
        self.assertEqual(
            analysis_input.true_bits.shape,
            (3, len(attack_input_no_agg["df_train_and_calib"])),
        )

    def test_run_attack_analysis_input_structure(self) -> None:
        """Test that the analysis input has the correct structure and data types."""
        num_resampling_times = 100
        lia_attack = LIAAttack(
            attack_input=self.attack_input,
            row_aggregation=AggregationType.MAX,
            y1_generation="target",
            num_resampling_times=num_resampling_times,
        )

        analysis_input = lia_attack.run_attack()
        expected_num_samples = len(self.attack_input["df_aggregated"])

        # Check data types and shapes
        self.assertIsInstance(analysis_input.predictions, np.ndarray)
        self.assertIsInstance(analysis_input.predictions_y1_generation, np.ndarray)
        self.assertIsInstance(analysis_input.true_bits, np.ndarray)
        self.assertIsInstance(analysis_input.y0, np.ndarray)
        self.assertIsInstance(analysis_input.y1, np.ndarray)
        self.assertIsInstance(analysis_input.received_labels, np.ndarray)

        # Check shapes
        self.assertEqual(analysis_input.predictions.shape, (expected_num_samples,))
        self.assertEqual(
            analysis_input.predictions_y1_generation.shape, (expected_num_samples,)
        )
        self.assertEqual(
            analysis_input.true_bits.shape, (num_resampling_times, expected_num_samples)
        )
        self.assertEqual(analysis_input.y0.shape, (expected_num_samples,))
        self.assertEqual(
            analysis_input.y1.shape, (num_resampling_times, expected_num_samples)
        )
        self.assertEqual(
            analysis_input.received_labels.shape,
            (num_resampling_times, expected_num_samples),
        )

        # Check that true_bits contains only 0s and 1s
        self.assertTrue(np.all(np.isin(analysis_input.true_bits, [0, 1])))

        # Check that y1 contains only 0s and 1s
        self.assertTrue(np.all(np.isin(analysis_input.y1, [0, 1])))

        # Check that mean of true_bits is close to 0.5 (should be approximately uniform random)
        true_bits_mean = np.mean(analysis_input.true_bits)
        self.assertAlmostEqual(true_bits_mean, 0.5, delta=0.1)

        # Check that received_labels logic is correct
        # When true_bits == 0, received_labels should equal y0
        # When true_bits == 1, received_labels should equal y1
        for i in range(analysis_input.true_bits.shape[0]):
            for j in range(analysis_input.true_bits.shape[1]):
                if analysis_input.true_bits[i, j] == 0:
                    self.assertEqual(
                        analysis_input.received_labels[i, j], analysis_input.y0[j]
                    )
                else:
                    self.assertEqual(
                        analysis_input.received_labels[i, j], analysis_input.y1[i, j]
                    )
