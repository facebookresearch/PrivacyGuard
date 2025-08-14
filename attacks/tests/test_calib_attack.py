# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict


import unittest

import pandas as pd

from numpy.testing import assert_almost_equal
from privacy_guard.analysis.mia.aggregate_analysis_input import AggregationType

from privacy_guard.attacks.calib_attack import CalibAttack, CalibScoreType


class TestCalibAttack(unittest.TestCase):
    def setUp(self) -> None:
        self.df_hold_out_train_json = """{"separable_id":{"0":100101280879201,"1":100101280879201,"2":100101280879201,"3":100113514211201,"4":100116367544001},"ad_id":{"0":120202291265480062,"1":23861966757870570,"2":120201337204280551,"3":6371985258328,"4":120200794973200645},"timestamp":{"0":1700972528,"1":1700972594,"2":1700972528,"3":1700928130,"4":1700933560},"impression_signature":{"0":39367.0,"1":55814.0,"2":304.0,"3":8246.0,"4":8514.0},"predictions":{"0":0.21985362,"1":0.10969869,"2":0.24854505,"3":0.0068224324,"4":0.004189688},"label":{"0":0.0,"1":1.0,"2":0.0,"3":1.0,"4":1.0}}"""
        self.df_hold_out_train = pd.read_json(self.df_hold_out_train_json)

        self.eval_output_table_settings = {
            "prediction_dtype": "array<float>",
            "prediction_name": "prediction_optimized",
            "eval_date_name": "ds",
            "oba_status_name": "user_oba_opt_out_status",
            "impression_signature_name": "impression_signature",
            "imp_device_type_name": "imp_device_type",
            "country_bucket_name": "country_bucket",
            "device_os_version_name": "device_os_version",
        }
        self.calibrated_calib_attack = CalibAttack(
            df_hold_out_train=self.df_hold_out_train,
            df_hold_out_test=self.df_hold_out_train,
            df_hold_out_train_calib=self.df_hold_out_train,
            df_hold_out_test_calib=self.df_hold_out_train,
            row_aggregation=AggregationType.MAX,
            should_calibrate_scores=True,
            score_type=CalibScoreType.LOSS,
        )

        self.calib_attack = CalibAttack(
            df_hold_out_train=self.df_hold_out_train,
            df_hold_out_test=self.df_hold_out_train,
            df_hold_out_train_calib=self.df_hold_out_train,
            df_hold_out_test_calib=self.df_hold_out_train,
            row_aggregation=AggregationType.MAX,
            should_calibrate_scores=False,
            score_type=CalibScoreType.LOSS,
        )

        super().setUp()

    def test_run_attack_presto_mocked(self) -> None:
        analysis_input = self.calib_attack.run_attack()
        self.assertTrue(analysis_input is not None)

        analysis_input = self.calibrated_calib_attack.run_attack()
        self.assertTrue(analysis_input is not None)

    def test_compute_score(self) -> None:
        _ = CalibAttack.compute_score(df=self.df_hold_out_train, score_type="loss")
        _ = CalibAttack.compute_score(df=self.df_hold_out_train, score_type="entropy")
        _ = CalibAttack.compute_score(
            df=self.df_hold_out_train, score_type="confidence"
        )
        _ = CalibAttack.compute_score(
            df=self.df_hold_out_train, score_type="scaled_logits"
        )
        with self.assertRaises(ValueError):
            _ = CalibAttack.compute_score(
                df=self.df_hold_out_train, score_type="no_score"
            )

    def test_scaled_logits_values(self) -> None:
        assert_almost_equal(
            CalibAttack.compute_score(
                df=self.df_hold_out_train, score_type="scaled_logits"
            ),
            [-1.26651961, 2.09382253, -1.10638714, 4.9806934, 5.47093052],
            decimal=7,
        )

    def test_column_validation(self) -> None:
        """Test that an IndexError is raised when a required column is missing."""
        # Create a dataframe missing one of the required columns
        df_missing_column = self.df_hold_out_train.drop(
            columns=["impression_signature"]
        )

        # Verify that an IndexError is raised when trying to create a CalibAttack instance
        with self.assertRaises(IndexError) as context:
            CalibAttack(
                df_hold_out_train=df_missing_column,
                df_hold_out_test=self.df_hold_out_train,
                df_hold_out_train_calib=self.df_hold_out_train,
                df_hold_out_test_calib=self.df_hold_out_train,
                row_aggregation=AggregationType.MAX,
                should_calibrate_scores=False,
                score_type=CalibScoreType.LOSS,
            )

        # Verify the error message
        self.assertIn(
            "column impression_signature not found in input dataframe(s)",
            str(context.exception),
        )
