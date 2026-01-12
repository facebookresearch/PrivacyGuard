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

import pandas as pd
from numpy.testing import assert_almost_equal
from privacy_guard.analysis.mia.aggregate_analysis_input import AggregationType
from privacy_guard.attacks.calib_attack import CalibAttack, CalibScoreType


class TestCalibAttack(unittest.TestCase):
    def setUp(self) -> None:
        self.df_hold_out_train_json = """{"user_id":{"0":00001,"1":00002,"2":00003,"3":00004,"4":00005},"sample_id":{"0":101,"1":102,"2":103,"3":104,"4":105},"timestamp":{"0":1000000001,"1":1000000002,"2":1000000003,"3":1000000004,"4":1000000005},"hash_id":{"0":30001.0,"1":30002.0,"2":50.0,"3":51.0,"4":52.0},"predictions":{"0":0.21985362,"1":0.10969869,"2":0.24854505,"3":0.0068224324,"4":0.004189688},"label":{"0":0.0,"1":1.0,"2":0.0,"3":1.0,"4":1.0}}"""
        self.df_hold_out_train = pd.read_json(self.df_hold_out_train_json)

        self.MERGE_COLUMNS = ["user_id", "sample_id", "timestamp", "hash_id"]

        self.user_id_key = "user_id"

        self.calibrated_calib_attack = CalibAttack(
            df_hold_out_train=self.df_hold_out_train,
            df_hold_out_test=self.df_hold_out_train,
            df_hold_out_train_calib=self.df_hold_out_train,
            df_hold_out_test_calib=self.df_hold_out_train,
            row_aggregation=AggregationType.MAX,
            should_calibrate_scores=True,
            user_id_key=self.user_id_key,
            score_type=CalibScoreType.LOSS,
            merge_columns=self.MERGE_COLUMNS,
        )

        self.calib_attack = CalibAttack(
            df_hold_out_train=self.df_hold_out_train,
            df_hold_out_test=self.df_hold_out_train,
            df_hold_out_train_calib=self.df_hold_out_train,
            df_hold_out_test_calib=self.df_hold_out_train,
            row_aggregation=AggregationType.MAX,
            should_calibrate_scores=False,
            user_id_key=self.user_id_key,
            score_type=CalibScoreType.LOSS,
            merge_columns=self.MERGE_COLUMNS,
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
        _ = CalibAttack.compute_score(df=self.df_hold_out_train, score_type="logits")
        with self.assertRaises(ValueError):
            _ = CalibAttack.compute_score(
                df=self.df_hold_out_train, score_type="no_score"
            )

    def test_logits_values(self) -> None:
        # Execute: compute logits score for the test data
        result = CalibAttack.compute_score(
            df=self.df_hold_out_train, score_type="logits"
        )

        # Assert: verify the logits values are correct
        # logits = -(log(predictions + 1e-30) - log(1 - predictions + 1e-30))
        # This is similar to scaled_logits but WITHOUT the (2*label - 1) factor
        assert_almost_equal(
            result,
            [1.26651961, 2.09382253, 1.10638714, 4.9806934, 5.47093052],
            decimal=7,
        )

    def test_logits_vs_scaled_logits(self) -> None:
        # Execute: compute both logits and scaled_logits
        logits = CalibAttack.compute_score(
            df=self.df_hold_out_train, score_type="logits"
        )

        copied_df = self.df_hold_out_train.copy()
        copied_df["label"] = 1
        scaled_logits = CalibAttack.compute_score(
            df=copied_df, score_type="scaled_logits"
        )

        assert_almost_equal(logits.values, scaled_logits.values, decimal=7)

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
        df_missing_column = self.df_hold_out_train.drop(columns=["hash_id"])

        # Verify that an IndexError is raised when trying to create a CalibAttack instance
        with self.assertRaises(IndexError) as context:
            CalibAttack(
                df_hold_out_train=df_missing_column,
                df_hold_out_test=self.df_hold_out_train,
                df_hold_out_train_calib=self.df_hold_out_train,
                df_hold_out_test_calib=self.df_hold_out_train,
                row_aggregation=AggregationType.MAX,
                should_calibrate_scores=False,
                user_id_key=self.user_id_key,
                score_type=CalibScoreType.LOSS,
                merge_columns=self.MERGE_COLUMNS,
            )

        # Verify the error message
        self.assertIn(
            "column hash_id not found in input dataframe(s)",
            str(context.exception),
        )
