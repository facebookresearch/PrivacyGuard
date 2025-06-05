# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict


import unittest
from unittest import mock

import pandas as pd
from privacy_guard.analysis.aggregate_analysis_input import (
    AggregateAnalysisInput,
    AggregationType,
)

from privacy_guard.attacks.lira_attack import LiraAttack

from privacy_guard.fb.query_helpers.hard_cut_utils import FilterDataSqlArgs


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
