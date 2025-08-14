# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import io
import unittest

import pandas as pd
import pkg_resources
import zstd
from privacy_guard.analysis.mia.aggregate_analysis_input import (
    AggregateAnalysisInput,
    AggregationType,
)
from scipy.stats import gmean


class TestAnalysisInput(unittest.TestCase):
    def setUp(self) -> None:
        json_path = pkg_resources.resource_filename(
            __name__, "test_data/df_train_merge.json.zst"
        )
        with open(json_path, "rb") as f:
            self.df_train_merge = pd.read_json(
                io.StringIO(
                    zstd.ZstdDecompressor().decompress(f.read()).decode("latin1")
                )
            )

        json_path = pkg_resources.resource_filename(
            __name__, "test_data/df_test_merge.json.zst"
        )
        with open(json_path, "rb") as f:
            self.df_test_merge = pd.read_json(
                io.StringIO(
                    zstd.ZstdDecompressor().decompress(f.read()).decode("latin1")
                )
            )

        super().setUp()

    def test_construct_analysis_input(self) -> None:
        _ = AggregateAnalysisInput(
            row_aggregation=AggregationType.MIN,
            df_train_merge=self.df_train_merge,
            df_test_merge=self.df_test_merge,
        )

        df_missing_col = self.df_train_merge.drop("score", axis=1)
        with self.assertRaises(ValueError) as ex:
            _ = AggregateAnalysisInput(
                row_aggregation=AggregationType.MIN,
                df_train_merge=df_missing_col,
                df_test_merge=self.df_test_merge,
            )
        self.assertIn("score", str(ex.exception))

        df_missing_col = self.df_test_merge.drop("score", axis=1).drop(
            "separable_id", axis=1
        )
        with self.assertRaises(ValueError) as ex:
            _ = AggregateAnalysisInput(
                row_aggregation=AggregationType.MIN,
                df_train_merge=df_missing_col,
                df_test_merge=pd.DataFrame(),
            )
        self.assertIn("score", str(ex.exception))
        self.assertIn("separable_id", str(ex.exception))

    def test_row_aggregation_cases(self) -> None:
        # for covered cases, we should not raise an exception
        user_agg_list = [
            AggregationType.MIN,
            AggregationType.MAX,
            AggregationType.AVG,
            AggregationType.LOGSUMEXP,
            AggregationType.GMEAN,
            AggregationType.NONE,
        ]
        for agg in user_agg_list:
            analysis_node = AggregateAnalysisInput(
                row_aggregation=agg,
                df_train_merge=self.df_train_merge,
                df_test_merge=self.df_test_merge,
            )
            self.assertFalse(analysis_node.df_train_user.empty)
            self.assertFalse(analysis_node.df_test_user.empty)
            self.assertEqual(analysis_node.row_aggregation, agg)

    def test_gmean_aggregation(self) -> None:
        agg = AggregationType.GMEAN

        analysis_input = AggregateAnalysisInput(
            row_aggregation=agg,
            df_train_merge=self.df_train_merge,
            df_test_merge=self.df_test_merge,
        )

        df_train_user_manual = pd.DataFrame(
            analysis_input.df_train_merge[["score", "separable_id"]]
            .groupby(["separable_id"], sort=False)
            .score.apply(lambda x: gmean(x.clip(lower=1e-3))),
            columns=["score"],
        )

        df_test_user_manual = pd.DataFrame(
            analysis_input.df_test_merge[["score", "separable_id"]]
            .groupby(["separable_id"], sort=False)
            .score.apply(lambda x: gmean(x.clip(lower=1e-3))),
            columns=["score"],
        )

        self.assertTrue(df_train_user_manual.equals(analysis_input.df_train_user))
        self.assertTrue(df_test_user_manual.equals(analysis_input.df_test_user))
