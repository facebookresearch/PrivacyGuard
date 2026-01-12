# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-strict

import pandas as pd
from privacy_guard.analysis.mia.aggregate_analysis_input import (
    AggregateAnalysisInput,
    AggregationType,
)
from privacy_guard.analysis.tests.base_test_analysis_node import BaseTestAnalysisNode
from scipy.stats import gmean


class TestAnalysisInput(BaseTestAnalysisNode):
    def setUp(self) -> None:
        super().setUp()

    def test_construct_analysis_input(self) -> None:
        _ = AggregateAnalysisInput(
            row_aggregation=AggregationType.MIN,
            df_train_merge=self.df_train_merge,
            df_test_merge=self.df_test_merge,
            user_id_key=self.user_id_key,
        )

        df_missing_col = self.df_train_merge.drop("score", axis=1)
        with self.assertRaises(ValueError) as ex:
            _ = AggregateAnalysisInput(
                row_aggregation=AggregationType.MIN,
                df_train_merge=df_missing_col,
                df_test_merge=self.df_test_merge,
                user_id_key=self.user_id_key,
            )
        self.assertIn("score", str(ex.exception))

        df_missing_col = self.df_test_merge.drop("score", axis=1).drop(
            self.user_id_key, axis=1
        )
        with self.assertRaises(ValueError) as ex:
            _ = AggregateAnalysisInput(
                row_aggregation=AggregationType.MIN,
                df_train_merge=df_missing_col,
                df_test_merge=pd.DataFrame(),
                user_id_key=self.user_id_key,
            )
        self.assertIn("score", str(ex.exception))
        self.assertIn(self.user_id_key, str(ex.exception))

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
                user_id_key=self.user_id_key,
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
            user_id_key=self.user_id_key,
        )

        df_train_user_manual = pd.DataFrame(
            analysis_input.df_train_merge[["score", self.user_id_key]]
            .groupby([self.user_id_key], sort=False)
            .score.apply(lambda x: gmean(x.clip(lower=1e-3))),
            columns=["score"],
        )

        df_test_user_manual = pd.DataFrame(
            analysis_input.df_test_merge[["score", self.user_id_key]]
            .groupby([self.user_id_key], sort=False)
            .score.apply(lambda x: gmean(x.clip(lower=1e-3))),
            columns=["score"],
        )

        self.assertTrue(df_train_user_manual.equals(analysis_input.df_train_user))
        self.assertTrue(df_test_user_manual.equals(analysis_input.df_test_user))
