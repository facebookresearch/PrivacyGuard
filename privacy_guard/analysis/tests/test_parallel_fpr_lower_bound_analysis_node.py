# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np

from privacy_guard.analysis.mia.aggregate_analysis_input import (
    AggregateAnalysisInput,
    AggregationType,
)
from privacy_guard.analysis.mia.fpr_lower_bound_analysis_node import (
    FPRLowerBoundAnalysisNodeOutput,
)
from privacy_guard.analysis.mia.parallel_fpr_lower_bound_analysis_node import (
    ParallelFPRLowerBoundAnalysisNode,
)

from privacy_guard.analysis.tests.base_test_analysis_node import BaseTestAnalysisNode


class TestParallelFPRLowerBoundAnalysisNode(BaseTestAnalysisNode):
    def setUp(self) -> None:
        super().setUp()

        self.analysis_input = AggregateAnalysisInput(
            row_aggregation=AggregationType.MAX,
            df_train_merge=self.df_train_merge,
            df_test_merge=self.df_test_merge,
            user_id_key=self.user_id_key,
        )

        self.analysis_node = ParallelFPRLowerBoundAnalysisNode(
            analysis_input=self.analysis_input,
            delta=0.000001,
            n_users_for_eval=5000,
            num_bootstrap_resampling_times=40,
            eps_computation_tasks_num=2,
        )

    def test_get_analysis_input(self) -> None:
        self.assertIsInstance(self.analysis_node.analysis_input, AggregateAnalysisInput)

    def test_timer_enabled(self) -> None:
        """
        Test that the timer works as expected.
        """
        test_timer_analysis_node = ParallelFPRLowerBoundAnalysisNode(
            analysis_input=self.analysis_input,
            delta=0.000001,
            n_users_for_eval=5000,
            num_bootstrap_resampling_times=40,
            with_timer=True,
            eps_computation_tasks_num=2,
        )

        test_timer_analysis_node.compute_outputs()

        self.assertIn("parallel_bootstrap", test_timer_analysis_node.get_timer_stats())

    def test_timer_disabled(self) -> None:
        """
        Test that the timer works as expected.
        """
        test_timer_analysis_node = ParallelFPRLowerBoundAnalysisNode(
            analysis_input=self.analysis_input,
            delta=0.000001,
            n_users_for_eval=5000,
            num_bootstrap_resampling_times=40,
            with_timer=False,
            eps_computation_tasks_num=2,
        )

        test_timer_analysis_node.compute_outputs()

        self.assertNotIn(
            "parallel_bootstrap", test_timer_analysis_node.get_timer_stats()
        )

    def test_compute_outputs(self) -> None:
        """
        Demonstrate that when test/train users are all sampled
        from the same distribution, the attack returns results
        close to random guessing.
        Epsilon close to zero, and accuracy close to 0.5

        Test and train data were sampled from same distribution
        """
        df_train_user_long, df_test_user_long = self.get_long_dataframes()

        analysis_input = AggregateAnalysisInput(
            row_aggregation=AggregationType.MAX,
            df_train_merge=df_train_user_long,
            df_test_merge=df_test_user_long,
            user_id_key=self.user_id_key,
        )

        analysis_node = ParallelFPRLowerBoundAnalysisNode(
            analysis_input=analysis_input,
            delta=0.000001,
            n_users_for_eval=5000,
            num_bootstrap_resampling_times=40,
            eps_computation_tasks_num=2,
        )

        outputs = analysis_node.compute_outputs()

        self.assertLessEqual(float(outputs["eps"]), 0.12)
        self.assertLessEqual(float(outputs["accuracy"]), 0.51)
        self.assertLessEqual(float(outputs["auc"]), 0.51)

    def test_compute_output_types(self) -> None:
        analysis_outputs = self.analysis_node.run_analysis()
        self.assertIsInstance(analysis_outputs, FPRLowerBoundAnalysisNodeOutput)
        analysis_outputs_dict = self.analysis_node.compute_outputs()
        self.assertIsInstance(analysis_outputs_dict, dict)
        self.assertIsInstance(analysis_outputs_dict["eps"], (float, np.floating))

        self.assertIsInstance(analysis_outputs_dict["eps_fpr_lb"], list)
        self.assertTrue(
            all(
                isinstance(x, (float, np.floating))
                for x in analysis_outputs_dict["eps_fpr_lb"]
            )
        )
        self.assertIsInstance(analysis_outputs_dict["eps_fpr_ub"], list)
        self.assertTrue(
            all(
                isinstance(x, (float, np.floating))
                for x in analysis_outputs_dict["eps_fpr_ub"]
            )
        )

        self.assertIsInstance(analysis_outputs_dict["eps_cp"], (float, np.floating))

        self.assertIsInstance(analysis_outputs_dict["eps_mean"], (float, np.floating))
        self.assertIsInstance(
            analysis_outputs_dict["eps_mean_lb"], (float, np.floating)
        )
        self.assertIsInstance(
            analysis_outputs_dict["eps_mean_ub"], (float, np.floating)
        )

        self.assertIsInstance(analysis_outputs_dict["accuracy"], (float, np.floating))
        self.assertIsInstance(analysis_outputs_dict["accuracy_ci"], list)
        self.assertTrue(
            all(
                isinstance(x, (float, np.floating))
                for x in analysis_outputs_dict["accuracy_ci"]
            )
        )

        self.assertIsInstance(analysis_outputs_dict["auc"], (float, np.floating))
        self.assertIsInstance(analysis_outputs_dict["auc_ci"], list)
        self.assertTrue(
            all(
                isinstance(x, (float, np.floating))
                for x in analysis_outputs_dict["auc_ci"]
            )
        )

        self.assertIsInstance(analysis_outputs_dict["data_size"], dict)
        self.assertTrue(
            {"train_size", "test_size", "bootstrap_size"}.issubset(
                analysis_outputs_dict["data_size"]
            )
        )
        self.assertTrue(
            all(isinstance(x, int) for x in analysis_outputs_dict["data_size"].values())
        )
