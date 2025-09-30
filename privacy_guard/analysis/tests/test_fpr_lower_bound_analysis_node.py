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

import numpy as np

from privacy_guard.analysis.mia.aggregate_analysis_input import (
    AggregateAnalysisInput,
    AggregationType,
)
from privacy_guard.analysis.mia.fpr_lower_bound_analysis_node import (
    FPRLowerBoundAnalysisNode,
    FPRLowerBoundAnalysisNodeOutput,
)

from privacy_guard.analysis.tests.base_test_analysis_node import BaseTestAnalysisNode


class TestFPRLowerBoundAnalysisNode(BaseTestAnalysisNode):
    def setUp(self) -> None:
        super().setUp()

        self.analysis_input = AggregateAnalysisInput(
            row_aggregation=AggregationType.MAX,
            df_train_merge=self.df_train_merge,
            df_test_merge=self.df_test_merge,
            user_id_key=self.user_id_key,
        )

        self.analysis_node = FPRLowerBoundAnalysisNode(
            analysis_input=self.analysis_input,
            delta=0.000001,
            n_users_for_eval=5000,
            num_bootstrap_resampling_times=40,
        )

    def test_get_analysis_input(self) -> None:
        self.assertIsInstance(self.analysis_node.analysis_input, AggregateAnalysisInput)

    def test_progress_bar(self) -> None:
        """
        Test that the progress bar works as expected.
        """
        test_progress_analysis_node = FPRLowerBoundAnalysisNode(
            analysis_input=self.analysis_input,
            delta=0.000001,
            n_users_for_eval=5000,
            num_bootstrap_resampling_times=40,
            show_progress=True,
        )

        test_progress_analysis_node.compute_outputs()

    def test_timer_enabled(self) -> None:
        """
        Test that the timer works as expected.
        """
        test_timer_analysis_node = FPRLowerBoundAnalysisNode(
            analysis_input=self.analysis_input,
            delta=0.000001,
            n_users_for_eval=5000,
            num_bootstrap_resampling_times=40,
            with_timer=True,
        )

        test_timer_analysis_node.compute_outputs()

        self.assertIn(
            "make_acc_auc_epsilon_array", test_timer_analysis_node.get_timer_stats()
        )
        self.assertIn(
            "make_epsilon_at_error_thresholds_array",
            test_timer_analysis_node.get_timer_stats(),
        )

    def test_timer_disabled(self) -> None:
        """
        Test that the timer works as expected.
        """
        test_timer_analysis_node = FPRLowerBoundAnalysisNode(
            analysis_input=self.analysis_input,
            delta=0.000001,
            n_users_for_eval=5000,
            num_bootstrap_resampling_times=40,
            with_timer=False,
        )

        test_timer_analysis_node.compute_outputs()

        self.assertNotIn(
            "make_acc_auc_epsilon_array", test_timer_analysis_node.get_timer_stats()
        )
        self.assertNotIn(
            "make_epsilon_at_error_thresholds_array",
            test_timer_analysis_node.get_timer_stats(),
        )

    def test_compute_outputs(self) -> None:
        """
        Demonstrate that when test/train users are all sampled
        from the same distribution, the attack returns results
        close to random guessing.
        Epsilon close to zero, and accuracy close to 0.5

        Test and train data were sampled from same distribution)
        """

        df_train_user_long, df_test_user_long = self.get_long_dataframes()

        analysis_input = AggregateAnalysisInput(
            row_aggregation=AggregationType.MAX,
            df_train_merge=df_train_user_long,
            df_test_merge=df_test_user_long,
            user_id_key=self.user_id_key,
        )

        analysis_node = FPRLowerBoundAnalysisNode(
            analysis_input=analysis_input,
            delta=0.000001,
            n_users_for_eval=5000,
            num_bootstrap_resampling_times=40,
        )

        outputs = analysis_node.compute_outputs()

        self.assertLessEqual(float(outputs["eps_cp"]), 0.1)
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

    def test_num_bootstrap_resampling(self) -> None:
        """
        Test that the number of bootstraps is set correctly.
        """
        num_bootstrap_resampling_times = 10
        test_num_bootstraps_analysis_node = FPRLowerBoundAnalysisNode(
            analysis_input=self.analysis_input,
            delta=0.000001,
            n_users_for_eval=10,
            num_bootstrap_resampling_times=num_bootstrap_resampling_times,
        )

        test_num_bootstraps_analysis_node.compute_outputs()

        self.assertEqual(
            len(test_num_bootstraps_analysis_node._make_metrics_array()),
            num_bootstrap_resampling_times,
        )
