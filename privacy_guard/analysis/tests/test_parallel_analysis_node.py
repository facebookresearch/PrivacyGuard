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
import pytest

from privacy_guard.analysis.base_analysis_node import compute_and_merge_outputs
from privacy_guard.analysis.mia.aggregate_analysis_input import (
    AggregateAnalysisInput,
    AggregationType,
)
from privacy_guard.analysis.mia.analysis_node import AnalysisNodeOutput
from privacy_guard.analysis.mia.parallel_analysis_node import ParallelAnalysisNode
from privacy_guard.analysis.mia.score_analysis_node import (
    ScoreAnalysisNode,
    ScoreAnalysisNodeOutput,
)

from privacy_guard.analysis.tests.base_test_analysis_node import BaseTestAnalysisNode


class TestParallelAnalysisNode(BaseTestAnalysisNode):
    def setUp(self) -> None:
        super().setUp()

        self.analysis_input = AggregateAnalysisInput(
            row_aggregation=AggregationType.MAX,
            df_train_merge=self.df_train_merge,
            df_test_merge=self.df_test_merge,
            user_id_key=self.user_id_key,
        )

        self.parallel_analysis_node = ParallelAnalysisNode(
            analysis_input=self.analysis_input,
            delta=0.000001,
            n_users_for_eval=5000,
            num_bootstrap_resampling_times=40,
            eps_computation_tasks_num=2,
        )

        self.score_analysis_node = ScoreAnalysisNode(analysis_input=self.analysis_input)

    def test_get_analysis_input(self) -> None:
        self.assertIsInstance(
            self.parallel_analysis_node.analysis_input, AggregateAnalysisInput
        )

    def test_timer_enabled(self) -> None:
        """
        Test that the timer works as expected.
        """
        test_timer_analysis_node = ParallelAnalysisNode(
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
        test_timer_analysis_node = ParallelAnalysisNode(
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

    @pytest.mark.skip(
        reason="This test hangs in stress runs. Skipping to not block workflow signals in OSS environment."
    )
    def test_compute_outputs(self) -> None:
        """
        Demonstrate that when test/train users are all sampled
        from the same distribution, the attack returns results
        close to random guessing.
        Epsilon close to zero, and accuracy close to 0.5

        Train user and test user are sampled from the same distribution.
        """
        df_train_user_long, df_test_user_long = self.get_long_dataframes()

        analysis_input = AggregateAnalysisInput(
            row_aggregation=AggregationType.MAX,
            df_train_merge=df_train_user_long,
            df_test_merge=df_test_user_long,
            user_id_key=self.user_id_key,
        )

        parallel_analysis_node = ParallelAnalysisNode(
            analysis_input=analysis_input,
            delta=0.000001,
            n_users_for_eval=1000,
            num_bootstrap_resampling_times=100,
            eps_computation_tasks_num=2,
        )

        outputs = parallel_analysis_node.compute_outputs()

        self.assertLessEqual(float(outputs["eps"]), 0.11)
        self.assertLessEqual(float(outputs["accuracy"]), 0.51)
        self.assertLessEqual(float(outputs["auc"]), 0.51)

    def test_compute_output_types(self) -> None:
        analysis_outputs = self.parallel_analysis_node.run_analysis()
        self.assertIsInstance(analysis_outputs, AnalysisNodeOutput)
        analysis_outputs_dict = self.parallel_analysis_node.compute_outputs()
        self.assertIsInstance(analysis_outputs_dict, dict)
        self.assertIsInstance(analysis_outputs_dict["eps"], (float, np.floating))
        self.assertIsInstance(analysis_outputs_dict["eps_lb"], (float, np.floating))
        self.assertIsInstance(
            analysis_outputs_dict["eps_fpr_max_ub"], (float, np.floating)
        )
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
        self.assertIsInstance(analysis_outputs_dict["eps_tpr_lb"], list)
        self.assertTrue(
            all(
                isinstance(x, (float, np.floating))
                for x in analysis_outputs_dict["eps_tpr_lb"]
            )
        )
        self.assertIsInstance(analysis_outputs_dict["eps_tpr_ub"], list)
        self.assertTrue(
            all(
                isinstance(x, (float, np.floating))
                for x in analysis_outputs_dict["eps_tpr_ub"]
            )
        )
        self.assertIsInstance(analysis_outputs_dict["eps_cp"], (float, np.floating))

        self.assertIsInstance(analysis_outputs_dict["accuracy"], (float, np.floating))
        self.assertIsInstance(analysis_outputs_dict["accuracy_ci"], list)
        self.assertEqual(len(analysis_outputs_dict["accuracy_ci"]), 2)
        self.assertTrue(
            all(
                isinstance(x, (float, np.floating))
                for x in analysis_outputs_dict["accuracy_ci"]
            )
        )

        self.assertIsInstance(analysis_outputs_dict["auc"], (float, np.floating))
        self.assertIsInstance(analysis_outputs_dict["auc_ci"], list)
        self.assertEqual(len(analysis_outputs_dict["auc_ci"]), 2)
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

    def test_score_analysis_node(self) -> None:
        """
        Compute outputs of the score_analysis_node
        """
        outputs = self.score_analysis_node.run_analysis()
        self.assertIsInstance(outputs, ScoreAnalysisNodeOutput)
        outputs = self.score_analysis_node.compute_outputs()
        self.assertIsInstance(outputs, dict)
        self.assertGreater(len(outputs["train_scores"]), 0)
        self.assertGreater(len(outputs["test_scores"]), 0)
        self.assertIsInstance(outputs["train_scores"], list)
        self.assertIsInstance(outputs["test_scores"], list)
        self.assertTrue(
            all(isinstance(x, (float, np.floating)) for x in outputs["train_scores"])
        )
        self.assertTrue(
            all(isinstance(x, (float, np.floating)) for x in outputs["test_scores"])
        )

    def test_compute_and_merge_outputs_single(self) -> None:
        outputs = compute_and_merge_outputs([])
        self.assertEqual(outputs, {})

        outputs = compute_and_merge_outputs([self.score_analysis_node])
        self.assertGreater(len(outputs["train_scores"]), 0)
        self.assertGreater(len(outputs["test_scores"]), 0)

    def test_compute_and_merge_outputs(self) -> None:
        """
        Compute outputs of multiple nodes and merge them into one dict.
        """

        node_list = [self.parallel_analysis_node, self.score_analysis_node]
        outputs = compute_and_merge_outputs(node_list)

        self.assertIn("train_scores", outputs)
        self.assertIn("test_scores", outputs)
        self.assertIn("eps", outputs)
        self.assertIn("accuracy", outputs)
        self.assertIn("auc", outputs)

    def test_compute_and_merge_overwrite(self) -> None:
        """
        Ensure that overwritten keys are supported.
        """
        node_list = [self.score_analysis_node, self.score_analysis_node]
        outputs = compute_and_merge_outputs(node_list)
        self.assertIn("train_scores", outputs)
        self.assertIn("test_scores", outputs)

    def test_negative_n_users_for_eval(self) -> None:
        """
        Negative number of users for eval should raise an error.
        """
        with self.assertRaisesRegex(ValueError, "must be nonnegative"):
            ParallelAnalysisNode(
                analysis_input=self.analysis_input,
                delta=0.000001,
                n_users_for_eval=-1,
                num_bootstrap_resampling_times=40,
                eps_computation_tasks_num=2,
            )

    def test_use_fnr_tnr_parameter(self) -> None:
        """Test that use_fnr_tnr parameter works correctly"""
        # Test with use_fnr_tnr=False (default)
        parallel_node_false = ParallelAnalysisNode(
            analysis_input=self.analysis_input,
            delta=0.000001,
            n_users_for_eval=100,
            num_bootstrap_resampling_times=10,
            eps_computation_tasks_num=2,
            use_fnr_tnr=False,
        )

        outputs_false = parallel_node_false.compute_outputs()

        # Test with use_fnr_tnr=True
        parallel_node_true = ParallelAnalysisNode(
            analysis_input=self.analysis_input,
            delta=0.000001,
            n_users_for_eval=100,
            num_bootstrap_resampling_times=10,
            eps_computation_tasks_num=2,
            use_fnr_tnr=True,
        )

        outputs_true = parallel_node_true.compute_outputs()

        # With use_fnr_tnr=False, should have 100 error thresholds (default)
        self.assertEqual(len(outputs_false["eps_fpr_lb"]), 100)
        self.assertEqual(len(outputs_false["eps_fpr_ub"]), 100)
        self.assertEqual(len(outputs_false["eps_tpr_lb"]), 100)
        self.assertEqual(len(outputs_false["eps_tpr_ub"]), 100)

        # With use_fnr_tnr=True, should have 99 error thresholds (filtered >= 1.0)
        self.assertEqual(len(outputs_true["eps_fpr_lb"]), 99)
        self.assertEqual(len(outputs_true["eps_fpr_ub"]), 99)
        self.assertEqual(len(outputs_true["eps_tpr_lb"]), 99)
        self.assertEqual(len(outputs_true["eps_tpr_ub"]), 99)

        # Arrays with use_fnr_tnr=True should be shorter due to filtering
        self.assertGreater(
            len(outputs_false["eps_fpr_lb"]), len(outputs_true["eps_fpr_lb"])
        )
        self.assertGreater(
            len(outputs_false["eps_fpr_ub"]), len(outputs_true["eps_fpr_ub"])
        )
        self.assertGreater(
            len(outputs_false["eps_tpr_lb"]), len(outputs_true["eps_tpr_lb"])
        )
        self.assertGreater(
            len(outputs_false["eps_tpr_ub"]), len(outputs_true["eps_tpr_ub"])
        )
