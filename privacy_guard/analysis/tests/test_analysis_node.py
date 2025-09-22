# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np
import pandas as pd

from privacy_guard.analysis.base_analysis_node import (
    BaseAnalysisInput,
    compute_and_merge_outputs,
)
from privacy_guard.analysis.mia.aggregate_analysis_input import (
    AggregateAnalysisInput,
    AggregationType,
)
from privacy_guard.analysis.mia.analysis_node import AnalysisNode, AnalysisNodeOutput
from privacy_guard.analysis.mia.score_analysis_node import (
    ScoreAnalysisNode,
    ScoreAnalysisNodeOutput,
)

from privacy_guard.analysis.tests.base_test_analysis_node import BaseTestAnalysisNode


class TestAnalysisNode(BaseTestAnalysisNode):
    def setUp(self) -> None:
        super().setUp()

        self.analysis_input = AggregateAnalysisInput(
            row_aggregation=AggregationType.MAX,
            df_train_merge=self.df_train_merge,
            df_test_merge=self.df_test_merge,
            user_id_key=self.user_id_key,
        )

        self.analysis_node = AnalysisNode(
            analysis_input=self.analysis_input,
            delta=0.000001,
            n_users_for_eval=5000,
            num_bootstrap_resampling_times=40,
        )

        self.score_analysis_node = ScoreAnalysisNode(analysis_input=self.analysis_input)

        # Benign setting where the test and train scores are separable
        separable_df_train = pd.DataFrame({"score": np.array([0.1, 0.1]).reshape(-1)})
        separable_df_test = pd.DataFrame({"score": np.array([0, 0]).reshape(-1)})
        self.separable_base_analysis_input = BaseAnalysisInput(
            separable_df_train, separable_df_test
        )

    def test_get_analysis_input(self) -> None:
        self.assertIsInstance(self.analysis_node.analysis_input, AggregateAnalysisInput)

    def test_progress_bar(self) -> None:
        """
        Test that the progress bar works as expected.
        """
        test_progress_analysis_node = AnalysisNode(
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
        test_timer_analysis_node = AnalysisNode(
            analysis_input=self.analysis_input,
            delta=0.000001,
            n_users_for_eval=5000,
            num_bootstrap_resampling_times=40,
            with_timer=True,
        )

        test_timer_analysis_node.compute_outputs()

        self.assertIn("make_metrics_array", test_timer_analysis_node.get_timer_stats())

    def test_timer_disabled(self) -> None:
        """
        Test that the timer works as expected.
        """
        test_timer_analysis_node = AnalysisNode(
            analysis_input=self.analysis_input,
            delta=0.000001,
            n_users_for_eval=5000,
            num_bootstrap_resampling_times=40,
            with_timer=False,
        )

        test_timer_analysis_node.compute_outputs()

        self.assertNotIn(
            "make_metrics_array", test_timer_analysis_node.get_timer_stats()
        )

    def test_turn_cap_eps_on(self) -> None:
        """
        Tests capping of computed epsilons. Under cap_eps=True and a seprable setting with two users, the max eps should be log(2) = 0.693.
        """
        analysis_node = AnalysisNode(
            self.separable_base_analysis_input,
            delta=0.00001,
            n_users_for_eval=2,
            cap_eps=True,
        )
        outputs = analysis_node.compute_outputs()
        eps_tpr_ub = max(
            outputs["eps_tpr_ub"]
        )  # max eps over all TPR thresholds, should be log(2) ~ 0.693
        assert abs(eps_tpr_ub - np.log(2)) < 1e-6

        eps_fpr_ub = max(
            outputs["eps_fpr_ub"]
        )  # max eps over all FPR thresholds, should be log(2) ~ 0.693
        assert abs(eps_fpr_ub - np.log(2)) < 1e-6

    def test_turn_cap_eps_off(self) -> None:
        """
        Tests capping of computed epsilons. Under cap_eps=False and a separable setting with two users, the max eps should be inf.
        """
        analysis_node = AnalysisNode(
            self.separable_base_analysis_input,
            delta=0.00001,
            n_users_for_eval=2,
            cap_eps=False,
        )
        outputs = analysis_node.compute_outputs()
        eps_tpr_ub = max(
            outputs["eps_tpr_ub"]
        )  # max eps over all TPR thresholds, should be inf
        assert eps_tpr_ub == float("inf")

        eps_fpr_ub = max(
            outputs["eps_fpr_ub"]
        )  # max eps over all FPR thresholds, should be inf
        print(outputs["eps_tpr_ub"], outputs["eps_fpr_ub"])
        assert eps_fpr_ub == float("inf")

    def test_num_bootstrap_resampling(self) -> None:
        """
        Test that the number of bootstraps is set correctly internally.
        """
        num_bootstrap_resampling_times = 10
        test_num_bootstraps_analysis_node = AnalysisNode(
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

    def test_compute_outputs(self) -> None:
        """
        Demonstrate that when test/train users are all sampled
        from the same distribution, the attack returns results
        close to random guessing.
        Epsilon close to zero, and accuracy close to 0.5

        Train and test data were sampled from the same distribution.
        """
        df_train_user_long, df_test_user_long = self.get_long_dataframes()

        self.analysis_input = AggregateAnalysisInput(
            row_aggregation=AggregationType.MAX,
            df_train_merge=df_train_user_long,
            df_test_merge=df_test_user_long,
            user_id_key=self.user_id_key,
        )

        self.analysis_node = AnalysisNode(
            analysis_input=self.analysis_input,
            delta=0.000001,
            n_users_for_eval=5000,
            num_bootstrap_resampling_times=40,
        )

        outputs = self.analysis_node.compute_outputs()

        self.assertLessEqual(float(outputs["eps_cp"]), 0.1)
        self.assertLessEqual(float(outputs["accuracy"]), 0.51)
        self.assertLessEqual(float(outputs["auc"]), 0.51)

    def test_compute_output_types(self) -> None:
        analysis_outputs = self.analysis_node.run_analysis()
        self.assertIsInstance(analysis_outputs, AnalysisNodeOutput)
        analysis_outputs_dict = self.analysis_node.compute_outputs()
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
        self.assertIsInstance(analysis_outputs_dict["eps_max_lb"], list)
        self.assertTrue(
            all(
                isinstance(x, (float, np.floating))
                for x in analysis_outputs_dict["eps_max_lb"]
            )
        )
        self.assertIsInstance(analysis_outputs_dict["eps_max_ub"], list)
        self.assertTrue(
            all(
                isinstance(x, (float, np.floating))
                for x in analysis_outputs_dict["eps_max_ub"]
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

        node_list = [self.analysis_node, self.score_analysis_node]
        outputs = compute_and_merge_outputs(node_list)

        self.assertIn("train_scores", outputs)
        self.assertIn("test_scores", outputs)
        self.assertIn("eps_cp", outputs)
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
            AnalysisNode(
                analysis_input=self.analysis_input,
                delta=0.000001,
                n_users_for_eval=-1,
                num_bootstrap_resampling_times=40,
            )

    def test_use_fnr_tnr_parameter_default(self) -> None:
        """Test that use_fnr_tnr defaults to False and works properly"""
        analysis_node = AnalysisNode(
            analysis_input=self.analysis_input,
            delta=0.000001,
            n_users_for_eval=100,
            num_bootstrap_resampling_times=10,
            # use_fnr_tnr not specified, should default to False
        )

        outputs = analysis_node.compute_outputs()

        # Should have normal behavior with all epsilon arrays having expected length
        self.assertIsInstance(outputs["eps_fpr_lb"], list)
        self.assertIsInstance(outputs["eps_fpr_ub"], list)
        self.assertIsInstance(outputs["eps_tpr_lb"], list)
        self.assertIsInstance(outputs["eps_tpr_ub"], list)
        self.assertIsInstance(outputs["eps_max_lb"], list)
        self.assertIsInstance(outputs["eps_max_ub"], list)

        # All arrays should have same length (100 error thresholds by default)
        self.assertEqual(len(outputs["eps_fpr_lb"]), 100)
        self.assertEqual(len(outputs["eps_fpr_ub"]), 100)
        self.assertEqual(len(outputs["eps_tpr_lb"]), 100)
        self.assertEqual(len(outputs["eps_tpr_ub"]), 100)
        self.assertEqual(len(outputs["eps_max_lb"]), 100)
        self.assertEqual(len(outputs["eps_max_ub"]), 100)

    def test_use_fnr_tnr_parameter_true(self) -> None:
        """Test that use_fnr_tnr=True filters error thresholds properly"""
        analysis_node = AnalysisNode(
            analysis_input=self.analysis_input,
            delta=0.000001,
            n_users_for_eval=100,
            num_bootstrap_resampling_times=10,
            use_fnr_tnr=True,
        )

        outputs = analysis_node.compute_outputs()

        # Should have normal behavior with all epsilon arrays
        self.assertIsInstance(outputs["eps_fpr_lb"], list)
        self.assertIsInstance(outputs["eps_fpr_ub"], list)
        self.assertIsInstance(outputs["eps_tpr_lb"], list)
        self.assertIsInstance(outputs["eps_tpr_ub"], list)
        self.assertIsInstance(outputs["eps_max_lb"], list)
        self.assertIsInstance(outputs["eps_max_ub"], list)

        # With use_fnr_tnr=True, error thresholds >= 1.0 should be filtered
        # Default error thresholds are np.linspace(0.01, 1, 100)
        # After filtering (< 1.0), we expect 99 elements
        self.assertEqual(len(outputs["eps_fpr_lb"]), 99)
        self.assertEqual(len(outputs["eps_fpr_ub"]), 99)
        self.assertEqual(len(outputs["eps_tpr_lb"]), 99)
        self.assertEqual(len(outputs["eps_tpr_ub"]), 99)
        self.assertEqual(len(outputs["eps_max_lb"]), 99)
        self.assertEqual(len(outputs["eps_max_ub"]), 99)

    def test_use_fnr_tnr_parameter_comparison(self) -> None:
        """Test comparison between use_fnr_tnr=False and use_fnr_tnr=True"""
        # Set random seed to ensure deterministic bootstrap sampling
        np.random.seed(42)

        # Test with use_fnr_tnr=False
        analysis_node_false = AnalysisNode(
            analysis_input=self.analysis_input,
            delta=0.000001,
            n_users_for_eval=100,
            num_bootstrap_resampling_times=10,
            use_fnr_tnr=False,
        )

        outputs_false = analysis_node_false.compute_outputs()

        # Reset seed to ensure same bootstrap sampling for the second run
        np.random.seed(42)

        # Test with use_fnr_tnr=True
        analysis_node_true = AnalysisNode(
            analysis_input=self.analysis_input,
            delta=0.000001,
            n_users_for_eval=100,
            num_bootstrap_resampling_times=10,
            use_fnr_tnr=True,
        )

        outputs_true = analysis_node_true.compute_outputs()

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
        self.assertGreater(
            len(outputs_false["eps_max_lb"]), len(outputs_true["eps_max_lb"])
        )
        self.assertGreater(
            len(outputs_false["eps_max_ub"]), len(outputs_true["eps_max_ub"])
        )

        # eps_cp should be the same in both cases (computed separately)
        self.assertAlmostEqual(
            outputs_false["eps_cp"], outputs_true["eps_cp"], places=10
        )

        # Other metrics should be the same
        self.assertAlmostEqual(
            outputs_false["accuracy"], outputs_true["accuracy"], places=10
        )
        self.assertAlmostEqual(outputs_false["auc"], outputs_true["auc"], places=10)
