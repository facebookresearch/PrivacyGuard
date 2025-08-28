# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

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
        self.assertIn("make_eps_tpr_array", test_timer_analysis_node.get_timer_stats())

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
        self.assertNotIn(
            "make_eps_tpr_array", test_timer_analysis_node.get_timer_stats()
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

        self.assertLessEqual(float(outputs["eps_ci"]), 0.1)
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
        self.assertIsInstance(analysis_outputs_dict["eps_ci"], (float, np.floating))

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
        self.assertIn("eps_ci", outputs)
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
