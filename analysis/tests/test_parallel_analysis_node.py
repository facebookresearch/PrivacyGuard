# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import io
import unittest

import numpy as np
import pandas as pd
import pkg_resources
import zstd
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


class TestParallelAnalysisNode(unittest.TestCase):
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

        self.user_id_key = "separable_id"

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

        super().setUp()

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

    def test_compute_outputs(self) -> None:
        """
        Demonstrate that when test/train users are all sampled
        from the same distribution, the attack returns results
        close to random guessing.
        Epsilon close to zero, and accuracy close to 0.5

        (json test and train data were sampled from same distribution)
        """

        outputs = self.parallel_analysis_node.compute_outputs()

        self.assertLessEqual(float(outputs["eps"]), 0.1)
        self.assertLessEqual(float(outputs["accuracy"]), 0.51)
        self.assertLessEqual(float(outputs["auc"]), 0.51)

    def test_compute_output_types(self) -> None:
        analysis_outputs = self.parallel_analysis_node.run_analysis()
        self.assertIsInstance(analysis_outputs, AnalysisNodeOutput)
        analysis_outputs_dict = self.parallel_analysis_node.compute_outputs()
        self.assertIsInstance(analysis_outputs_dict, dict)
        self.assertIsInstance(analysis_outputs_dict["eps"], (float, np.floating))
        self.assertIsInstance(
            analysis_outputs_dict["eps_geo_split"], (float, np.floating)
        )
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
