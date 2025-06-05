# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import os
import tempfile

import unittest
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

from privacy_guard.analysis.factored_analysis_node import FactoredAnalysisNode


class TestFactoredAnalysisNode(unittest.TestCase):
    def sample_normal_distribution(
        self, mean: float = 0.0, std_dev: float = 1.0, num_samples: int = 20000
    ) -> pd.DataFrame:
        scores = np.random.normal(loc=mean, scale=std_dev, size=num_samples)
        return pd.DataFrame({"score": scores})

    def test_same_distribution_low_eps(self) -> None:
        df_train_user = self.sample_normal_distribution(0.5, 0.25, 20000)
        df_test_user = self.sample_normal_distribution(0.5, 0.25, 20000)
        score_train = df_train_user["score"]
        score_test = df_test_user["score"]
        loss_train = torch.from_numpy(score_train.to_numpy())
        loss_test = torch.from_numpy(score_test.to_numpy())

        with tempfile.TemporaryDirectory() as tmpdirname:
            train_filename = os.path.join(tmpdirname, "train.pd")
            test_filename = os.path.join(tmpdirname, "test.pd")
            torch.save(loss_train, train_filename)
            torch.save(loss_test, test_filename)

            with ProcessPoolExecutor() as pool:
                partial_results = list(
                    pool.map(
                        FactoredAnalysisNode.compute_partial_results,
                        [
                            (
                                train_filename,
                                test_filename,
                                0.000001,
                                100,
                            )
                            for _ in range(10)
                        ],
                    )
                )

            metrics_array = []
            eps_tpr_array = []
            for metrics_result, eps_tpr_result in partial_results:
                metrics_array.extend(metrics_result)
                eps_tpr_array.extend(eps_tpr_result)

            results = FactoredAnalysisNode.merge_results(
                True, metrics_array, eps_tpr_array
            )

        self.assertLessEqual(results["eps"], 0.5)

    def test_different_distribution_high_eps(self) -> None:
        df_test_user = self.sample_normal_distribution(0.75, 0.1, 20000)
        df_train_user = self.sample_normal_distribution(0.5, 0.25, 20000)
        score_train = df_train_user["score"]
        score_test = df_test_user["score"]
        loss_train = torch.from_numpy(score_train.to_numpy())
        loss_test = torch.from_numpy(score_test.to_numpy())

        with tempfile.TemporaryDirectory() as tmpdirname:
            train_filename = os.path.join(tmpdirname, "train.pd")
            test_filename = os.path.join(tmpdirname, "test.pd")

            torch.save(loss_train, train_filename)
            torch.save(loss_test, test_filename)

            with ProcessPoolExecutor() as pool:
                partial_results = list(
                    pool.map(
                        FactoredAnalysisNode.compute_partial_results,
                        [
                            (
                                train_filename,
                                test_filename,
                                0.000001,
                                100,
                            )
                            for _ in range(10)
                        ],
                    )
                )

            metrics_array = []
            eps_tpr_array = []
            for metrics_result, eps_tpr_result in partial_results:
                metrics_array.extend(metrics_result)
                eps_tpr_array.extend(eps_tpr_result)

            results = FactoredAnalysisNode.merge_results(
                True, metrics_array, eps_tpr_array
            )

        self.assertGreaterEqual(results["eps"], 0.5)
