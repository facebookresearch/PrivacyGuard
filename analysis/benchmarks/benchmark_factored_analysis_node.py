# pyre-unsafe
import logging
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import tabulate
import torch
from privacy_guard.analysis.analysis_node import AnalysisNode
from privacy_guard.analysis.base_analysis_input import BaseAnalysisInput

from privacy_guard.analysis.mia.factored_analysis_node import FactoredAnalysisNode


def compute_epsilon_using_analysis_node(
    df_train_user: pd.DataFrame, df_test_user: pd.DataFrame
):
    analysis_input = BaseAnalysisInput(
        df_train_user=df_train_user, df_test_user=df_test_user
    )
    analysis_node = AnalysisNode(
        analysis_input=analysis_input,
        delta=0.000001,
        n_users_for_eval=15000,
        num_bootstrap_resampling_times=1000,
    )
    analysis_node.compute_outputs()


def compute_epsilon_using_factored_analysis_node(
    df_train_user: pd.DataFrame, df_test_user: pd.DataFrame
):
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

        FactoredAnalysisNode.merge_results(True, metrics_array, eps_tpr_array)


def sample_normal_distribution(
    mean: float = 0.0, std_dev: float = 1.0, num_samples: int = 20000
) -> pd.DataFrame:
    scores = np.random.normal(loc=mean, scale=std_dev, size=num_samples)
    return pd.DataFrame({"score": scores})


def main():
    logging.basicConfig(level=logging.INFO)
    results = {}

    for i in range(20):
        num_rows = 10000 * (i + 1)
        df_train_user = sample_normal_distribution(0.5, 0.25, num_rows)
        df_test_user = sample_normal_distribution(0.5, 0.25, num_rows)

        start = time.time()
        compute_epsilon_using_analysis_node(df_train_user, df_test_user)
        end = time.time()
        analysis_node_time = end - start

        start = time.time()
        compute_epsilon_using_factored_analysis_node(df_train_user, df_test_user)
        end = time.time()
        factored_analysis_node_time = end - start

        results[num_rows] = (analysis_node_time, factored_analysis_node_time)

        logging.info(
            f"Num samples: {num_rows}, Analysis Node Time: {analysis_node_time}, Factored Analysis Node Time: {factored_analysis_node_time}"
        )

    headers = [
        "Number of rows",
        "Analysis Node Time",
        "Factored Analysis Node Time",
        "Speedup",
    ]
    rows = []
    for num_rows, (analysis_node_time, factored_analysis_node_time) in results.items():
        speedup = analysis_node_time / factored_analysis_node_time
        rows.append(
            [num_rows, analysis_node_time, factored_analysis_node_time, speedup]
        )

    print(tabulate.tabulate(rows, headers=headers))
