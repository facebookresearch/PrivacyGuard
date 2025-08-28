# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import unittest

from typing import Tuple

import numpy as np
import pandas as pd


class BaseTestAnalysisNode(unittest.TestCase):
    """
    Util test class which sets up common dataframes for use in testing.
    """

    def sample_normal_distribution(
        self, mean: float = 0.0, std_dev: float = 1.0, num_samples: int = 20000
    ) -> pd.DataFrame:
        scores = np.random.normal(loc=mean, scale=std_dev, size=num_samples)
        user_ids = list(range(0, num_samples))
        return pd.DataFrame({"user_id": user_ids, "score": scores})

    def setUp(self) -> None:
        self.df_train_merge = pd.DataFrame(
            {
                "user_id": [
                    123456,
                    123456,
                    789012,
                    345678,
                    901234,
                ],
                "score": [0.8, 0.7, 0.6, 0.9, 0.5],
            }
        )

        # Create sample data for testing (same structure but different values)
        self.df_test_merge = pd.DataFrame(
            {
                "user_id": [
                    567890,
                    567890,
                    112233,
                    445566,
                    778899,
                ],
                "score": [0.2, 0.3, 0.4, 0.1, 0.5],
            }
        )

        self.user_id_key = "user_id"

        super().setUp()

    def get_long_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        np.random.seed(0)
        df_train_user_long = self.sample_normal_distribution(0.5, 0.1, 10000)
        df_test_user_long = self.sample_normal_distribution(0.5, 0.1, 10000)

        return (df_train_user_long, df_test_user_long)
