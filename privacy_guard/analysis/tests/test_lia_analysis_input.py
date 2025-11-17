# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import unittest

import numpy as np
import pandas as pd
from privacy_guard.analysis.lia.lia_analysis_input import LIAAnalysisInput


class TestLIAAnalysisInput(unittest.TestCase):
    def setUp(self) -> None:
        self.predictions = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.predictions_y1_generation = np.array([0.15, 0.25, 0.35, 0.45, 0.55])
        self.true_bits = np.array([0, 1, 0, 1, 0])
        self.y0 = np.array([0, 1, 0, 1, 0])
        self.y1 = np.array([1, 0, 1, 0, 1])
        self.received_labels = np.array([1, 0, 1, 0, 1])

        self.lia_input = LIAAnalysisInput(
            predictions=self.predictions,
            predictions_y1_generation=self.predictions_y1_generation,
            true_bits=self.true_bits,
            y0=self.y0,
            y1=self.y1,
            received_labels=self.received_labels,
        )
        super().setUp()

    def test_lia_analysis_input_init(self) -> None:
        # Test that all attributes are set correctly
        np.testing.assert_array_equal(self.lia_input.predictions, self.predictions)
        np.testing.assert_array_equal(
            self.lia_input.predictions_y1_generation, self.predictions_y1_generation
        )
        np.testing.assert_array_equal(self.lia_input.true_bits, self.true_bits)
        np.testing.assert_array_equal(self.lia_input.y0, self.y0)
        np.testing.assert_array_equal(self.lia_input.y1, self.y1)
        np.testing.assert_array_equal(
            self.lia_input.received_labels, self.received_labels
        )

        # Test that DataFrame is created correctly
        self.assertIsInstance(self.lia_input.df_train_user, pd.DataFrame)
        self.assertIsInstance(self.lia_input.df_test_user, pd.DataFrame)
        np.testing.assert_array_equal(
            self.lia_input.df_train_user["predictions"].values, self.predictions
        )
