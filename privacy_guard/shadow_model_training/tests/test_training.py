# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
"""
Tests for the training module in shadow_model_training.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from privacy_guard.shadow_model_training.training import (
    evaluate_model,
    get_softmax_scores,
    get_transformed_logits,
    prepare_lira_data,
    prepare_rmia_data,
    train_model,
)
from torch.utils.data import DataLoader, TensorDataset


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TestTraining(unittest.TestCase):
    """Test cases for the training module."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create a simple model
        self.model = SimpleModel()

        # Create simple datasets
        x_train = torch.randn(20, 10)
        y_train = torch.randint(0, 2, (20,))
        x_test = torch.randn(10, 10)
        y_test = torch.randint(0, 2, (10,))

        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)

        self.train_loader = DataLoader(train_dataset, batch_size=5)
        self.test_loader = DataLoader(test_dataset, batch_size=5)

        # Use CPU for testing
        self.device = torch.device("cpu")

    def test_evaluate_model(self) -> None:
        """Test that evaluate_model returns an accuracy value."""
        accuracy = evaluate_model(self.model, self.test_loader, self.device)

        # Check that accuracy is a float between 0 and 100
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 100.0)

    @patch("torch.optim.SGD")
    @patch("torch.optim.lr_scheduler.CosineAnnealingLR")
    def test_train_model(
        self, mock_scheduler: MagicMock, mock_optimizer: MagicMock
    ) -> None:
        """Test that train_model trains the model and returns it."""
        # Configure mocks
        mock_optimizer.return_value = MagicMock()
        mock_scheduler.return_value = MagicMock()

        # Train for just 1 epoch to speed up the test
        trained_model = train_model(
            self.model,
            self.train_loader,
            self.test_loader,
            epochs=1,
            device=self.device,
        )

        # Check that the returned model is the same instance
        self.assertIs(trained_model, self.model)

        # Check that optimizer and scheduler were called
        mock_optimizer.assert_called_once()
        mock_scheduler.assert_called_once()

    def test_get_transformed_logits(self) -> None:
        """Test that get_transformed_logits returns logits of the expected shape."""
        logits = get_transformed_logits(self.model, self.test_loader, self.device)

        # Check that logits is a numpy array with the expected shape
        self.assertIsInstance(logits, np.ndarray)

        # The shape should be (num_samples, 1) since we're returning likelihood ratios
        expected_samples = len(self.test_loader.dataset)
        self.assertEqual(logits.shape, (expected_samples, 1))

    def test_prepare_lira_data(self) -> None:
        """Test that prepare_lira_data returns DataFrames with the expected structure."""
        # Create mock data
        target_logits = np.random.randn(100, 1)
        shadow_logits = np.random.randn(3, 100, 1)  # 3 shadow models
        user_id_key = "custom_id"
        # Create mock membership arrays
        target_membership = np.zeros(100, dtype=bool)
        target_membership[:50] = True  # First 50 samples are members

        shadow_memberships = []
        for _ in range(3):
            membership = np.zeros(100, dtype=bool)
            membership[np.random.choice(100, 50, replace=False)] = True
            shadow_memberships.append(membership)

        # Create mock datasets with proper typing
        from torch.utils.data import Subset

        mock_subset = MagicMock(spec=Subset)

        # Create mock datasets with proper typing
        target_dataset = (mock_subset, target_membership)
        shadow_datasets = [
            (mock_subset, membership) for membership in shadow_memberships
        ]

        # Call prepare_lira_data
        df_train_online, df_test_online, df_train_offline, df_test_offline = (
            prepare_lira_data(
                target_logits,
                shadow_logits,
                target_dataset,
                shadow_datasets,
                user_id_key,
            )
        )

        # Check that the returned objects are DataFrames
        self.assertIsInstance(df_train_online, pd.DataFrame)
        self.assertIsInstance(df_test_online, pd.DataFrame)
        self.assertIsInstance(df_train_offline, pd.DataFrame)
        self.assertIsInstance(df_test_offline, pd.DataFrame)

        # Check that the DataFrames have the expected number of rows
        self.assertEqual(len(df_train_online), 50)  # 50 members
        self.assertEqual(len(df_test_online), 50)  # 50 non-members
        self.assertEqual(len(df_train_offline), 50)  # 50 members
        self.assertEqual(len(df_test_offline), 50)  # 50 non-members

        # Check that the DataFrames have the expected columns
        self.assertIn(user_id_key, df_train_online.columns)
        self.assertIn("score_orig", df_train_online.columns)
        self.assertIn("score_mean_in", df_train_online.columns)
        self.assertIn("score_std_in", df_train_online.columns)
        self.assertIn("score_mean_out", df_train_online.columns)
        self.assertIn("score_std_out", df_train_online.columns)

        self.assertIn(user_id_key, df_train_offline.columns)
        self.assertIn("score_orig", df_train_offline.columns)
        self.assertIn("score_mean", df_train_offline.columns)
        self.assertIn("score_std", df_train_offline.columns)

    def test_get_softmax_scores(self) -> None:
        """Test that get_softmax_scores returns softmax scores of the expected shape."""
        # Test with default temperature
        scores = get_softmax_scores(self.model, self.test_loader, device=self.device)

        # Check that scores is a numpy array with the expected shape
        self.assertIsInstance(scores, np.ndarray)
        expected_samples = len(self.test_loader.dataset)
        self.assertEqual(scores.shape, (expected_samples,))

        # Check that scores are in the range [0, 1] (softmax probabilities)
        self.assertTrue(np.all(scores >= 0))
        self.assertTrue(np.all(scores <= 1))

        # Test with custom temperature
        custom_temperature = 2.0
        scores_custom = get_softmax_scores(
            self.model,
            self.test_loader,
            temperature=custom_temperature,
            device=self.device,
        )

        # Check that custom temperature produces different results
        self.assertIsInstance(scores_custom, np.ndarray)
        self.assertEqual(scores_custom.shape, (expected_samples,))

        # Scores with different temperatures should generally be different
        # (unless the model produces identical outputs, which is unlikely with random data)
        self.assertFalse(np.array_equal(scores, scores_custom))

    def test_prepare_rmia_data(self) -> None:
        """Test that prepare_rmia_data returns DataFrames with the expected structure."""
        # Create mock data
        n_samples = 100
        n_refs = 3

        # Create target scores
        target_scores_train = np.random.randn(n_samples)
        target_scores_population = np.random.randn(50)

        # Create reference scores - shape should be (n_samples, n_refs)
        ref_scores_train = np.random.randn(n_samples, n_refs)
        ref_scores_population = np.random.randn(50, n_refs)

        # Create mock membership arrays and datasets
        from torch.utils.data import Subset

        mock_subset = MagicMock(spec=Subset)

        # Create target dataset with membership (first 60 samples are in training)
        target_train_indices = list(range(60))
        target_dataset = (mock_subset, np.zeros(n_samples, dtype=bool))
        target_dataset[0].indices = target_train_indices

        # Create reference datasets with different membership patterns
        reference_datasets = []
        for ref_idx in range(n_refs):
            ref_mock_subset = MagicMock(spec=Subset)
            # Each reference model has different samples in training
            ref_train_indices = list(range(ref_idx * 20, (ref_idx + 1) * 20 + 20))
            ref_mock_subset.indices = ref_train_indices
            ref_membership = np.zeros(n_samples, dtype=bool)
            reference_datasets.append((ref_mock_subset, ref_membership))

        # Call prepare_rmia_data
        df_train_merge, df_test_merge, df_population = prepare_rmia_data(
            target_scores_train,
            target_scores_population,
            ref_scores_train,
            ref_scores_population,
            target_dataset,
            reference_datasets,
        )

        # Check that the returned objects are DataFrames
        self.assertIsInstance(df_train_merge, pd.DataFrame)
        self.assertIsInstance(df_test_merge, pd.DataFrame)
        self.assertIsInstance(df_population, pd.DataFrame)

        # Check that the DataFrames have the expected number of rows
        self.assertEqual(len(df_train_merge), 60)  # 60 members in target training
        self.assertEqual(len(df_test_merge), 40)  # 40 non-members in target training
        self.assertEqual(len(df_population), 50)  # 50 population samples

        # Check that the DataFrames have the expected columns
        expected_columns_train = ["user_id", "score_orig"]
        for ref_idx in range(n_refs):
            expected_columns_train.extend(
                [f"score_ref_{ref_idx}", f"member_ref_{ref_idx}"]
            )

        for col in expected_columns_train:
            self.assertIn(col, df_train_merge.columns)
            self.assertIn(col, df_test_merge.columns)

        # Check population DataFrame columns (no user_id)
        expected_columns_population = ["score_orig"]
        for ref_idx in range(n_refs):
            expected_columns_population.extend(
                [f"score_ref_{ref_idx}", f"member_ref_{ref_idx}"]
            )

        for col in expected_columns_population:
            self.assertIn(col, df_population.columns)

        # Check data types
        self.assertTrue(df_train_merge["score_orig"].dtype in [np.float32, np.float64])
        self.assertTrue(df_test_merge["score_orig"].dtype in [np.float32, np.float64])
        self.assertTrue(df_population["score_orig"].dtype in [np.float32, np.float64])

        # Check that reference membership columns are boolean
        for ref_idx in range(n_refs):
            self.assertEqual(df_train_merge[f"member_ref_{ref_idx}"].dtype, bool)
            self.assertEqual(df_test_merge[f"member_ref_{ref_idx}"].dtype, bool)
            self.assertEqual(df_population[f"member_ref_{ref_idx}"].dtype, bool)

        # Check that population reference memberships are all False
        for ref_idx in range(n_refs):
            self.assertTrue(np.all(~df_population[f"member_ref_{ref_idx}"]))

        # Test with custom user_id_key
        custom_user_id_key = "custom_id"
        df_train_custom, df_test_custom, _ = prepare_rmia_data(
            target_scores_train,
            target_scores_population,
            ref_scores_train,
            ref_scores_population,
            target_dataset,
            reference_datasets,
            user_id_key=custom_user_id_key,
        )

        # Check that custom user_id_key is used
        self.assertIn(custom_user_id_key, df_train_custom.columns)
        self.assertIn(custom_user_id_key, df_test_custom.columns)


if __name__ == "__main__":
    unittest.main()
