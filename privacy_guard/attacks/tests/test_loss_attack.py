# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-strict

import unittest
from unittest.mock import MagicMock

import pandas as pd
import torch
import torch.nn as nn

from privacy_guard.attacks.loss_attack import LossAttack
from torch.utils.data import DataLoader, TensorDataset


class TestLossAttack(unittest.TestCase):
    def setUp(self) -> None:
        # Define constants
        self.batch_size = 2
        self.len_train = 10
        self.len_test = 8
        self.num_features = 3
        self.num_classes = 2

        # Create mock model
        self.model = MagicMock(spec=nn.Module)

        # Create mock data for train and holdout sets
        train_data = torch.randn(self.len_train, self.num_features)
        train_labels = torch.randint(0, self.num_classes, (self.len_train,))
        holdout_data = torch.randn(self.len_test, self.num_features)
        holdout_labels = torch.randint(0, self.num_classes, (self.len_test,))

        # Create DataLoaders
        train_dataset = TensorDataset(train_data, train_labels)
        holdout_dataset = TensorDataset(holdout_data, holdout_labels)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size)
        self.holdout_dataloader = DataLoader(
            holdout_dataset, batch_size=self.batch_size
        )
        super().setUp()

    def test_init(self) -> None:
        """Test initialization of LossAttack."""
        custom_compute_loss = MagicMock()

        attack = LossAttack(
            private_model=self.model,
            private_train=self.train_dataloader,
            private_holdout=self.holdout_dataloader,
            compute_loss=custom_compute_loss,
        )
        self.assertEqual(attack.private_model, self.model)
        self.assertEqual(attack.private_train, self.train_dataloader)
        self.assertEqual(attack.private_holdout, self.holdout_dataloader)
        self.assertEqual(attack.compute_loss, custom_compute_loss)

    def test_run_attack(self) -> None:
        """Test run_attack method."""
        # Set up mock compute_loss function
        mock_compute_loss = MagicMock()
        train_losses = torch.rand(self.len_train)
        holdout_losses = torch.rand(self.len_test)
        mock_compute_loss.side_effect = [train_losses, holdout_losses]

        # Create attack instance
        attack = LossAttack(
            private_model=self.model,
            private_train=self.train_dataloader,
            private_holdout=self.holdout_dataloader,
            compute_loss=mock_compute_loss,
        )

        # Run attack
        result = attack.run_attack()

        # Verify compute_loss was called with correct arguments
        mock_compute_loss.assert_any_call(self.model, self.train_dataloader)
        mock_compute_loss.assert_any_call(self.model, self.holdout_dataloader)

        # Verify result structure
        self.assertEqual(len(result.df_train_user), self.len_train)
        self.assertEqual(len(result.df_test_user), self.len_test)
        self.assertTrue("user_id" in result.df_train_user.columns)
        self.assertTrue("score" in result.df_train_user.columns)
        self.assertTrue("user_id" in result.df_test_user.columns)
        self.assertTrue("score" in result.df_test_user.columns)

        # Verify scores are negated losses
        pd.testing.assert_series_equal(
            result.df_train_user["score"],
            pd.Series(-train_losses.numpy(), name="score"),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            result.df_test_user["score"],
            pd.Series(-holdout_losses.numpy(), name="score"),
            check_names=False,
        )

    def test_with_cross_entropy_loss(self) -> None:
        """Test compute_loss_cross_entropy function."""
        # Configure the mock model to return appropriate outputs for the test
        self.model.return_value = torch.randn(self.batch_size, self.num_classes)

        # loss functin will be the default compute_loss_cross_entropy
        attack = LossAttack(
            private_model=self.model,
            private_train=self.train_dataloader,
            private_holdout=self.holdout_dataloader,
        )

        result = attack.run_attack()

        self.assertEqual(len(result.df_train_user), self.len_train)
        self.assertEqual(len(result.df_test_user), self.len_test)

    def test_empty_dataloaders(self) -> None:
        """Test that LossAttack raises ValueError when initialized with empty dataloaders."""
        # Create empty datasets
        empty_dataset = TensorDataset(torch.Tensor([]), torch.Tensor([]))
        empty_train_dataloader = DataLoader(empty_dataset, batch_size=1)
        empty_holdout_dataloader = DataLoader(empty_dataset, batch_size=1)

        # Test with empty train dataloader
        with self.assertRaises(ValueError):
            LossAttack(
                private_model=self.model,
                private_train=empty_train_dataloader,
                private_holdout=self.holdout_dataloader,
                compute_loss=MagicMock(),
            )

        # Test with empty holdout dataloader
        with self.assertRaises(ValueError):
            LossAttack(
                private_model=self.model,
                private_train=self.train_dataloader,
                private_holdout=empty_holdout_dataloader,
                compute_loss=MagicMock(),
            )

        # Test with both empty dataloaders
        with self.assertRaises(ValueError):
            LossAttack(
                private_model=self.model,
                private_train=empty_train_dataloader,
                private_holdout=empty_holdout_dataloader,
                compute_loss=MagicMock(),
            )
