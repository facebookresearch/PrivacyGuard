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

from typing import Callable

import pandas as pd
import torch
import torch.nn as nn

from privacy_guard.analysis.base_analysis_input import BaseAnalysisInput
from privacy_guard.attacks.base_attack import BaseAttack
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_loss_cross_entropy(
    model: nn.Module, dataloader: DataLoader
) -> torch.Tensor:
    """
    Computes the losses given by the model over the dataloader.
    Uses cross entropy loss for classification tasks.
    """
    losses = []
    criterion = nn.CrossEntropyLoss(reduction="none")

    for img, target in dataloader:
        outputs = model(img)
        batch_losses = criterion(outputs, target)
        losses += batch_losses.tolist()

    return torch.Tensor(losses)


class LossAttack(BaseAttack):
    """
    This is an implementation of a Membership Inference Attack based on loss values.

    Given a function to compute the loss:
        - Computes the losses of the private model on both the private
          train and holdout sets
        - Returns a BaseAnalysisInput object to analyze the results
    """

    private_model: nn.Module
    private_train: DataLoader
    private_holdout: DataLoader
    compute_loss: Callable[[nn.Module, DataLoader], torch.Tensor]

    def __init__(
        self,
        private_model: nn.Module,
        private_train: DataLoader,
        private_holdout: DataLoader,
        compute_loss: Callable[
            [nn.Module, DataLoader], torch.Tensor
        ] = compute_loss_cross_entropy,
    ) -> None:
        """
        Initialize the LossAttack.

        Args:
            private_model: The model to attack
            private_train: DataLoader for the training data
            private_holdout: DataLoader for the holdout data. Holdout and train data can have different lengths.
            compute_loss: Function to compute loss for a model on a dataloader
        """
        self.private_model = private_model
        self.private_train = private_train
        self.private_holdout = private_holdout
        self.compute_loss = compute_loss

        # Check if train or holdout datasets are empty
        if len(private_train) == 0 or len(private_holdout) == 0:
            raise ValueError(
                "private_train and private_holdout datasets cannot be empty"
            )

    def run_attack(self) -> BaseAnalysisInput:
        """
        Execute loss-based membership inference attack on input data.

        Returns:
            BaseAnalysisInput: Output containing train and holdout (test) data,
            ready for consumption by different analyses.
        """
        # Compute losses for train and holdout data
        losses_train = self.compute_loss(self.private_model, self.private_train)
        losses_holdout = self.compute_loss(self.private_model, self.private_holdout)

        # Convert to pandas DataFrames
        df_train = pd.DataFrame(
            {
                "user_id": range(len(losses_train)),
                "score": -losses_train.numpy(),  # Negate so higher score = more likely member
            }
        )

        df_test = pd.DataFrame(
            {
                "user_id": range(len(losses_holdout)),
                "score": -losses_holdout.numpy(),  # Negate so higher score = more likely member
            }
        )

        # Create and return analysis input
        analysis_input = BaseAnalysisInput(
            df_train_user=df_train,
            df_test_user=df_test,
        )

        return analysis_input
