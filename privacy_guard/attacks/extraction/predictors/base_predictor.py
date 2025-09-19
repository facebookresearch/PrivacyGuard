# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Base predictor interface for GenAI extraction attacks.
"""

from abc import ABC, abstractmethod
from typing import Any, List

import torch


class BasePredictor(ABC):
    """
    Abstract base class for all predictor implementations used in extraction attacks.
    """

    @abstractmethod
    def generate(self, prompts: List[str], **generation_kwargs: Any) -> List[str]:
        """
        Generate text continuations for given prompts.

        Args:
            prompts: List of input prompts to generate continuations for
            **generation_kwargs: Generation parameters (top_p, top_k, temperature, etc.)

        Returns:
            List of generated text continuations, one per input prompt
        """
        pass

    @abstractmethod
    def get_logits(
        self, prompts: List[str], targets: List[str], batch_size: int = 1
    ) -> List[torch.Tensor]:
        """
        Compute logits for target sequences given prompts.

        Args:
            prompts: List of input prompts
            targets: List of target sequences to compute logits for
            batch_size: Number of sequences to process in each batch (default: 1)

        Returns:
            List of tensors, each with shape (target_length, vocab_size) for the
            corresponding prompt-target pair
        """
        pass

    @abstractmethod
    def get_logprobs(
        self, prompts: List[str], targets: List[str], **generation_kwargs: Any
    ) -> List[torch.Tensor]:
        """
        Compute log probabilities for target sequences given prompts.

        Args:
            prompts: List of input prompts
            targets: List of target sequences to compute log probabilities for
            **generation_kwargs: Generation parameters (temperature, top_k, etc.)

        Returns:
            List of tensors, each containing log probabilities for the corresponding
            prompt-target pair
        """
        pass
