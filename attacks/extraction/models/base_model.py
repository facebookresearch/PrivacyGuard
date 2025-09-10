# pyre-strict

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
Base model interface for GenAI extraction attacks.
"""

from abc import ABC, abstractmethod
from typing import Any, List

import torch


class BaseModel(ABC):
    """
    Abstract base class for all model implementations used in extraction attacks.
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
    def get_logits(self, prompts: List[str], targets: List[str]) -> torch.Tensor:
        """
        Compute logits for target sequences given prompts.

        Args:
            prompts: List of input prompts
            targets: List of target sequences to compute logits for

        Returns:
            Tensor of logits for target sequences
        """
        pass

    @abstractmethod
    def get_logprobs(self, prompts: List[str], targets: List[str]) -> torch.Tensor:
        """
        Compute log probabilities for target sequences given prompts.

        Args:
            prompts: List of input prompts
            targets: List of target sequences to compute log probabilities for

        Returns:
            Tensor of log probabilities for target sequences
        """
        pass
