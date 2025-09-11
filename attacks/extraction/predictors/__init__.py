# pyre-strict

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
Predictors module for privacy attacks.

This module provides predictor abstractions for different types of generative AI models
used in privacy evaluation attacks.
"""

from .base_predictor import BasePredictor
from .huggingface_predictor import HuggingFacePredictor

__all__ = [
    "BasePredictor",
    "HuggingFacePredictor",
]
