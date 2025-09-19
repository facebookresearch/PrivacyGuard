# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

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
