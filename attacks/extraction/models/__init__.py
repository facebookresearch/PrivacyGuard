# pyre-strict

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
Model interfaces for GenAI extraction attacks.

This module provides model abstractions for different types of generative AI models
used in privacy evaluation attacks.
"""

from .base_model import BaseModel
from .huggingface_model import HuggingFaceModel

__all__ = [
    "BaseModel",
    "HuggingFaceModel",
]
