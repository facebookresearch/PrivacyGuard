# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
"""
Tests for the model module in shadow_model_training.
"""

import unittest

import torch
from privacy_guard.shadow_model_training.model import (
    create_model,
    DeepCNN,
    ResidualUnit,
)


class TestModel(unittest.TestCase):
    """Test cases for the model module."""

    def test_residual_unit(self) -> None:
        """Test that ResidualUnit can be instantiated and forward pass works."""
        # Test with same input and output channels
        unit = ResidualUnit(64, 64, downsample=False)
        x = torch.randn(2, 64, 32, 32)
        y = unit(x)
        self.assertEqual(y.shape, (2, 64, 32, 32))

        # Test with different input and output channels and downsampling
        unit = ResidualUnit(64, 128, downsample=True)
        x = torch.randn(2, 64, 32, 32)
        y = unit(x)
        self.assertEqual(y.shape, (2, 128, 16, 16))

    def test_deep_cnn(self) -> None:
        """Test that DeepCNN can be instantiated and forward pass works."""
        model = DeepCNN(num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        self.assertEqual(y.shape, (2, 10))

    def test_create_model(self) -> None:
        """Test that create_model returns a DeepCNN instance."""
        model = create_model()
        self.assertIsInstance(model, DeepCNN)
        self.assertEqual(model.classifier.out_features, 10)


if __name__ == "__main__":
    unittest.main()
