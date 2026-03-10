# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-strict
"""
Tests for the model module in shadow_model_training.
"""

import unittest

import torch
from privacy_guard.shadow_model_training.model import (
    create_mlp_model,
    create_model,
    DeepCNN,
    ResidualUnit,
    SimpleMLP,
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

    def test_create_model_custom_classes(self) -> None:
        """Test that create_model accepts custom num_classes."""
        model = create_model(num_classes=5)
        self.assertIsInstance(model, DeepCNN)
        self.assertEqual(model.classifier.out_features, 5)

    def test_deep_cnn_custom_input_channels(self) -> None:
        """Test DeepCNN with custom input channels."""
        model = DeepCNN(num_classes=3, input_channels=1)
        x = torch.randn(2, 1, 32, 32)
        y = model(x)
        self.assertEqual(y.shape, (2, 3))

    def test_simple_mlp(self) -> None:
        """Test that SimpleMLP forward pass works."""
        model = SimpleMLP(input_dim=20, num_classes=5)
        x = torch.randn(4, 20)
        y = model(x)
        self.assertEqual(y.shape, (4, 5))

    def test_simple_mlp_custom_hidden(self) -> None:
        """Test SimpleMLP with custom hidden dimensions."""
        model = SimpleMLP(input_dim=50, num_classes=3, hidden_dims=[64, 32, 16])
        x = torch.randn(4, 50)
        y = model(x)
        self.assertEqual(y.shape, (4, 3))

    def test_create_mlp_model(self) -> None:
        """Test that create_mlp_model returns a SimpleMLP instance."""
        model = create_mlp_model(input_dim=10, num_classes=4)
        self.assertIsInstance(model, SimpleMLP)
        x = torch.randn(2, 10)
        y = model(x)
        self.assertEqual(y.shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
