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
Model definitions for shadow model training.

This module provides neural network model definitions for privacy attack experiments.
"""

import torch
import torch.nn as nn


class ResidualUnit(nn.Module):
    """
    A Residual unit for deep networks with skip connections.
    """

    def __init__(
        self, input_channels: int, output_channels: int, downsample: bool = False
    ) -> None:
        super().__init__()

        # Determine stride based on whether we need to downsample
        self.stride: int = 2 if downsample else 1

        # Main path
        self.conv_path: nn.Sequential = nn.Sequential(
            # First convolution with optional downsampling
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=self.stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            # Second convolution always with stride 1
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
        )

        # Skip connection path
        self.skip_path: nn.Module = nn.Identity()
        if downsample or input_channels != output_channels:
            self.skip_path = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    stride=self.stride,
                    bias=False,
                ),
                nn.BatchNorm2d(output_channels),
            )

        # Final activation
        self.activation: nn.ReLU = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main path
        main_path = self.conv_path(x)

        # Skip connection
        skip_path = self.skip_path(x)

        # Combine paths and activate
        return self.activation(main_path + skip_path)


class DeepCNN(nn.Module):
    """
    A deep convolutional neural network with residual connections.
    Architecture inspired by ResNet.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        # Initial feature extraction
        self.input_block: nn.Sequential = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Residual blocks with increasing feature dimensions
        self.stage1: nn.Sequential = self._create_stage(
            64, 64, blocks=2, downsample_first=False
        )
        self.stage2: nn.Sequential = self._create_stage(
            64, 128, blocks=2, downsample_first=True
        )
        self.stage3: nn.Sequential = self._create_stage(
            128, 256, blocks=2, downsample_first=True
        )
        self.stage4: nn.Sequential = self._create_stage(
            256, 512, blocks=2, downsample_first=True
        )

        # Global pooling and classification
        self.global_pool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier: nn.Linear = nn.Linear(512, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _create_stage(
        self,
        input_channels: int,
        output_channels: int,
        blocks: int,
        downsample_first: bool,
    ) -> nn.Sequential:
        """Create a stage with multiple residual units."""
        layers = []

        # First block may need to downsample
        layers.append(
            ResidualUnit(input_channels, output_channels, downsample=downsample_first)
        )

        # Remaining blocks maintain dimensions
        for _ in range(1, blocks):
            layers.append(
                ResidualUnit(output_channels, output_channels, downsample=False)
            )

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """Initialize model weights for better training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = self.input_block(x)

        # Process through residual stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Global pooling and classification
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


def create_model() -> DeepCNN:
    """Create a deep CNN model for CIFAR-10 classification."""
    return DeepCNN(num_classes=10)
