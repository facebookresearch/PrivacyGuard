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
Tests for the dataset module in shadow_model_training.
"""

import unittest

import numpy as np
import torch
from privacy_guard.shadow_model_training.dataset import (
    create_rmia_datasets,
    create_shadow_datasets,
    CustomDataset,
    get_cifar10_transforms,
)
from torch.utils.data import Dataset
from torchvision import transforms


class MockCIFAR10(Dataset):
    """Mock CIFAR10 dataset for testing."""

    def __init__(self, size: int = 100) -> None:
        self.size = size
        self.data: torch.Tensor = torch.randn(size, 3, 32, 32)
        self.targets: torch.Tensor = torch.randint(0, 10, (size,))

    def __len__(self) -> int:
        return self.size

    # pyrefly: ignore [bad-param-name-override]
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


class TestDataset(unittest.TestCase):
    """Test cases for the dataset module."""

    def test_get_cifar10_transforms(self) -> None:
        """Test that get_cifar10_transforms returns the expected transforms."""
        train_transform, test_transform = get_cifar10_transforms()

        # Check that transforms are of the correct type
        self.assertIsInstance(train_transform, transforms.Compose)
        self.assertIsInstance(test_transform, transforms.Compose)

        # Check that train_transform has more transforms than test_transform
        self.assertGreater(
            len(train_transform.transforms), len(test_transform.transforms)
        )

        # Check that both transforms include ToTensor and Normalize
        train_transform_types = [type(t) for t in train_transform.transforms]
        test_transform_types = [type(t) for t in test_transform.transforms]

        self.assertIn(transforms.ToTensor, train_transform_types)
        self.assertIn(transforms.Normalize, train_transform_types)
        self.assertIn(transforms.ToTensor, test_transform_types)
        self.assertIn(transforms.Normalize, test_transform_types)

    def test_create_shadow_datasets(self) -> None:
        """Test that create_shadow_datasets creates the expected datasets."""
        mock_dataset = MockCIFAR10(size=1000)
        n_shadows = 4
        pkeep = 0.5

        shadow_datasets, target_dataset = create_shadow_datasets(
            mock_dataset, n_shadows=n_shadows, pkeep=pkeep, seed=42
        )

        # Check that we have the expected number of shadow datasets
        self.assertEqual(len(shadow_datasets), n_shadows - 1)

        # Check that target_dataset is a tuple of (Subset, np.ndarray)
        self.assertIsInstance(target_dataset, tuple)
        self.assertEqual(len(target_dataset), 2)

        # Check that each shadow dataset is a tuple of (Subset, np.ndarray)
        for shadow_dataset in shadow_datasets:
            self.assertIsInstance(shadow_dataset, tuple)
            self.assertEqual(len(shadow_dataset), 2)

            # Check that the subset has the expected size (approximately)
            subset, keep = shadow_dataset
            expected_size = int(pkeep * len(mock_dataset))
            self.assertAlmostEqual(
                len(subset), expected_size, delta=expected_size * 0.2
            )

            # Check that the keep array has the expected shape
            self.assertEqual(keep.shape, (len(mock_dataset),))

    def test_create_rmia_datasets(self) -> None:
        """Test that create_rmia_datasets creates the expected datasets."""
        train_dataset = MockCIFAR10(size=1000)
        test_dataset = MockCIFAR10(size=200)
        num_references = 5
        population_size = 100

        reference_datasets, target_dataset, population_dataset = create_rmia_datasets(
            train_dataset, test_dataset, num_references, population_size
        )

        # Check that we have the expected number of reference datasets
        self.assertEqual(len(reference_datasets), num_references - 1)

        # Check that target_dataset is a tuple of (Subset, np.ndarray)
        self.assertIsInstance(target_dataset, tuple)
        self.assertEqual(len(target_dataset), 2)
        target_subset, target_membership = target_dataset
        self.assertEqual(len(target_membership), len(train_dataset))
        # Check target subset has valid indices
        self.assertTrue(all(idx < len(train_dataset) for idx in target_subset.indices))

        # Check that each reference dataset is a tuple of (Subset, np.ndarray)
        for ref_dataset in reference_datasets:
            self.assertIsInstance(ref_dataset, tuple)
            self.assertEqual(len(ref_dataset), 2)
            ref_subset, ref_membership = ref_dataset
            self.assertEqual(len(ref_membership), len(train_dataset))
            # Check ref subset has valid indices
            self.assertTrue(all(idx < len(train_dataset) for idx in ref_subset.indices))

        # Check population dataset size
        self.assertEqual(len(population_dataset), population_size)

        # Check that population dataset indices are valid
        self.assertTrue(
            all(idx < len(test_dataset) for idx in population_dataset.indices)
        )

    def test_create_rmia_datasets_minimum_references(self) -> None:
        """Test that create_rmia_datasets fails with less than 3 references."""
        train_dataset = MockCIFAR10(size=100)
        test_dataset = MockCIFAR10(size=50)

        with self.assertRaises(AssertionError):
            create_rmia_datasets(train_dataset, test_dataset, num_references=2)


class TestCustomDataset(unittest.TestCase):
    """Test cases for the CustomDataset class."""

    def test_construction_and_properties(self) -> None:
        """Test creation from numpy/tensors and num_classes/input_shape."""
        # From numpy arrays
        data_np = np.random.randn(100, 10).astype(np.float32)
        targets_np = np.array([0, 1, 2, 3, 4] * 20)
        ds = CustomDataset(data_np, targets_np)
        self.assertEqual(len(ds), 100)
        self.assertEqual(ds.num_classes, 5)
        self.assertEqual(ds.input_shape, torch.Size([10]))
        sample, label = ds[0]
        self.assertIsInstance(sample, torch.Tensor)
        self.assertIsInstance(label, torch.Tensor)

        # From torch tensors with image-like shape
        ds2 = CustomDataset(torch.randn(50, 3, 32, 32), torch.randint(0, 5, (50,)))
        self.assertEqual(len(ds2), 50)
        self.assertEqual(ds2.input_shape, torch.Size([3, 32, 32]))

    def test_validation_errors(self) -> None:
        """Test that invalid inputs raise ValueError."""
        with self.assertRaises(ValueError):
            CustomDataset(
                np.zeros((100, 10), dtype=np.float32), np.zeros(50, dtype=np.int64)
            )
        with self.assertRaises(ValueError):
            CustomDataset(
                np.zeros((0, 10), dtype=np.float32), np.array([], dtype=np.int64)
            )

    def test_transform_and_shadow_integration(self) -> None:
        """Test transforms and compatibility with create_shadow_datasets."""
        data = np.random.randn(200, 10).astype(np.float32)
        targets = np.random.randint(0, 3, size=200)
        transform_called: list[bool] = [False]

        def my_transform(x: torch.Tensor) -> torch.Tensor:
            transform_called[0] = True
            return x * 2.0

        dataset = CustomDataset(data, targets, transform=my_transform)
        dataset[0]
        self.assertTrue(transform_called[0])

        shadow_datasets, target_dataset = create_shadow_datasets(
            dataset, n_shadows=4, pkeep=0.5, seed=42
        )
        self.assertEqual(len(shadow_datasets), 3)
        self.assertIsInstance(target_dataset, tuple)


if __name__ == "__main__":
    unittest.main()
