# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
"""
Tests for the dataset module in shadow_model_training.
"""

import unittest

import torch
from privacy_guard.shadow_model_training.dataset import (
    create_rmia_datasets,
    create_shadow_datasets,
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


if __name__ == "__main__":
    unittest.main()
