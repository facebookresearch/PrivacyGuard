# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
"""
Dataset utilities for shadow model training.

This module provides functions for loading datasets and creating shadow datasets
for privacy attack experiments.
"""

from pathlib import Path
from typing import cast, List, Optional, Protocol, Sequence, Tuple, TypeVar

import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from typing_extensions import Sized


class SizedDataset(Protocol):
    """Protocol for datasets that have a __len__ method."""

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> object: ...


T: TypeVar = TypeVar("T")
DatasetT: TypeVar = TypeVar("DatasetT", bound=Dataset)


def get_cifar10_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get transforms for CIFAR-10 dataset.

    Returns:
        A tuple containing (train_transform, test_transform)
    """
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            # Normalize using CIFAR-10 mean and std values per RGB channel
            # These are the standard values used in the PyTorch community for CIFAR-10
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # Normalize using CIFAR-10 mean and std values per RGB channel
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
        ]
    )

    return train_transform, test_transform


def load_cifar10(data_dir: Optional[Path] = None) -> Tuple[CIFAR10, CIFAR10]:
    """
    Load CIFAR-10 dataset.

    Args:
        data_dir: Optional custom directory for dataset storage.
                 Defaults to ~/opt/data/cifar if not specified.

    Returns:
        A tuple containing (train_dataset, test_dataset)
    """
    if data_dir is None:
        data_dir = Path.home() / "opt/data/cifar"

    train_transform, test_transform = get_cifar10_transforms()

    train_dataset = CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    return train_dataset, test_dataset


def create_shadow_datasets(
    train_dataset: Dataset, n_shadows: int = 16, pkeep: float = 0.5, seed: int = 0
) -> Tuple[List[Tuple[Subset, np.ndarray]], Tuple[Subset, np.ndarray]]:
    """
    Create shadow model datasets.

    Args:
        train_dataset: The full training dataset
        n_shadows: Number of shadow models
        pkeep: Fraction of data to keep for each shadow model
        seed: Random seed for reproducibility

    Returns:
        List of (train_subsets, keep) tuples for each shadow model and the target model
    """
    np.random.seed(seed)
    dataset_size = len(cast(Sized, train_dataset))

    # Generate random values for each example and shadow model
    random_values = np.random.uniform(0, 1, size=(n_shadows, dataset_size))

    # Sort values to determine which examples are kept for each shadow model
    sorted_indices = random_values.argsort(axis=0)

    # Create a boolean mask for each shadow model
    keep_mask = sorted_indices < int(pkeep * n_shadows)

    shadow_datasets = []
    for shadow_index in range(n_shadows):
        # Get indices for examples included in this shadow model
        included_indices = np.where(keep_mask[shadow_index])[0]

        # Create subsets for in examples (members)
        shadow_subset = Subset(train_dataset, list(included_indices))

        shadow_datasets.append((shadow_subset, keep_mask[shadow_index]))

    return shadow_datasets[:-1], shadow_datasets[-1]


def create_rmia_datasets(
    train_dataset: Dataset,
    test_dataset: Dataset,
    num_references: int,
    population_size: int = 10000,
) -> Tuple[
    List[Tuple[Subset, np.ndarray]],
    Tuple[Subset, np.ndarray],
    Subset,
]:
    """
    Creates reference datasets, target dataset, and population dataset for RMIA.

    Args:
        train_dataset: The training dataset
        test_dataset: The test dataset
        num_references: Number of reference models to create
        population_size: Size of the population dataset

    Returns:
        Tuple containing:
        - reference_datasets: List of tuples with reference datasets and their labels
        - target_dataset: Tuple with target dataset and its labels
        - population_dataset: Dataset used for population estimation
    """

    assert num_references >= 3, (
        "Number of references must be at least 3 (2 shadows + 1 target)"
    )
    # Create reference datasets and target dataset
    reference_datasets: List[Tuple[Subset, np.ndarray]]
    target_dataset: Tuple[Subset, np.ndarray]
    reference_datasets, target_dataset = create_shadow_datasets(
        train_dataset, n_shadows=num_references
    )

    # Create population dataset (using test data as population for this example)
    # In practice, this should be a separate dataset not used for training or testing
    population_indices: Sequence[int] = np.random.choice(
        len(cast(Sized, test_dataset)), population_size, replace=False
    ).tolist()
    population_dataset: Subset = Subset(test_dataset, population_indices)

    # Print dataset sizes
    for i, (ref_in, _) in enumerate(reference_datasets):
        print(f"Reference {i}: {len(ref_in)} in-samples")

    print(f"Target: {len(target_dataset[0])} in-samples")
    print(f"Population: {len(population_dataset)} samples")

    return reference_datasets, target_dataset, population_dataset
