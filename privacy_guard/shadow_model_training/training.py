# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
"""
Training utilities for shadow model training.

This module provides functions for training models, evaluating models,
extracting logits, and preparing data for LiRA attacks.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


# Default device
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 5,
    device: torch.device = DEFAULT_DEVICE,
) -> nn.Module:
    """
    Train a model on the given dataset.

    Args:
        model: The model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        epochs: Number of epochs to train for
        device: Device to train on (cuda or cpu)

    Returns:
        Trained model
    """
    # Initialize optimizer and scheduler
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        print(f"Epoch {epoch + 1}/{epochs}")
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Print epoch summary
        avg_loss = train_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        print(f"Training - Loss: {avg_loss:.4f}, Acc: {train_acc:.2f}%")

        # Evaluate on test set
        test_acc = evaluate_model(model, test_loader, device)
        print(f"Testing - Acc: {test_acc:.2f}%")

        scheduler.step()

    return model


def evaluate_model(
    model: nn.Module, data_loader: DataLoader, device: torch.device = DEFAULT_DEVICE
) -> float:
    """
    Evaluate model accuracy on a dataset.

    Args:
        model: The model to evaluate
        data_loader: DataLoader for evaluation data
        device: Device to evaluate on (cuda or cpu)

    Returns:
        Accuracy as a percentage
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total


@torch.no_grad()
def get_transformed_logits(
    model: nn.Module, data_loader: DataLoader, device: torch.device = DEFAULT_DEVICE
) -> np.ndarray:
    """
    Extract and process model outputs for LiRA attack.

    This function:
    1. Collects raw logits from the model
    2. Converts logits to probability distributions
    3. Calculates likelihood ratios between correct and incorrect classes

    Args:
        model: Neural network model to evaluate
        data_loader: DataLoader containing evaluation samples
        device: Device to run inference on (cuda or cpu)

    Returns:
        Likelihood ratio scores for membership inference
    """
    # Set model to evaluation mode
    model.eval()

    # Collect raw model outputs
    raw_outputs = []
    true_labels = []
    for batch, labels in data_loader:
        batch = batch.to(device)
        batch_outputs = model(batch)
        raw_outputs.append(batch_outputs.cpu().numpy())
        true_labels.append(labels.numpy())

    # Combine all batches
    combined_outputs = np.concatenate(raw_outputs)
    true_labels = np.concatenate(true_labels)

    # Reshape to add query dimension (batch_size, num_queries=1, num_classes)
    batch_size, num_classes = combined_outputs.shape
    reshaped_outputs = combined_outputs.reshape(batch_size, 1, num_classes)

    # Convert logits to probabilities using stable softmax implementation
    # First shift by max value for numerical stability
    shifted_logits = reshaped_outputs - np.max(reshaped_outputs, axis=2, keepdims=True)
    exp_logits = np.exp(shifted_logits).astype(np.float64)
    probabilities = exp_logits / np.sum(exp_logits, axis=2, keepdims=True)

    # Calculate model accuracy
    predicted_labels = np.argmax(probabilities[:, 0, :], axis=1)
    accuracy = np.mean(predicted_labels == true_labels)
    print(f"Model accuracy: {accuracy:.4f}")

    # Extract probabilities for correct classes
    indices = np.arange(probabilities.shape[0])
    correct_class_probs = probabilities[indices, :, true_labels]

    # Calculate sum of probabilities for incorrect classes
    incorrect_probs = probabilities.copy()
    incorrect_probs[indices, :, true_labels] = 0
    incorrect_class_probs = np.sum(incorrect_probs, axis=2)

    # Calculate log likelihood ratio with small epsilon to prevent log(0)
    epsilon = 1e-45
    likelihood_ratios = np.log(correct_class_probs + epsilon) - np.log(
        incorrect_class_probs + epsilon
    )

    return likelihood_ratios


@torch.no_grad()
def get_softmax_scores(
    model: nn.Module,
    data_loader: DataLoader,
    temperature: float = 1.0,
    device: torch.device = DEFAULT_DEVICE,
) -> np.ndarray:
    """
    Extract and process model outputs for RmiaAttack.
    Args:
        model: Neural network model to evaluate
        data_loader: DataLoader containing evaluation samples
        temperature: Temperature parameter for softmax transformation
        device: Device to run inference on (cuda or cpu)
    Returns:
        Softmax scores for membership inference
    """

    model.eval()

    softmax_outputs = []
    for batch, labels in data_loader:
        batch = batch.to(device)
        batch_outputs = model(batch)
        labels = labels.to(device)
        temp_signals = torch.div(batch_outputs, temperature)
        max_logit_signals = torch.max(temp_signals, dim=1)[0]

        logit_signals = torch.sub(temp_signals, max_logit_signals.reshape(-1, 1))
        exp_logit_signals = torch.exp(logit_signals)
        exp_logit_sum = exp_logit_signals.sum(dim=1).reshape(-1, 1)
        true_exp_logit = exp_logit_signals.gather(1, labels.reshape(-1, 1))
        softmax_outputs.append(torch.div(true_exp_logit, exp_logit_sum).cpu().numpy())

    softmax_outputs = np.concatenate(softmax_outputs).squeeze()

    return softmax_outputs


def prepare_rmia_data(
    target_scores_train: np.ndarray,
    target_scores_population: np.ndarray,
    ref_scores_train: np.ndarray,
    ref_scores_population: np.ndarray,
    target_dataset: Tuple[Subset, np.ndarray],
    reference_datasets: List[Tuple[Subset, np.ndarray]],
    user_id_key: str = "user_id",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare data for RMIA attack by creating dataframes with required columns.
    Args:
        target_scores_train: Target model scores for training data
        target_scores_population: Target model scores for population data
        ref_scores_train: Reference model scores for training data
        ref_scores_population: Reference model scores for population data
        target_dataset: Target dataset with membership information
        reference_datasets: Reference datasets with membership information
        user_id_key: Name of the column containing user IDs
    Returns:
        Tuple of (df_train_merge, df_test_merge, df_population)
    """
    n_samples = len(target_scores_train)
    n_refs = len(reference_datasets)

    # Create membership indicators for reference models
    # For each sample and each reference model, check if sample was in training set
    target_in_indices = set(target_dataset[0].indices)
    ref_memberships = np.zeros((n_samples, n_refs), dtype=bool)

    for ref_idx, (ref_in, _) in enumerate(reference_datasets):
        ref_memberships[ref_in.indices, ref_idx] = True

    # Create training data (members of target model)
    train_indices = list(target_in_indices)
    df_train_data = {
        user_id_key: np.arange(len(train_indices)),
        "score_orig": target_scores_train[train_indices].flatten(),
    }

    # Add reference scores and membership indicators
    for ref_idx in range(n_refs):
        df_train_data[f"score_ref_{ref_idx}"] = ref_scores_train[
            train_indices, ref_idx
        ].flatten()
        df_train_data[f"member_ref_{ref_idx}"] = ref_memberships[
            train_indices, ref_idx
        ].flatten()

    df_train_merge = pd.DataFrame.from_dict(df_train_data)

    # Create test data (non-members of target model)
    all_indices = set(range(n_samples))
    test_indices = list(all_indices - target_in_indices)
    df_test_data = {
        user_id_key: np.arange(len(test_indices)),
        "score_orig": target_scores_train[test_indices].flatten(),
    }

    # Add reference scores and membership indicators
    for ref_idx in range(n_refs):
        df_test_data[f"score_ref_{ref_idx}"] = ref_scores_train[
            test_indices, ref_idx
        ].flatten()
        df_test_data[f"member_ref_{ref_idx}"] = ref_memberships[
            test_indices, ref_idx
        ].flatten()

    df_test_merge = pd.DataFrame.from_dict(df_test_data)

    # Create population data
    n_population = len(target_scores_population)
    df_population_data = {
        "score_orig": target_scores_population.flatten(),
    }

    # Add reference scores for population (no membership indicators needed)
    for ref_idx in range(n_refs):
        df_population_data[f"score_ref_{ref_idx}"] = ref_scores_population[
            :, ref_idx
        ].flatten()
        df_population_data[f"member_ref_{ref_idx}"] = np.zeros(
            n_population, dtype=bool
        ).flatten()

    df_population = pd.DataFrame.from_dict(df_population_data)

    return df_train_merge, df_test_merge, df_population


def prepare_lira_data(
    target_logits: np.ndarray,
    shadow_logits: np.ndarray,
    target_dataset: Tuple[Subset, np.ndarray],
    shadow_datasets: List[Tuple[Subset, np.ndarray]],
    user_id_key: str = "user_id",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare data for online and offline LiRA attacks.

    Args:
        target_logits: Logits from target model
        shadow_logits: Logits from shadow models
        target_dataset: Target dataset with membership information
        shadow_datasets: Shadow datasets with membership information

    Returns:
        DataFrames for training and testing (online and offline variants)
    """
    n_samples = len(target_logits)

    # Create membership arrays for each shadow model
    memberships = np.array([keep for _, keep in shadow_datasets])

    # Calculate mean and std of logits for in and out samples
    score_mean_in = np.zeros(n_samples)
    score_std_in = np.zeros(n_samples)
    score_mean_out = np.zeros(n_samples)
    score_std_out = np.zeros(n_samples)

    # Vectorized computation of mean and std for in and out logits
    # Create masks for in and out samples
    in_mask = memberships[:, np.arange(n_samples)]
    out_mask = ~in_mask

    # Create 3D arrays where we can use broadcasting
    # Shape: (n_shadows, n_samples)
    expanded_logits = shadow_logits

    # Calculate means using masked operations and np.nanmean
    # First create masked arrays where invalid entries are set to NaN
    in_masked = np.where(in_mask[:, :, np.newaxis], expanded_logits, np.nan)
    out_masked = np.where(out_mask[:, :, np.newaxis], expanded_logits, np.nan)

    # Calculate means along shadow models dimension (axis=0)
    score_mean_in = np.nanmean(in_masked, axis=0).flatten()
    score_mean_out = np.nanmean(out_masked, axis=0).flatten()

    # Calculate standard deviations
    # Add a small constant (1e-8) to avoid division by zero
    score_std_in = np.nanstd(in_masked, axis=0).flatten()
    score_std_in = score_std_in + 1e-8  # Add epsilon as a separate step
    score_std_out = np.nanstd(out_masked, axis=0).flatten()
    score_std_out = score_std_out + 1e-8  # Add epsilon as a separate step

    # Handle NaN values for samples with no in or out examples
    score_mean_in = np.nan_to_num(score_mean_in)
    score_std_in = np.nan_to_num(score_std_in, nan=1e-8)
    score_mean_out = np.nan_to_num(score_mean_out)
    score_std_out = np.nan_to_num(score_std_out, nan=1e-8)

    target_membership = target_dataset[-1]

    # Create training dataframe for online LiRA attack
    df_train_online = pd.DataFrame(
        {
            user_id_key: range(sum(target_membership)),
            "score_orig": target_logits[target_membership].flatten(),
            "score_mean_in": score_mean_in[target_membership],
            "score_std_in": score_std_in[target_membership],
            "score_mean_out": score_mean_out[target_membership],
            "score_std_out": score_std_out[target_membership],
        }
    )

    # Create testing dataframe for online LiRA attack
    df_test_online = pd.DataFrame(
        {
            user_id_key: range(sum(~target_membership)),
            "score_orig": target_logits[~target_membership].flatten(),
            "score_mean_in": score_mean_in[~target_membership],
            "score_std_in": score_std_in[~target_membership],
            "score_mean_out": score_mean_out[~target_membership],
            "score_std_out": score_std_out[~target_membership],
        }
    )

    # Create training dataframe for offline LiRA attack
    df_train_offline = pd.DataFrame(
        {
            user_id_key: range(sum(target_membership)),
            "score_orig": target_logits[target_membership].flatten(),
            "score_mean": score_mean_out[target_membership],
            "score_std": score_std_out[target_membership],
        }
    )

    # Create testing dataframe for offline LiRA attack
    df_test_offline = pd.DataFrame(
        {
            user_id_key: range(sum(~target_membership)),
            "score_orig": target_logits[~target_membership].flatten(),
            "score_mean": score_mean_out[~target_membership],
            "score_std": score_std_out[~target_membership],
        }
    )

    return df_train_online, df_test_online, df_train_offline, df_test_offline
