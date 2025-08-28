# pyre-strict
#!/usr/bin/env -S grimaldi --kernel bento_kernel_privacy_guard
# FILE_UID: 3eb7d962-95ba-4ce9-8fa7-c288845d3cbf
# NOTEBOOK_NUMBER: N7714854 (1978195306257478)

""":md
# LiRA (Likelihood Ratio Attack) with PrivacyGuard for CIFAR-10

## Introduction

This tutorial demonstrates how to use the Privacy Guard framework to perform Likelihood Ratio Attacks (LiRA) on machine learning models trained on the CIFAR-10 dataset. We'll cover both online and offline variants of the attack. For more details, refer to the paper: https://arxiv.org/pdf/2112.03570.

### What is a Likelihood Ratio Attack (LiRA)?

LiRA is a powerful membership inference attack that uses the likelihood ratio test to determine whether a specific data point was used to train a machine learning model. Unlike simpler attacks that rely on a single threshold, LiRA uses multiple shadow models to estimate the distribution of model outputs for members and non-members of the training set.

### Online vs. Offline LiRA

- **Online LiRA**: Uses shadow models trained both with and without the target examples to estimate the likelihood distributions.
- **Offline LiRA**: Uses only shadow models trained without the target examples, making it more practical in real-world scenarios where attackers don't have access to models trained with specific examples.

### What We'll Cover

In this tutorial, we will:
1. Set up the CIFAR-10 dataset and create training/testing splits
2. Train a target model and multiple shadow models
3. Extract logits from all models
4. Perform both online and offline LiRA attacks using Privacy Guard's `LiraAttack` class
5. Analyze the attack results and evaluate privacy risks
6. Compare the effectiveness of online and offline attacks

Let's get started!
"""

""":md
## Setup and Imports

First, let's import the necessary libraries and set up our environment.
"""

""":py '1106694958073797'"""
# %local-changes # pragma: uncomment

""":py '1263338561945640'"""
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.typing import NDArray

from privacy_guard.analysis.mia.aggregate_analysis_input import AggregationType
from privacy_guard.attacks.lira_attack import LiraAttack
from privacy_guard.shadow_model_training import (
    analyze_attack,
    create_model,
    create_shadow_datasets,
    get_transformed_logits,
    load_cifar10,
    plot_roc_curve,
    plot_score_distributions,
    prepare_lira_data,
    train_model,
)
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Use CUDA if available
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

""":md
## Dataset Preparation

Now, let's load the CIFAR-10 dataset and prepare it for our experiments.
"""

""":py"""
# Load the dataset
train_dataset: CIFAR10
test_dataset: CIFAR10
train_dataset, test_dataset = load_cifar10()
print(f"Training set size: {len(train_dataset)}")
print(f"Test set size: {len(test_dataset)}")

""":md
## Creating Shadow Datasets

For LiRA, we need to create multiple shadow datasets. Each shadow model will be trained on a different subset of the training data.
"""

""":py '725665850462310'"""
# Create shadow datasets
num_shadows: int = 8  # Using fewer shadows for tutorial speed
shadow_datasets: List[Tuple[Subset, NDArray[np.float64]]]
target_dataset: Tuple[Subset, NDArray[np.float64]]
shadow_datasets, target_dataset = create_shadow_datasets(
    train_dataset, n_shadows=num_shadows
)

# Print shadow dataset sizes
for i, (shadow_in, _) in enumerate(shadow_datasets):
    print(f"Shadow {i}: {len(shadow_in)} in-samples")

print(
    f"Target: {len(target_dataset[0])} in-samples, {len(target_dataset[1])} out-samples"
)

""":md
## Training Target and Shadow Models

Let's train our target model and shadow models.
"""

""":py '1713389759292778'"""
# Create data loaders
batch_size: int = 256
test_loader: DataLoader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)

EPOCHS: int = 50
# Train target model (using target_in samples)
print("Training target model...")
target_model: nn.Module = create_model().to(DEVICE)
target_in: Subset = target_dataset[0]
target_loader: DataLoader = DataLoader(
    target_in, batch_size=batch_size, shuffle=True, num_workers=2
)
target_model = train_model(
    target_model, target_loader, test_loader, epochs=EPOCHS, device=DEVICE
)

# Train shadow models
shadow_models: List[nn.Module] = []
for i, (shadow_in, _) in enumerate(shadow_datasets):
    print(f"Training shadow model {i+1}/{len(shadow_datasets)}...")
    shadow_loader = DataLoader(
        shadow_in, batch_size=batch_size, shuffle=True, num_workers=2
    )
    shadow_model = create_model().to(DEVICE)
    shadow_model = train_model(
        shadow_model, shadow_loader, test_loader, epochs=EPOCHS, device=DEVICE
    )
    shadow_models.append(shadow_model)

""":md
## Extracting Logits

Now, let's extract logits from all models for the entire training dataset. These logits will be used for the LiRA attack.
"""

""":py '2547346078950993'"""
# Create a DataLoader for the entire training dataset (without shuffling)
train_eval_loader: DataLoader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)

# Get logits from target model
print("Getting logits from target model...")
target_logits: NDArray[np.float64] = get_transformed_logits(
    target_model, train_eval_loader, DEVICE
)

# Get logits from shadow models
shadow_logits_list: List[NDArray[np.float64]] = []
num_shadows = len(shadow_models)
for i, shadow_model in enumerate(shadow_models):
    print(f"Getting logits from shadow model {i+1}/{num_shadows}...")
    shadow_logits_list.append(
        get_transformed_logits(shadow_model, train_eval_loader, DEVICE)
    )

shadow_logits: NDArray[np.float64] = np.array(shadow_logits_list)

""":md
## Preparing Data for LiRA Attack

Now, let's prepare the data for the LiRA attack using the Privacy Guard framework.
"""

""":py '739573058862447'"""
# Prepare data for LiRA attack
df_train_online: pd.DataFrame
df_test_online: pd.DataFrame
df_train_offline: pd.DataFrame
df_test_offline: pd.DataFrame
df_train_online, df_test_online, df_train_offline, df_test_offline = prepare_lira_data(
    target_logits, shadow_logits, target_dataset, shadow_datasets
)

print(f"Training data shape: {df_train_online.shape}")
print(f"Testing data shape: {df_test_online.shape}")

""":md
## Performing LiRA Attacks

Now, let's perform both online and offline LiRA attacks using the Privacy Guard framework.
"""

""":py"""
# Online LiRA attack
online_lira_attack = LiraAttack(
    df_train_merge=df_train_online,
    df_test_merge=df_test_online,
    row_aggregation=AggregationType.NONE,
    use_fixed_variance=True,
    std_dev_type="mix",
    online_attack=True,
)

# Run the attack
online_attack_results: Any = online_lira_attack.run_attack()

# Offline LiRA attack
offline_lira_attack = LiraAttack(
    df_train_merge=df_train_offline,
    df_test_merge=df_test_offline,
    row_aggregation=AggregationType.NONE,
    use_fixed_variance=True,
    std_dev_type="shadows_out",
    online_attack=False,
)

# Run the attack
offline_attack_results: Any = offline_lira_attack.run_attack()

""":md
## Analyzing Attack Results

Let's analyze the results of our attacks using the `AnalysisNode` class.
"""

""":py '724491577158681'"""
# Analyze online attack
online_analysis: Dict[str, Any] = analyze_attack(online_attack_results, "Online LiRA")

# Analyze offline attack
offline_analysis: Dict[str, Any] = analyze_attack(
    offline_attack_results, "Offline LiRA"
)

""":md
## Visualizing Attack Results

Let's visualize the distribution of scores for members and non-members for both attacks.
"""

""":py '764890462598611'"""
# Plot score distributions
plot_score_distributions(online_attack_results, "Online LiRA")
plot_score_distributions(offline_attack_results, "Offline LiRA")

""":md
## Comparing ROC Curves

Let's compare the ROC curves for both attacks to visualize their performance.
"""

""":py '728737329798389'"""
# Plot ROC curves
plot_roc_curve(online_analysis, online_attack_results, title="Online LiRA")
plot_roc_curve(offline_analysis, offline_attack_results, title="Offline LiRA")

""":md
## Conclusion

In this tutorial, we demonstrated how to use the Privacy Guard framework to perform Likelihood Ratio Attacks (LiRA) on machine learning models trained on the CIFAR-10 dataset. We covered both online and offline variants of the attack and analyzed their effectiveness.

### Key Takeaways

1. **LiRA is a powerful membership inference attack** that uses multiple shadow models to estimate the distribution of model outputs for members and non-members.

2. **Online LiRA typically outperforms offline LiRA** because it has access to more information (shadow models trained both with and without target examples).

3. **Privacy Guard provides a comprehensive framework** for implementing and analyzing membership inference attacks, making it easier to assess the privacy risks of machine learning models.

### Mitigating Privacy Risks

To mitigate the privacy risks identified by LiRA attacks, consider the following approaches:

1. **Differential Privacy**: Train models with differential privacy guarantees to limit the influence of individual training examples.

2. **Regularization**: Apply stronger regularization techniques to reduce overfitting, which can help reduce the gap between model behavior on training and testing data.

3. **Model Pruning**: Reduce model complexity to prevent memorization of training data.

4. **Ensemble Methods**: Use ensemble methods to average predictions across multiple models, which can help reduce the variance in predictions.

By understanding and addressing these privacy risks, you can build more privacy-preserving machine learning models.
"""
