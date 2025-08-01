# pyre-ignore-all-errors
#!/usr/bin/env -S grimaldi --kernel bento_kernel_empirical_dp
# FILE_UID: 65090375-6bbb-4bbb-b05f-3e0dd15dbbff
# NOTEBOOK_NUMBER: N7208437 (9923413411059195)

""":md
# Loss-Based Membership Inference Attack with PrivacyGuard

## Introduction

We showcase a loss-based membership inference attack using PrivacyGuard. This tutorial will guide you through the process of conducting a loss-based membership inference attack on a machine learning model and analyzing the results.

### What is a Membership Inference Attack?

A membership inference attack (MIA) is a privacy attack that aims to determine whether a specific data point was used to train a machine learning model. These attacks exploit the fact that models often behave differently on data they were trained on compared to data they haven't seen before.

### Why Loss-Based Attacks?

Loss-based attacks are a common type of membership inference attack that use the loss value (or prediction error) of a model on a data point as a signal for membership. The intuition is that models typically have lower loss values on training data compared to unseen data. This difference can be exploited to infer membership.

### What We'll Cover

In this tutorial, we will:
1. Set up a simple classification problem using Gaussian data
2. Train a model on a subset of the data
3. Perform a loss-based membership inference attack using the `LossAttack` class
4. Analyze the attack results using the `LossAnalysisNode` class
5. Interpret the results and discuss privacy implications

Let's get started!
"""

""":md
## Setup and Imports

First, let's import the necessary libraries and set up our environment.
"""

""":py '1178243353777211'"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from privacy_guard.analysis.analysis_node import AnalysisNode
from privacy_guard.attacks.loss_attack import LossAttack
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

""":md
## Generating Synthetic Data

We'll create a synthetic dataset for a binary classification problem using Gaussian distributions. This will serve as our example dataset for demonstrating the membership inference attack.
"""

""":py"""
# Define parameters for the Gaussian mixture
n_class = 10  # Number of classes
n_train = 20 * n_class  # Number of training samples
n_test = 20 * n_class  # Number of test samples
sigma = 10  # Standard deviation of the Gaussian distributions
d = 1000  # Dimensionality of the feature space


""":py '1292343756230394'"""
# Generate training data
train_y = torch.arange(0, n_class).repeat_interleave(n_train // n_class)
class_centers = torch.randn(n_class, d)  # Random centers for each class
train_x = class_centers[train_y] + sigma * torch.randn(n_train, d)

# Generate test data
test_y = torch.arange(0, n_class).repeat_interleave(n_test // n_class)
test_x = class_centers[test_y] + sigma * torch.randn(n_test, d)

# Create datasets
trainset = TensorDataset(train_x, train_y)
testset = TensorDataset(test_x, test_y)

print(f"Training set shape: {train_x.shape}, {train_y.shape}")
print(f"Testing set shape: {test_x.shape}, {test_y.shape}")

""":py"""
# Create data loaders
train_loader = DataLoader(trainset, batch_size=64)
test_loader = DataLoader(testset, batch_size=n_test)

""":md
## Training the Model

Now, let's train our model on the training data. We'll use cross-entropy loss and the Adam optimizer.
"""

""":py '1203302637926066'"""
# Create a linear model
model = nn.Linear(d, n_class)
print(model)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()

# Training loop
losses = []
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

for _ in range(2):  # Train for 2 epochs
    for inp, out in train_loader:
        loss = criterion(model(inp), out)
        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.item())
        optimizer.step()

""":md
Let's visualize the training loss over timesteps:
"""

""":py '1281109070052753'"""
plt.plot(losses)
plt.ylabel("Loss")
plt.xlabel("Step")
plt.title("Training Loss")
plt.show()

""":md
## Performing a Loss-Based Membership Inference Attack

Now that we have trained our model, we can perform a loss-based membership inference attack using the `LossAttack` class from the Privacy Guard framework.

### Understanding the Attack

The loss-based attack works by computing the loss of the model on both the training data (members) and the testing data (non-members). The intuition is that the model will have lower loss values on the training data it has seen before compared to the testing data it hasn't seen.

By analyzing the distribution of these loss values, an attacker can potentially distinguish between members and non-members of the training set, thus inferring membership.

We run the attack via the `LossAttack` class. The class takes as input a function that computes loss given a model and a dataloader. In our case we define the `cross_entropy_loss` function. 
"""

""":py"""


# define the loss computation function
@torch.no_grad()
def compute_loss_cross_entropy(
    model: nn.Module, dataloader: DataLoader
) -> torch.Tensor:
    """
    Computes the losses given by the model over the dataloader.
    """
    losses = []
    criterion = nn.CrossEntropyLoss(reduction="none")

    for img, target in dataloader:
        outputs = model(img)
        batch_losses = criterion(outputs, target)
        losses += batch_losses.tolist()

    return torch.Tensor(losses)


""":py '1566909348045740'"""
# Perform the loss-based membership inference attack
loss_attack = LossAttack(
    private_model=model,
    private_train=train_loader,
    private_holdout=test_loader,
    compute_loss=compute_loss_cross_entropy,
)

# Run the attack
attack_results = loss_attack.run_attack()

print(
    f"Attack results shape - Train: {attack_results.df_train_user.shape}, Test: {attack_results.df_test_user.shape}"
)

""":md
Let's visualize the distribution of scores (negative loss values) for the training and testing data:
"""

""":py '983447870307524'"""
plt.figure(figsize=(10, 6))

# Get the scores
train_scores = attack_results.df_train_user["score"]
test_scores = attack_results.df_test_user["score"]

# Plot histograms
plt.hist(
    train_scores,
    bins=30,
    alpha=0.5,
    label="Training Data (Members)",
    color="tab:orange",
)
plt.hist(
    test_scores,
    bins=30,
    alpha=0.5,
    label="Testing Data (Non-members)",
    color="tab:blue",
)

plt.xlabel("Score (Negative Loss)")
plt.ylabel("Frequency")
plt.title("Distribution of Scores for Members vs. Non-members")
plt.legend()
plt.grid(True)
plt.show()

""":md
We see that it is easy to separate the members from non-members using a threshold.
"""

""":md
## Analyzing the Attack Results with AnalysisNode

Now, let's use the `AnalysisNode` class to analyze the results of our attack. This class provides various metrics to evaluate the effectiveness of the attack, such as accuracy, AUC, and privacy leakage measured by epsilon.
"""

""":py '1418987862856247'"""
# Create a LossAnalysisNode
analysis_node = AnalysisNode(
    analysis_input=attack_results,
    delta=1e-5,  # Small delta value for privacy calculations
    n_users_for_eval=min(
        len(attack_results.df_train_user), len(attack_results.df_test_user)
    ),
    num_bootstrap_resampling_times=1000,
    show_progress=True,
)

# Run the analysis
analysis_results = analysis_node.compute_outputs()

# Print the results
print(
    f"Attack Accuracy: {analysis_results["accuracy"]:.4f} (95% CI: [{analysis_results["accuracy_ci"][0]:.4f}, {analysis_results["accuracy_ci"][1]:.4f}])"
)
print(
    f"Attack AUC: {analysis_results["auc"]:.4f} (95% CI: [{analysis_results["auc_ci"][0]:.4f}, {analysis_results["auc_ci"][1]:.4f}])"
)
print(f"Epsilon at TPR=1% (Upper Bound): {analysis_results["eps_tpr_ub"][0]:.4f}")
print(f"Epsilon at TPR=1% (Lower Bound): {analysis_results["eps_tpr_lb"][0]:.4f}")

""":md
## Interpreting the Results

Let's interpret the results of our membership inference attack:

### Accuracy and AUC

- **Accuracy**: This measures the proportion of correctly classified samples (members and non-members). An accuracy of 0.5 is equivalent to random guessing, while an accuracy of 1.0 means perfect classification.
- **AUC (Area Under the ROC Curve)**: This measures the ability of the attack to distinguish between members and non-members across different thresholds. An AUC of 0.5 indicates no discriminative ability, while an AUC of 1.0 indicates perfect discrimination. As AUC is measured using different thresholds, it is thought of as an "average case" performance of the attack.

### Privacy Leakage (Epsilon)

- **Epsilon**: This is a measure of privacy leakage in the context of differential privacy. A higher epsilon value indicates more privacy leakage. In practical terms, it quantifies how much more likely a member is to be correctly identified compared to a non-member.
- **Epsilon at TPR=1%**: This is the epsilon value at a true positive rate (TPR) of 1%. It represents the privacy leakage when the attack correctly identifies only 1% of the members. 

### Interpreting Our Results

Based on the results we obtained:

1. The accuracy is significantly above 0.5, which indicates that the attack is successful in distinguishing between members and non-members.
2. The AUC is significantly above 0.5, which indicates that the attack has good discriminative ability.
3. The epsilon value is high, which indicates significant privacy leakage.

### Mitigating Privacy Risks

To mitigate the privacy risks identified by this attack, you could consider:

1. **Regularization**: Apply stronger regularization to the model to reduce overfitting, which can help reduce the gap between training and testing loss.
2. **Differential Privacy**: Train the model with differential privacy guarantees, which add noise to the training process to limit the influence of individual training samples.
3. **Model Pruning**: Reduce the complexity of the model to prevent it from memorizing the training data.
4. **Early Stopping**: Stop training before the model starts to overfit the training data.
"""
