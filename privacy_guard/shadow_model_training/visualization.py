# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
"""
Visualization utilities for privacy attack results.

This module provides functions for visualizing and analyzing the results of
privacy attacks, including ROC curves and score distributions.
"""

from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from privacy_guard.analysis.mia.analysis_node import AnalysisNode
from sklearn.metrics import roc_curve


def analyze_attack(attack_results: Any, attack_name: str) -> Dict[str, Any]:
    """
    Analyze attack results using AnalysisNode.

    Args:
        attack_results: Results from a privacy attack
        attack_name: Name of the attack for display purposes

    Returns:
        Dictionary containing analysis results
    """
    analysis_node = AnalysisNode(
        analysis_input=attack_results,
        delta=1e-5,
        n_users_for_eval=min(
            len(attack_results.df_train_user), len(attack_results.df_test_user)
        ),
        num_bootstrap_resampling_times=1000,
        show_progress=False,
    )

    # Run the analysis
    analysis_results = analysis_node.compute_outputs()

    # Print the results
    print(f"\n{attack_name} Attack Results:")
    print(
        f"Attack Accuracy: {analysis_results['accuracy']:.4f} (95% CI: [{analysis_results['accuracy_ci'][0]:.4f}, {analysis_results['accuracy_ci'][1]:.4f}])"
    )
    print(
        f"Attack AUC: {analysis_results['auc']:.4f} (95% CI: [{analysis_results['auc_ci'][0]:.4f}, {analysis_results['auc_ci'][1]:.4f}])"
    )
    print(f"Epsilon at TPR=1% (Upper Bound): {analysis_results['eps_tpr_ub'][0]:.4f}")
    print(f"Epsilon at TPR=1% (Lower Bound): {analysis_results['eps_tpr_lb'][0]:.4f}")

    return analysis_results


def plot_score_distributions(attack_results: Any, title: str) -> None:
    """
    Plot score distributions for members and non-members.

    Args:
        attack_results: Results from a privacy attack
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))

    # Get the scores
    train_scores = attack_results.df_train_user["score"]
    test_scores = attack_results.df_test_user["score"]

    # Create a boxplot of scores
    plt.figure(figsize=(10, 6))
    boxplot_data = [train_scores, test_scores]
    plt.boxplot(
        boxplot_data, labels=["Training Data (Members)", "Test Data (Non-members)"]
    )
    plt.ylabel("Score")
    plt.title(f"{title} - Boxplot of Scores for Members vs. Non-members")
    plt.grid(True, axis="y")
    plt.show()

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
        label="Test Data (Non-members)",
        color="tab:blue",
    )

    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.title(f"{title} - Distribution of Scores for Members vs. Non-members")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_roc_curve(
    attack_analysis: Dict[str, Any], attack_results: Any, title: str = "LiRA"
) -> None:
    """
    Plot the ROC curve for the LiRA attack.

    Args:
        attack_analysis: The output from analysis_node.compute_outputs()
        attack_results: The output from lira_attack.run_attack()
        title: Title for the plot
    """
    plt.figure(figsize=(10, 8))

    # Get the user scores
    train_scores = attack_results.df_train_user["score"]
    test_scores = attack_results.df_test_user["score"]

    # Combine scores and create labels
    all_scores = np.concatenate([train_scores, test_scores])
    all_labels = np.concatenate(
        [np.ones(len(train_scores)), np.zeros(len(test_scores))]
    )

    # Calculate ROC curve points using scikit-learn
    fpr, tpr, _ = roc_curve(all_labels, all_scores)

    # Filter for fpr < 0.1
    mask = fpr < 0.1
    fpr = fpr[mask]
    tpr = tpr[mask]

    # Plot ROC curve with semilog x-axis
    plt.loglog(fpr, tpr, "b-", linewidth=2)
    plt.loglog([0, 0.1], [0, 0.1], "k--", linewidth=1)  # Random guess line

    # Add annotations
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title(f"{title} ROC Curve", fontsize=16)

    # Add text with attack metrics
    plt.text(
        0.02,
        0.97,
        f"AUC: {attack_analysis['auc']:.4f}\n"
        f"Accuracy: {attack_analysis['accuracy']:.4f}\n"
        f"Epsilon at TPR=1%: {attack_analysis['eps_tpr_ub'][0]:.4f}",
        transform=plt.gca().transAxes,
        fontsize=14,
        verticalalignment="top",
    )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
