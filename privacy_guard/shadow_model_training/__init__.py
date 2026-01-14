# pyre-strict
"""
Shadow model training package for privacy attacks.

This package provides utilities for training shadow models and performing privacy attacks.
"""

from privacy_guard.shadow_model_training.dataset import (
    create_shadow_datasets,
    load_cifar10,
)
from privacy_guard.shadow_model_training.model import create_model
from privacy_guard.shadow_model_training.training import (
    evaluate_model,
    get_softmax_scores,
    get_transformed_logits,
    prepare_lira_data,
    prepare_rmia_data,
    train_model,
)
from privacy_guard.shadow_model_training.visualization import (
    analyze_attack,
    plot_roc_curve,
    plot_score_distributions,
)

__all__ = [
    "analyze_attack",
    "create_model",
    "create_shadow_datasets",
    "evaluate_model",
    "get_softmax_scores",
    "get_transformed_logits",
    "load_cifar10",
    "plot_roc_curve",
    "plot_score_distributions",
    "prepare_rmia_data",
    "prepare_lira_data",
    "train_model",
]
