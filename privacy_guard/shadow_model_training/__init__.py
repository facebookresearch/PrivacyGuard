# pyre-strict
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Shadow model training package for privacy attacks.

from privacy_guard.shadow_model_training.dataset import (
    create_shadow_datasets,
    load_cifar10,
)
from privacy_guard.shadow_model_training.model import create_model
from privacy_guard.shadow_model_training.training import (
    get_transformed_logits,
    prepare_lira_data,
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
    "get_transformed_logits",
    "load_cifar10",
    "plot_roc_curve",
    "plot_score_distributions",
    "prepare_lira_data",
    "train_model",
]
