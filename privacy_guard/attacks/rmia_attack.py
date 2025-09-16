# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
from typing import Optional

import numpy as np
import pandas as pd

from privacy_guard.analysis.mia.aggregate_analysis_input import (
    AggregateAnalysisInput,
    AggregationType,
)
from privacy_guard.attacks.base_attack import BaseAttack
from sklearn.metrics import auc, roc_curve


class RmiaAttack(BaseAttack):
    """
    This is an implementation of Robust Membership Inference Attack (RMIA)
    based on the paper: https://arxiv.org/abs/2312.03262.
    Implementation incorporated into PrivacyGuard based on the official implementation from https://github.com/privacytrustlab/ml_privacy_meter/tree/master


    The attack estimates population probability distributions using reference models and compares
    the target model's predictions against estimated population averages to determine membership.

    The attack leverages:
        - Multiple reference models trained without target samples
        - Population samples to establish baseline probability distributions
        - Ratio-based scoring for membership inference
    """

    # Default column patterns for reference model data
    REF_SCORE_PREFIX = "score_ref_"
    REF_MEMBER_PREFIX = "member_ref_"

    def __init__(
        self,
        df_train_merge: pd.DataFrame,
        df_test_merge: pd.DataFrame,
        df_population: pd.DataFrame,
        row_aggregation: AggregationType,
        num_reference_models: Optional[int] = None,
        alpha_coefficient: float = 0.3,
        enable_auto_tuning: bool = False,
        user_id_key: str = "user_id",
    ) -> None:
        """
        Initialize the RMIA attack.

        Args:
            df_train_merge: Training data with target and reference model scores
            df_test_merge: Test data with target and reference model scores
            df_population: Population data for probability distribution estimation
            row_aggregation: User aggregation strategy specification
            alpha_coefficient: Approximation coefficient for population probability estimation
            num_reference_models: Number of reference models to use in attack (default: half of available models)
            enable_auto_tuning: Enable automatic tuning of alpha_coefficient
        """
        self.df_train_merge: pd.DataFrame = df_train_merge.copy()
        self.df_test_merge: pd.DataFrame = df_test_merge.copy()
        self.df_population: pd.DataFrame = df_population.copy()
        self.row_aggregation: AggregationType = row_aggregation
        self.alpha_coefficient: float = alpha_coefficient

        if num_reference_models is None:
            # estimate number of reference models based on the number of columns in the dataframe
            num_reference_models = max(df_population.shape[1] // 4, 1)
        else:
            # ensure that num_reference_models is not greater than the number of reference models available
            num_reference_models = min(
                num_reference_models, df_population.shape[1] // 2
            )

        self.num_reference_models: int = num_reference_models
        self.enable_auto_tuning: bool = enable_auto_tuning
        self.user_id_key: str = user_id_key

        # Validate input data integrity
        self._validate_input_data()

    def _validate_input_data(self) -> None:
        """
        Validate that input dataframes contain required columns and data.

        Raises:
            ValueError: If required columns are missing or data is empty
        """
        for df_name, df in [
            ("df_train_merge", self.df_train_merge),
            ("df_test_merge", self.df_test_merge),
            ("df_population", self.df_population),
        ]:
            if df.empty:
                raise ValueError(f"{df_name} cannot be empty")

            if "score_orig" not in df.columns:
                raise ValueError(f"{df_name} must contain 'score_orig' column")

        # Check for reference model columns
        ref_score_cols = [
            col
            for col in self.df_train_merge.columns
            if col.startswith(self.REF_SCORE_PREFIX)
        ]
        ref_member_cols = [
            col
            for col in self.df_train_merge.columns
            if col.startswith(self.REF_MEMBER_PREFIX)
        ]

        if not ref_score_cols:
            raise ValueError(
                f"No reference score columns found (expected {self.REF_SCORE_PREFIX}*)"
            )
        if not ref_member_cols:
            raise ValueError(
                f"No reference membership columns found (expected {self.REF_MEMBER_PREFIX}*)"
            )

    @classmethod
    def compute_ref_signal_averages(
        cls,
        ref_signals: np.ndarray,
        ref_memberships: np.ndarray,
        num_models: Optional[int] = None,
        alpha: float = 0.3,
    ) -> np.ndarray:
        """
        Compute average prediction probabilities from reference models excluding target samples.

        Args:
            ref_signals: Prediction scores from reference models
            ref_memberships: Boolean membership matrix indicating training inclusion
            num_models: Number of reference models to consider
            alpha: Approximation coefficient for population probability

        Returns:
            Averaged prediction scores excluding membership samples
        """
        non_member_mask = ~ref_memberships
        out_ref_signals = ref_signals * non_member_mask

        if num_models is None:
            num_models = ref_signals.shape[1] // 2

        if num_models > 1:
            # Select top non-zero signals for each sample
            out_ref_signals = np.sort(out_ref_signals, axis=1)[:, -num_models:]
        else:
            # Apply single model approximation formula
            if alpha != 0:
                approximation = ((ref_signals + alpha - 1) / alpha) * ref_memberships
                out_ref_signals += approximation
            else:
                # Default fallback approximation w/ alpha=0.3
                fallback = ((ref_signals - 0.7) / 0.3) * ref_memberships
                out_ref_signals += fallback

        return out_ref_signals

    def _auto_tune_alpha_coefficient(
        self,
        target_model_idx: int,
        ref_scores: np.ndarray,
        ref_memberships: np.ndarray,
        population_ref_scores: np.ndarray,
    ) -> float:
        """Auto-tune alpha coefficient using cross-validation on reference models.
        Args:
            target_model_idx: Index of target model in reference scores
            ref_scores: Reference model prediction scores
            ref_memberships: Reference model membership matrix
            population_ref_scores: Population reference model scores
        Returns:
            Tuned alpha coefficient
        """
        if ref_scores.shape[1] < 2:
            print("Not enough reference models for auto-tuning")
            return self.alpha_coefficient

        # Select validation model
        val_idx = target_model_idx
        val_scores = ref_scores[:, val_idx]

        # Set model for evaluation
        eval_idx = [(val_idx + 1) % ref_scores.shape[1]]
        eval_scores = ref_scores[:, eval_idx]
        eval_memberships = ref_memberships[:, eval_idx]
        eval_population_scores = population_ref_scores[:, eval_idx]

        best_alpha, best_auc = 0.0, 0.0

        for alpha in np.arange(0, 1.1, 0.1):
            scores = self._compute_membership_scores(
                val_scores,
                eval_scores,
                eval_memberships,
                population_ref_scores[:, val_idx],
                eval_population_scores,
                alpha,
                1,
            )

            fpr, tpr, _ = roc_curve(eval_memberships.astype(int), scores)
            current_auc = auc(fpr, tpr)

            if current_auc > best_auc:
                best_auc = current_auc
                best_alpha = alpha
            print(f"Alpha ={alpha:.2f}, AUC={current_auc:.4f}")

        print(f"Alpha tuning: best={best_alpha:.2f}, AUC={best_auc:.4f}")
        return best_alpha

    def _compute_membership_scores(
        self,
        target_scores: np.ndarray,
        ref_scores: np.ndarray,
        ref_memberships: np.ndarray,
        population_target_scores: np.ndarray,
        population_ref_scores: np.ndarray,
        alpha: float,
        num_models: int,
    ) -> np.ndarray:
        """
        Execute core membership inference scoring algorithm.

        Args:
            target_scores: Target model prediction scores
            ref_scores: Reference model prediction scores
            ref_memberships: Reference model membership matrix
            population_target_scores: Population target model scores
            population_ref_scores: Population reference model scores
            alpha: Population probability approximation coefficient
            num_models: Number of reference models to use

        Returns:
            Membership inference scores (higher indicates greater membership likelihood)
        """

        # Compute reference signals for target dataset
        target_ref_mean = self.compute_ref_signal_averages(
            ref_scores, ref_memberships, num_models, alpha
        )
        target_mean_out = np.mean(target_ref_mean, axis=1)
        target_population_estimate = (1 + alpha) / 2 * target_mean_out + (1 - alpha) / 2
        target_probability_ratios = target_scores.ravel() / target_population_estimate

        # Compute reference signals for population dataset
        # Population samples are not used for training, so we set the membership to False
        population_memberships = np.zeros_like(population_ref_scores).astype(bool)
        population_ref_mean = self.compute_ref_signal_averages(
            population_ref_scores, population_memberships, num_models, alpha
        )
        population_mean_out = np.mean(population_ref_mean, axis=1)
        population_estimate = (1 + alpha) / 2 * population_mean_out + (1 - alpha) / 2
        population_probability_ratios = (
            population_target_scores.ravel() / population_estimate
        )

        # Calculate final membership scores
        ratio_comparisons = (
            target_probability_ratios[:, np.newaxis] / population_probability_ratios
        )
        membership_scores = np.average(ratio_comparisons > 1.0, axis=1)

        return membership_scores

    def _extract_model_data(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract and validate model prediction data from input dataframe.

        Args:
            df: Input dataframe containing model scores and membership data

        Returns:
            Tuple containing target scores, reference scores, and membership indicators

        Raises:
            ValueError: If required columns are missing from dataframe
        """
        # Validate and extract target model scores
        if "score_orig" not in df.columns:
            raise ValueError("Missing required 'score_orig' column in dataframe")
        target_scores = df["score_orig"].values

        # Extract reference model score columns
        ref_score_columns = [
            col for col in df.columns if col.startswith(self.REF_SCORE_PREFIX)
        ]
        if not ref_score_columns:
            raise ValueError(
                f"No reference score columns found (expected {self.REF_SCORE_PREFIX}*)"
            )
        ref_scores = df[ref_score_columns].values

        # Extract reference model membership columns
        ref_member_columns = [
            col for col in df.columns if col.startswith(self.REF_MEMBER_PREFIX)
        ]
        if not ref_member_columns:
            raise ValueError(
                f"No membership columns found (expected {self.REF_MEMBER_PREFIX}*)"
            )
        ref_memberships = df[ref_member_columns].values.astype(bool)

        return target_scores, ref_scores, ref_memberships

    def run_attack(self) -> AggregateAnalysisInput:
        """
        Execute relative membership inference attack on input data.

        Returns:
            AggregateAnalysisInput: Output containing train and test data,
            ready for consumption by different analyses
        """
        # Validate and extract model data from input dataframes
        train_target_scores, train_ref_scores, train_ref_memberships = (
            self._extract_model_data(self.df_train_merge)
        )
        test_target_scores, test_ref_scores, test_ref_memberships = (
            self._extract_model_data(self.df_test_merge)
        )
        population_target_scores, population_ref_scores, _ = self._extract_model_data(
            self.df_population
        )

        # Auto-tune alpha coefficient if requested
        current_alpha = self.alpha_coefficient
        if self.enable_auto_tuning:
            # Use reference data for alpha tuning
            # Combine training and test reference data for more robust tuning
            combined_ref_scores = np.vstack([train_ref_scores, test_ref_scores])
            combined_ref_memberships = np.vstack(
                [train_ref_memberships, test_ref_memberships]
            )

            current_alpha = self._auto_tune_alpha_coefficient(
                target_model_idx=0,  # Default target model index
                ref_scores=combined_ref_scores,
                ref_memberships=combined_ref_memberships,
                population_ref_scores=population_ref_scores,
            )
        # Compute membership scores for training data
        train_membership_scores = self._compute_membership_scores(
            target_scores=train_target_scores,
            ref_scores=train_ref_scores,
            ref_memberships=train_ref_memberships,
            population_target_scores=population_target_scores,
            population_ref_scores=population_ref_scores,
            alpha=current_alpha,
            num_models=self.num_reference_models,
        )

        # Compute membership scores for test data
        test_membership_scores = self._compute_membership_scores(
            target_scores=test_target_scores,
            ref_scores=test_ref_scores,
            ref_memberships=test_ref_memberships,
            population_target_scores=population_target_scores,
            population_ref_scores=population_ref_scores,
            alpha=current_alpha,
            num_models=self.num_reference_models,
        )

        # Create output dataframes with computed scores
        self.df_train_merge["score"] = train_membership_scores
        self.df_test_merge["score"] = test_membership_scores

        # Return analysis input for downstream processing
        analysis_input = AggregateAnalysisInput(
            row_aggregation=self.row_aggregation,
            df_train_merge=self.df_train_merge,
            df_test_merge=self.df_test_merge,
            user_id_key=self.user_id_key,
        )

        return analysis_input
