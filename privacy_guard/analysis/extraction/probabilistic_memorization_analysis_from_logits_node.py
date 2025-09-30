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
import math
import warnings
from dataclasses import dataclass
from typing import Any, cast, Dict, List, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from privacy_guard.analysis.base_analysis_node import BaseAnalysisNode
from privacy_guard.analysis.base_analysis_output import BaseAnalysisOutput
from privacy_guard.analysis.extraction.probabilistic_memorization_analysis_from_logits_input import (
    ProbabilisticMemorizationAnalysisFromLogitsInput,
)

from tqdm import tqdm

tqdm.pandas()


@dataclass
class ProbabilisticMemorizationAnalysisFromLogitsNodeOutput(BaseAnalysisOutput):
    num_samples: int
    model_probability: pd.Series
    above_probability_threshold: pd.Series
    n_probabilities: Optional[pd.Series]
    augmented_output_dataset: pd.DataFrame


def apply_sampling_params(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """
    Apply sampling parameters (temperature, top_k, top_p) to logits.

    Args:
        logits: Input logits tensor of shape (..., vocab_size)
        temperature: Temperature for scaling logits (default: 1.0)
        top_k: Keep only top k logits, set others to -inf (optional)
        top_p: Keep top p probability mass, set others to -inf (optional)

    Returns:
        Modified logits tensor with sampling parameters applied
    """
    # Warn if both top_k and top_p are specified (not typical usage)
    if top_k is not None and top_p is not None:
        warnings.warn(
            "Both top_k and top_p sampling parameters are specified. "
            "While both will be applied sequentially, this is not typical usage. "
            "Consider using only one sampling method.",
            UserWarning,
            stacklevel=2,
        )

    # Apply temperature scaling
    if temperature != 1.0:
        logits = logits / temperature

    # Apply top_k filtering if specified
    if top_k is not None and top_k > 0:
        # Keep only top_k logits, set others to -inf
        top_k_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
        kth_scores = top_k_logits[..., -1].unsqueeze(-1)
        # Set logits below kth score to -inf
        logits = torch.where(
            logits < kth_scores,
            torch.full_like(logits, float("-inf")),
            logits,
        )

    # Apply top_p (nucleus) filtering if specified
    if top_p is not None and 0.0 < top_p < 1.0:
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

        # Convert to probabilities
        sorted_probs = F.softmax(sorted_logits, dim=-1)

        # Calculate cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Create mask for tokens to keep (cumulative probability <= top_p)
        # We keep the first token that exceeds top_p to ensure we always have at least one token
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift right to keep the first token that exceeds top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Convert back to original indices
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)

        # Set removed indices to -inf
        logits = torch.where(
            indices_to_remove, torch.full_like(logits, float("-inf")), logits
        )

    return logits


def _compute_model_probability_from_logits(
    row: pd.Series,
    logits_column: str,
    target_tokens_column: str,
    **generation_kwargs: Any,
) -> float:
    """Compute model probability from prediction logits using actual target tokens and generation parameters.

    Args:
        row (pd.Series): A row of a DataFrame containing the logits and target tokens columns.
        logits_column (str): Name of the column containing logits.
        target_tokens_column (str): Name of the column containing target token IDs.
        **generation_kwargs: Generation parameters (e.g., temperature, top_k, top_p).

    Returns:
        float: The model probability for the target.
    """
    prediction_logits = row[logits_column]
    target_tokens = row[target_tokens_column]

    # Check for None values and raise error instead of returning 0
    if prediction_logits is None:
        raise ValueError(f"Logits column '{logits_column}' contains None value")
    if target_tokens is None:
        raise ValueError(
            f"Target tokens column '{target_tokens_column}' contains None value"
        )

    # Convert to torch tensor if it's not already
    if not isinstance(prediction_logits, torch.Tensor):
        if isinstance(prediction_logits, list):
            # Convert 2D list to 2D tensor
            prediction_logits = torch.tensor(prediction_logits, dtype=torch.float)
        else:
            raise ValueError(
                f"Invalid format for {logits_column}. Expected tensor or 2D list"
            )

    # Convert target_tokens to tensor if needed
    if not isinstance(target_tokens, torch.Tensor):
        if isinstance(target_tokens, list):
            target_tokens = torch.tensor(target_tokens, dtype=torch.long)
        else:
            raise ValueError(
                f"Invalid format for {target_tokens_column}. Expected tensor or list"
            )

    # Validate tensor dimensions
    if prediction_logits.dim() != 2:
        raise ValueError(
            f"Expected 2D logits tensor (seq_len, vocab_size), got {prediction_logits.dim()}D"
        )

    if target_tokens.dim() != 1:
        raise ValueError(
            f"Expected 1D target tokens tensor, got {target_tokens.dim()}D"
        )

    # Check sequence length match
    if prediction_logits.shape[0] != target_tokens.shape[0]:
        raise ValueError(
            f"Sequence length mismatch: logits {prediction_logits.shape[0]} vs tokens {target_tokens.shape[0]}"
        )

    # Extract generation parameters
    temperature = generation_kwargs.get("temperature", 1.0)
    top_k = generation_kwargs.get("top_k", None)
    top_p = generation_kwargs.get("top_p", None)

    # Apply sampling parameters to logits
    modified_logits = apply_sampling_params(
        prediction_logits, temperature=temperature, top_k=top_k, top_p=top_p
    )

    # Convert to log probabilities
    log_probs = F.log_softmax(modified_logits, dim=-1)

    # Get the log probabilities for the actual target tokens
    token_indices = torch.arange(len(target_tokens))
    token_log_probs = log_probs[token_indices, target_tokens]

    # Sum the log probabilities and convert to probability
    total_log_prob = torch.sum(token_log_probs).item()
    model_prob = math.exp(total_log_prob)
    return model_prob


def _compute_n_probabilities_dict(
    model_prob: float, n_values: List[int]
) -> Dict[int, float]:
    """Compute n-based probabilities for all n values using the formula p = 1 - (1 - model_prob)**n.

    Args:
        model_prob (float): The model probability.
        n_values (List[int]): List of n values for calculations.

    Returns:
        Dict[int, float]: Dictionary mapping n to its corresponding probability.
    """
    return {n: 1.0 - ((1.0 - model_prob) ** n) for n in n_values}


def _check_above_probability_threshold(model_prob: float, threshold: float) -> bool:
    return model_prob > threshold


class ProbabilisticMemorizationAnalysisFromLogitsNode(BaseAnalysisNode):
    def __init__(
        self, analysis_input: ProbabilisticMemorizationAnalysisFromLogitsInput
    ) -> None:
        self.generation_kwargs: Dict[str, Any] = analysis_input.generation_kwargs
        self.prob_threshold: float = analysis_input.prob_threshold
        self.n_values: List[int] = analysis_input.n_values
        self.generation_df: pd.DataFrame = analysis_input.generation_df
        self.logits_column: str = analysis_input.logits_column
        self.target_tokens_column: str = analysis_input.target_tokens_column

        super().__init__(analysis_input=analysis_input)

    def run_analysis(self) -> ProbabilisticMemorizationAnalysisFromLogitsNodeOutput:
        """Run the probabilistic memorization analysis from logits."""
        analysis_input: ProbabilisticMemorizationAnalysisFromLogitsInput = cast(
            ProbabilisticMemorizationAnalysisFromLogitsInput, self.analysis_input
        )
        generation_df = analysis_input.generation_df.copy()

        # Compute model probabilities from logits using generation parameters
        model_probability = generation_df.progress_apply(
            lambda row: _compute_model_probability_from_logits(
                row,
                self.logits_column,
                self.target_tokens_column,
                **self.generation_kwargs,
            ),
            axis=1,
        )
        generation_df["model_probability"] = model_probability

        # Check if above threshold
        above_probability_threshold = model_probability.progress_apply(
            lambda prob: _check_above_probability_threshold(prob, self.prob_threshold)
        )
        generation_df["above_probability_threshold"] = above_probability_threshold

        # Compute n-based probabilities if n_values is provided
        n_probabilities = None
        if self.n_values:
            n_probabilities = model_probability.progress_apply(
                lambda prob: _compute_n_probabilities_dict(prob, self.n_values)
            )
            generation_df["n_probabilities"] = n_probabilities

        return ProbabilisticMemorizationAnalysisFromLogitsNodeOutput(
            num_samples=len(generation_df),
            model_probability=model_probability,
            above_probability_threshold=above_probability_threshold,
            n_probabilities=n_probabilities,
            augmented_output_dataset=generation_df,
        )
