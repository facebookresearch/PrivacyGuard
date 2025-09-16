# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
import math
from dataclasses import dataclass
from typing import cast, Dict, List, Optional

import pandas as pd
from privacy_guard.analysis.base_analysis_node import BaseAnalysisNode
from privacy_guard.analysis.base_analysis_output import BaseAnalysisOutput
from privacy_guard.analysis.extraction.probabilistic_memorization_analysis_input import (
    ProbabilisticMemorizationAnalysisInput,
)

from tqdm import tqdm

tqdm.pandas()


@dataclass
class ProbabilisticMemorizationAnalysisNodeOutput(BaseAnalysisOutput):
    num_samples: int
    model_probability: pd.Series
    above_probability_threshold: pd.Series
    n_probabilities: Optional[pd.Series]
    augmented_output_dataset: pd.DataFrame


def _compute_model_probability(row: pd.Series) -> float:
    """Compute model probability by summing up the prediction logprobs for each row. Currently only works for a single target (the most common use case).

    Args:
        row (pd.Series): A row of a DataFrame containing the "prediction_logprobs" column (each entry is a 2D list).

    Returns:
        float: The model probability for the target computed by summing up the logprobs.
    """
    prediction_logprobs = row["prediction_logprobs"]
    # Sum up the logprobs for this row
    if isinstance(prediction_logprobs, list) and len(prediction_logprobs) > 0:
        if isinstance(prediction_logprobs[0], list):
            if len(prediction_logprobs) > 1:
                raise ValueError(
                    "Invalid format for prediction_logprobs. Expected a 1D list of numbers or a nested list of a single list."
                )
            prediction_logprobs = prediction_logprobs[0]  # Unnest the list

        # Check if it's a 1D list (all elements are numbers)
        total_logprob = sum(prediction_logprobs)
        model_prob = math.exp(total_logprob)
        return model_prob
    else:
        return 0.0


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


class ProbabilisticMemorizationAnalysisNode(BaseAnalysisNode):
    def __init__(self, analysis_input: ProbabilisticMemorizationAnalysisInput) -> None:
        self.prob_threshold: float = analysis_input.prob_threshold
        self.n_values: List[int] = analysis_input.n_values
        self.generation_df: pd.DataFrame = analysis_input.generation_df

        super().__init__(analysis_input=analysis_input)

    def run_analysis(self) -> ProbabilisticMemorizationAnalysisNodeOutput:
        analysis_input: ProbabilisticMemorizationAnalysisInput = cast(
            ProbabilisticMemorizationAnalysisInput, self.analysis_input
        )
        generation_df = analysis_input.generation_df.copy()

        # Compute model probabilities
        model_probability = generation_df.progress_apply(
            _compute_model_probability, axis=1
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

        return ProbabilisticMemorizationAnalysisNodeOutput(
            num_samples=len(generation_df),
            model_probability=model_probability,
            above_probability_threshold=above_probability_threshold,
            n_probabilities=n_probabilities,
            augmented_output_dataset=generation_df,
        )
