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

import argparse
import logging
from typing import List, Optional

import pandas as pd
from privacy_guard.analysis.extraction.probabilistic_memorization_analysis_input import (
    ProbabilisticMemorizationAnalysisInput,
)
from privacy_guard.analysis.extraction.probabilistic_memorization_analysis_node import (
    ProbabilisticMemorizationAnalysisNode,
)
from tqdm import tqdm

tqdm.pandas()


def dump_augmented_df(df: pd.DataFrame, jsonl_output_path: str) -> None:
    jsonl_data = df.to_json(orient="records", lines=True)
    with open(jsonl_output_path, "w") as f:
        f.write(jsonl_data)


def run_probabilistic_memorization_analysis(
    generation_df_path: str,
    prob_threshold: float,
    output_path: str,
    n_values: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Runs the probabilistic memorization analysis on the given dataframe and returns the results.

    Args:
        generation_df_path: Path to the generation dataframe JSONL file
        prob_threshold: Probability threshold (p) for determining if a sample is memorized
        output_path: Path to save the output JSONL file
        n_values: Optional list of n values for computing corresponding probabilities,
            where n is the number of attempts in which we compute the probability of outputting the target

    Returns:
        The augmented dataframe with analysis results
    """
    generation_df = pd.read_json(generation_df_path, lines=True)

    analysis_input = ProbabilisticMemorizationAnalysisInput(
        generation_df=generation_df,
        prob_threshold=prob_threshold,
        n_values=n_values,
    )

    analysis_node = ProbabilisticMemorizationAnalysisNode(analysis_input=analysis_input)
    results = analysis_node.compute_outputs()

    augmented_df = results["augmented_output_dataset"]

    logging.info(f"Number of samples: {results['num_samples']}")
    logging.info(
        f"Samples above threshold: {results['above_probability_threshold'].sum()}"
    )
    logging.info(
        f"Percentage above threshold: {results['above_probability_threshold'].mean() * 100:.2f}%"
    )

    if n_values:
        logging.info(f"N-values analyzed: {n_values}")

    dump_augmented_df(df=augmented_df, jsonl_output_path=output_path)
    logging.info(f"Wrote analysis results to {output_path}")

    return augmented_df


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Run probabilistic memorization analysis on generation data with prediction logprobs"
    )
    parser.add_argument(
        "--generation_path",
        help="Path to the generation dataframe JSONL file containing prediction_logprobs",
        required=True,
    )
    parser.add_argument(
        "--prob_threshold",
        help="Threshold for comparing model probabilities",
        default=0.01,
        type=float,
        required=True,
    )
    parser.add_argument(
        "--output_path",
        help="Path to save the output JSONL file",
        required=True,
    )
    parser.add_argument(
        "--n_values",
        help="Comma-separated list of n values for computing corresponding probabilities (e.g., '10,100,1000')",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    # Parse n_values if provided
    n_values = None
    if args.n_values:
        n_values = [int(n.strip()) for n in args.n_values.split(",")]

    run_probabilistic_memorization_analysis(
        generation_df_path=args.generation_path,
        prob_threshold=args.prob_threshold,
        output_path=args.output_path,
        n_values=n_values,
    )


if __name__ == "__main__":
    main()
