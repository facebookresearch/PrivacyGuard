# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import argparse

import pandas as pd
from privacy_guard.analysis.reference_model_comparison_input import (
    ReferenceModelComparisonInput,
)
from privacy_guard.analysis.reference_model_comparison_node import (
    ReferenceModelComparisonNode,
)

from tqdm import tqdm

tqdm.pandas()


def dump_augmented_df(df: pd.DataFrame, jsonl_output_path: str) -> None:
    """
    Save the dataframe to a JSONL file.
    """
    jsonl_data = df.to_json(orient="records", lines=True)
    # Save JSONL data to file
    with open(jsonl_output_path, "w") as f:
        f.write(jsonl_data)


def run_comparison_analysis(
    target_df_path: str,
    reference_df_path: str,
    output_path: str,
    result_key: str = "decision_prompt",
) -> pd.DataFrame:
    """
    Runs the reference model comparison analysis on the given dataframes and returns the results.

    Args:
        target_df_path: Path to the target dataframe JSONL file
        reference_df_path: Path to the reference dataframe JSONL file
        output_path: Path to save the output JSONL file
        result_key: Column name to use for comparison (default: "decision_prompt")

    Returns:
        The augmented dataframe with comparison results
    """
    # Load dataframes
    target_df = pd.read_json(target_df_path, lines=True)
    reference_df = pd.read_json(reference_df_path, lines=True)

    # Create input
    analysis_input = ReferenceModelComparisonInput(
        target_df=target_df,
        reference_df=reference_df,
        result_key=result_key,
    )

    # Create node and run analysis
    analysis_node = ReferenceModelComparisonNode(analysis_input=analysis_input)
    results = analysis_node.compute_outputs()

    augmented_df = results["augmented_output_dataset"]

    # Save results
    dump_augmented_df(df=augmented_df, jsonl_output_path=output_path)
    print(f"Wrote comparison results to {output_path}")

    return augmented_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_path",
        help="Path to the target dataframe JSONL file",
        required=True,
    )
    parser.add_argument(
        "--reference_path",
        help="Path to the reference dataframe JSONL file",
        required=True,
    )
    parser.add_argument(
        "--output_path",
        help="Path to save the output JSONL file",
        required=True,
    )
    parser.add_argument(
        "--result_key",
        help="Column name to use for comparison (default: decision_prompt)",
        default="decision_prompt",
    )

    args = parser.parse_args()

    run_comparison_analysis(
        target_df_path=args.target_path,
        reference_df_path=args.reference_path,
        output_path=args.output_path,
        result_key=args.result_key,
    )


if __name__ == "__main__":
    main()
