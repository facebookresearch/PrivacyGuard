# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3
"""
Script for running various NLP tasks using the PrivacyGuard generation package.

Usage:
    buck run -c fbcode.enable_gpu_sections=true -c hpc_comms.use_nccl=2.18.3 privacy_guard/attacks/extraction/utils:generation_main --task [task_name] [options]

Tasks:
    - keyword_extraction: Extract keywords from text
    - paraphrase: Paraphrase text
    - summary: Summarize text

Options:
    --input_file: Path to the input file
    --output_file: Path to the output file
    --input_format: Format of the input file (jsonl, csv, json)
    --output_format: Format of the output file (jsonl, csv, json)
    --model_path: Path to the model
    --device: Device to use (cuda, cpu)
    --input_column: Name of the input column
    --output_column: Name of the output column
    --batch_size: Batch size for processing
    --max_new_tokens: Maximum number of new tokens to generate
"""

import argparse
import os

from privacy_guard.attacks.extraction.generation_attack import GenerationAttack
from privacy_guard.attacks.extraction.predictors import HuggingFacePredictor


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run NLP using PrivacyGuard GenerationAttack."
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--input_file",
        type=str,
        default="/tmp/input_data.jsonl",
        help="Path to the input file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/tmp/output_data.jsonl",
        help="Path to the output file. ",
    )
    parser.add_argument(
        "--input_format",
        type=str,
        default="jsonl",
        choices=["jsonl", "csv", "json"],
        help="Format of the input file",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="jsonl",
        choices=["jsonl", "csv", "json"],
        help="Format of the output file",
    )
    # Get current user for default model path
    current_user = os.environ.get("USER", "default_user")

    parser.add_argument(
        "--model_path",
        type=str,
        default=f"/home/{current_user}/models/Llama-3.2-1B",
        help="Path to the model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for processing",
    )
    parser.add_argument(
        "--input_column",
        type=str,
        default="input_col",
        help="Name of the input column in the data",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="target",
        help="Name of the target column in the data",
    )
    parser.add_argument(
        "--output_column",
        type=str,
        default="output_col",
        help="Name of the output column in the data.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate",
    )

    return parser.parse_args()


def main() -> int:
    """Main function to run the generation."""

    # Parse arguments
    args = parse_args()

    # Create a HuggingFace predictor instance
    predictor = HuggingFacePredictor(
        model_name=args.model_path,
        device=args.device,
    )

    print("Predictor created...")

    generation_attack = GenerationAttack(
        input_file=args.input_file,
        output_file=args.output_file,
        predictor=predictor,
        input_format=args.input_format,
        output_format=args.output_format,
        input_column=args.input_column,
        target_column=args.target_column,
        output_column=args.output_column,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    print("Running generation attack...")

    _ = generation_attack.run_attack()

    return 0


if __name__ == "__main__":
    main()
