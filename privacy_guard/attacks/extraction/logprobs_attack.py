# pyre-strict

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import logging

from typing import Any, Dict, List, Literal

import pandas as pd

from privacy_guard.analysis.extraction.probabilistic_memorization_analysis_input import (
    ProbabilisticMemorizationAnalysisInput,
)
from privacy_guard.attacks.base_attack import BaseAttack

from privacy_guard.attacks.extraction.utils.data_utils import (
    load_data,
    load_model_and_tokenizer,
    save_results,
)


def setup_logger() -> logging.Logger:
    """Set up the logger for the script."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Remove any existing handlers to avoid duplicates
    logger.handlers.clear()

    logger.propagate = False

    # Create console handler and set level
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


logger: logging.Logger = setup_logger()


FormatType = Literal["jsonl", "csv", "json"]


class LogProbsAttack(BaseAttack):
    """
    This attack, given prompt-target pairs and a path to a LLM, extracts log probabilities and
    prepares "ProbabilisticMemorizationAnalysisInput" for analysis.

    Options:
        --input_file: Path to the input file containing prompt-target pairs
        --output_file: Optional path to the output file. If none, outputs are not saved to file
            and instead are saved only in the ProbabilisticMemorizationAnalysisInput.
        --input_format: Format of the input file (jsonl, csv, json)
        --output_format: Format of the output file (jsonl, csv, json)
        --model_path: Path to the model
        --device: Device to use (cuda, cpu)
        --prompt_column: Name of the prompt column in the input file
        --target_column: Name of the target column in the input file
        --output_column: Name of the output column in the output dataframe
        --batch_size: Batch size for processing
        --prob_threshold: Threshold for comparing model probabilities
        --n_values: optional list of n values for computing corresponding probabilities of model outputting the target in n attempts. Refer to https://arxiv.org/abs/2410.19482 for details.
    """

    def __init__(
        self,
        input_file: str,
        output_file: str | None,
        model_path: str,
        input_format: FormatType = "jsonl",
        output_format: FormatType | None = "jsonl",
        device: str = "cuda",
        prompt_column: str = "prompt",
        target_column: str = "target",
        output_column: str = "prediction_logprobs",
        batch_size: int = 4,
        prob_threshold: float = 0.5,
        n_values: List[int] | None = None,
        **model_kwargs: Dict[str, Any],
    ) -> None:
        if output_file is None and output_format is not None:
            logger.warning(
                'LogProbs attack argument "output_format" is unused when output file is not specified'
            )

        self.input_file = input_file
        self.output_file = output_file
        self.input_format = input_format
        self.output_format = output_format
        self.model_path = model_path
        self.device = device
        self.prompt_column = prompt_column
        self.target_column = target_column
        self.output_column = output_column
        self.batch_size = batch_size
        self.prob_threshold = prob_threshold
        self.n_values: List[int] = n_values or []
        self.model_kwargs: Dict[str, Any] = model_kwargs

        # Load data
        logger.info(f"Loading data from {input_file}")
        self.input_df: pd.DataFrame = load_data(input_file, format=input_format)
        logger.info(f"Loaded {len(self.input_df)} rows")

        # Validate required columns
        required_columns = {self.prompt_column, self.target_column}
        missing_columns = required_columns - set(self.input_df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Load model and tokenizer
        logger.info(f"Loading model: {model_path}")
        self.model, self.tokenizer = load_model_and_tokenizer(  # pyre-ignore
            model_path, device=device
        )
        logger.info("Model loaded successfully. LogProbs attack is ready to run")

    def run_attack(self) -> ProbabilisticMemorizationAnalysisInput:
        # Process data - TODO: Implement log probabilities extraction logic
        logger.info("Executing log probabilities attack")

        # TODO: Replace with actual logprobs extraction using self.model.get_logprobs()
        processed_df = self.input_df.copy()
        processed_df[self.output_column] = [
            [] for _ in range(len(processed_df))
        ]  # Placeholder

        logger.info("Processing complete")

        # Save results
        if self.output_file is not None:
            output_path: str = self.output_file
            output_format: str = str(self.output_format)
            logger.info(f"Saving results to {output_path}")
            save_results(df=processed_df, output_path=output_path, format=output_format)
            logger.info("Results saved successfully")
        else:
            logger.info("No output file specified, not saving results to disk")

        return ProbabilisticMemorizationAnalysisInput(
            generation_df=processed_df,
            prob_threshold=self.prob_threshold,
            n_values=self.n_values,
        )
