# pyre-strict

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import logging

from typing import Any, Dict, Literal

import pandas as pd

from privacy_guard.analysis.extraction.text_inclusion_analysis_input import (
    LCSBoundConfig,
    TextInclusionAnalysisInput,
)
from privacy_guard.attacks.base_attack import BaseAttack

from privacy_guard.attacks.extraction.utils.data_utils import (
    load_data,
    load_model_and_tokenizer,
    save_results,
)

from privacy_guard.attacks.extraction.utils.model_inference import process_dataframe


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


TaskType = Literal["pretrain", "instruct"]
FormatType = Literal["jsonl", "csv", "json"]


class GenerationAttack(BaseAttack):
    """
    This attack, given a prompt set and a path to a LLM, executes generations and
    prepares "TextInclusionAnalysisInput" for analysis.


    Options:
        --input_file: Path to the input file
        --output_file: Optional to the output file. If none, outputs are not saved to file
            and instead are saved only in the TextInclusionAnalysisInput.
        --input_format: Format of the input file (jsonl, csv, json)
        --output_format: Format of the output file (jsonl, csv, json)
        --model_path: Path to the model
        --device: Device to use (cuda, cpu)
        --input_column: Name of the input column in the input file
        --output_column: Name of the output column in the output dataframe
        --batch_size: Batch size for processing
        --max_new_tokens: Maximum number of new tokens to generate

    """

    def __init__(
        self,
        task: TaskType,
        input_file: str,
        output_file: str | None,
        model_path: str,
        input_format: FormatType = "jsonl",
        output_format: FormatType | None = "jsonl",
        device: str = "cuda",
        input_column: str = "prompt",
        output_column: str = "prediction",
        batch_size: int = 4,
        max_new_tokens: int = 512,
        **generation_kwargs: Dict[str, Any],
    ) -> None:
        if output_file is None and output_format is not None:
            logger.warning(
                'Generation attack argument "output_format" is unused when output file is not specified'
            )

        self.task = task
        self.input_file = input_file
        self.output_file = output_file
        self.input_format = input_format
        self.output_format = output_format
        self.model_path = model_path
        self.device = device
        self.input_column = input_column
        self.output_column = output_column
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.generation_kwargs: Dict[str, Any] = generation_kwargs

        # Load data
        logger.info(f"Loading data from {input_file}")
        self.input_df: pd.DataFrame = load_data(input_file, format=input_format)
        logger.info(f"Loaded {len(self.input_df)} rows")

        # Load model and tokenizer
        logger.info(f"Loading model: {model_path}")
        self.model, self.tokenizer = load_model_and_tokenizer(  # pyre-ignore
            model_path, device=device
        )
        logger.info("Model loaded successfully. Generation attack is ready to run")

    def run_attack(self) -> TextInclusionAnalysisInput:
        # Process data
        logger.info(f"Executing generation attack: {self.task}")
        processed_df = process_dataframe(
            df=self.input_df,
            input_column=self.input_column,
            output_column=self.output_column,
            model=self.model,
            tokenizer=self.tokenizer,
            task=self.task,
            batch_size=self.batch_size,
            max_new_tokens=self.max_new_tokens,
            **self.generation_kwargs,
        )

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

        # TODO: Alternatively, call "TextInclusionAttack::run_attack()" directly
        return TextInclusionAnalysisInput(
            generation_df=processed_df,
            disable_similarity=True,
            prompt_key=self.input_column,
            target_key="target",
            generation_key=self.output_column,
            lcs_bound_config=LCSBoundConfig(lcs_len_target=150, fp_len_target=50),
        )
        logger.info("Task completed successfully")
