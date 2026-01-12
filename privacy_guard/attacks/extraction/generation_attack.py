# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-strict

import logging
from typing import Any, Dict, Literal

import pandas as pd
from privacy_guard.analysis.extraction.text_inclusion_analysis_input import (
    LCSBoundConfig,
    TextInclusionAnalysisInput,
)
from privacy_guard.attacks.base_attack import BaseAttack
from privacy_guard.attacks.extraction.utils.data_utils import load_data, save_results

from .predictors.base_predictor import BasePredictor


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
    This attack, given prompt sets and a predictor, executes generations and
    prepares "TextInclusionAnalysisInput" for analysis.

    Options:
        --input_file: Path to the input file containing prompts
        --output_file: Optional path to the output file. If none, outputs are not saved to file
            and instead are saved only in the TextInclusionAnalysisInput.
        --input_format: Format of the input file (jsonl, csv, json)
        --output_format: Format of the output file (jsonl, csv, json)
        --predictor: Predictor instance to use for generations
        --input_column: Name of the input column in the input file
        --target_column: Name of the target column in the input file
        --output_column: Name of the output column in the output dataframe
        --batch_size: Batch size for processing
        --**generation_kwargs: Generation parameters (temperature, top_k, top_p, etc.) passed to the predictor
    """

    def __init__(
        self,
        input_file: str,
        output_file: str | None,
        predictor: BasePredictor,
        input_format: FormatType = "jsonl",
        output_format: FormatType | None = "jsonl",
        input_column: str = "prompt",
        target_column: str = "target",
        output_column: str = "prediction",
        batch_size: int = 1,
        **generation_kwargs: Any,
    ) -> None:
        if output_file is None and output_format is not None:
            logger.warning(
                'Generation attack argument "output_format" is unused when output file is not specified'
            )

        self.input_file = input_file
        self.output_file = output_file
        self.input_format = input_format
        self.output_format = output_format
        self.predictor: BasePredictor = predictor
        self.input_column = input_column
        self.target_column = target_column
        self.output_column = output_column
        self.batch_size = batch_size
        self.generation_kwargs: Dict[str, Any] = generation_kwargs

        # Load data
        logger.info(f"Loading data from {input_file}")
        self.input_df: pd.DataFrame = load_data(input_file, format=input_format)
        logger.info(f"Loaded {len(self.input_df)} rows")

        # Validate required columns
        required_columns = {self.input_column}
        missing_columns = required_columns - set(self.input_df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        logger.info("Generation attack is ready to run")

    def run_attack(self) -> TextInclusionAnalysisInput:
        # Process data
        logger.info("Executing generation attack")

        prompts = self.input_df[self.input_column].tolist()

        # Generate text using the predictor
        logger.info(f"Generating text for {len(prompts)} prompts")
        generations = self.predictor.generate(
            prompts=prompts,
            batch_size=self.batch_size,
            **self.generation_kwargs,
        )

        processed_df = self.input_df.copy()
        processed_df[self.output_column] = generations

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

        return TextInclusionAnalysisInput(
            generation_df=processed_df,
            disable_similarity=True,
            prompt_key=self.input_column,
            target_key=self.target_column,
            generation_key=self.output_column,
            lcs_bound_config=LCSBoundConfig(lcs_len_target=150, fp_len_target=50),
        )
