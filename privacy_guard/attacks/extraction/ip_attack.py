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
from privacy_guard.analysis.llm_judge.llm_judge_analysis_input import (
    LLMJudgeAnalysisInput,
)
from privacy_guard.analysis.llm_judge.llm_judge_config import LLMJudgeConfig
from privacy_guard.attacks.base_attack import BaseAttack
from privacy_guard.attacks.extraction.predictors.base_predictor import BasePredictor
from privacy_guard.attacks.extraction.utils.data_utils import load_data, save_results


def setup_logger() -> logging.Logger:
    """Set up the logger for the script."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    logger.handlers.clear()
    logger.propagate = False

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


logger: logging.Logger = setup_logger()


FormatType = Literal["jsonl", "csv", "json"]


class IPAttack(BaseAttack):
    """Attack for evaluating IP content generation risk using an LLM judge.

    Given a set of prompts designed to probe a target model for IP content
    reproduction, this attack:
    1. Generates text from the target model via a ``BasePredictor``.
    2. Returns an ``LLMJudgeAnalysisInput`` that can be fed into
       ``LLMJudgeAnalysisNode`` for scoring the generations against
       IP-related criteria (e.g. verbatim reproduction, paraphrasing).

    Args:
        input_file: Path to the input file containing probing prompts.
        output_file: Optional path to save generations. When ``None``,
            results are only returned in the analysis input.
        predictor: Predictor instance used to generate text from the
            target model.
        judge_config: ``LLMJudgeConfig`` specifying the judge provider,
            model, and scoring criteria for IP evaluation.
        input_format: Format of the input file.
        output_format: Format of the output file.
        prompt_key: Column name for the input prompt.
        generation_key: Column name for the generated text (written
            into the output dataframe).
        reference_key: Column name for reference/ground-truth text.
            Set to ``None`` when no reference text is available.
        batch_size: Batch size for generation.
        **generation_kwargs: Additional generation parameters forwarded
            to the predictor (temperature, top_k, top_p, etc.).
    """

    def __init__(
        self,
        input_file: str,
        output_file: str | None,
        predictor: BasePredictor,
        judge_config: LLMJudgeConfig,
        input_format: FormatType = "jsonl",
        output_format: FormatType = "jsonl",
        prompt_key: str = "prompt",
        generation_key: str = "generation",
        reference_key: str = "reference_text",
        batch_size: int = 1,
        **generation_kwargs: Any,
    ) -> None:
        self.input_file = input_file
        self.output_file = output_file
        self.input_format = input_format
        self.output_format = output_format
        self.predictor: BasePredictor = predictor
        self.judge_config = judge_config
        self.prompt_key = prompt_key
        self.generation_key = generation_key
        self.reference_key = reference_key
        self.batch_size = batch_size
        self.generation_kwargs: Dict[str, Any] = generation_kwargs

        logger.info(f"Loading data from {input_file}")
        self.input_df: pd.DataFrame = load_data(input_file, format=input_format)
        logger.info(f"Loaded {len(self.input_df)} rows")

        required_columns = {self.prompt_key}
        missing_columns = required_columns - set(self.input_df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        logger.info("IP attack is ready to run")

    def run_attack(self) -> LLMJudgeAnalysisInput:
        """Execute the IP content probing attack.

        Generates text from the target model and returns an analysis
        input ready for LLM judge evaluation.
        """
        logger.info("Executing IP content attack")

        prompts = self.input_df[self.prompt_key].tolist()

        logger.info(f"Generating text for {len(prompts)} prompts")
        generations = self.predictor.generate(
            prompts=prompts,
            batch_size=self.batch_size,
            **self.generation_kwargs,
        )

        processed_df = self.input_df.copy()
        processed_df[self.generation_key] = generations

        logger.info("Generation complete")

        if self.output_file is not None:
            output_path: str = self.output_file
            output_format: str = str(self.output_format)
            logger.info(f"Saving results to {output_path}")
            save_results(df=processed_df, output_path=output_path, format=output_format)
            logger.info("Results saved successfully")
        else:
            logger.info("No output file specified, not saving results to disk")

        return LLMJudgeAnalysisInput(
            generation_df=processed_df,
            config=self.judge_config,
            prompt_key=self.prompt_key,
            generation_key=self.generation_key,
            reference_key=self.reference_key,
        )
