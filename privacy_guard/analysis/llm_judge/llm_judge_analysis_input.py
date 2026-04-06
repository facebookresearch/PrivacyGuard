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

import logging

import pandas as pd
from privacy_guard.analysis.base_analysis_input import BaseAnalysisInput
from privacy_guard.analysis.llm_judge.llm_judge_config import LLMJudgeConfig


logger: logging.Logger = logging.getLogger(__name__)


class LLMJudgeAnalysisInput(BaseAnalysisInput):
    """Input for LLM-as-judge evaluation.

    Takes a single dataframe containing at minimum a ``prompt`` column and a
    ``generation`` column.  A ``reference_text`` column is **optional** — when
    absent the judge evaluates solely based on the configured scoring criteria.

    Args:
        generation_df: DataFrame with prompt/generation (and optionally
            reference_text) columns.
        config: ``LLMJudgeConfig`` specifying the provider, model, eval
            prompt template, and scoring criteria.
        prompt_key: Column name for the input prompt.
        generation_key: Column name for the model-generated text.
        reference_key: Column name for the ground-truth reference text.
            Set to ``None`` when no reference text is available.
    """

    REQUIRED_COLUMNS: list[str] = ["prompt", "generation"]

    def __init__(
        self,
        generation_df: pd.DataFrame,
        config: LLMJudgeConfig,
        prompt_key: str = "prompt",
        generation_key: str = "generation",
        reference_key: str | None = "reference_text",
    ) -> None:
        columns = generation_df.columns.tolist()
        assert prompt_key in columns, (
            f"Prompt key '{prompt_key}' not found in dataframe columns {columns}"
        )
        assert generation_key in columns, (
            f"Generation key '{generation_key}' not found in "
            f"dataframe columns {columns}"
        )
        if reference_key is not None and reference_key not in columns:
            logger.warning(
                f"Reference key '{reference_key}' not found in dataframe "
                f"columns {columns}. Proceeding without reference text."
            )
            reference_key = None

        self.prompt_key = prompt_key
        self.generation_key = generation_key
        self.reference_key = reference_key
        self.config = config

        super().__init__(df_train_user=generation_df, df_test_user=pd.DataFrame())

    @property
    def generation_df(self) -> pd.DataFrame:
        """Property accessor for the generation dataframe."""
        return self._df_train_user

    @property
    def has_reference(self) -> bool:
        """Whether reference text is available for evaluation."""
        return self.reference_key is not None
