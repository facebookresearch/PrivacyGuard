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


import logging
import math
from dataclasses import dataclass, field
from typing import cast

import pandas as pd
from privacy_guard.analysis.base_analysis_node import BaseAnalysisNode
from privacy_guard.analysis.base_analysis_output import BaseAnalysisOutput
from privacy_guard.analysis.code_similarity.code_similarity_analysis_input import (
    LLMJudgeAnalysisInput,
)
from privacy_guard.analysis.code_similarity.llm_judge_prompts import JudgePromptSpec

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class LLMJudgeNodeOutput(BaseAnalysisOutput):
    """Output of :class:`LLMJudgeNode`.

    Attributes:
        num_samples: total number of sample rows.
        num_unparsed: rows whose judge response yielded no score.
        per_sample_judge_score: DataFrame with a ``judge_score`` column;
            unparsed rows are NaN.
        avg_judge_score: mean score over parsed rows, NaN if none parsed.
        avg_judge_score_by_language: per-language mean score, or ``None``
            when no ``language`` column is present.
    """

    num_samples: int
    num_unparsed: int
    per_sample_judge_score: pd.DataFrame = field(repr=False)
    avg_judge_score: float
    avg_judge_score_by_language: dict[str, float] | None


class LLMJudgeNode(BaseAnalysisNode):
    """Score code pairs from the raw responses of an LLM judge.

    This node runs no model inference.  You render the prompts with a
    :class:`JudgePromptSpec`, run them through the judge model of your
    choice, and put the raw replies in a ``judge_raw_response`` column;
    the node parses and aggregates them into the same output shape as
    the other code-similarity nodes, so results can be merged with
    :func:`privacy_guard.analysis.base_analysis_node.compute_and_merge_outputs`.
    See
    :mod:`privacy_guard.analysis.code_similarity.llm_judge_prompts`
    for the prompts and end-to-end usage.

    Unparseable responses become NaN and are excluded from the averages
    rather than scored 0.0 -- a failed parse is missing data, and scoring
    it as zero would bias similarity downwards.  Check ``num_unparsed``
    before trusting a mean.

    Args:
        analysis_input: an :class:`LLMJudgeAnalysisInput` holding the
            judge responses.
        prompt_spec: the :class:`JudgePromptSpec` used to produce those
            responses; supplies the response parser.
    """

    def __init__(
        self,
        analysis_input: LLMJudgeAnalysisInput,
        prompt_spec: JudgePromptSpec,
    ) -> None:
        super().__init__(analysis_input=analysis_input)
        self._prompt_spec = prompt_spec

    def run_analysis(self) -> LLMJudgeNodeOutput:
        analysis_input = cast(LLMJudgeAnalysisInput, self.analysis_input)
        df = analysis_input.generation_df

        parse = self._prompt_spec.parse
        scores = [
            parse(response) if isinstance(response, str) else None
            for response in df["judge_raw_response"]
        ]
        per_sample = pd.DataFrame(
            {"judge_score": [math.nan if s is None else s for s in scores]},
            index=df.index,
        )

        num_unparsed = sum(1 for score in scores if score is None)
        if num_unparsed:
            logger.warning(
                f"{num_unparsed} of {len(scores)} judge responses could not be "
                f"parsed with prompt '{self._prompt_spec.name}'; "
                "they are excluded from the averages."
            )

        avg_by_lang: dict[str, float] | None = None
        if "language" in df.columns:
            per_sample["language"] = df["language"].values
            avg_by_lang = per_sample.groupby("language")["judge_score"].mean().to_dict()

        # pandas skips NaN, and yields NaN when nothing is left to average.
        return LLMJudgeNodeOutput(
            num_samples=len(df),
            num_unparsed=num_unparsed,
            per_sample_judge_score=per_sample,
            avg_judge_score=float(per_sample["judge_score"].mean()),
            avg_judge_score_by_language=avg_by_lang,
        )
