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

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, cast

import pandas as pd
import requests
from privacy_guard.analysis.base_analysis_node import BaseAnalysisNode
from privacy_guard.analysis.base_analysis_output import BaseAnalysisOutput
from privacy_guard.analysis.llm_judge.llm_judge_analysis_input import (
    LLMJudgeAnalysisInput,
)
from privacy_guard.analysis.llm_judge.llm_judge_config import (
    LLMJudgeConfig,
    LLMProvider,
)
from tqdm import tqdm


logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default prompt template
# ---------------------------------------------------------------------------

_DEFAULT_EVAL_PROMPT_WITH_REFERENCE = """You are an impartial judge evaluating the quality of an AI-generated response.

## Input Prompt
{prompt}

## AI-Generated Response
{generation}

## Reference Text
{reference_text}

## Evaluation Criteria
{criteria}

Evaluate the AI-generated response against the reference text on each criterion.
For each criterion, assign an integer score from 1 (worst) to 5 (best).

Return your evaluation as JSON with this exact structure:
```json
{{
    "scores": {{"criterion_name": score, ...}},
    "overall_score": <float average of all scores>,
    "reasoning": "<brief justification for your scores>"
}}
```"""

_DEFAULT_EVAL_PROMPT_WITHOUT_REFERENCE = """You are an impartial judge evaluating the quality of an AI-generated response.

## Input Prompt
{prompt}

## AI-Generated Response
{generation}

## Evaluation Criteria
{criteria}

Evaluate the AI-generated response on each criterion.
For each criterion, assign an integer score from 1 (worst) to 5 (best).

Return your evaluation as JSON with this exact structure:
```json
{{
    "scores": {{"criterion_name": score, ...}},
    "overall_score": <float average of all scores>,
    "reasoning": "<brief justification for your scores>"
}}
```"""

_DEFAULT_CRITERIA = ["accuracy", "relevance", "fluency", "completeness"]


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class LLMJudgeAnalysisOutput(BaseAnalysisOutput):
    """Encapsulates the outputs of LLMJudgeAnalysisNode."""

    num_samples: int
    avg_overall_score: float
    per_sample_overall_scores: list[float]
    per_criteria_avg_scores: dict[str, float]
    per_sample_criteria_scores: list[dict[str, float]]
    per_sample_reasoning: list[str]
    num_failed: int
    provider: str
    model: str
    augmented_output_dataset: pd.DataFrame = field(repr=False)


# ---------------------------------------------------------------------------
# API callers — one per provider
# ---------------------------------------------------------------------------


def _get_api_key(config: LLMJudgeConfig) -> str:
    """Retrieve the API key from the environment.

    The API key is expected to be stored in the following environment variables:
        ANTHROPIC_API_KEY,
        OPENAI_API_KEY, or
        GEMINI_API_KEY
    depending on the provider.
    """
    key = os.environ.get(config.api_key_env_var, None)
    if not key:
        raise ValueError(
            f"API key not found. Set the '{config.api_key_env_var}' "
            f"environment variable."
        )
    return key


def _call_anthropic(
    prompt: str,
    config: LLMJudgeConfig,
    api_key: str,
) -> dict[str, Any]:
    """Call the Anthropic Messages API and return parsed JSON."""
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": config.model,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    text: str = data["content"][0]["text"]
    return _parse_json_response(text)


def _call_openai(
    prompt: str,
    config: LLMJudgeConfig,
    api_key: str,
) -> dict[str, Any]:
    """Call the OpenAI Chat Completions API and return parsed JSON."""
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.model,
            "max_completion_tokens": config.max_tokens,
            "temperature": config.temperature,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an evaluation judge. Always respond with valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    text: str = data["choices"][0]["message"]["content"]
    return _parse_json_response(text)


def _call_gemini(
    prompt: str,
    config: LLMJudgeConfig,
    api_key: str,
) -> dict[str, Any]:
    """Call the Gemini generateContent API and return parsed JSON."""
    url = (
        f"https://generativelanguage.googleapis.com/v1/models/"
        f"{config.model}:generateContent"
    )
    response = requests.post(
        url,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": config.temperature,
                "maxOutputTokens": config.max_tokens,
            },
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    text: str = data["candidates"][0]["content"]["parts"][0]["text"]
    return _parse_json_response(text)


_PROVIDER_CALLERS: dict[
    LLMProvider,
    Callable[[str, LLMJudgeConfig, str], dict[str, Any]],
] = {
    LLMProvider.ANTHROPIC: _call_anthropic,
    LLMProvider.OPENAI: _call_openai,
    LLMProvider.GEMINI: _call_gemini,
}


def _parse_json_response(text: str) -> dict[str, Any]:
    """Extract and parse JSON from a model response that may contain markdown fences."""
    cleaned = text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1]
    if "```" in cleaned:
        cleaned = cleaned.split("```", 1)[0]
    cleaned = cleaned.strip()
    result: dict[str, Any] = json.loads(cleaned)
    return result


def call_llm_judge(
    prompt: str,
    config: LLMJudgeConfig,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Dispatch a judge evaluation call to the configured provider.

    Retries on transient failures with exponential back-off.
    """
    api_key = _get_api_key(config)
    caller = _PROVIDER_CALLERS[config.provider]

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return caller(prompt, config, api_key)
        except (
            requests.RequestException,
            json.JSONDecodeError,
            KeyError,
            IndexError,
        ) as exc:
            last_error = exc
            wait = min(2**attempt, 8)
            logger.warning(
                f"Judge call attempt {attempt + 1}/{max_retries} failed: {exc}. "
                f"Retrying in {wait}s."
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Judge evaluation failed after {max_retries} attempts. "
        f"Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Analysis node
# ---------------------------------------------------------------------------


class LLMJudgeAnalysisNode(BaseAnalysisNode):
    """Evaluates generation quality using an LLM-as-judge.

    For each row in the input dataframe the node:
    1. Builds an evaluation prompt from the configured template.
    2. Sends the prompt to the configured LLM provider.
    3. Parses the structured JSON verdict (per-criteria scores + reasoning).
    4. Aggregates per-sample scores into summary metrics.
    """

    def __init__(self, analysis_input: LLMJudgeAnalysisInput) -> None:
        tqdm.pandas()
        self._config: LLMJudgeConfig = analysis_input.config
        super().__init__(analysis_input=analysis_input)

    def _build_eval_prompt(self, row: pd.Series) -> str:
        """Build the evaluation prompt for a single sample."""
        analysis_input = cast(LLMJudgeAnalysisInput, self.analysis_input)
        config = self._config

        prompt_text: str = str(row[analysis_input.prompt_key])
        generation_text: str = str(row[analysis_input.generation_key])
        reference_text: str = ""
        if analysis_input.has_reference and analysis_input.reference_key is not None:
            reference_text = str(row[analysis_input.reference_key])

        criteria = config.scoring_criteria or _DEFAULT_CRITERIA
        criteria_text = "\n".join(f"- {c}" for c in criteria)

        if config.eval_prompt:
            return config.eval_prompt.format(
                prompt=prompt_text,
                generation=generation_text,
                reference_text=reference_text,
                criteria=criteria_text,
            )

        if analysis_input.has_reference:
            return _DEFAULT_EVAL_PROMPT_WITH_REFERENCE.format(
                prompt=prompt_text,
                generation=generation_text,
                reference_text=reference_text,
                criteria=criteria_text,
            )

        return _DEFAULT_EVAL_PROMPT_WITHOUT_REFERENCE.format(
            prompt=prompt_text,
            generation=generation_text,
            criteria=criteria_text,
        )

    def _evaluate_single(self, row: pd.Series) -> dict[str, Any]:
        """Run judge evaluation on a single row and return parsed result."""
        eval_prompt = self._build_eval_prompt(row)
        try:
            return call_llm_judge(eval_prompt, self._config)
        except RuntimeError as exc:
            logger.error(f"Judge evaluation failed for row: {exc}")
            criteria = self._config.scoring_criteria or _DEFAULT_CRITERIA
            return {
                "scores": dict.fromkeys(criteria, 0),
                "overall_score": 0.0,
                "reasoning": f"Evaluation failed: {exc}",
            }

    def run_analysis(self) -> LLMJudgeAnalysisOutput:
        """Execute judge evaluation across all samples."""
        analysis_input = cast(LLMJudgeAnalysisInput, self.analysis_input)
        df = analysis_input.generation_df.copy()

        logger.info(
            f"Starting LLM judge evaluation: {len(df)} samples, "
            f"provider={self._config.provider.value}, "
            f"model={self._config.model}"
        )

        results: list[dict[str, Any]] = df.progress_apply(
            self._evaluate_single, axis=1
        ).tolist()

        # Extract per-sample metrics
        per_sample_overall: list[float] = []
        per_sample_criteria: list[dict[str, float]] = []
        per_sample_reasoning: list[str] = []
        num_failed = 0

        for result in results:
            score = float(result.get("overall_score", 0.0))
            per_sample_overall.append(score)
            per_sample_criteria.append(result.get("scores", {}))
            per_sample_reasoning.append(result.get("reasoning", ""))
            if score == 0.0:
                num_failed += 1

        # Aggregate per-criteria averages (excluding failed evaluations)
        criteria = self._config.scoring_criteria or _DEFAULT_CRITERIA
        criteria_totals: dict[str, float] = dict.fromkeys(criteria, 0.0)
        criteria_counts: dict[str, int] = {c: 0 for c in criteria}  # noqa: C420

        for sample_scores in per_sample_criteria:
            for c in criteria:
                val = sample_scores.get(c, 0)
                if val > 0:
                    criteria_totals[c] += float(val)
                    criteria_counts[c] += 1

        per_criteria_avg: dict[str, float] = {
            c: (
                criteria_totals[c] / criteria_counts[c]
                if criteria_counts[c] > 0
                else 0.0
            )
            for c in criteria
        }

        valid_scores = [s for s in per_sample_overall if s > 0]
        avg_overall = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        # Augment the output dataset with judge results
        df["judge_overall_score"] = per_sample_overall
        df["judge_reasoning"] = per_sample_reasoning
        for c in criteria:
            df[f"judge_{c}_score"] = [s.get(c, 0) for s in per_sample_criteria]

        logger.info(
            f"Judge evaluation complete: avg_score={avg_overall:.2f}, "
            f"failed={num_failed}/{len(df)}"
        )

        return LLMJudgeAnalysisOutput(
            num_samples=len(df),
            avg_overall_score=avg_overall,
            per_sample_overall_scores=per_sample_overall,
            per_criteria_avg_scores=per_criteria_avg,
            per_sample_criteria_scores=per_sample_criteria,
            per_sample_reasoning=per_sample_reasoning,
            num_failed=num_failed,
            provider=self._config.provider.value,
            model=self._config.model,
            augmented_output_dataset=df,
        )
