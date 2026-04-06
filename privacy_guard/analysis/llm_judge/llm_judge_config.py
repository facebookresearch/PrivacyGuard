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

from dataclasses import dataclass, field
from enum import Enum


class LLMProvider(Enum):
    """Supported LLM providers for judge evaluation."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"


class AnthropicModel(Enum):
    """Available Anthropic models."""

    CLAUDE_4_OPUS = "claude-opus-4-20250514"
    CLAUDE_4_SONNET = "claude-sonnet-4-20250514"
    CLAUDE_3_7_SONNET = "claude-3-7-sonnet-20250219"
    CLAUDE_3_5_HAIKU = "claude-3-5-haiku-20241022"


class OpenAIModel(Enum):
    """Available OpenAI models."""

    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_1 = "gpt-4.1"
    GPT_4_1_MINI = "gpt-4.1-mini"
    O3 = "o3"
    O3_MINI = "o3-mini"
    O4_MINI = "o4-mini"


class GeminiModel(Enum):
    """Available Gemini models."""

    GEMINI_2_5_PRO = "gemini-2.5-pro-preview-05-06"
    GEMINI_2_5_FLASH = "gemini-2.5-flash-preview-04-17"
    GEMINI_2_0_FLASH = "gemini-2.0-flash"


# Map provider to its default model
DEFAULT_MODELS: dict[LLMProvider, str] = {
    LLMProvider.ANTHROPIC: AnthropicModel.CLAUDE_4_SONNET.value,
    LLMProvider.OPENAI: OpenAIModel.GPT_4O.value,
    LLMProvider.GEMINI: GeminiModel.GEMINI_2_5_PRO.value,
}

# Lookup table: all valid model strings grouped by provider
VALID_MODELS: dict[LLMProvider, set[str]] = {
    LLMProvider.ANTHROPIC: {m.value for m in AnthropicModel},
    LLMProvider.OPENAI: {m.value for m in OpenAIModel},
    LLMProvider.GEMINI: {m.value for m in GeminiModel},
}


@dataclass
class LLMJudgeConfig:
    """Configuration for an LLM-as-judge evaluation run.

    Attributes:
        provider: Which LLM provider to use.
        model: Model identifier string. Must belong to the chosen provider.
            When left as empty string the provider's default model is used.
        eval_prompt: The prompt template sent to the judge. Use placeholders
            ``{prompt}``, ``{generation}``, and optionally ``{reference_text}``
            and ``{criteria}`` which will be filled at evaluation time.
            Leave empty to use the built-in default prompt.
        scoring_criteria: List of criteria the judge should evaluate
            (e.g. ["accuracy", "fluency", "relevance"]).
        temperature: Sampling temperature for the judge model.
        max_tokens: Maximum tokens in the judge response.
        api_key_env_var: Name of the environment variable holding the API key
            for the chosen provider.
    """

    provider: LLMProvider = LLMProvider.ANTHROPIC
    model: str = ""
    eval_prompt: str = ""
    scoring_criteria: list[str] = field(default_factory=list)
    temperature: float = 0.0
    max_tokens: int = 1024
    api_key_env_var: str = ""

    def __post_init__(self) -> None:
        # Resolve default model when none is specified
        if not self.model:
            self.model = DEFAULT_MODELS[self.provider]

        # Validate that the model belongs to the chosen provider
        if self.model not in VALID_MODELS[self.provider]:
            valid = ", ".join(sorted(VALID_MODELS[self.provider]))
            raise ValueError(
                f"Model '{self.model}' is not valid for provider "
                f"'{self.provider.value}'. Valid models: {valid}"
            )

        # Resolve default env var name when none is specified
        if not self.api_key_env_var:
            self.api_key_env_var = {
                LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
                LLMProvider.OPENAI: "OPENAI_API_KEY",
                LLMProvider.GEMINI: "GEMINI_API_KEY",
            }[self.provider]
