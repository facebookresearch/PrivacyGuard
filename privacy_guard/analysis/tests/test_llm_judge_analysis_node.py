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


import json
import unittest
from unittest.mock import patch

import pandas as pd
from privacy_guard.analysis.llm_judge.llm_judge_analysis_input import (
    LLMJudgeAnalysisInput,
)
from privacy_guard.analysis.llm_judge.llm_judge_analysis_node import (
    _call_anthropic,
    _call_gemini,
    _call_openai,
    _parse_json_response,
    call_llm_judge,
    LLMJudgeAnalysisNode,
    LLMJudgeAnalysisOutput,
)
from privacy_guard.analysis.llm_judge.llm_judge_config import (
    LLMJudgeConfig,
    LLMProvider,
)


def _make_judge_response(
    criteria: list[str] | None = None,
) -> dict[str, object]:
    """Build a realistic judge response dict for testing."""
    criteria = criteria or ["accuracy", "relevance", "fluency", "completeness"]
    scores = {c: 4 for c in criteria}
    return {
        "scores": scores,
        "overall_score": 4.0,
        "reasoning": "Good quality response.",
    }


class TestParseJsonResponse(unittest.TestCase):
    def test_clean_json(self) -> None:
        raw = json.dumps({"scores": {"accuracy": 5}, "overall_score": 5.0})
        result = _parse_json_response(raw)
        self.assertEqual(result["overall_score"], 5.0)
        self.assertEqual(result["scores"]["accuracy"], 5)

    def test_json_with_markdown_fences(self) -> None:
        raw = '```json\n{"overall_score": 3.5}\n```'
        result = _parse_json_response(raw)
        self.assertEqual(result["overall_score"], 3.5)

    def test_json_with_surrounding_whitespace(self) -> None:
        raw = '  \n  {"overall_score": 2.0}  \n  '
        result = _parse_json_response(raw)
        self.assertEqual(result["overall_score"], 2.0)

    def test_invalid_json_raises(self) -> None:
        with self.assertRaises(json.JSONDecodeError):
            _parse_json_response("not valid json")


class TestGetApiKey(unittest.TestCase):
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key-123"})
    def test_returns_key_when_present(self) -> None:
        from privacy_guard.analysis.llm_judge.llm_judge_analysis_node import (
            _get_api_key,
        )

        config = LLMJudgeConfig(provider=LLMProvider.ANTHROPIC)
        self.assertEqual(_get_api_key(config), "test-key-123")

    @patch.dict("os.environ", {}, clear=True)
    def test_raises_when_key_missing(self) -> None:
        from privacy_guard.analysis.llm_judge.llm_judge_analysis_node import (
            _get_api_key,
        )

        config = LLMJudgeConfig(
            provider=LLMProvider.ANTHROPIC, api_key_env_var="MISSING_KEY"
        )
        with self.assertRaises(ValueError):
            _get_api_key(config)


class TestCallLLMJudge(unittest.TestCase):
    def setUp(self) -> None:
        self.config = LLMJudgeConfig(provider=LLMProvider.ANTHROPIC)
        self.expected = _make_judge_response()
        super().setUp()

    @patch(
        "privacy_guard.analysis.llm_judge.llm_judge_analysis_node._get_api_key",
        return_value="fake-key",
    )
    @patch("privacy_guard.analysis.llm_judge.llm_judge_analysis_node._PROVIDER_CALLERS")
    def test_success_on_first_attempt(
        self,
        mock_callers: unittest.mock.MagicMock,
        _mock_key: unittest.mock.MagicMock,
    ) -> None:
        mock_callers.__getitem__ = unittest.mock.MagicMock(
            return_value=unittest.mock.MagicMock(return_value=self.expected)
        )
        result = call_llm_judge("test prompt", self.config)
        self.assertEqual(result["overall_score"], 4.0)

    @patch(
        "privacy_guard.analysis.llm_judge.llm_judge_analysis_node._get_api_key",
        return_value="fake-key",
    )
    @patch("privacy_guard.analysis.llm_judge.llm_judge_analysis_node._PROVIDER_CALLERS")
    @patch("privacy_guard.analysis.llm_judge.llm_judge_analysis_node.time.sleep")
    def test_retries_on_transient_failure(
        self,
        _mock_sleep: unittest.mock.MagicMock,
        mock_callers: unittest.mock.MagicMock,
        _mock_key: unittest.mock.MagicMock,
    ) -> None:
        caller = unittest.mock.MagicMock(
            side_effect=[json.JSONDecodeError("err", "", 0), self.expected]
        )
        mock_callers.__getitem__ = unittest.mock.MagicMock(return_value=caller)
        result = call_llm_judge("test prompt", self.config, max_retries=3)
        self.assertEqual(result["overall_score"], 4.0)
        self.assertEqual(caller.call_count, 2)

    @patch(
        "privacy_guard.analysis.llm_judge.llm_judge_analysis_node._get_api_key",
        return_value="fake-key",
    )
    @patch("privacy_guard.analysis.llm_judge.llm_judge_analysis_node._PROVIDER_CALLERS")
    @patch("privacy_guard.analysis.llm_judge.llm_judge_analysis_node.time.sleep")
    def test_raises_after_all_retries_fail(
        self,
        _mock_sleep: unittest.mock.MagicMock,
        mock_callers: unittest.mock.MagicMock,
        _mock_key: unittest.mock.MagicMock,
    ) -> None:
        caller = unittest.mock.MagicMock(side_effect=json.JSONDecodeError("err", "", 0))
        mock_callers.__getitem__ = unittest.mock.MagicMock(return_value=caller)
        with self.assertRaises(RuntimeError):
            call_llm_judge("test prompt", self.config, max_retries=2)
        self.assertEqual(caller.call_count, 2)


class TestCallAnthropic(unittest.TestCase):
    """Tests for _call_anthropic with mocked requests.post returning realistic API responses."""

    def setUp(self) -> None:
        self.config = LLMJudgeConfig(provider=LLMProvider.ANTHROPIC)
        self.api_key = "fake-anthropic-key"
        super().setUp()

    @patch("privacy_guard.analysis.llm_judge.llm_judge_analysis_node.requests.post")
    def test_successful_response(self, mock_post: unittest.mock.MagicMock) -> None:
        mock_response = unittest.mock.MagicMock()
        mock_response.json.return_value = {
            "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "scores": {
                                "accuracy": 4,
                                "relevance": 5,
                                "fluency": 4,
                                "completeness": 3,
                            },
                            "overall_score": 4.0,
                            "reasoning": "Good quality response.",
                        }
                    ),
                }
            ],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 25, "output_tokens": 150},
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = _call_anthropic("test prompt", self.config, self.api_key)

        self.assertEqual(result["overall_score"], 4.0)
        self.assertEqual(result["scores"]["accuracy"], 4)
        self.assertEqual(result["scores"]["relevance"], 5)
        self.assertEqual(result["reasoning"], "Good quality response.")

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        self.assertEqual(call_kwargs.args[0], "https://api.anthropic.com/v1/messages")
        self.assertEqual(call_kwargs.kwargs["headers"]["x-api-key"], self.api_key)
        self.assertEqual(call_kwargs.kwargs["json"]["model"], self.config.model)

    @patch("privacy_guard.analysis.llm_judge.llm_judge_analysis_node.requests.post")
    def test_response_with_markdown_fences(
        self, mock_post: unittest.mock.MagicMock
    ) -> None:
        judge_json = json.dumps(
            {
                "scores": {
                    "accuracy": 5,
                    "relevance": 5,
                    "fluency": 5,
                    "completeness": 5,
                },
                "overall_score": 5.0,
                "reasoning": "Excellent response.",
            }
        )
        mock_response = unittest.mock.MagicMock()
        mock_response.json.return_value = {
            "id": "msg_02ABC",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": f"```json\n{judge_json}\n```"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 30, "output_tokens": 160},
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = _call_anthropic("test prompt", self.config, self.api_key)
        self.assertEqual(result["overall_score"], 5.0)

    @patch("privacy_guard.analysis.llm_judge.llm_judge_analysis_node.requests.post")
    def test_http_error_raises(self, mock_post: unittest.mock.MagicMock) -> None:
        import requests as req

        mock_response = unittest.mock.MagicMock()
        mock_response.raise_for_status.side_effect = req.HTTPError("401 Unauthorized")
        mock_post.return_value = mock_response

        with self.assertRaises(req.HTTPError):
            _call_anthropic("test prompt", self.config, self.api_key)


class TestCallOpenAI(unittest.TestCase):
    """Tests for _call_openai with mocked requests.post returning realistic API responses."""

    def setUp(self) -> None:
        self.config = LLMJudgeConfig(provider=LLMProvider.OPENAI)
        self.api_key = "fake-openai-key"
        super().setUp()

    @patch("privacy_guard.analysis.llm_judge.llm_judge_analysis_node.requests.post")
    def test_successful_response(self, mock_post: unittest.mock.MagicMock) -> None:
        mock_response = unittest.mock.MagicMock()
        mock_response.json.return_value = {
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "scores": {
                                    "accuracy": 4,
                                    "relevance": 5,
                                    "fluency": 4,
                                    "completeness": 3,
                                },
                                "overall_score": 4.0,
                                "reasoning": "Good quality response.",
                            }
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 25,
                "completion_tokens": 150,
                "total_tokens": 175,
            },
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = _call_openai("test prompt", self.config, self.api_key)

        self.assertEqual(result["overall_score"], 4.0)
        self.assertEqual(result["scores"]["accuracy"], 4)
        self.assertEqual(result["scores"]["relevance"], 5)
        self.assertEqual(result["reasoning"], "Good quality response.")

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        self.assertEqual(
            call_kwargs.args[0],
            "https://api.openai.com/v1/chat/completions",
        )
        self.assertEqual(
            call_kwargs.kwargs["headers"]["Authorization"],
            f"Bearer {self.api_key}",
        )
        self.assertEqual(call_kwargs.kwargs["json"]["model"], self.config.model)

    @patch("privacy_guard.analysis.llm_judge.llm_judge_analysis_node.requests.post")
    def test_response_with_markdown_fences(
        self, mock_post: unittest.mock.MagicMock
    ) -> None:
        judge_json = json.dumps(
            {
                "scores": {
                    "accuracy": 3,
                    "relevance": 4,
                    "fluency": 3,
                    "completeness": 4,
                },
                "overall_score": 3.5,
                "reasoning": "Decent response.",
            }
        )
        mock_response = unittest.mock.MagicMock()
        mock_response.json.return_value = {
            "id": "chatcmpl-def456",
            "object": "chat.completion",
            "created": 1677858300,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"```json\n{judge_json}\n```",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 30,
                "completion_tokens": 160,
                "total_tokens": 190,
            },
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = _call_openai("test prompt", self.config, self.api_key)
        self.assertEqual(result["overall_score"], 3.5)

    @patch("privacy_guard.analysis.llm_judge.llm_judge_analysis_node.requests.post")
    def test_http_error_raises(self, mock_post: unittest.mock.MagicMock) -> None:
        import requests as req

        mock_response = unittest.mock.MagicMock()
        mock_response.raise_for_status.side_effect = req.HTTPError(
            "429 Too Many Requests"
        )
        mock_post.return_value = mock_response

        with self.assertRaises(req.HTTPError):
            _call_openai("test prompt", self.config, self.api_key)


class TestCallGemini(unittest.TestCase):
    """Tests for _call_gemini with mocked requests.post returning realistic API responses."""

    def setUp(self) -> None:
        self.config = LLMJudgeConfig(provider=LLMProvider.GEMINI)
        self.api_key = "fake-gemini-key"
        super().setUp()

    @patch("privacy_guard.analysis.llm_judge.llm_judge_analysis_node.requests.post")
    def test_successful_response(self, mock_post: unittest.mock.MagicMock) -> None:
        mock_response = unittest.mock.MagicMock()
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps(
                                    {
                                        "scores": {
                                            "accuracy": 4,
                                            "relevance": 5,
                                            "fluency": 4,
                                            "completeness": 3,
                                        },
                                        "overall_score": 4.0,
                                        "reasoning": "Good quality response.",
                                    }
                                )
                            }
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 25,
                "candidatesTokenCount": 150,
                "totalTokenCount": 175,
            },
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = _call_gemini("test prompt", self.config, self.api_key)

        self.assertEqual(result["overall_score"], 4.0)
        self.assertEqual(result["scores"]["accuracy"], 4)
        self.assertEqual(result["scores"]["relevance"], 5)
        self.assertEqual(result["reasoning"], "Good quality response.")

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        self.assertIn(self.config.model, call_kwargs.args[0])
        self.assertIn("generateContent", call_kwargs.args[0])
        self.assertEqual(call_kwargs.kwargs["headers"]["x-goog-api-key"], self.api_key)

    @patch("privacy_guard.analysis.llm_judge.llm_judge_analysis_node.requests.post")
    def test_response_with_markdown_fences(
        self, mock_post: unittest.mock.MagicMock
    ) -> None:
        judge_json = json.dumps(
            {
                "scores": {
                    "accuracy": 2,
                    "relevance": 3,
                    "fluency": 2,
                    "completeness": 2,
                },
                "overall_score": 2.25,
                "reasoning": "Poor response.",
            }
        )
        mock_response = unittest.mock.MagicMock()
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": f"```json\n{judge_json}\n```"}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 30,
                "candidatesTokenCount": 160,
                "totalTokenCount": 190,
            },
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = _call_gemini("test prompt", self.config, self.api_key)
        self.assertEqual(result["overall_score"], 2.25)

    @patch("privacy_guard.analysis.llm_judge.llm_judge_analysis_node.requests.post")
    def test_http_error_raises(self, mock_post: unittest.mock.MagicMock) -> None:
        import requests as req

        mock_response = unittest.mock.MagicMock()
        mock_response.raise_for_status.side_effect = req.HTTPError(
            "500 Internal Server Error"
        )
        mock_post.return_value = mock_response

        with self.assertRaises(req.HTTPError):
            _call_gemini("test prompt", self.config, self.api_key)


class TestLLMJudgeAnalysisNode(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame(
            {
                "prompt": ["What is AI?", "Explain ML"],
                "generation": ["AI is...", "ML is..."],
                "reference_text": [
                    "Artificial intelligence is...",
                    "Machine learning is...",
                ],
            }
        )
        self.config = LLMJudgeConfig(
            provider=LLMProvider.ANTHROPIC,
            scoring_criteria=["accuracy", "fluency"],
        )
        self.analysis_input = LLMJudgeAnalysisInput(
            generation_df=self.df, config=self.config
        )
        self.node = LLMJudgeAnalysisNode(analysis_input=self.analysis_input)
        super().setUp()

    def test_build_eval_prompt_with_reference(self) -> None:
        row = self.df.iloc[0]
        prompt_text = self.node._build_eval_prompt(row)
        self.assertIn("What is AI?", prompt_text)
        self.assertIn("AI is...", prompt_text)
        self.assertIn("Artificial intelligence is...", prompt_text)
        self.assertIn("accuracy", prompt_text)
        self.assertIn("fluency", prompt_text)

    def test_build_eval_prompt_without_reference(self) -> None:
        df_no_ref = self.df[["prompt", "generation"]].copy()
        config = LLMJudgeConfig(
            provider=LLMProvider.ANTHROPIC,
            scoring_criteria=["accuracy"],
        )
        analysis_input = LLMJudgeAnalysisInput(
            generation_df=df_no_ref, config=config, reference_key=None
        )
        node = LLMJudgeAnalysisNode(analysis_input=analysis_input)
        prompt_text = node._build_eval_prompt(df_no_ref.iloc[0])
        self.assertIn("What is AI?", prompt_text)
        self.assertIn("AI is...", prompt_text)
        self.assertNotIn("Reference Text", prompt_text)

    def test_build_eval_prompt_custom_template(self) -> None:
        config = LLMJudgeConfig(
            provider=LLMProvider.ANTHROPIC,
            eval_prompt="P: {prompt} G: {generation} R: {reference_text} C: {criteria}",
            scoring_criteria=["clarity"],
        )
        analysis_input = LLMJudgeAnalysisInput(generation_df=self.df, config=config)
        node = LLMJudgeAnalysisNode(analysis_input=analysis_input)
        prompt_text = node._build_eval_prompt(self.df.iloc[0])
        self.assertIn("P: What is AI?", prompt_text)
        self.assertIn("G: AI is...", prompt_text)
        self.assertIn("R: Artificial intelligence is...", prompt_text)
        self.assertIn("- clarity", prompt_text)

    @patch("privacy_guard.analysis.llm_judge.llm_judge_analysis_node.call_llm_judge")
    def test_evaluate_single_success(self, mock_judge: unittest.mock.MagicMock) -> None:
        expected = _make_judge_response(["accuracy", "fluency"])
        mock_judge.return_value = expected
        result = self.node._evaluate_single(self.df.iloc[0])
        self.assertEqual(result["overall_score"], 4.0)
        self.assertIn("accuracy", result["scores"])

    @patch(
        "privacy_guard.analysis.llm_judge.llm_judge_analysis_node.call_llm_judge",
        side_effect=RuntimeError("API down"),
    )
    def test_evaluate_single_failure_returns_fallback(
        self, _mock_judge: unittest.mock.MagicMock
    ) -> None:
        result = self.node._evaluate_single(self.df.iloc[0])
        self.assertEqual(result["overall_score"], 0.0)
        self.assertIn("failed", result["reasoning"].lower())

    @patch("privacy_guard.analysis.llm_judge.llm_judge_analysis_node.call_llm_judge")
    def test_run_analysis_output_structure(
        self, mock_judge: unittest.mock.MagicMock
    ) -> None:
        mock_judge.return_value = _make_judge_response(["accuracy", "fluency"])
        output = self.node.run_analysis()
        self.assertIsInstance(output, LLMJudgeAnalysisOutput)
        self.assertEqual(output.num_samples, 2)
        self.assertEqual(output.num_failed, 0)
        self.assertEqual(output.provider, "anthropic")
        self.assertEqual(len(output.per_sample_overall_scores), 2)
        self.assertEqual(len(output.per_sample_criteria_scores), 2)
        self.assertEqual(len(output.per_sample_reasoning), 2)
        self.assertIn("accuracy", output.per_criteria_avg_scores)
        self.assertIn("fluency", output.per_criteria_avg_scores)

    @patch("privacy_guard.analysis.llm_judge.llm_judge_analysis_node.call_llm_judge")
    def test_run_analysis_avg_overall_score(
        self, mock_judge: unittest.mock.MagicMock
    ) -> None:
        mock_judge.return_value = _make_judge_response(["accuracy", "fluency"])
        output = self.node.run_analysis()
        self.assertAlmostEqual(output.avg_overall_score, 4.0)

    @patch("privacy_guard.analysis.llm_judge.llm_judge_analysis_node.call_llm_judge")
    def test_run_analysis_augmented_dataset(
        self, mock_judge: unittest.mock.MagicMock
    ) -> None:
        mock_judge.return_value = _make_judge_response(["accuracy", "fluency"])
        output = self.node.run_analysis()
        aug_df = output.augmented_output_dataset
        self.assertIn("judge_overall_score", aug_df.columns)
        self.assertIn("judge_reasoning", aug_df.columns)
        self.assertIn("judge_accuracy_score", aug_df.columns)
        self.assertIn("judge_fluency_score", aug_df.columns)
        self.assertEqual(len(aug_df), 2)

    @patch("privacy_guard.analysis.llm_judge.llm_judge_analysis_node.call_llm_judge")
    def test_run_analysis_with_failures(
        self, mock_judge: unittest.mock.MagicMock
    ) -> None:
        success = _make_judge_response(["accuracy", "fluency"])
        mock_judge.side_effect = [success, RuntimeError("fail")]
        output = self.node.run_analysis()
        self.assertEqual(output.num_failed, 1)
        self.assertEqual(output.per_sample_overall_scores[0], 4.0)
        self.assertEqual(output.per_sample_overall_scores[1], 0.0)

    @patch("privacy_guard.analysis.llm_judge.llm_judge_analysis_node.call_llm_judge")
    def test_compute_outputs_returns_dict(
        self, mock_judge: unittest.mock.MagicMock
    ) -> None:
        mock_judge.return_value = _make_judge_response(["accuracy", "fluency"])
        outputs = self.node.compute_outputs()
        self.assertIsInstance(outputs, dict)
        self.assertIn("avg_overall_score", outputs)
        self.assertIn("num_samples", outputs)
        self.assertIn("num_failed", outputs)

    @patch("privacy_guard.analysis.llm_judge.llm_judge_analysis_node.call_llm_judge")
    def test_run_analysis_per_criteria_averages(
        self, mock_judge: unittest.mock.MagicMock
    ) -> None:
        response_1: dict[str, object] = {
            "scores": {"accuracy": 5, "fluency": 3},
            "overall_score": 4.0,
            "reasoning": "Good.",
        }
        response_2: dict[str, object] = {
            "scores": {"accuracy": 3, "fluency": 5},
            "overall_score": 4.0,
            "reasoning": "Good.",
        }
        mock_judge.side_effect = [response_1, response_2]
        output = self.node.run_analysis()
        self.assertAlmostEqual(output.per_criteria_avg_scores["accuracy"], 4.0)
        self.assertAlmostEqual(output.per_criteria_avg_scores["fluency"], 4.0)
