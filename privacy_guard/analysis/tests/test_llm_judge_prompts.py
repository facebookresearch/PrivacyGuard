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


import unittest

from privacy_guard.analysis.code_similarity.llm_judge_prompts import (
    FUNCTIONAL_EQUIVALENCE,
    get_judge_prompts,
)

REFERENCE = "def add(a, b):\n    return a + b\n"
GENERATED = "def sum_two(x, y):\n    return x + y\n"

# One well-formed response per prompt, all scoring 0.75.
RESPONSES: dict[str, str] = {
    "song_2024": "[[0.750]]",
    "nikiema_2025": (
        '{"similarity_rating": 0.75, "category": "similar", '
        '"reasoning": "Both add two numbers."}'
    ),
    "functional_equivalence": (
        '{"equivalence_score": 0.75, "reasoning": "Both add two numbers."}'
    ),
}


class TestJudgePrompts(unittest.TestCase):
    def test_render_substitutes_placeholders(self) -> None:
        """Every placeholder is filled, in both system and user messages."""
        for name, spec in get_judge_prompts().items():
            with self.subTest(prompt=name):
                system, user = spec.render(REFERENCE, GENERATED, language="C++")
                combined = system + user
                self.assertNotIn("__LANGUAGE__", combined)
                self.assertNotIn("{reference_function}", combined)
                self.assertNotIn("{generated_function}", combined)
                self.assertIn(REFERENCE, user)
                self.assertIn(GENERATED, user)
                self.assertIn("C++", combined)

    def test_render_preserves_literal_json_braces(self) -> None:
        """Rendering uses str.replace, so literal braces survive."""
        _system, user = FUNCTIONAL_EQUIVALENCE.render(REFERENCE, GENERATED)
        self.assertIn('{"equivalence_score": 0.0 to 1.0', user)

    def test_parse_well_formed_response(self) -> None:
        for name, spec in get_judge_prompts().items():
            with self.subTest(prompt=name):
                self.assertEqual(spec.parse(RESPONSES[name]), 0.75)

    def test_parse_response_echoing_the_prompt(self) -> None:
        """Predictors echo the prompt back with the generation by default.

        Song's prompt contains "[[0.777]]" and the functional-equivalence
        prompt contains a JSON-shaped example, so a parser taking the
        first match would score the instructions instead of the answer.
        """
        for name, spec in get_judge_prompts().items():
            with self.subTest(prompt=name):
                _system, user = spec.render(REFERENCE, GENERATED)
                echoed = f"{user}\n{RESPONSES[name]}"
                self.assertEqual(spec.parse(echoed), 0.75)

    def test_parse_unparseable_returns_none(self) -> None:
        for name, spec in get_judge_prompts().items():
            for response in ["", "I cannot answer that.", "0.75"]:
                with self.subTest(prompt=name, response=response):
                    self.assertIsNone(spec.parse(response))

    def test_parse_clamps_out_of_range_score(self) -> None:
        self.assertEqual(get_judge_prompts()["song_2024"].parse("[[1.5]]"), 1.0)
        self.assertEqual(
            FUNCTIONAL_EQUIVALENCE.parse('{"equivalence_score": -0.5}'), 0.0
        )
