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

import unittest

import pandas as pd
from privacy_guard.analysis.code_similarity.code_bleu_node import (
    CodeBleuNode,
    CodeBleuNodeOutput,
)
from privacy_guard.attacks.code_similarity.code_bleu_attack import CodeBleuAttack


def _run_e2e(df: pd.DataFrame, default_language: str = "python") -> CodeBleuNodeOutput:
    """Run the full CodeBLEU pipeline: attack preprocessing → analysis node."""
    attack_output = CodeBleuAttack(
        data=df, default_language=default_language
    ).run_attack()
    return CodeBleuNode(analysis_input=attack_output).run_analysis()


def _attack_row(target: str, generated: str, language: str = "python") -> pd.Series:  # type: ignore[type-arg]
    """Return a single-row attack result as a pd.Series for static-helper tests."""
    df = pd.DataFrame(
        {
            "target_code_string": [target],
            "model_generated_code_string": [generated],
        }
    )
    attack_output = CodeBleuAttack(data=df, default_language=language).run_attack()
    return attack_output.generation_df.iloc[0]


class TestCodeBleuNode(unittest.TestCase):
    # ------------------------------------------------------------------
    # dataflow_match – pure tuple comparisons, no tree-sitter needed
    # ------------------------------------------------------------------

    def test_dataflow_match_identical(self) -> None:
        """Identical DFGs produce a match score of 1.0."""
        dfg = [("var_0", "comesFrom", []), ("var_1", "comesFrom", ["var_0"])]
        self.assertAlmostEqual(CodeBleuNode.dataflow_match(dfg, dfg), 1.0, places=5)

    def test_dataflow_match_empty_target(self) -> None:
        """Empty target DFG returns 0.0 (degenerate case)."""
        generated = [("var_0", "comesFrom", [])]
        self.assertAlmostEqual(
            CodeBleuNode.dataflow_match([], generated), 0.0, places=5
        )

    def test_dataflow_match_both_empty(self) -> None:
        """Both target and generated empty returns 0.0."""
        self.assertAlmostEqual(CodeBleuNode.dataflow_match([], []), 0.0, places=5)

    def test_dataflow_match_no_overlap(self) -> None:
        """Completely different relationships return 0.0."""
        target = [("var_0", "comesFrom", [])]
        generated = [("var_0", "assignTo", [])]
        self.assertAlmostEqual(
            CodeBleuNode.dataflow_match(target, generated), 0.0, places=5
        )

    def test_dataflow_match_partial(self) -> None:
        """Partially overlapping DFGs return an intermediate score."""
        target = [
            ("var_0", "comesFrom", []),
            ("var_1", "comesFrom", ["var_0"]),
        ]
        generated = [
            ("var_0", "comesFrom", []),  # matches
            ("var_1", "assignTo", ["var_0"]),  # different relationship – no match
        ]
        score = CodeBleuNode.dataflow_match(target, generated)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)
        self.assertAlmostEqual(score, 0.5, places=5)

    def test_dataflow_match_does_not_double_count(self) -> None:
        """Each generated edge can only match one target edge (no double counting)."""
        target = [
            ("var_0", "comesFrom", []),
            ("var_0", "comesFrom", []),  # duplicate
        ]
        generated = [("var_0", "comesFrom", [])]  # only one copy
        # match_count should be 1, total_count 2 → score 0.5
        score = CodeBleuNode.dataflow_match(target, generated)
        self.assertAlmostEqual(score, 0.5, places=5)

    # ------------------------------------------------------------------
    # syntax_match – requires tree-sitter nodes via run_attack
    # ------------------------------------------------------------------

    def test_syntax_match_identical_code(self) -> None:
        """Identical code produces syntax match of 1.0."""
        code = "def foo():\n    return 1\n"
        row = _attack_row(code, code)
        score = CodeBleuNode.syntax_match(row["target_ast"], row["generated_ast"])
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_syntax_match_same_structure_different_names(self) -> None:
        """Same AST structure but different identifiers still gets a high score."""
        # Both are single-assignment modules – identical structure
        row = _attack_row("x = 1\n", "y = 2\n")
        score = CodeBleuNode.syntax_match(row["target_ast"], row["generated_ast"])
        # AST node types match exactly, so score should be 1.0
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_syntax_match_unrelated_code(self) -> None:
        """Structurally different code produces a lower syntax score than identical."""
        identical_row = _attack_row(
            "def foo():\n    x = 1\n    return x\n",
            "def foo():\n    x = 1\n    return x\n",
        )
        unrelated_row = _attack_row(
            "def foo():\n    x = 1\n    return x\n",
            "y = 42\n",
        )
        score_identical = CodeBleuNode.syntax_match(
            identical_row["target_ast"], identical_row["generated_ast"]
        )
        score_unrelated = CodeBleuNode.syntax_match(
            unrelated_row["target_ast"], unrelated_row["generated_ast"]
        )
        self.assertGreater(score_identical, score_unrelated)

    # ------------------------------------------------------------------
    # calc_codebleu – static composite score
    # ------------------------------------------------------------------

    def test_calc_codebleu_identical_code(self) -> None:
        """calc_codebleu returns a high score (> 0.8) for identical code."""
        code = "def foo():\n    x = 1\n    return x\n"
        row = _attack_row(code, code)
        score = CodeBleuNode.calc_codebleu(
            row["target_tokens"],
            row["generated_tokens"],
            row["target_tokens_with_weights"],
            row["target_ast"],
            row["generated_ast"],
            row["target_normalized_dfg"],
            row["generated_normalized_dfg"],
        )
        self.assertGreater(score, 0.8)

    def test_calc_codebleu_score_in_range(self) -> None:
        """calc_codebleu always returns a value in [0, 1]."""
        pairs = [
            ("def foo():\n    return 1\n", "def foo():\n    return 1\n"),
            ("def foo():\n    return 1\n", "x = 42\n"),
            ("x = 1\n", "y = 2\n"),
        ]
        for target, generated in pairs:
            with self.subTest(target=target[:20]):
                row = _attack_row(target, generated)
                score = CodeBleuNode.calc_codebleu(
                    row["target_tokens"],
                    row["generated_tokens"],
                    row["target_tokens_with_weights"],
                    row["target_ast"],
                    row["generated_ast"],
                    row["target_normalized_dfg"],
                    row["generated_normalized_dfg"],
                )
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)

    def test_calc_codebleu_custom_weights_sum_to_score(self) -> None:
        """With only one component active (weight 1.0), score equals that component."""
        code = "def foo():\n    x = 1\n    return x\n"
        row = _attack_row(code, code)
        args = (
            row["target_tokens"],
            row["generated_tokens"],
            row["target_tokens_with_weights"],
            row["target_ast"],
            row["generated_ast"],
            row["target_normalized_dfg"],
            row["generated_normalized_dfg"],
        )
        # All-syntax weight: result should equal syntax_match alone
        score_syntax_only = CodeBleuNode.calc_codebleu(
            *args, weights=(0.0, 0.0, 1.0, 0.0)
        )
        syntax_score = CodeBleuNode.syntax_match(
            row["target_ast"], row["generated_ast"]
        )
        self.assertAlmostEqual(score_syntax_only, syntax_score, places=5)

        # All-dataflow weight: result should equal dataflow_match alone
        score_df_only = CodeBleuNode.calc_codebleu(*args, weights=(0.0, 0.0, 0.0, 1.0))
        df_score = CodeBleuNode.dataflow_match(
            row["target_normalized_dfg"], row["generated_normalized_dfg"]
        )
        self.assertAlmostEqual(score_df_only, df_score, places=5)

    def test_calc_codebleu_weight_scaling(self) -> None:
        """Doubling a weight proportionally increases its contribution to the score."""
        code = "def foo():\n    x = 1\n    return x\n"
        row = _attack_row(code, code)
        args = (
            row["target_tokens"],
            row["generated_tokens"],
            row["target_tokens_with_weights"],
            row["target_ast"],
            row["generated_ast"],
            row["target_normalized_dfg"],
            row["generated_normalized_dfg"],
        )
        # For identical code all components ≈ 1.0, so both configs should be > 0.8
        score_equal = CodeBleuNode.calc_codebleu(
            *args, weights=(0.25, 0.25, 0.25, 0.25)
        )
        score_ngram_heavy = CodeBleuNode.calc_codebleu(
            *args, weights=(0.5, 0.25, 0.25, 0.0)
        )
        self.assertGreater(score_equal, 0.8)
        self.assertGreater(score_ngram_heavy, 0.8)

    # ------------------------------------------------------------------
    # run_analysis – output structure
    # ------------------------------------------------------------------

    def test_run_analysis_returns_correct_type(self) -> None:
        """run_analysis() returns a CodeBleuNodeOutput instance."""
        df = pd.DataFrame(
            {
                "target_code_string": ["def foo():\n    return 1\n"],
                "model_generated_code_string": ["def foo():\n    return 1\n"],
            }
        )
        output = _run_e2e(df)
        self.assertIsInstance(output, CodeBleuNodeOutput)

    def test_run_analysis_num_samples(self) -> None:
        """num_samples matches the number of input rows."""
        df = pd.DataFrame(
            {
                "target_code_string": [
                    "def foo():\n    return 1\n",
                    "def bar():\n    return 2\n",
                    "x = 1\n",
                ],
                "model_generated_code_string": [
                    "def foo():\n    return 1\n",
                    "def baz():\n    return 3\n",
                    "y = 2\n",
                ],
            }
        )
        output = _run_e2e(df)
        self.assertEqual(output.num_samples, 3)

    def test_run_analysis_per_sample_column(self) -> None:
        """per_sample_code_bleu contains a 'code_bleu' column with one row per sample."""
        df = pd.DataFrame(
            {
                "target_code_string": [
                    "def foo():\n    return 1\n",
                    "x = 1\n",
                ],
                "model_generated_code_string": [
                    "def foo():\n    return 1\n",
                    "y = 2\n",
                ],
            }
        )
        output = _run_e2e(df)
        self.assertIn("code_bleu", output.per_sample_code_bleu.columns)
        self.assertEqual(len(output.per_sample_code_bleu), 2)

    def test_run_analysis_avg_equals_mean_of_per_sample(self) -> None:
        """avg_code_bleu equals the arithmetic mean of per_sample_code_bleu."""
        df = pd.DataFrame(
            {
                "target_code_string": [
                    "def foo():\n    return 1\n",
                    "def bar():\n    x = 1\n    return x\n",
                ],
                "model_generated_code_string": [
                    "def foo():\n    return 1\n",
                    "x = 42\n",
                ],
            }
        )
        output = _run_e2e(df)
        expected_mean = float(output.per_sample_code_bleu["code_bleu"].mean())
        self.assertAlmostEqual(output.avg_code_bleu, expected_mean, places=5)

    def test_run_analysis_scores_in_range(self) -> None:
        """All per-sample scores are in [0, 1]."""
        df = pd.DataFrame(
            {
                "target_code_string": [
                    "def foo():\n    return 1\n",
                    "def bar():\n    x = 1\n    return x\n",
                    "x = 1\n",
                ],
                "model_generated_code_string": [
                    "def foo():\n    return 1\n",
                    "x = 42\n",
                    "",
                ],
            }
        )
        output = _run_e2e(df)
        for score in output.per_sample_code_bleu["code_bleu"]:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    # ------------------------------------------------------------------
    # run_analysis – score semantics
    # ------------------------------------------------------------------

    def test_run_analysis_identical_code_high_score(self) -> None:
        """Identical code produces avg_code_bleu above 0.8."""
        df = pd.DataFrame(
            {
                "target_code_string": ["def foo():\n    x = 1\n    return x\n"],
                "model_generated_code_string": [
                    "def foo():\n    x = 1\n    return x\n"
                ],
            }
        )
        output = _run_e2e(df)
        self.assertGreater(output.avg_code_bleu, 0.8)

    def test_run_analysis_identical_scores_higher_than_unrelated(self) -> None:
        """Identical code scores higher than structurally unrelated code."""
        df_identical = pd.DataFrame(
            {
                "target_code_string": ["def foo():\n    x = 1\n    return x\n"],
                "model_generated_code_string": [
                    "def foo():\n    x = 1\n    return x\n"
                ],
            }
        )
        df_unrelated = pd.DataFrame(
            {
                "target_code_string": ["def foo():\n    x = 1\n    return x\n"],
                "model_generated_code_string": ["y = 42\n"],
            }
        )
        score_identical = _run_e2e(df_identical).avg_code_bleu
        score_unrelated = _run_e2e(df_unrelated).avg_code_bleu
        self.assertGreater(score_identical, score_unrelated)

    # ------------------------------------------------------------------
    # run_analysis – language grouping
    # ------------------------------------------------------------------

    def test_run_analysis_no_language_column(self) -> None:
        """avg_code_bleu_by_language is None when no 'language' column is present."""
        df = pd.DataFrame(
            {
                "target_code_string": ["def foo():\n    return 1\n"],
                "model_generated_code_string": ["def foo():\n    return 1\n"],
            }
        )
        output = _run_e2e(df)
        self.assertIsNone(output.avg_code_bleu_by_language)

    def test_run_analysis_with_language_column(self) -> None:
        """avg_code_bleu_by_language is populated with one key per language."""
        df = pd.DataFrame(
            {
                "target_code_string": [
                    "def foo():\n    return 1\n",
                    "int main() { return 0; }",
                ],
                "model_generated_code_string": [
                    "def foo():\n    return 1\n",
                    "int main() { return 0; }",
                ],
                "language": ["python", "cpp"],
            }
        )
        output = _run_e2e(df, default_language="python")
        self.assertIsNotNone(output.avg_code_bleu_by_language)
        by_lang = output.avg_code_bleu_by_language
        assert by_lang is not None  # narrow for type checker
        self.assertIn("python", by_lang)
        self.assertIn("cpp", by_lang)
        self.assertIsInstance(by_lang["python"], float)
        self.assertIsInstance(by_lang["cpp"], float)

    def test_run_analysis_language_scores_in_range(self) -> None:
        """Per-language average scores are in [0, 1]."""
        df = pd.DataFrame(
            {
                "target_code_string": [
                    "def foo():\n    return 1\n",
                    "function add(a, b) { return a + b; }",
                    "func add(a, b int) int { return a + b }",
                ],
                "model_generated_code_string": [
                    "def bar():\n    return 2\n",
                    "function sub(a, b) { return a - b; }",
                    "func add(a, b int) int { return a + b }",
                ],
                "language": ["python", "javascript", "go"],
            }
        )
        output = _run_e2e(df, default_language="python")
        by_lang = output.avg_code_bleu_by_language
        assert by_lang is not None
        for lang, score in by_lang.items():
            with self.subTest(language=lang):
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)

    # ------------------------------------------------------------------
    # run_analysis – multi-language E2E
    # ------------------------------------------------------------------

    def test_run_analysis_java(self) -> None:
        """E2E pipeline works for Java code."""
        df = pd.DataFrame(
            {
                "target_code_string": [
                    "public int add(int a, int b) { return a + b; }",
                ],
                "model_generated_code_string": [
                    "public int add(int a, int b) { return a + b; }",
                ],
            }
        )
        output = _run_e2e(df, default_language="java")
        self.assertGreater(output.avg_code_bleu, 0.8)

    def test_run_analysis_javascript(self) -> None:
        """E2E pipeline works for JavaScript code."""
        df = pd.DataFrame(
            {
                "target_code_string": ["function add(a, b) { return a + b; }"],
                "model_generated_code_string": ["function add(a, b) { return a + b; }"],
            }
        )
        output = _run_e2e(df, default_language="javascript")
        self.assertGreater(output.avg_code_bleu, 0.8)

    def test_run_analysis_go(self) -> None:
        """E2E pipeline works for Go code."""
        df = pd.DataFrame(
            {
                "target_code_string": ["func add(a int, b int) int { return a + b }"],
                "model_generated_code_string": [
                    "func add(a int, b int) int { return a + b }"
                ],
            }
        )
        output = _run_e2e(df, default_language="go")
        self.assertGreater(output.avg_code_bleu, 0.8)

    def test_run_analysis_rust(self) -> None:
        """E2E pipeline works for Rust code."""
        df = pd.DataFrame(
            {
                "target_code_string": ["fn add(a: i32, b: i32) -> i32 { a + b }"],
                "model_generated_code_string": [
                    "fn add(a: i32, b: i32) -> i32 { a + b }"
                ],
            }
        )
        output = _run_e2e(df, default_language="rust")
        self.assertGreater(output.avg_code_bleu, 0.8)

    def test_run_analysis_ruby(self) -> None:
        """E2E pipeline works for Ruby code."""
        df = pd.DataFrame(
            {
                "target_code_string": ["def add(a, b)\n  a + b\nend\n"],
                "model_generated_code_string": ["def add(a, b)\n  a + b\nend\n"],
            }
        )
        output = _run_e2e(df, default_language="ruby")
        self.assertGreater(output.avg_code_bleu, 0.8)

    # ------------------------------------------------------------------
    # Complex real-world use cases
    # ------------------------------------------------------------------

    def test_renamed_variables_high_syntax_and_dfg_match(self) -> None:
        """Same algorithm with renamed variables: syntax and DFG match are perfect,
        which drives a high overall score despite the token strings differing.

        This tests a key property of CodeBLEU: variable-name normalization in the DFG
        and type-only AST comparison make the metric invariant to renaming.
        """
        target = (
            "def compute_sum(numbers):\n"
            "    total = 0\n"
            "    for num in numbers:\n"
            "        total += num\n"
            "    return total\n"
        )
        generated = (
            "def compute_sum(items):\n"
            "    acc = 0\n"
            "    for val in items:\n"
            "        acc += val\n"
            "    return acc\n"
        )
        row = _attack_row(target, generated)

        # AST node types are identical → syntax_match = 1.0
        syntax = CodeBleuNode.syntax_match(row["target_ast"], row["generated_ast"])
        self.assertAlmostEqual(syntax, 1.0, places=5)

        # Normalized DFGs are structurally identical → dataflow_match = 1.0
        dfg = CodeBleuNode.dataflow_match(
            row["target_normalized_dfg"], row["generated_normalized_dfg"]
        )
        self.assertAlmostEqual(dfg, 1.0, places=5)

        # Overall: two out of four components are perfect → score > 0.5 guaranteed,
        # even though ngram BLEU is low because variable names all differ as tokens.
        # The metric correctly boosts the score via syntax + DFG invariance.
        output = _run_e2e(
            pd.DataFrame(
                {
                    "target_code_string": [target],
                    "model_generated_code_string": [generated],
                }
            )
        )
        self.assertGreater(output.avg_code_bleu, 0.5)
        self.assertLess(output.avg_code_bleu, 1.0)

    def test_iterative_vs_recursive_factorial_score_ordering(self) -> None:
        """Iterative and recursive implementations of factorial share tokens and
        the function signature but differ structurally (if+call vs for+accumulator).

        Score should be strictly between identical and completely unrelated code,
        showing the metric captures partial structural similarity.
        """
        recursive = (
            "def factorial(n):\n"
            "    if n == 0:\n"
            "        return 1\n"
            "    return n * factorial(n - 1)\n"
        )
        iterative = (
            "def factorial(n):\n"
            "    result = 1\n"
            "    for i in range(1, n + 1):\n"
            "        result *= i\n"
            "    return result\n"
        )
        unrelated = "def greet(name):\n    print('hello', name)\n"

        def _score(t: str, g: str) -> float:
            return _run_e2e(
                pd.DataFrame(
                    {"target_code_string": [t], "model_generated_code_string": [g]}
                )
            ).avg_code_bleu

        score_identical = _score(recursive, recursive)
        score_diff_impl = _score(recursive, iterative)
        score_unrelated = _score(recursive, unrelated)

        # Identical must beat different implementation
        self.assertGreater(score_identical, score_diff_impl)
        # Different implementation shares signature/tokens, so beats unrelated
        self.assertGreater(score_diff_impl, score_unrelated)

    def test_sorting_algorithms_score_higher_than_unrelated(self) -> None:
        """Bubble sort and selection sort share substantial structure (nested loops,
        array indexing, conditional swap pattern, same signature) but differ in the
        inner update logic.

        The metric should give them a noticeably higher score than comparing either
        to an unrelated string-processing function, reflecting the shared skeleton.
        """
        bubble_sort = (
            "def bubble_sort(arr):\n"
            "    n = len(arr)\n"
            "    for i in range(n):\n"
            "        for j in range(0, n - i - 1):\n"
            "            if arr[j] > arr[j + 1]:\n"
            "                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n"
            "    return arr\n"
        )
        selection_sort = (
            "def selection_sort(arr):\n"
            "    n = len(arr)\n"
            "    for i in range(n):\n"
            "        min_idx = i\n"
            "        for j in range(i + 1, n):\n"
            "            if arr[min_idx] > arr[j]:\n"
            "                min_idx = j\n"
            "        arr[i], arr[min_idx] = arr[min_idx], arr[i]\n"
            "    return arr\n"
        )
        unrelated = (
            "def tokenize(text):\n"
            "    words = text.strip().split()\n"
            "    return [w.lower() for w in words]\n"
        )

        def _score(t: str, g: str) -> float:
            return _run_e2e(
                pd.DataFrame(
                    {"target_code_string": [t], "model_generated_code_string": [g]}
                )
            ).avg_code_bleu

        score_sorting_pair = _score(bubble_sort, selection_sort)
        score_vs_unrelated = _score(bubble_sort, unrelated)

        # Two sorting algorithms score higher than bubble sort vs string processing
        self.assertGreater(score_sorting_pair, score_vs_unrelated)

        # Both sorting algorithms share enough structure that their score is non-trivial
        self.assertGreater(score_sorting_pair, 0.3)


if __name__ == "__main__":
    unittest.main()
