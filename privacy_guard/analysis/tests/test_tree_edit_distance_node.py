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
from privacy_guard.analysis.code_similarity.tree_edit_distance_node import (
    TreeEditDistanceNode,
    TreeEditDistanceNodeOutput,
)
from privacy_guard.attacks.code_similarity.py_tree_sitter_attack import (
    PyTreeSitterAttack,
)


def _run_e2e(
    df: pd.DataFrame,
    default_language: str = "python",
) -> TreeEditDistanceNodeOutput:
    """Helper: run attack then analysis end-to-end."""
    attack = PyTreeSitterAttack(data=df, default_language=default_language)
    analysis_input = attack.run_attack()
    node = TreeEditDistanceNode(analysis_input=analysis_input)
    return node.run_analysis()


class TestTreeEditDistanceNode(unittest.TestCase):
    def test_similarity_values(self) -> None:
        """Identical code should yield ~1.0; different code should be low."""
        with self.subTest("identical_python"):
            code = "def foo():\n    return 1\n"
            df = pd.DataFrame(
                {
                    "target_code_string": [code],
                    "model_generated_code_string": [code],
                }
            )
            output = _run_e2e(df)
            self.assertIsInstance(output, TreeEditDistanceNodeOutput)
            self.assertAlmostEqual(output.avg_similarity, 1.0, places=5)
            self.assertEqual(output.num_both_parsed, 1)

        with self.subTest("different_python"):
            df = pd.DataFrame(
                {
                    "target_code_string": ["def foo():\n    return 1\n"],
                    "model_generated_code_string": [
                        "class Bar:\n    def __init__(self):\n"
                        "        self.x = 1\n"
                        "    def method(self, a, b):\n"
                        "        return a + b\n"
                    ],
                }
            )
            output = _run_e2e(df)
            self.assertLess(output.avg_similarity, 0.5)

        with self.subTest("cpp_similarity"):
            df = pd.DataFrame(
                {
                    "target_code_string": ["int add(int a, int b) { return a + b; }"],
                    "model_generated_code_string": [
                        "int sum(int x, int y) { return x + y; }"
                    ],
                }
            )
            output = _run_e2e(df, default_language="cpp")
            self.assertGreater(output.avg_similarity, 0.7)

        with self.subTest("partial_parse_high_similarity"):
            # Generated code contains the same function as the target
            # but is surrounded by syntax errors.  After error-node
            # filtering the partial AST should still yield high
            # similarity against the clean target.
            target = "def foo():\n    x = 1\n    return x\n"
            generated = "))))\ndef foo():\n    x = 1\n    @@@@\n    return x\n(((\n"
            df = pd.DataFrame(
                {
                    "target_code_string": [target],
                    "model_generated_code_string": [generated],
                }
            )
            output = _run_e2e(df)
            # Partial parse still produces a similarity score (not NaN)
            self.assertEqual(output.num_both_parsed, 1)
            self.assertGreater(output.avg_similarity, 0.5)

        with self.subTest("ast_equivalence_different_strings"):
            # Two code snippets that are syntactically equivalent but
            # differ in identifier names and string literals should
            # yield similarity ≈ 1.0 because tree-sitter AST nodes are
            # labelled by grammar category (e.g. "identifier", "string"),
            # not by the actual text content.
            target = 'def compute():\n    result = "hello"\n    return result\n'
            generated = 'def process():\n    output = "world"\n    return output\n'
            df = pd.DataFrame(
                {
                    "target_code_string": [target],
                    "model_generated_code_string": [generated],
                }
            )
            output = _run_e2e(df)
            self.assertAlmostEqual(output.avg_similarity, 1.0, places=5)

    def test_avg_similarity_by_language(self) -> None:
        """Mixed Python+C++ input produces per-language averages."""
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
        output = _run_e2e(df)
        assert output.avg_similarity_by_language is not None
        by_lang = output.avg_similarity_by_language
        self.assertIn("python", by_lang)
        self.assertIn("cpp", by_lang)
        self.assertAlmostEqual(by_lang["python"], 1.0, places=5)
        self.assertAlmostEqual(by_lang["cpp"], 1.0, places=5)

    def test_compute_similarity_static_method(self) -> None:
        """TreeEditDistanceNode.compute_similarity works standalone."""
        node1, _ = PyTreeSitterAttack.parse_code("x = 1\n", language="python")
        node2, _ = PyTreeSitterAttack.parse_code("x = 1\n", language="python")

        sim = TreeEditDistanceNode.compute_similarity(node1, node2)
        self.assertAlmostEqual(sim, 1.0, places=5)
