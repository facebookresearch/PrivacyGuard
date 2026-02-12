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
from privacy_guard.analysis.code_similarity.code_similarity_analysis_input import (
    CodeSimilarityAnalysisInput,
)
from privacy_guard.attacks.code_similarity.py_tree_sitter_attack import (
    PyTreeSitterAttack,
)
from zss import Node as ZSSNode, simple_distance


def _trees_identical(a: ZSSNode, b: ZSSNode) -> bool:
    """Return True when two zss trees have zero edit distance."""
    return simple_distance(a, b) == 0


class TestPyTreeSitterAttack(unittest.TestCase):
    def test_run_attack_and_languages(self) -> None:
        """Test run_attack for Python and C++ code, and malformed code."""
        with self.subTest("python"):
            df = pd.DataFrame(
                {
                    "target_code_string": ["def foo():\n    return 1\n"],
                    "model_generated_code_string": ["def bar():\n    return 2\n"],
                }
            )
            attack = PyTreeSitterAttack(data=df, default_language="python")
            result = attack.run_attack()

            self.assertIsInstance(result, CodeSimilarityAnalysisInput)
            gen_df = result.generation_df
            self.assertEqual(gen_df["target_parse_status"].iloc[0], "success")
            self.assertEqual(gen_df["generated_parse_status"].iloc[0], "success")
            self.assertIsNotNone(gen_df["target_ast"].iloc[0])
            self.assertIsNotNone(gen_df["generated_ast"].iloc[0])

        with self.subTest("cpp"):
            df = pd.DataFrame(
                {
                    "target_code_string": ["int main() { return 0; }"],
                    "model_generated_code_string": [
                        "int add(int a, int b) { return a + b; }"
                    ],
                }
            )
            attack = PyTreeSitterAttack(data=df, default_language="cpp")
            result = attack.run_attack()

            gen_df = result.generation_df
            self.assertEqual(gen_df["target_parse_status"].iloc[0], "success")
            self.assertEqual(gen_df["generated_parse_status"].iloc[0], "success")

        with self.subTest("malformed_code_partial_parse"):
            df = pd.DataFrame(
                {
                    "target_code_string": ["def foo(:\n    return"],
                    "model_generated_code_string": ["def bar():\n    return 1\n"],
                }
            )
            attack = PyTreeSitterAttack(data=df, default_language="python")
            result = attack.run_attack()

            gen_df = result.generation_df
            self.assertEqual(gen_df["target_parse_status"].iloc[0], "partial")
            # A partial AST is still returned (not None)
            self.assertIsNotNone(gen_df["target_ast"].iloc[0])
            # The well-formed generated code should parse cleanly
            self.assertEqual(gen_df["generated_parse_status"].iloc[0], "success")

        with self.subTest("malformed_similar_errors_around_valid_code"):
            # Generated code is identical to target but wrapped in syntax
            # errors before, after, and in between statements.  After
            # filtering error nodes the partial AST should closely
            # resemble the target AST.
            target = "def foo():\n    x = 1\n    return x\n"
            generated = (
                "))))\n"  # errors before
                "def foo():\n"
                "    x = 1\n"
                "    @@@@\n"  # errors in between
                "    return x\n"
                "(((\n"  # errors after
            )
            df = pd.DataFrame(
                {
                    "target_code_string": [target],
                    "model_generated_code_string": [generated],
                }
            )
            attack = PyTreeSitterAttack(data=df, default_language="python")
            result = attack.run_attack()

            gen_df = result.generation_df
            self.assertEqual(gen_df["target_parse_status"].iloc[0], "success")
            self.assertEqual(gen_df["generated_parse_status"].iloc[0], "partial")
            # Both ASTs should be present and structurally close
            t_ast: ZSSNode = gen_df["target_ast"].iloc[0]
            g_ast: ZSSNode = gen_df["generated_ast"].iloc[0]
            dist = simple_distance(t_ast, g_ast)
            # The filtered partial AST should be very close to the
            # target (small edit distance relative to tree size).
            self.assertLessEqual(dist, 5)

        with self.subTest("ast_equivalence_different_identifiers"):
            # Two code snippets that differ only in identifier names and
            # string literals should produce identical ASTs because
            # tree-sitter node types capture grammar categories, not
            # the actual text content.
            code_a = 'def compute():\n    result = "hello"\n    return result\n'
            code_b = 'def process():\n    output = "world"\n    return output\n'
            ast_a, status_a = PyTreeSitterAttack.parse_code(code_a)
            ast_b, status_b = PyTreeSitterAttack.parse_code(code_b)
            self.assertEqual(status_a, "success")
            self.assertEqual(status_b, "success")
            self.assertTrue(
                _trees_identical(ast_a, ast_b),
                "ASTs should be identical when code differs only in "
                "identifier names and string literals",
            )

    def test_missing_column_raises(self) -> None:
        """Missing required columns should raise ValueError."""
        with self.subTest("missing_target"):
            df = pd.DataFrame({"model_generated_code_string": ["x"]})
            with self.assertRaises(ValueError):
                PyTreeSitterAttack(data=df)

        with self.subTest("missing_generated"):
            df = pd.DataFrame({"target_code_string": ["x"]})
            with self.assertRaises(ValueError):
                PyTreeSitterAttack(data=df)

    def test_parse_code_static_method(self) -> None:
        """parse_code() works as a standalone static method."""
        with self.subTest("python_parse"):
            node, status = PyTreeSitterAttack.parse_code("x = 1\n", language="python")
            self.assertEqual(status, "success")
            self.assertEqual(node.label, "module")

        with self.subTest("cpp_parse"):
            node, status = PyTreeSitterAttack.parse_code("int x = 1;", language="cpp")
            self.assertEqual(status, "success")
            self.assertEqual(node.label, "translation_unit")

        with self.subTest("malformed_returns_partial"):
            node, status = PyTreeSitterAttack.parse_code("def foo(:", language="python")
            self.assertEqual(status, "partial")
            # A partial AST is still returned
            self.assertEqual(node.label, "module")
