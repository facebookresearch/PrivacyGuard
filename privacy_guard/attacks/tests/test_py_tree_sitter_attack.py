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
from privacy_guard.analysis.code_similarity.tree_edit_distance_node import (
    TreeEditDistanceNode,
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

        with self.subTest("c"):
            df = pd.DataFrame(
                {
                    "target_code_string": ["int add(int a, int b) { return a + b; }"],
                    "model_generated_code_string": [
                        "int sub(int a, int b) { return a - b; }"
                    ],
                }
            )
            attack = PyTreeSitterAttack(data=df, default_language="c")
            result = attack.run_attack()
            gen_df = result.generation_df
            self.assertEqual(gen_df["target_parse_status"].iloc[0], "success")
            self.assertEqual(gen_df["generated_parse_status"].iloc[0], "success")

        with self.subTest("java"):
            df = pd.DataFrame(
                {
                    "target_code_string": [
                        "class Foo { int add(int a, int b) { return a + b; } }"
                    ],
                    "model_generated_code_string": [
                        "class Bar { int sub(int a, int b) { return a - b; } }"
                    ],
                }
            )
            attack = PyTreeSitterAttack(data=df, default_language="java")
            result = attack.run_attack()
            gen_df = result.generation_df
            self.assertEqual(gen_df["target_parse_status"].iloc[0], "success")
            self.assertEqual(gen_df["generated_parse_status"].iloc[0], "success")

        with self.subTest("rust"):
            df = pd.DataFrame(
                {
                    "target_code_string": ["fn add(a: i32, b: i32) -> i32 { a + b }"],
                    "model_generated_code_string": [
                        "fn sub(a: i32, b: i32) -> i32 { a - b }"
                    ],
                }
            )
            attack = PyTreeSitterAttack(data=df, default_language="rust")
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

        with self.subTest("c_parse"):
            node, status = PyTreeSitterAttack.parse_code(
                "int add(int a, int b) { return a + b; }", language="c"
            )
            self.assertEqual(status, "success")
            self.assertEqual(node.label, "translation_unit")

        with self.subTest("java_parse"):
            node, status = PyTreeSitterAttack.parse_code(
                "class Foo { int add(int a, int b) { return a + b; } }",
                language="java",
            )
            self.assertEqual(status, "success")
            self.assertEqual(node.label, "program")

        with self.subTest("rust_parse"):
            node, status = PyTreeSitterAttack.parse_code(
                "fn add(a: i32, b: i32) -> i32 { a + b }", language="rust"
            )
            self.assertEqual(status, "success")
            self.assertEqual(node.label, "source_file")

        with self.subTest("javascript_parse"):
            node, status = PyTreeSitterAttack.parse_code(
                "function add(a, b) { return a + b; }", language="javascript"
            )
            self.assertEqual(status, "success")
            self.assertEqual(node.label, "program")

        with self.subTest("go_parse"):
            node, status = PyTreeSitterAttack.parse_code(
                "package main\nfunc add(a int, b int) int { return a + b }",
                language="go",
            )
            self.assertEqual(status, "success")
            self.assertEqual(node.label, "source_file")

        with self.subTest("ruby_parse"):
            node, status = PyTreeSitterAttack.parse_code(
                "def add(a, b)\n  a + b\nend", language="ruby"
            )
            self.assertEqual(status, "success")
            self.assertEqual(node.label, "program")

        with self.subTest("php_parse"):
            node, status = PyTreeSitterAttack.parse_code(
                "<?php function add($a, $b) { return $a + $b; } ?>",
                language="php",
            )
            self.assertEqual(status, "success")
            self.assertEqual(node.label, "program")

        with self.subTest("c_sharp_parse"):
            node, status = PyTreeSitterAttack.parse_code(
                "class Foo { int Add(int a, int b) { return a + b; } }",
                language="c_sharp",
            )
            self.assertEqual(status, "success")
            self.assertEqual(node.label, "compilation_unit")

        with self.subTest("malformed_returns_partial"):
            node, status = PyTreeSitterAttack.parse_code("def foo(:", language="python")
            self.assertEqual(status, "partial")
            # A partial AST is still returned
            self.assertEqual(node.label, "module")


def _sim(code1: str, code2: str, lang: str) -> tuple[float, str, str]:
    ast1, s1 = PyTreeSitterAttack.parse_code(code1, language=lang)
    ast2, s2 = PyTreeSitterAttack.parse_code(code2, language=lang)
    return TreeEditDistanceNode.compute_similarity(ast1, ast2), s1, s2


class TestTreeEditSimilarityComprehensive(unittest.TestCase):
    """End-to-end parse + similarity tests across languages and difficulty levels."""

    def _check_lang(
        self, lang: str, pairs: list[tuple[str, str, str, float, float]]
    ) -> None:
        for code1, code2, desc, lo, hi in pairs:
            with self.subTest(f"{lang}_{desc}"):
                sim, s1, s2 = _sim(code1, code2, lang)
                self.assertEqual(s1, "success")
                self.assertEqual(s2, "success")
                self.assertGreaterEqual(sim, lo)
                self.assertLessEqual(sim, hi)

    def test_python_similarity_levels(self) -> None:
        self._check_lang(
            "python",
            [
                ("x = 1", "x = 1", "trivial_identical", 0.99, 1.01),
                (
                    "def add(a, b):\n    return a + b\n",
                    "def add(a, b):\n    return a + b\n",
                    "identical_func",
                    0.99,
                    1.01,
                ),
                (
                    "def add(a, b):\n    return a + b\n",
                    "def sum(x, y):\n    return x + y\n",
                    "same_struct_diff_names",
                    0.9,
                    1.01,
                ),
                (
                    "def foo(x):\n    return x * 2\n",
                    "def foo(x):\n    if x > 0:\n        return x\n    return -x\n",
                    "different_body",
                    0.2,
                    0.8,
                ),
                (
                    "def f(lst):\n    out = []\n    for x in lst:\n        out.append(x*2)\n    return out\n",
                    "def f(lst):\n    return [x*2 for x in lst]\n",
                    "loop_vs_comprehension",
                    0.1,
                    0.7,
                ),
                (
                    "class Foo:\n    def bar(self):\n        return 1\n",
                    "def bar():\n    return 1\n",
                    "class_vs_func",
                    0.1,
                    0.7,
                ),
                (
                    "def f(x):\n    if x > 0:\n        for i in range(x):\n            if i % 2 == 0:\n                print(i)\n",
                    "def f(x):\n    if x > 0:\n        for i in range(x):\n            if i % 2 == 0:\n                print(i)\n",
                    "nested_identical",
                    0.99,
                    1.01,
                ),
                (
                    "def f():\n    try:\n        x = 1/0\n    except ZeroDivisionError:\n        x = 0\n    return x\n",
                    "def f():\n    try:\n        x = 1/0\n    except ZeroDivisionError:\n        x = 0\n    return x\n",
                    "try_except_identical",
                    0.99,
                    1.01,
                ),
                (
                    '@staticmethod\ndef f(x):\n    """doc"""\n    return x\n',
                    "def f(x):\n    return x\n",
                    "decorator_docstring_vs_plain",
                    0.3,
                    0.9,
                ),
                (
                    "import os\nfor f in os.listdir('.'):\n    print(f)\n",
                    "class Config:\n    DEBUG = True\n    PORT = 8080\n",
                    "completely_different",
                    0.0,
                    0.4,
                ),
            ],
        )

    def test_c_similarity_levels(self) -> None:
        self._check_lang(
            "c",
            [
                ("int x = 1;", "int x = 1;", "trivial_identical", 0.99, 1.01),
                (
                    "int add(int a, int b) { return a + b; }",
                    "int add(int a, int b) { return a + b; }",
                    "identical_func",
                    0.99,
                    1.01,
                ),
                (
                    "int add(int a, int b) { return a + b; }",
                    "int sum(int x, int y) { return x + y; }",
                    "same_struct_diff_names",
                    0.9,
                    1.01,
                ),
                (
                    "int f(int x) { return x * 2; }",
                    "int f(int x) { if (x > 0) return x; return -x; }",
                    "different_body",
                    0.2,
                    0.8,
                ),
                (
                    "int fib(int n) { if(n<=1) return n; return fib(n-1)+fib(n-2); }",
                    "int fib(int n) { if(n<=1) return n; return fib(n-1)+fib(n-2); }",
                    "recursive_identical",
                    0.99,
                    1.01,
                ),
                (
                    "void swap(int* a, int* b) { int t=*a; *a=*b; *b=t; }",
                    "void swap(int* a, int* b) { int t=*a; *a=*b; *b=t; }",
                    "swap_identical",
                    0.99,
                    1.01,
                ),
                (
                    "int max(int a, int b) { return a > b ? a : b; }",
                    "int min(int a, int b) { return a < b ? a : b; }",
                    "max_vs_min",
                    0.8,
                    1.01,
                ),
                (
                    '#include <stdio.h>\nint main() { printf("hello"); return 0; }',
                    "typedef struct { int x; int y; } Vec2; float dot(Vec2 a, Vec2 b) { return a.x*b.x+a.y*b.y; }",
                    "completely_different",
                    0.0,
                    0.5,
                ),
            ],
        )

    def test_java_similarity_levels(self) -> None:
        self._check_lang(
            "java",
            [
                (
                    "class A { int add(int a, int b) { return a + b; } }",
                    "class A { int add(int a, int b) { return a + b; } }",
                    "identical",
                    0.99,
                    1.01,
                ),
                (
                    "class A { int add(int a, int b) { return a + b; } }",
                    "class B { int sum(int x, int y) { return x + y; } }",
                    "same_struct_diff_names",
                    0.9,
                    1.01,
                ),
                (
                    "class A { int f(int x) { return x * 2; } }",
                    "class A { int f(int x) { if (x > 0) return x; return -x; } }",
                    "different_body",
                    0.3,
                    0.85,
                ),
                (
                    "class A { int fib(int n) { if(n<=1) return n; return fib(n-1)+fib(n-2); } }",
                    "class A { int fib(int n) { if(n<=1) return n; return fib(n-1)+fib(n-2); } }",
                    "recursive_identical",
                    0.99,
                    1.01,
                ),
                (
                    "class A { int max(int a, int b) { return a > b ? a : b; } }",
                    "class A { int min(int a, int b) { return a < b ? a : b; } }",
                    "max_vs_min",
                    0.8,
                    1.01,
                ),
                (
                    'import java.util.*;\nclass Main { public static void main(String[] args) { System.out.println("hello"); } }',
                    "class Config { static final boolean DEBUG = true; static final int PORT = 8080; }",
                    "completely_different",
                    0.0,
                    0.5,
                ),
            ],
        )

    def test_rust_similarity_levels(self) -> None:
        self._check_lang(
            "rust",
            [
                (
                    "fn add(a: i32, b: i32) -> i32 { a + b }",
                    "fn add(a: i32, b: i32) -> i32 { a + b }",
                    "identical",
                    0.99,
                    1.01,
                ),
                (
                    "fn add(a: i32, b: i32) -> i32 { a + b }",
                    "fn sum(x: i32, y: i32) -> i32 { x + y }",
                    "same_struct_diff_names",
                    0.9,
                    1.01,
                ),
                (
                    "fn f(x: i32) -> i32 { x * 2 }",
                    "fn f(x: i32) -> i32 { if x > 0 { x } else { -x } }",
                    "different_body",
                    0.2,
                    0.8,
                ),
                (
                    "fn fib(n: u32) -> u32 { if n <= 1 { n } else { fib(n-1) + fib(n-2) } }",
                    "fn fib(n: u32) -> u32 { if n <= 1 { n } else { fib(n-1) + fib(n-2) } }",
                    "recursive_identical",
                    0.99,
                    1.01,
                ),
                (
                    "fn max(a: i32, b: i32) -> i32 { if a > b { a } else { b } }",
                    "fn min(a: i32, b: i32) -> i32 { if a < b { a } else { b } }",
                    "max_vs_min",
                    0.8,
                    1.01,
                ),
                (
                    'use std::io;\nfn main() { println!("hello"); }',
                    "struct Config { debug: bool, port: u16 }",
                    "completely_different",
                    0.0,
                    0.5,
                ),
            ],
        )

    def test_cpp_similarity_levels(self) -> None:
        self._check_lang(
            "cpp",
            [
                (
                    "int add(int a, int b) { return a + b; }",
                    "int add(int a, int b) { return a + b; }",
                    "identical",
                    0.99,
                    1.01,
                ),
                (
                    "int add(int a, int b) { return a + b; }",
                    "int sum(int x, int y) { return x + y; }",
                    "same_struct_diff_names",
                    0.9,
                    1.01,
                ),
                (
                    "int fib(int n) { if(n<=1) return n; return fib(n-1)+fib(n-2); }",
                    "int fib(int n) { if(n<=1) return n; return fib(n-1)+fib(n-2); }",
                    "recursive_identical",
                    0.99,
                    1.01,
                ),
                (
                    "int max(int a, int b) { return a > b ? a : b; }",
                    "int min(int a, int b) { return a < b ? a : b; }",
                    "max_vs_min",
                    0.8,
                    1.01,
                ),
                (
                    '#include <iostream>\nint main() { std::cout << "hello"; return 0; }',
                    "struct Config { bool debug = true; int port = 8080; };",
                    "completely_different",
                    0.0,
                    0.5,
                ),
            ],
        )
