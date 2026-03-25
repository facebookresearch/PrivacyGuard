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

import importlib.resources
import unittest

import pandas as pd
from codebleu.dataflow_match import (  # @manual=fbsource//third-party/pypi/codebleu:codebleu
    dfg_function,
    get_data_flow,
    normalize_dataflow,
)
from privacy_guard.attacks.code_similarity.code_bleu_attack import CodeBleuAttack

# pyre-ignore[21]: tree-sitter doesn't have properly exposed type stubs
from tree_sitter import (  # @manual=fbsource//third-party/pypi/tree-sitter:tree-sitter
    Language,
    Parser,
)


# pyre-ignore[11]: Annotation `Parser` is not defined as a type
def _make_parser(language: str) -> Parser:
    tree_sitter_language = Language(
        importlib.resources.files("codebleu") / "my-languages.so", language
    )
    # pyre-ignore[16]: Module `tree_sitter` has no attribute `Parser`
    parser = Parser()
    parser.set_language(tree_sitter_language)
    return parser


def _run_attack(target: str, generated: str, language: str = "python") -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "target_code_string": [target],
            "model_generated_code_string": [generated],
        }
    )
    return CodeBleuAttack(data=df, default_language=language).run_attack().generation_df


class CodeBleuAttackTest(unittest.TestCase):
    # ------------------------------------------------------------------
    # tokenizer
    # ------------------------------------------------------------------

    def test_tokenizer(self) -> None:
        self.assertEqual(
            CodeBleuAttack.tokenizer("x = 1 + 2"),
            ["x", "=", "1", "+", "2"],
        )
        self.assertEqual(CodeBleuAttack.tokenizer(""), [])

    # ------------------------------------------------------------------
    # make_weights
    # ------------------------------------------------------------------

    def test_make_weights(self) -> None:
        tokens = ["def", "return", "foo", "x"]
        keywords = ["def", "return", "if", "for"]
        weights = CodeBleuAttack.make_weights(tokens, keywords)
        self.assertEqual(weights["def"], 1)
        self.assertEqual(weights["return"], 1)
        self.assertAlmostEqual(weights["foo"], 0.2)
        self.assertAlmostEqual(weights["x"], 0.2)
        self.assertEqual(set(weights.keys()), set(tokens))

    # ------------------------------------------------------------------
    # get_data_flow (from codebleu package)
    # ------------------------------------------------------------------

    def test_get_data_flow_none_dfg_func_returns_empty(self) -> None:
        """codebleu's get_data_flow returns [] when dfg_func is None.
        Note: run_attack() now raises ValueError before reaching this state;
        this test documents the underlying library behaviour only.
        """
        parser = _make_parser("python")
        self.assertEqual(get_data_flow("x = 1 + 2", [parser, None]), [])

    def test_get_data_flow_python_captures_dependency(self) -> None:
        """Python DFG captures that y depends on x in 'x=1; y=x+2'."""
        parser = _make_parser("python")
        code = "x = 1\ny = x + 2"
        raw_dfg = get_data_flow(code, [parser, dfg_function["python"]])
        normalized = normalize_dataflow(raw_dfg)

        # x appears first → var_0; y depends on x, so var_0 should appear as a parent
        parent_sets = [set(item[2]) for item in normalized]
        self.assertTrue(
            any("var_0" in parents for parents in parent_sets),
            f"Expected var_0 (x) as a parent in DFG, got: {normalized}",
        )

    def test_get_data_flow_java_captures_dependency(self) -> None:
        """Java DFG captures that y depends on x in an equivalent snippet."""
        parser = _make_parser("java")
        code = "class T { void f() { int x = 1; int y = x + 2; } }"
        raw_dfg = get_data_flow(code, [parser, dfg_function["java"]])
        normalized = normalize_dataflow(raw_dfg)

        # At least one variable must list another as its source
        self.assertTrue(
            any(len(item[2]) > 0 for item in normalized),
            f"Expected at least one data-flow dependency in Java DFG, got: {normalized}",
        )

    # ------------------------------------------------------------------
    # normalize_dataflow (from codebleu package)
    # ------------------------------------------------------------------

    def test_normalize_dataflow_renames_variables(self) -> None:
        # Raw DFG item format: (name, index, relationship, [parent_names], [...])
        raw_dfg = [
            ("x", 0, "comesFrom", [], []),
            ("y", 1, "comesFrom", ["x"], []),
            ("z", 2, "comesFrom", ["x", "y"], []),
        ]
        normalized = normalize_dataflow(raw_dfg)
        self.assertEqual(normalized[0], ("var_0", "comesFrom", []))
        self.assertEqual(normalized[1], ("var_1", "comesFrom", ["var_0"]))
        self.assertEqual(normalized[2], ("var_2", "comesFrom", ["var_0", "var_1"]))

    def test_normalize_dataflow_consistent_renaming(self) -> None:
        """Same structural DFG normalizes identically regardless of original variable names."""
        raw_a = [
            ("alpha", 0, "comesFrom", [], []),
            ("beta", 1, "comesFrom", ["alpha"], []),
        ]
        raw_b = [("foo", 0, "comesFrom", [], []), ("bar", 1, "comesFrom", ["foo"], [])]
        self.assertEqual(
            normalize_dataflow(raw_a),
            normalize_dataflow(raw_b),
        )

    # ------------------------------------------------------------------
    # run_attack – tokens (exact values)
    # ------------------------------------------------------------------

    def test_run_attack_exact_tokens(self) -> None:
        """Tokenization of known code produces the exact expected token list."""
        gen_df = _run_attack("x = 1", "y = 2")
        self.assertEqual(gen_df["target_tokens"].iloc[0], ["x", "=", "1"])
        self.assertEqual(gen_df["generated_tokens"].iloc[0], ["y", "=", "2"])

    def test_run_attack_tokens_with_weights_python_keywords(self) -> None:
        """Python keywords 'def'/'return' get weight 1; identifiers get 0.2."""
        gen_df = _run_attack("def foo(): return 1", "x = 1")
        tokens, weight_dict = gen_df["target_tokens_with_weights"].iloc[0]
        self.assertEqual(tokens, ["def", "foo():", "return", "1"])
        self.assertEqual(weight_dict["def"], 1)
        self.assertEqual(weight_dict["return"], 1)
        self.assertAlmostEqual(weight_dict["foo():"], 0.2)

    # ------------------------------------------------------------------
    # run_attack – AST content for Python and Java
    # ------------------------------------------------------------------

    def test_run_attack_ast_python(self) -> None:
        """Python AST root is a 'module' containing a 'function_definition'."""
        gen_df = _run_attack(
            "def add(a, b):\n    return a + b\n",
            "def add(a, b):\n    return a + b\n",
        )
        ast = gen_df["target_ast"].iloc[0]
        self.assertEqual(ast.type, "module")
        child_types = {child.type for child in ast.children}
        self.assertIn("function_definition", child_types)

    def test_run_attack_ast_java(self) -> None:
        """Java AST root is a 'program' containing a 'class_declaration'."""
        code = "class Foo { int add(int a, int b) { return a + b; } }"
        gen_df = _run_attack(code, code, language="java")
        ast = gen_df["target_ast"].iloc[0]
        self.assertEqual(ast.type, "program")
        child_types = {child.type for child in ast.children}
        self.assertIn("class_declaration", child_types)

    def test_run_attack_ast_same_code_same_structure(self) -> None:
        """Identical target and generated code produce ASTs with the same structure."""
        code = "def foo(x):\n    return x * 2\n"
        gen_df = _run_attack(code, code)
        target_ast = gen_df["target_ast"].iloc[0]
        generated_ast = gen_df["generated_ast"].iloc[0]
        self.assertEqual(target_ast.type, generated_ast.type)
        self.assertEqual(
            [c.type for c in target_ast.children],
            [c.type for c in generated_ast.children],
        )

    def test_run_attack_ast_different_code_different_structure(self) -> None:
        """Structurally different code (function def vs assignment) yields different AST child types."""
        gen_df = _run_attack(
            "def foo(x):\n    return x\n",
            "x = 1\n",
        )
        target_child_types = {c.type for c in gen_df["target_ast"].iloc[0].children}
        generated_child_types = {
            c.type for c in gen_df["generated_ast"].iloc[0].children
        }
        self.assertIn("function_definition", target_child_types)
        self.assertNotIn("function_definition", generated_child_types)

    # ------------------------------------------------------------------
    # run_attack – DFG content for Python and Java
    # ------------------------------------------------------------------

    def test_run_attack_normalized_dfg_python(self) -> None:
        """Python normalized DFG for 'x=1; y=x+2' shows var_0 as y's parent."""
        gen_df = _run_attack(
            "def foo():\n    x = 1\n    y = x + 2\n    return y\n",
            "def foo():\n    x = 1\n    y = x + 2\n    return y\n",
        )
        normalized_dfg = gen_df["target_normalized_dfg"].iloc[0]
        parent_sets = [set(item[2]) for item in normalized_dfg]
        self.assertTrue(
            any("var_0" in p for p in parent_sets),
            f"Expected var_0 as a parent in Python DFG, got: {normalized_dfg}",
        )

    def test_run_attack_normalized_dfg_java(self) -> None:
        """Java normalized DFG for equivalent snippet shows at least one data-flow edge."""
        code = "class T { void f() { int x = 1; int y = x + 2; } }"
        gen_df = _run_attack(code, code, language="java")
        normalized_dfg = gen_df["target_normalized_dfg"].iloc[0]
        self.assertTrue(
            any(len(item[2]) > 0 for item in normalized_dfg),
            f"Expected at least one DFG dependency in Java, got: {normalized_dfg}",
        )

    def test_run_attack_identical_code_same_normalized_dfg(self) -> None:
        """Identical target and generated code produce equal normalized DFGs."""
        code = "def foo():\n    x = 1\n    y = x + 2\n    return y\n"
        gen_df = _run_attack(code, code)
        self.assertEqual(
            gen_df["target_normalized_dfg"].iloc[0],
            gen_df["generated_normalized_dfg"].iloc[0],
        )

    # ------------------------------------------------------------------
    # run_attack – error handling and language column
    # ------------------------------------------------------------------

    def test_run_attack_missing_columns_raise(self) -> None:
        with self.subTest("missing_target"):
            with self.assertRaises(ValueError):
                CodeBleuAttack(pd.DataFrame({"model_generated_code_string": ["x = 1"]}))
        with self.subTest("missing_generated"):
            with self.assertRaises(ValueError):
                CodeBleuAttack(pd.DataFrame({"target_code_string": ["x = 1"]}))

    def test_run_attack_unsupported_language_raises(self) -> None:
        """run_attack raises ValueError for a language not in AVAILABLE_LANGS."""
        df = pd.DataFrame(
            {
                "target_code_string": ["x = 1"],
                "model_generated_code_string": ["y = 2"],
            }
        )
        with self.assertRaises(ValueError):
            CodeBleuAttack(data=df, default_language="cobol").run_attack()

    def test_run_attack_language_column_overrides_default(self) -> None:
        """Per-row 'language' column controls which parser is used."""
        df = pd.DataFrame(
            {
                "target_code_string": [
                    "def foo():\n    return 1\n",
                    "class Foo { int add(int a, int b) { return a + b; } }",
                ],
                "model_generated_code_string": [
                    "def foo():\n    return 1\n",
                    "class Foo { int add(int a, int b) { return a + b; } }",
                ],
                "language": ["python", "java"],
            }
        )
        gen_df = (
            CodeBleuAttack(data=df, default_language="python")
            .run_attack()
            .generation_df
        )
        self.assertEqual(gen_df["target_ast"].iloc[0].type, "module")
        self.assertEqual(gen_df["target_ast"].iloc[1].type, "program")


if __name__ == "__main__":
    unittest.main()
