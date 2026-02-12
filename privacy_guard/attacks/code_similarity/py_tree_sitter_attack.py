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

import logging
from types import ModuleType
from typing import Any

import pandas as pd
import tree_sitter_cpp  # @manual=fbsource//third-party/pypi/tree-sitter-cpp:tree-sitter-cpp
import tree_sitter_python  # @manual=fbsource//third-party/pypi/tree-sitter-python:tree-sitter-python
from privacy_guard.analysis.code_similarity.code_similarity_analysis_input import (
    CodeSimilarityAnalysisInput,
)
from privacy_guard.attacks.base_attack import BaseAttack
from tree_sitter import (  # @manual=fbsource//third-party/pypi/tree-sitter:tree-sitter
    Language,
    Parser,
)
from zss import Node as ZSSNode


logger: logging.Logger = logging.getLogger(__name__)

# Maps user-facing language strings to tree-sitter language modules.
_LANGUAGE_REGISTRY: dict[str, ModuleType] = {
    "python": tree_sitter_python,
    "py": tree_sitter_python,
    "c++": tree_sitter_cpp,
    "cpp": tree_sitter_cpp,
}


def _get_parser(language: str) -> Parser:
    """Create a tree-sitter Parser for the given language.

    Args:
        language: a key in _LANGUAGE_REGISTRY (e.g. "python", "cpp")

    Returns:
        A configured tree-sitter Parser instance.

    Raises:
        ValueError: if the language is not supported.
    """
    lang_key = language.lower()
    ts_module = _LANGUAGE_REGISTRY.get(lang_key)
    if ts_module is None:
        raise ValueError(
            f"Unsupported language '{language}'. "
            f"Supported: {sorted(_LANGUAGE_REGISTRY.keys())}"
        )

    ts_language = Language(ts_module.language())  # type: ignore[attr-defined]
    parser = Parser(ts_language)
    return parser


class PyTreeSitterAttack(BaseAttack):
    """Parse target and generated code into ASTs using tree-sitter.

    Expects a DataFrame with ``target_code_string`` and
    ``model_generated_code_string`` columns.  Produces a
    :class:`CodeSimilarityAnalysisInput` with additional AST columns
    ready for downstream similarity analysis.

    Args:
        data: DataFrame with code string columns.
        default_language: default language for parsing (e.g. "python", "cpp").
            Rows may override this via a ``language`` column.
    """

    REQUIRED_COLUMNS: list[str] = [
        "target_code_string",
        "model_generated_code_string",
    ]

    def __init__(
        self,
        data: pd.DataFrame,
        default_language: str = "python",
    ) -> None:
        missing = set(self.REQUIRED_COLUMNS) - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self._data: pd.DataFrame = data.copy()
        self._default_language: str = default_language

    # ------------------------------------------------------------------
    # Public static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ts_node_to_zss_node(ts_node: Any, filter_errors: bool = False) -> ZSSNode:
        """Recursively convert a tree-sitter Node into a zss Node.

        Each zss node is labelled with the tree-sitter node's ``type``
        string (e.g. ``"function_definition"``, ``"identifier"``).

        Args:
            ts_node: tree-sitter Node to convert.
            filter_errors: when True, skip children that are ERROR or
                MISSING nodes (tree-sitter error-recovery artefacts).
        """
        zss_node = ZSSNode(ts_node.type)
        for child in ts_node.children:
            if filter_errors and (child.is_error or child.is_missing):
                continue
            zss_node.addkid(
                PyTreeSitterAttack._ts_node_to_zss_node(child, filter_errors)
            )
        return zss_node

    @staticmethod
    def parse_code(code: str, language: str = "python") -> tuple[ZSSNode, str]:
        """Parse a single code snippet and return a zss Node tree.

        Tree-sitter always produces a parse tree, even for malformed
        code.  When syntax errors are present the parser inserts ERROR
        and MISSING nodes.  This method filters those nodes out and
        returns the valid portion of the AST so that downstream
        similarity analysis can still operate on partially-correct code.

        Args:
            code: source code string.
            language: language identifier (see ``_LANGUAGE_REGISTRY``).

        Returns:
            Tuple of ``(root_node, parse_status)`` where *root_node* is
            the root :class:`zss.Node` and *parse_status* is
            ``"success"`` when the code parsed without errors or
            ``"partial"`` when error/missing nodes were filtered out.
        """
        parser = _get_parser(language)
        tree = parser.parse(code.encode("utf-8"))
        if not tree.root_node.has_error:
            return (
                PyTreeSitterAttack._ts_node_to_zss_node(tree.root_node),
                "success",
            )
        return (
            PyTreeSitterAttack._ts_node_to_zss_node(tree.root_node, filter_errors=True),
            "partial",
        )

    # ------------------------------------------------------------------
    # BaseAttack interface
    # ------------------------------------------------------------------

    def run_attack(self) -> CodeSimilarityAnalysisInput:
        """Parse every row's code strings into ASTs.

        Adds the following columns to the DataFrame:
            - ``target_ast``: zss Node (always present)
            - ``generated_ast``: zss Node (always present)
            - ``target_parse_status``: ``"success"`` or ``"partial"``
            - ``generated_parse_status``: ``"success"`` or ``"partial"``

        Returns:
            A :class:`CodeSimilarityAnalysisInput` wrapping the
            augmented DataFrame.
        """
        df = self._data

        has_language_col = "language" in df.columns

        target_asts: list[ZSSNode] = []
        generated_asts: list[ZSSNode] = []
        target_parse_statuses: list[str] = []
        generated_parse_statuses: list[str] = []

        for _idx, row in df.iterrows():
            lang = str(row["language"]) if has_language_col else self._default_language

            t_ast, t_status = self.parse_code(str(row["target_code_string"]), lang)
            target_asts.append(t_ast)
            target_parse_statuses.append(t_status)

            g_ast, g_status = self.parse_code(
                str(row["model_generated_code_string"]), lang
            )
            generated_asts.append(g_ast)
            generated_parse_statuses.append(g_status)

        df["target_ast"] = target_asts
        df["generated_ast"] = generated_asts
        df["target_parse_status"] = target_parse_statuses
        df["generated_parse_status"] = generated_parse_statuses

        return CodeSimilarityAnalysisInput(generation_df=df)
