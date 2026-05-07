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
import logging
from typing import Any

import pandas as pd
from codebleu.codebleu import (  # @manual=fbsource//third-party/pypi/codebleu:codebleu
    AVAILABLE_LANGS,
)
from codebleu.dataflow_match import (  # @manual=fbsource//third-party/pypi/codebleu:codebleu
    dfg_function,
    get_data_flow,
    normalize_dataflow,
)
from codebleu.parser import (  # @manual=fbsource//third-party/pypi/codebleu:codebleu
    remove_comments_and_docstrings,
)
from privacy_guard.analysis.code_similarity.code_similarity_analysis_input import (
    CodeBleuAnalysisInput,
)
from privacy_guard.attacks.base_attack import BaseAttack

# pyre-ignore[21]: tree-sitter doesn't have properly exposed type stubs
from tree_sitter import (  # @manual=fbsource//third-party/pypi/tree-sitter:tree-sitter
    Language,
    Node,
    Parser,
)

logger: logging.Logger = logging.getLogger(__name__)


class CodeBleuAttack(BaseAttack):
    """Prepare target and generated code for similarity analysis using CodeBLEU.
    CodeBLEU combines BLEU scores, syntax similarity through AST and semantic similarity using data flow (DFG).
    See: https://arxiv.org/pdf/2009.10297, or https://github.com/k4black/codebleu/tree/main.

    Expects a DataFrame with ``target_code_string`` and
    ``model_generated_code_string`` columns.  Produces a
    :class:`CodeBleuAnalysisInput` with additional AST and DFG columns
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
    def tokenizer(s: str) -> list[str]:
        return s.split()

    @staticmethod
    def make_weights(
        reference_tokens: list[str], key_word_list: list[str]
    ) -> dict[str, float]:
        return {
            token: 1 if token in key_word_list else 0.2 for token in reference_tokens
        }

    # ------------------------------------------------------------------
    # BaseAttack interface
    # ------------------------------------------------------------------

    def run_attack(self) -> CodeBleuAnalysisInput:
        """Parse every row's code strings into ASTs and extract normalized dataflows.

        Adds the following columns to the DataFrame:
            - ``target_tokens``: List[str]
            - ``generated_tokens``: List[str]
            - ``target_tokens_with_weights``: List
            - ``target_ast``: tree_sitter.Node
            - ``generated_ast``: tree_sitter.Node
            - ``target_normalized_dfg``: list of normalized dataflow items
            - ``generated_normalized_dfg``: list of normalized dataflow items

        Returns:
            A :class:`CodeBleuAnalysisInput` wrapping the
            augmented DataFrame.
        """
        df = self._data
        has_language_col = "language" in df.columns

        target_tokens: list[list[str]] = []
        generated_tokens: list[list[str]] = []
        target_tokens_with_weights: list[list[Any]] = []
        # pyre-ignore[11]: Annotation `Node` is not defined as a type
        target_asts: list[Node] = []
        generated_asts: list[Node] = []
        target_normalized_dfgs: list[Any] = []
        generated_normalized_dfgs: list[Any] = []

        # keep a cache for parser and keywords
        # pyre-ignore[11]: Annotation `Parser` is not defined as a type
        parser_cache: dict[str, Parser] = {}
        keywords_cache: dict[str, list[str]] = {}

        for _, row in df.iterrows():
            lang = str(row["language"]) if has_language_col else self._default_language

            # Get parser and DFG function for this language
            if lang not in parser_cache:
                if lang not in AVAILABLE_LANGS:
                    raise ValueError(f"Language {lang} not supported by CodeBLEU.")
                tree_sitter_language = Language(
                    # pyrefly: ignore [bad-argument-type]
                    importlib.resources.files("codebleu") / "my-languages.so",
                    lang,
                )
                # pyre-ignore[16]: Module `tree_sitter` has no attribute `Parser`.
                parser = Parser()
                parser.set_language(tree_sitter_language)
                parser_cache[lang] = parser
            parser = parser_cache[lang]
            dfg_func = dfg_function.get(lang)
            if dfg_func is None:
                raise ValueError(f"No DFG function available for language: {lang}")

            if lang not in keywords_cache:
                keywords_file = (
                    importlib.resources.files("codebleu") / "keywords" / f"{lang}.txt"
                )
                keywords_cache[lang] = keywords_file.read_text(
                    encoding="utf-8"
                ).splitlines()
            keywords = keywords_cache[lang]

            # (1) Process target code
            target_str = str(row["target_code_string"]).strip()

            # get the (weighted) tokens to compute BLEU, only needed for the target code
            raw_target_tokens = self.tokenizer(target_str)
            target_tokens.append(raw_target_tokens)
            target_tokens_with_weights.append(
                [raw_target_tokens, self.make_weights(raw_target_tokens, keywords)]
            )

            # get the AST
            target_code = remove_comments_and_docstrings(target_str, lang)
            target_tree = parser.parse(bytes(target_code, "utf8")).root_node
            target_asts.append(target_tree)

            # get the data flow
            target_dfg = get_data_flow(target_code, [parser, dfg_func])
            target_normalized_dfgs.append(normalize_dataflow(target_dfg))

            # (2) Process generated code
            generated_str = str(row["model_generated_code_string"]).strip()

            # get the tokens to compute BLEU
            generated_tokens.append(self.tokenizer(generated_str))

            # get the AST
            generated_code = remove_comments_and_docstrings(generated_str, lang)
            generated_tree = parser.parse(bytes(generated_code, "utf8")).root_node
            generated_asts.append(generated_tree)

            # get the data flow
            generated_dfg = get_data_flow(generated_code, [parser, dfg_func])
            generated_normalized_dfgs.append(normalize_dataflow(generated_dfg))

        df["target_tokens"] = target_tokens
        df["generated_tokens"] = generated_tokens
        df["target_tokens_with_weights"] = target_tokens_with_weights
        df["target_ast"] = target_asts
        df["generated_ast"] = generated_asts
        df["target_normalized_dfg"] = target_normalized_dfgs
        df["generated_normalized_dfg"] = generated_normalized_dfgs

        return CodeBleuAnalysisInput(generation_df=df)
