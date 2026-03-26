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
from dataclasses import dataclass, field
from typing import Any, cast

import pandas as pd
from codebleu.bleu import (  # @manual=fbsource//third-party/pypi/codebleu:codebleu
    corpus_bleu,
)
from codebleu.weighted_ngram_match import (  # @manual=fbsource//third-party/pypi/codebleu:codebleu
    corpus_bleu as corpus_bleu_weighted,
)
from privacy_guard.analysis.base_analysis_node import BaseAnalysisNode
from privacy_guard.analysis.base_analysis_output import BaseAnalysisOutput
from privacy_guard.analysis.code_similarity.code_similarity_analysis_input import (
    CodeBleuAnalysisInput,
)

# pyre-ignore[21]: tree-sitter doesn't have properly exposed type stubs
from tree_sitter import Node

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class CodeBleuNodeOutput(BaseAnalysisOutput):
    """Output of :class:`CodeBleuNode`.

    Attributes:
        num_samples: total number of sample rows.
        per_sample_code_bleu: DataFrame with a ``code_bleu`` column.
        avg_code_bleu: average CodeBLEU across all pairs.
        avg_code_bleu_by_language: per-language average similarity, or
            ``None`` when no ``language`` column is present.
    """

    num_samples: int
    per_sample_code_bleu: pd.DataFrame = field(repr=False)
    avg_code_bleu: float
    avg_code_bleu_by_language: dict[str, float] | None


class CodeBleuNode(BaseAnalysisNode):
    """Compute CodeBLEU similarity between two pieces of code.

    The metric represents a weighted sum of the following components:
        - alpha * ngram_match_score
        - beta * weighted_ngram_match_score, where language-specific, generic tokens are given less weight
        - gamma * syntax_match_score, where syntax_match_score is computed using the distance between ASTs of the code
        - theta * dataflow_match_score, where dataflow_match_score is computed using the distance between data flows of the code

    Args:
        analysis_input: a :class:`CodeBleuAnalysisInput` produced
            by :class:`CodeBleuAttack`.
    """

    def __init__(self, analysis_input: CodeBleuAnalysisInput) -> None:
        super().__init__(analysis_input=analysis_input)

    @staticmethod
    # pyre-ignore[11]: Annotation `Node` is not defined as a type.
    def syntax_match(target_tree: Node, generated_tree: Node) -> float:
        def _node_sexp(node: Node) -> str:
            """Build a position-independent s-expression string for a subtree."""
            if not node.children:
                return node.type
            return f"({node.type} {' '.join(_node_sexp(c) for c in node.children)})"

        def get_all_sub_trees(root_node: Node) -> list[str]:
            node_stack: list[Node] = [root_node]
            sub_tree_sexp_list = []
            while node_stack:
                cur_node = node_stack.pop()
                sub_tree_sexp_list.append(_node_sexp(cur_node))
                for child_node in cur_node.children:
                    if child_node.children:
                        node_stack.append(child_node)
            return sub_tree_sexp_list

        target_sexps = get_all_sub_trees(target_tree)
        generated_sexps = get_all_sub_trees(generated_tree)

        # Per §3.2 of https://arxiv.org/pdf/2009.10297.pdf:
        # Match(T_candidate, T_reference) = |ST(T_candidate) ∩ ST(T_reference)| / |ST(T_reference)|
        # Iterate over generated (candidate) subtrees and count matches in target (reference).
        # this follows their definition in the paper and addresses the TODO mentioned in their code
        if len(target_sexps) == 0:
            logger.warning("Empty target AST, syntax match score degenerates to 0.")
            return 0.0

        target_sexps_copy = list(target_sexps)
        match_count = 0
        for sub_tree in generated_sexps:
            if sub_tree in target_sexps_copy:
                match_count += 1
                target_sexps_copy.remove(sub_tree)

        return match_count / len(target_sexps)

    @staticmethod
    def dataflow_match(target_dfg: Any, generated_dfg: Any) -> float:
        total_count = len(target_dfg)

        if total_count == 0:
            logger.warning("Empty target DFG, dataflow match score degenerates to 0.")
            return 0.0

        generated_dfg_copy = list(generated_dfg)  # Shallow copy to avoid mutating input
        match_count = 0

        for dataflow in target_dfg:
            if dataflow in generated_dfg_copy:
                match_count += 1
                generated_dfg_copy.remove(dataflow)

        return match_count / total_count

    @staticmethod
    def calc_codebleu(
        target_tokens: list[str],
        generated_tokens: list[str],
        target_tokens_with_weights: tuple[list[str], dict[str, float]],
        target_ast: Node,
        generated_ast: Node,
        target_normalized_dataflow: Any,
        generated_normalized_dataflow: Any,
        weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
    ) -> float:
        """Calculate the CodeBLEU similarity score between target and generated code.

        CodeBLEU is a composite metric that combines lexical, syntactic, and semantic
        similarity measures. The final score is a weighted sum of four components:

            score = α * ngram_match + β * weighted_ngram_match + γ * syntax_match + θ * dataflow_match

        Where:
            - ngram_match: Standard BLEU score measuring n-gram overlap
            - weighted_ngram_match: BLEU score with reduced weight (0.2) for non-keyword tokens (specific per language)
            - syntax_match: Fraction of target AST subtrees found in generated AST
            - dataflow_match: Fraction of target dataflow edges found in generated code

        Note that, if the target AST or DFG is empty, the syntax and dataflow matchs are set to 0.

        See: https://arxiv.org/pdf/2009.10297

        Args:
            target_tokens: Tokenized target (reference) code.
            generated_tokens: Tokenized generated (hypothesis) code.
            target_tokens_with_weights: Target tokens with keyword weight dict [tokens, {token: weight}].
            target_ast: Parsed AST root node for target code.
            generated_ast: Parsed AST root node for generated code.
            target_normalized_dataflow: Normalized dataflow graph for target code.
            generated_normalized_dataflow: Normalized dataflow graph for generated code.
            weights: Tuple of (α, β, γ, θ) weights for the four components.
                Defaults to equal weighting (0.25, 0.25, 0.25, 0.25).

        Returns:
            CodeBLEU similarity score in the range [0, 1], where 1 indicates
            identical code.
        """

        ngram_match_score = corpus_bleu([[target_tokens]], [generated_tokens])

        weighted_ngram_match_score = corpus_bleu_weighted(
            [[target_tokens_with_weights]], [generated_tokens]
        )

        # calculate syntax match
        syntax_match_score = CodeBleuNode.syntax_match(target_ast, generated_ast)

        # calculate dataflow match
        dataflow_match_score = CodeBleuNode.dataflow_match(
            target_normalized_dataflow, generated_normalized_dataflow
        )

        alpha, beta, gamma, theta = weights
        code_bleu_score = (
            alpha * ngram_match_score
            + beta * weighted_ngram_match_score
            + gamma * syntax_match_score
            + theta * dataflow_match_score
        )

        return code_bleu_score

    # ------------------------------------------------------------------
    # BaseAnalysisNode interface
    # ------------------------------------------------------------------

    def run_analysis(self) -> CodeBleuNodeOutput:
        analysis_input = cast(CodeBleuAnalysisInput, self.analysis_input)
        df = analysis_input.generation_df

        def _row_similarity(row: pd.Series) -> float:  # type: ignore[type-arg]
            return CodeBleuNode.calc_codebleu(
                row["target_tokens"],
                row["generated_tokens"],
                row["target_tokens_with_weights"],
                row["target_ast"],
                row["generated_ast"],
                row["target_normalized_dfg"],
                row["generated_normalized_dfg"],
            )

        similarities = df.apply(_row_similarity, axis=1)
        per_sample = pd.DataFrame({"code_bleu": similarities})

        avg_code_bleu = float(similarities.mean()) if len(similarities) > 0 else 0.0

        avg_by_lang: dict[str, float] | None = None
        if "language" in df.columns:
            per_sample["language"] = df["language"].values
            grouped = per_sample.groupby("language")["code_bleu"].mean()
            avg_by_lang = grouped.to_dict()

        return CodeBleuNodeOutput(
            num_samples=len(df),
            per_sample_code_bleu=per_sample,
            avg_code_bleu=avg_code_bleu,
            avg_code_bleu_by_language=avg_by_lang,
        )
