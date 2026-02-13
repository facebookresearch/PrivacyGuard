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
from typing import cast

import pandas as pd
from privacy_guard.analysis.base_analysis_node import BaseAnalysisNode
from privacy_guard.analysis.base_analysis_output import BaseAnalysisOutput
from privacy_guard.analysis.code_similarity.code_similarity_analysis_input import (
    CodeSimilarityAnalysisInput,
)
from zss import Node as ZSSNode, simple_distance


logger: logging.Logger = logging.getLogger(__name__)


def _count_nodes(node: ZSSNode) -> int:
    """Recursively count the number of nodes in a zss tree."""
    count = 1
    for child in node.children:
        count += _count_nodes(child)
    return count


@dataclass
class TreeEditDistanceNodeOutput(BaseAnalysisOutput):
    """Output of :class:`TreeEditDistanceNode`.

    Attributes:
        num_samples: total number of sample rows.
        num_both_parsed: number of rows where both target and generated
            code produced an AST (always equals *num_samples* since the
            attack now returns partial ASTs for malformed code).
        per_sample_similarity: DataFrame with a ``similarity`` column.
        avg_similarity: average similarity across all pairs.
        avg_similarity_by_language: per-language average similarity, or
            ``None`` when no ``language`` column is present.
    """

    num_samples: int
    num_both_parsed: int
    per_sample_similarity: pd.DataFrame = field(repr=False)
    avg_similarity: float
    avg_similarity_by_language: dict[str, float] | None


class TreeEditDistanceNode(BaseAnalysisNode):
    """Compute tree-edit-distance similarity between AST pairs.

    Uses the Zhang-Shasha algorithm (via ``zss.simple_distance``) to
    compute edit distance, then normalises to a 0-1 similarity score::

        similarity = max(1 - distance / max(n1, n2), 0)

    where *n1* and *n2* are the node counts of the two trees.

    Args:
        analysis_input: a :class:`CodeSimilarityAnalysisInput` produced
            by :class:`PyTreeSitterAttack`.
    """

    def __init__(self, analysis_input: CodeSimilarityAnalysisInput) -> None:
        super().__init__(analysis_input=analysis_input)

    # ------------------------------------------------------------------
    # Public static helper
    # ------------------------------------------------------------------

    @staticmethod
    def compute_similarity(tree1: ZSSNode, tree2: ZSSNode) -> float:
        """Compute normalised tree-edit-distance similarity.

        Args:
            tree1: first zss Node tree.
            tree2: second zss Node tree.

        Returns:
            Similarity in [0, 1] where 1.0 means identical trees.
        """
        dist: int = simple_distance(tree1, tree2)
        n1 = _count_nodes(tree1)
        n2 = _count_nodes(tree2)
        max_nodes = max(n1, n2)
        if max_nodes == 0:
            return 1.0
        return max(1.0 - dist / max_nodes, 0.0)

    # ------------------------------------------------------------------
    # BaseAnalysisNode interface
    # ------------------------------------------------------------------

    def run_analysis(self) -> TreeEditDistanceNodeOutput:
        analysis_input = cast(CodeSimilarityAnalysisInput, self.analysis_input)
        df = analysis_input.generation_df

        def _row_similarity(row: pd.Series) -> float:  # type: ignore[type-arg]
            return TreeEditDistanceNode.compute_similarity(
                row["target_ast"], row["generated_ast"]
            )

        similarities = df.apply(_row_similarity, axis=1)
        per_sample = pd.DataFrame({"similarity": similarities})

        num_both_parsed = len(similarities)
        avg_similarity = float(similarities.mean()) if num_both_parsed > 0 else 0.0

        avg_by_lang: dict[str, float] | None = None
        if "language" in df.columns:
            per_sample["language"] = df["language"].values
            grouped = per_sample.groupby("language")["similarity"].mean()
            avg_by_lang = grouped.to_dict()

        return TreeEditDistanceNodeOutput(
            num_samples=len(df),
            num_both_parsed=num_both_parsed,
            per_sample_similarity=per_sample,
            avg_similarity=avg_similarity,
            avg_similarity_by_language=avg_by_lang,
        )
