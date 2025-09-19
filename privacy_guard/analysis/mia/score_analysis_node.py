# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from dataclasses import dataclass
from typing import List

from privacy_guard.analysis.base_analysis_node import BaseAnalysisNode
from privacy_guard.analysis.base_analysis_output import BaseAnalysisOutput


@dataclass
class ScoreAnalysisNodeOutput(BaseAnalysisOutput):
    """
    A dataclass to encapsulate the outputs of ScoreAnalysisNode.
    Attributes:
        score_train (List[float]): List of scores for the training data.
        score_test (List[float]): List of scores for the test data.
    """

    train_scores: List[float]
    test_scores: List[float]


class ScoreAnalysisNode(BaseAnalysisNode):
    """
    ScoreAnalysisNode class for Privacy Guard.

    Returns the full "score_train" and "score_test" lists as
    output of the analysis. These are large objects, and limit
    the readibility of the analysis output.

    Args:
        analysis_input: BaseAnalysisInput object containing the
            training and testing data
    """

    def run_analysis(self) -> ScoreAnalysisNodeOutput:
        df_train_user = self.analysis_input.df_train_user
        df_test_user = self.analysis_input.df_test_user
        score_train = df_train_user["score"]
        score_test = df_test_user["score"]

        outputs = ScoreAnalysisNodeOutput(
            train_scores=score_train.tolist(),
            test_scores=score_test.tolist(),
        )

        return outputs
