# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
import logging
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from privacy_guard.analysis.base_analysis_input import BaseAnalysisInput
from privacy_guard.analysis.base_analysis_output import BaseAnalysisOutput


logger: logging.Logger = logging.getLogger(__name__)


class BaseAnalysisNode(ABC):
    """
    Base Analysis Node class for Privacy Guard.

    Implementations of this class should implement the methods
    (1) "run_analysis" to use the data in
    BaseAnalysisInput to produce different output metrics as a structured dataclass and
    (2) "compute_outputs" method that calls "run_analysis" and returns the output as a dictionary.
    The latter method is useful for merging the outputs of multiple analysis nodes

    Args:
        analysis_input: BaseAnalysisInput object containing the
            training and testing data
    """

    def __init__(self, analysis_input: BaseAnalysisInput) -> None:
        """
        args:
            user_aggregation: specifies user aggregation strategy
        """
        self._analysis_input = analysis_input

    @property
    def analysis_input(self) -> BaseAnalysisInput:
        """
        Return the BaseAnalysisInput object passed in the constructor
        """
        return self._analysis_input

    @abstractmethod
    # TODO: For now, allow a dictionary output to be returned as we do in InclusionAnalysisNode. In the future, create a dataclass for the output of InclusionAnalysisNode.
    def run_analysis(self) -> BaseAnalysisOutput:
        """
        Computes the analysis outputs and stores the results as a dataclass extending BaseAnalysisOutput
        """

    def compute_outputs(self) -> dict[str, Any]:
        """
        Returns the analysis outputs as a dictionary
        """
        dataclass_output = self.run_analysis()
        return dataclass_output.to_dict()


AnalysisNodeType = TypeVar("AnalysisNodeType", bound=BaseAnalysisNode)


def compute_and_merge_outputs(nodes: list[AnalysisNodeType]) -> dict[str, Any]:
    """
    Compute outputs of multiple nodes and merge them into one dict.

    If the analysis output of different nodes share the same key, the
    latter value will override the former.

    args:
        nodes: list of analysis nodes which implement BaseAnalysisNode
    """
    outputs = {}
    for node in nodes:
        new_outputs = node.compute_outputs()
        overwritten_keys = set(outputs.keys()).intersection(set(new_outputs.keys()))
        if overwritten_keys:
            logger.info(
                "The following keys are being overwritten by another analysis: "
                + str(overwritten_keys)
            )

        outputs.update(new_outputs)

    return outputs
