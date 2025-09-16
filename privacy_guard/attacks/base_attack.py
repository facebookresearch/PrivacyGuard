# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from abc import ABC, abstractmethod

from privacy_guard.analysis.base_analysis_input import BaseAnalysisInput


class BaseAttack(ABC):
    """
    BaseAttack class for PrivacyGuard.

    Generic module supporting implementation of a wide suite of
    Privacy Attacks accross different use cases. Extensible for continued
    addition of SotA privacy attacks.

    Attacks produce AnalysisInput, containing two dataframes with
    training and testing data, as well
    as an aggregation strategy to aggregate the data accross the dataframes.
    See privacy_guard/analysis/analysis_input.py for more details.

    The AnalysisInput created from each attack is forwarded as input for
    different implementations of privacy_guard/analysis/BaseAnalysisNode
    """

    @abstractmethod
    def run_attack(self) -> BaseAnalysisInput:
        """
        Execute privacy attack on input data

        Returns:
            BaseAnalysisInput: output containing train and test
            data, ready for consumption by different analyses.
        """
