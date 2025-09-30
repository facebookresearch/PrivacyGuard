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
