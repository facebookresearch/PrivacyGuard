# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from privacy_guard.analysis.base_analysis_node import BaseAnalysisNode
from privacy_guard.analysis.base_analysis_output import BaseAnalysisOutput
from scipy.stats import norm


@dataclass
class FDPAnalysisNodeOutput(BaseAnalysisOutput):
    """
    A dataclass to encapsulate the outputs of FDPAnalsyisNode.
    Attributes:
        eps (float): Epsilon value
    """

    # Empirical epsilon
    eps: float


class FDPAnalysisNode(BaseAnalysisNode):
    def __init__(
        self,
        m: int,
        c: int,
        c_cap: int,
        target_noise: float = 0.001,
        threshold: float = 0.05,
        k: int = 2,
        delta: float = 1e-6,
    ) -> None:
        """
        Class to implement the FDP analysis in "Auditing f -Differential Privacy in One Run" (https://arxiv.org/abs/2410.22235)

        :param m: Total number of canaries.
        :param c: Number of correct guesses.
        :param c_cap: Number of total guesses.
        :param target_noise: Initial noise level for candidate noises.
        :param threshold: Probability threshold value for auditing.
        :param k: alphabet size (k=2 for membership inference attacks).
        :param delta: Delta value for DP.
        """
        self.candidate_noises: list[float] = [
            target_noise + i * 0.01 for i in range(1000)
        ]

        self.inverse_blow_up_functions: list[Callable[[float], float]] = [
            self.gaussianDP_blow_up_inverse(noise) for noise in self.candidate_noises
        ]

        self.threshold = threshold
        self.delta = delta
        self.k = k

        self.m = m
        self.c = c
        self.c_cap = c_cap

    @staticmethod
    def gaussianDP_blow_up_function(noise: float) -> Callable[[float], float]:
        """
        Creates a blow-up function for Gaussian differential privacy.

        :param noise: Noise level for the function.
        :return: A callable blow-up function.
        """

        def blow_up_function(x: float) -> float:
            threshold = norm.ppf(x)
            blown_up_threshold = threshold + 1 / noise
            return norm.cdf(blown_up_threshold)

        return blow_up_function

    @staticmethod
    def gaussianDP_blow_up_inverse(noise: float) -> Callable[[float], float]:
        """
        Creates an inverse blow-up function for Gaussian differential privacy.

        :param noise: Noise level for the function.
        :return: A callable inverse blow-up function.
        """

        def blow_up_inverse_function(x: float) -> float:
            threshold = norm.ppf(x)
            blown_up_threshold = threshold - 1 / noise
            return norm.cdf(blown_up_threshold)

        return blow_up_inverse_function

    def calculate_delta_gaussian(self, noise: float, epsilon: float) -> float:
        """
        Calculates the delta value for Gaussian differential privacy.

        :param noise: Noise level for the calculation.
        :param epsilon: Epsilon value for the calculation.
        :return: Calculated delta value.
        """
        delta = norm.cdf(-epsilon * noise + 1 / (2 * noise)) - np.exp(
            epsilon
        ) * norm.cdf(-epsilon * noise - 1 / (2 * noise))
        return delta

    def calculate_epsilon_gaussian(self, noise: float) -> float:
        """
        Calculates the epsilon value for Gaussian differential privacy.

        :param noise: Noise level for the calculation.
        :return: Calculated epsilon value.
        """
        epsilon_upper = 100
        epsilon_lower = 0
        while epsilon_upper - epsilon_lower > 0.001:
            epsilon_middle = (epsilon_upper + epsilon_lower) / 2
            if self.calculate_delta_gaussian(noise, epsilon_middle) > self.delta:
                epsilon_lower = epsilon_middle
            else:
                epsilon_upper = epsilon_middle
        return epsilon_upper

    def rh_with_cap(
        self,
        inverse_blow_up_function: Callable[[float], float],
        alpha: float,
        beta: float,
        j: int,
        m: int,
        c_cap: int,
    ) -> Tuple[list[float], list[float]]:
        """
        Computes the r and h values with a cap.

        :param inverse_blow_up_function: Inverse blow-up function to use.
        :param alpha: Initial alpha value.
        :param beta: Initial beta value.
        :param j: Index for calculations.
        :param m: Total number of canaries.
        :param c_cap: Number of total guesses.
        :return: Tuple of r and h values.
        """
        h = [0.0 for _ in range(j + 1)]
        r = [0.0 for _ in range(j + 1)]
        h[j] = beta
        r[j] = alpha
        for i in range(j - 1, -1, -1):
            h[i] = max(h[i + 1], (self.k - 1) * inverse_blow_up_function(r[i + 1]))
            r[i] = r[i + 1] + (i / (c_cap - i)) * (h[i] - h[i + 1])
        return (r, h)

    def audit_rh_with_cap(
        self,
        inverse_blow_up_function: Callable[[float], float],
        m: int,
        c: int,
        c_cap: int,
    ) -> bool:
        """
        Audits the r and h values with a cap to check if they meet the threshold.

        :param inverse_blow_up_function: Inverse blow-up function to use.
        :param m: Total number of canaries.
        :param c: Number of correct guesses.
        :param c_cap: Number of total guesses.
        :return: Boolean indicating if the audit passed.
        """
        threshold = self.threshold * c_cap / m
        alpha = threshold * c / c_cap
        beta = threshold * (c_cap - c) / c_cap
        r, h = self.rh_with_cap(inverse_blow_up_function, alpha, beta, c, m, c_cap)

        return r[0] + h[0] <= c_cap / m

    def run_analysis_with_parameters(
        self,
        m: int,
        c: int,
        c_cap: int,
    ) -> FDPAnalysisNodeOutput:
        """
        Computes the epsilon value for the given parameters.

        :param m: Total number of canaries.
        :param c: Number of correct guesses.
        :param c_cap: Number of total guesses.
        :return: Calculated epsilon value.
        """
        assert c_cap <= m, "m must be greater than c_cap"
        assert c <= c_cap, "c_cap must be greater than c"

        empirical_privacy_index = 0
        while empirical_privacy_index < len(
            self.inverse_blow_up_functions
        ) and self.audit_rh_with_cap(
            self.inverse_blow_up_functions[empirical_privacy_index],
            m,
            c,
            c_cap,
        ):
            empirical_privacy_index += 1
        empirical_noise = self.candidate_noises[
            min(empirical_privacy_index, len(self.candidate_noises) - 1)
        ]
        empirical_eps = self.calculate_epsilon_gaussian(empirical_noise)
        output = FDPAnalysisNodeOutput(eps=empirical_eps)
        return output

    def run_analysis(
        self,
    ) -> FDPAnalysisNodeOutput:
        """
        Runs analysis with default parameter arguments.

        :return: Calculated epsilon value.
        """
        return self.run_analysis_with_parameters(m=self.m, c=self.c, c_cap=self.c_cap)
