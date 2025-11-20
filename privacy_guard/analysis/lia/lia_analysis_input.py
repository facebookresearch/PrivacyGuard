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

import pandas as pd
from numpy.typing import NDArray
from privacy_guard.analysis.base_analysis_input import BaseAnalysisInput


class LIAAnalysisInput(BaseAnalysisInput):
    def __init__(
        self,
        predictions: NDArray[float],
        predictions_y1_generation: NDArray[float],
        true_bits: NDArray[int],
        y0: NDArray[int],
        y1: NDArray[int],
        received_labels: NDArray[int],
    ) -> None:
        """
        Input to the LIA analysis.
        args:
            predictions: target model's predictions on the training data
            predictions_y1_generation: predictions on training features used for generating y1 (reconstructed labels)
            true_bits: array of 0s and 1s indicating whether a sample is from training or reconstrcuted
            y0: training labels
            y1: reconstructed labels
            received_labels: labels received by the adversary (y0 or y1)
        """
        self.predictions = predictions
        self.true_bits = true_bits
        self.y0 = y0
        self.y1 = y1
        self.received_labels = received_labels
        self.predictions_y1_generation = predictions_y1_generation
        # Create minimal DataFrame for compatibility with BaseAnalysisInput
        df_small = pd.DataFrame({"predictions": predictions})
        super().__init__(df_train_user=df_small, df_test_user=df_small)
