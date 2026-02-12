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
from privacy_guard.analysis.base_analysis_input import BaseAnalysisInput


class CodeSimilarityAnalysisInput(BaseAnalysisInput):
    """
    Analysis input for code similarity analysis.

    Stores a generation DataFrame containing target and model-generated code strings
    along with their parsed ASTs.

    Required columns:
        - target_code_string: the original target code
        - model_generated_code_string: the model's generated code
        - target_ast: parsed AST (zss Node) for the target code
        - generated_ast: parsed AST (zss Node) for the generated code
        - target_parse_status: "success" or "partial" (error nodes filtered)
        - generated_parse_status: "success" or "partial" (error nodes filtered)

    Args:
        generation_df: DataFrame containing code strings and parsed ASTs
    """

    REQUIRED_COLUMNS: list[str] = [
        "target_code_string",
        "model_generated_code_string",
        "target_ast",
        "generated_ast",
        "target_parse_status",
        "generated_parse_status",
    ]

    def __init__(self, generation_df: pd.DataFrame) -> None:
        missing = set(self.REQUIRED_COLUMNS) - set(generation_df.columns)
        if missing:
            raise ValueError(f"Missing required columns in generation_df: {missing}")

        super().__init__(df_train_user=generation_df, df_test_user=pd.DataFrame())

    @property
    def generation_df(self) -> pd.DataFrame:
        """Property accessor for the generation DataFrame."""
        return self._df_train_user
