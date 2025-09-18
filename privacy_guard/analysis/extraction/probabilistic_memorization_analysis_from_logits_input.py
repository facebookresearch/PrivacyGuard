# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from privacy_guard.analysis.base_analysis_input import BaseAnalysisInput


logger: logging.Logger = logging.getLogger(__name__)


class ProbabilisticMemorizationAnalysisFromLogitsInput(BaseAnalysisInput):
    """
    Takes in a single dataframe of generation data with prediction logits and target tokens.

    args:
        generation_df: dataframe containing logits column and target_tokens column
        prob_threshold: threshold for comparing model probabilities
        n_values: optional list of n values for computing corresponding probabilities of model outputting the target in n attempts. Refer to https://arxiv.org/abs/2410.19482 for details.
        logits_column: name of the column containing logits (default: "prediction_logits")
        target_tokens_column: name of the column containing target tokens (default: "target_tokens")
        **generation_kwargs: keyword arguments for generation (e.g., temp, top_k)
    """

    def __init__(
        self,
        generation_df: pd.DataFrame,
        prob_threshold: float,
        n_values: Optional[List[int]] = None,
        logits_column: str = "prediction_logits",
        target_tokens_column: str = "target_tokens",
        **generation_kwargs: Any,
    ) -> None:
        self.generation_kwargs: Dict[str, Any] = generation_kwargs
        self.prob_threshold: float = prob_threshold
        self.n_values: List[int] = n_values or []
        self.logits_column: str = logits_column
        self.target_tokens_column: str = target_tokens_column

        # Validate required columns
        required_columns = {logits_column, target_tokens_column}
        missing_columns = required_columns - set(generation_df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        super().__init__(df_train_user=generation_df, df_test_user=pd.DataFrame())

    @property
    def generation_df(self) -> pd.DataFrame:
        """
        Property accessor for generation_df
        """
        return self._df_train_user
