# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import List

import numpy as np
import pandas as pd
from privacy_guard.analysis.base_analysis_input import BaseAnalysisInput

from privacy_guard.analysis.mia.aggregate_analysis_input import (
    AggregateAnalysisInput,
    AggregationType,
)
from privacy_guard.attacks.base_attack import BaseAttack
from python.migrations.py310 import StrEnum310


class CalibScoreType(StrEnum310):
    LOSS = "loss"
    ENTROPY = "entropy"
    CONFIDENCE = "confidence"
    SCALED_LOGITS = "scaled_logits"


class CalibAttack(BaseAttack):
    """
    This is an implementation of an MIA attack
    Lightweight calibration attack
    """

    def __init__(
        self,
        df_hold_out_train: pd.DataFrame,
        df_hold_out_test: pd.DataFrame,
        df_hold_out_train_calib: pd.DataFrame,
        df_hold_out_test_calib: pd.DataFrame,
        row_aggregation: AggregationType,
        should_calibrate_scores: bool,
        score_type: CalibScoreType,
        user_id_key: str = "user_id",
        merge_columns: List[str] | None = None,
    ) -> None:
        """
        args:
            df_hold_out_train: Training dataset
            df_hold_out_test: Testing dataset
            df_hold_out_train_calib: Calibrated training dataset
            df_hold_out_test_calib: Calibrated testing dataset
                Dataframes have columns "label" and "predictions"
            row_aggregation: specifies user aggregation strategy
            should_calibrate_scores: Whether to calibrate scores with df_hold_out_*_calib or not.
            score_type: type of score to use.
            user_id_key: key representing user ids, to use in aggregation.
            merge_columns: list of columns to merge dataset and calibrated
                dataset on. If None, will default to user_id_key inly
        """
        self.df_hold_out_train = df_hold_out_train
        self.df_hold_out_test = df_hold_out_test
        self.df_hold_out_train_calib = df_hold_out_train_calib
        self.df_hold_out_test_calib = df_hold_out_test_calib

        self.row_aggregation: AggregationType = row_aggregation

        self.should_calibrate_scores = should_calibrate_scores

        self.score_type = score_type

        self.user_id_key = user_id_key
        self.merge_columns: List[str] = merge_columns or [user_id_key]

        for column in self.merge_columns:
            for columns in [
                df_hold_out_train.columns,
                df_hold_out_test.columns,
                df_hold_out_train_calib,
                df_hold_out_test_calib,
            ]:
                if column not in columns:
                    raise IndexError(f"column {column} not found in input dataframe(s)")

    @classmethod
    def compute_loss(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute log-likelihood. Lower means model fits more
        """
        return -(
            df["label"] * np.log(1e-30 + df["predictions"])
            + (1 - df["label"]) * np.log(1e-30 + 1 - df["predictions"])
        )

    @classmethod
    def compute_entropy(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute entropy. Lower means model fits more
        """
        return -(
            df["predictions"] * np.log(1e-30 + df["predictions"])
            + (1 - df["predictions"]) * np.log(1e-30 + 1 - df["predictions"])
        )

    @classmethod
    def compute_confidence(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute (negative) confidence. Lower means model fits more
        """
        return -(
            df["label"] * df["predictions"]
            + (1 - df["label"]) * (1 - df["predictions"])
        )

    @classmethod
    def compute_scaled_logits(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute (negative) scaled logits. Lower means model fits more
        """

        return -(
            np.log(1e-30 + df["predictions"]) - np.log(1e-30 + 1 - df["predictions"])
        ) * (2 * df["label"] - 1)

    @classmethod
    def compute_score(cls, df: pd.DataFrame, score_type: str) -> pd.DataFrame:
        """
        Compute scores as DataFrame
        """

        match score_type:
            case "loss":
                score = cls.compute_loss(df)
            case "entropy":
                score = cls.compute_entropy(df)
            case "confidence":
                score = cls.compute_confidence(df)
            case "scaled_logits":
                score = cls.compute_scaled_logits(df)
            case _:
                raise ValueError(f"{score_type} is not a valid score type.")

        return score

    def run_attack(self) -> BaseAnalysisInput:
        """
        Execute lightweight calibration attack on input data
        """
        ##check that train/test users scores are loaded
        assert self.df_hold_out_train.shape[0] > 0
        assert self.df_hold_out_test.shape[0] > 0
        assert self.df_hold_out_train_calib.shape[0] > 0
        assert self.df_hold_out_test_calib.shape[0] > 0

        # SCORE_TYPES: compute score
        self.df_hold_out_train["score"] = self.compute_score(
            self.df_hold_out_train, self.score_type
        )
        self.df_hold_out_test["score"] = self.compute_score(
            self.df_hold_out_test, self.score_type
        )

        # ATTACK_TYPES: Calibrate if needed
        if self.should_calibrate_scores:
            self.df_hold_out_train_calib["score"] = self.compute_score(
                self.df_hold_out_train_calib, self.score_type
            )

            self.df_hold_out_test_calib["score"] = self.compute_score(
                self.df_hold_out_test_calib, self.score_type
            )

            df_train_merge = pd.merge(
                self.df_hold_out_train,
                self.df_hold_out_train_calib,
                on=self.merge_columns,
                suffixes=("_valid", "_calib"),
            )
            df_test_merge = pd.merge(
                self.df_hold_out_test,
                self.df_hold_out_test_calib,
                on=self.merge_columns,
                suffixes=("_valid", "_calib"),
            )

            # calibrate
            df_train_merge["score"] = (
                df_train_merge["score_valid"] - df_train_merge["score_calib"]
            )
            df_test_merge["score"] = (
                df_test_merge["score_valid"] - df_test_merge["score_calib"]
            )

            df_train_merge["score"] = -df_train_merge["score"]
            df_test_merge["score"] = -df_test_merge["score"]
        else:
            df_train_merge = self.df_hold_out_train
            df_test_merge = self.df_hold_out_test
            df_train_merge["score"] = -df_train_merge["score"]
            df_test_merge["score"] = -df_test_merge["score"]

        analysis_input = AggregateAnalysisInput(
            row_aggregation=self.row_aggregation,
            df_train_merge=df_train_merge,
            df_test_merge=df_test_merge,
            user_id_key=self.user_id_key,
        )

        return analysis_input
