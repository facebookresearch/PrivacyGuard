# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
from typing import Dict, List

import numpy as np
import pandas as pd
from privacy_guard.analysis.lia.lia_analysis_input import LIAAnalysisInput

from privacy_guard.analysis.mia.aggregate_analysis_input import AggregationType
from privacy_guard.attacks.base_attack import BaseAttack


class LIAAttackInput:
    ADS_MERGE_COLUMNS = [
        "separable_id",
        "ad_id",
        "timestamp",
        "impression_signature",
        "label",
    ]

    def __init__(
        self,
        df_hold_out_train: pd.DataFrame,
        df_hold_out_train_calib: pd.DataFrame,
        row_aggregation: AggregationType,
        merge_columns: List[str] | None = None,
    ) -> None:
        """
        args:
            df_hold_out_train: Subset of training set containing canaries for the attack
            df_hold_out_train_calib: Samples of df_hold_out_train evaluated on a calibration model/snapshot.
            row_aggregation: specifies aggregation strategy for aggregating rows for each user.
            merge_columns: columns to merge on for df_hold_out_train and df_hold_out_train_calib.
        """
        self.df_hold_out_train = df_hold_out_train
        self.df_hold_out_train_calib = df_hold_out_train_calib
        self.row_aggregation = row_aggregation
        self.merge_columns: List[str] = merge_columns or self.ADS_MERGE_COLUMNS

        if self.df_hold_out_train.shape[0] == 0:
            raise ValueError("df_hold_out_train must be non-empty")
        if self.df_hold_out_train_calib.shape[0] == 0:
            raise ValueError("df_hold_out_train_calib must be non-empty")

        for column in self.merge_columns:
            for columns in [
                df_hold_out_train.columns,
                df_hold_out_train_calib.columns,
            ]:
                if column not in columns:
                    raise IndexError(f"column {column} not found in input dataframe(s)")

        if "predictions" not in self.df_hold_out_train.columns:
            raise ValueError("predictions column not found in df_hold_out_train")
        if "predictions" not in self.df_hold_out_train_calib.columns:
            raise ValueError("predictions column not found in df_hold_out_train_calib")

    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate all samples pertaining to one user by selecting the easiest sample to target.
        """
        print("Aggregating tables...")
        if self.row_aggregation == AggregationType.ABS_MAX:
            df["abs_score"] = df["score"].abs()
            return df.loc[df.groupby("separable_id")["abs_score"].idxmax()]
        elif self.row_aggregation == AggregationType.MAX:
            return df.loc[df.groupby("separable_id")["score"].idxmax()]
        elif self.row_aggregation == AggregationType.MIN:
            return df.loc[df.groupby("separable_id")["score"].idxmin()]
        elif self.row_aggregation == AggregationType.NONE:
            return df
        else:
            raise ValueError(f"Unknown aggregation type {self.row_aggregation}")

    def prepare_attack_input(self) -> Dict[str, pd.DataFrame]:
        """
        Prepare input for label inference attack.
        """
        # add predictions from calibration model
        df_train_merge = pd.merge(
            self.df_hold_out_train,
            self.df_hold_out_train_calib,
            on=self.merge_columns,
            suffixes=("", "_calib"),
        )

        # aggregate tables
        # calculate score used for aggregation
        df_train_merge["score"] = (
            df_train_merge["predictions"] - df_train_merge["predictions_calib"]
        )
        df_train_merge_agg = self.aggregate(df_train_merge)
        print("Aggregation complete!")

        attack_input_dict = {
            "df_train_and_calib": df_train_merge,
            "df_aggregated": df_train_merge_agg,
        }

        return attack_input_dict


class LIAAttack(BaseAttack):
    """
    This class implements LIA: label inference attack.
    """

    def __init__(
        self,
        attack_input: Dict[str, pd.DataFrame],
        row_aggregation: AggregationType,
        y1_generation: str = "calibration",
        num_resampling_times: int = 100,
    ) -> None:
        """
        args:
            attack_input: dictionary containing dataframes for the attack, must contain keys "df_train_and_calib" and "df_aggregated"
            row_aggregation: specifies aggregation strategy for aggregating rows for each user
            y1_generation: strategy for generating the labels y1 (reconstructed labels)
            num_resampling_times: Number of times to instantiate the LIA game (for confidence interval estimation)
        """
        self.attack_input = attack_input
        self.row_aggregation = row_aggregation
        self.y1_generation = y1_generation
        self.num_resampling_times = num_resampling_times

    def get_y1_predictions(self, df_attack: pd.DataFrame) -> np.ndarray:
        """
        Get predictions used for y1 (reconstructed label) generation for the attack.
        args:
            df_attack: dataframe used for the attack, contains columns "predictions" and "predictions_calib" or "predictions_reference"
        returns:
            predictions_y1_generation: predictions used generating reconstructed labels y1
        """
        predictions_y1_generation = None
        if self.y1_generation == "target":
            predictions_y1_generation = df_attack["predictions"].values
            print("Using target predictions for y1 generation")
        elif self.y1_generation == "calibration":
            if "predictions_calib" not in df_attack.columns:
                raise ValueError(
                    "predictions_calib column not found in df_attack. Please provide calibration predictions."
                )
            predictions_y1_generation = df_attack["predictions_calib"].values
            print("Using calibration predictions for y1 generation")
        elif self.y1_generation == "reference":
            if "predictions_reference" not in df_attack.columns:
                raise ValueError(
                    "predictions_reference column not found in df_attack. Please provide reference predictions."
                )
            predictions_y1_generation = df_attack["predictions_reference"].values
            print("Using reference predictions for y1 generation")
        else:
            combo_factor = float(self.y1_generation)
            if "predictions_calib" not in df_attack.columns:
                raise ValueError(
                    "predictions_calib column not found in df_attack. Please provide calibration predictions."
                )
            predictions_y1_generation = (
                combo_factor * df_attack["predictions"].values
                + (1 - combo_factor) * df_attack["predictions_calib"].values
            )
            print(
                "Using combo of target and calibration predictions for y1 generation with factor ",
                combo_factor,
            )

        return predictions_y1_generation

    def run_attack(self) -> LIAAnalysisInput:
        """
        Run LIA attack.
        """
        # choose table for attack based on aggregation
        if self.row_aggregation == AggregationType.NONE:
            df_attack = self.attack_input["df_train_and_calib"]
        else:
            df_attack = self.attack_input["df_aggregated"]

        y0 = df_attack["label"].values
        predictions = df_attack["predictions"].values
        predictions_y1_generation = self.get_y1_predictions(df_attack)
        true_bits_all_reps = np.random.randint(
            2, size=(self.num_resampling_times, len(df_attack))
        )
        random_floats = np.random.rand(self.num_resampling_times, len(df_attack))
        y1_all_reps = (random_floats < predictions_y1_generation).astype(int)
        received_labels_all_reps = np.where(true_bits_all_reps == 0, y0, y1_all_reps)

        # Create analysis input object
        analysis_input = LIAAnalysisInput(
            predictions=predictions,
            predictions_y1_generation=predictions_y1_generation,
            true_bits=true_bits_all_reps,
            y0=y0,
            y1=y1_all_reps,
            received_labels=received_labels_all_reps,
        )

        return analysis_input
