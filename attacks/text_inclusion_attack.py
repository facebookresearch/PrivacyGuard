# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import logging
import os

import pandas as pd

from privacy_guard.analysis.text_inclusion_analysis_input import (
    LCSBoundConfig,
    TextInclusionAnalysisInput,
)

from privacy_guard.attacks.base_attack import BaseAttack

logger: logging.Logger = logging.getLogger(__name__)


class TextInclusionAttack(BaseAttack):
    """
    This loads a LLM generation file and prepares it for text inclusion analysis.
    """

    def __init__(
        self,
        llm_generation_file: str | None = None,
        data: pd.DataFrame | None = None,
        num_rows: int | None = None,
        bound_lcs: bool = False,
    ) -> None:
        """
        args:
            mast_job_name: name of MAST job to pull results from.
                In its job definition, the job should have a "dataset_dir" arg
            oilfs_mnt: Prefix to oil fuse mount on the machine this attack is ran from.
        """

        self.llm_generation_file = llm_generation_file

        if self.llm_generation_file is not None and data is None:
            assert ".jsonl" in self.llm_generation_file
            if os.access(self.llm_generation_file, os.W_OK):
                logger.warning(
                    f"WARNING: Write permission should not be allowed for {self.llm_generation_file}."
                    "\n This risks overwriting or removing the file and losing Eval results permanently."
                )

            logger.info(f"Loading LLM generation file... '{self.llm_generation_file}'")
            self.data: pd.DataFrame = pd.read_json(self.llm_generation_file, lines=True)
        elif self.llm_generation_file is None and data is not None:
            self.data: pd.DataFrame = data
        else:
            raise ValueError(
                "TextInclusionAttack must specify exactly one of 'llm_generation_file'"
                + " or 'data'. "
            )

        self.num_rows = num_rows
        self.bound_lcs = bound_lcs

    def preprocess_data(self) -> pd.DataFrame:
        """
        Changes "prompt" column to text if it is in a dict type.
        """
        df = self.data

        for _, row in df.iterrows():
            prompt = row["prompt"]
            if not isinstance(prompt, str):
                if len(prompt["dialog"]) != 1:
                    raise NotImplementedError("Multi-Turn support not yet implemented")
                elif prompt["dialog"][0]["source"] != "user":
                    raise ValueError("First message of prompt is not from user")

        df["prompt"] = df["prompt"].progress_apply(
            lambda x: x if isinstance(x, str) else x["dialog"][0]["body"]
        )

        return df

    def run_analysis_on_json_data_impl(
        self,
    ) -> TextInclusionAnalysisInput:
        data = self.preprocess_data()

        if self.num_rows:
            data = data.head(self.num_rows)

        # first_sentence_instruct => "prompt"
        data["output_text"] = data["prediction"]
        data["target"] = data["targets"].apply(lambda x: x[0])

        # For cases where lcs is too expensive, only compute at target len thresholds.
        lcs_bound_config: LCSBoundConfig | None = (
            None
            if not self.bound_lcs
            else LCSBoundConfig(lcs_len_target=150, fp_len_target=50)
        )
        analysis_input = TextInclusionAnalysisInput(
            generation_df=data,
            disable_similarity=True,
            target_key="targets",
            lcs_bound_config=lcs_bound_config,
        )
        return analysis_input

    def run_attack(self) -> TextInclusionAnalysisInput:
        """
        Pull results from MAST and prepare input for text inclusion analysis.
        """
        return self.run_analysis_on_json_data_impl()
