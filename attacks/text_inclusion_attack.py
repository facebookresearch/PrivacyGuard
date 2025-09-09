# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import logging
import os

from typing import Dict

import pandas as pd

from privacy_guard.analysis.extraction.text_inclusion_analysis_input import (
    LCSBoundConfig,
    TextInclusionAnalysisInput,
    TextInclusionAnalysisInputBatch,
)

from privacy_guard.attacks.base_attack import BaseAttack


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
            llm_generation_file: Path to .jsonl file with prompts, targets, and generations
            data: Alternatively, can pass in dataframe directly.
            num_rows: If provided, only run analysis on the first num_rows rows of each result.
            bound_lcs: If True, bound the LCS computation to the target threshold length.
        """

        self.llm_generation_file = llm_generation_file

        if self.llm_generation_file is not None and data is None:
            assert ".jsonl" in self.llm_generation_file
            if os.access(self.llm_generation_file, os.W_OK):
                logging.warning(
                    f"WARNING: Write permission should not be allowed for {self.llm_generation_file}."
                    "\n This risks overwriting or removing the file and losing Eval results permanently."
                )

            logging.info(f"Loading LLM generation file... '{self.llm_generation_file}'")
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
            try:
                prompt = row["prompt"]
            except KeyError:
                logging.warning(
                    f"Prompt column not found in data file {self.llm_generation_file}."
                    " Double check that your directory is pointing to the LLM generation files."
                )
                raise
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
            target_key="target",
            lcs_bound_config=lcs_bound_config,
        )
        return analysis_input

    def run_attack(self) -> TextInclusionAnalysisInput:
        """
        Prepare input for text inclusion analysis.
        """
        return self.run_analysis_on_json_data_impl()


class TextInclusionAttackBatch(BaseAttack):
    """
    TextInclusionAttackBatch loads a directory of LLM generation files, and runs TextInclusionAttack
    on each file, preparing them for TextInclusionAnalysisNode.

    Batch is used to signify that a group of LLM generation files are meant to be analyzed together.
    For example, if mutliple prompt types or datasets were used on the same model, they can be processed
    as a batch to compare the dataset's performance and keep the outputs together.
    """

    def __init__(
        self,
        dump_dir_str: str,
        result_name_filter: str | None = None,
        num_rows: int | None = None,
        bound_lcs: bool = False,
    ) -> None:
        """
        args:
            dump_dir_str: Directory path containing raw_result LLM generation files
            result_name_filter: If provided, only run analysis on results that match this filter.
            num_rows: If provided, only run analysis on the first num_rows rows of each result.
            bound_lcs: If True, bound the LCS computation to the target threshold length.
        """
        self.dump_dir_str = dump_dir_str
        self.result_name_filter: str | None = result_name_filter

        self.num_rows = num_rows
        self.bound_lcs = bound_lcs

    def load_results_from_mnt(self) -> TextInclusionAnalysisInputBatch:
        dump_dir_str = self.dump_dir_str
        assert os.path.isdir(dump_dir_str)

        result_files = []
        for root, _, dump_dir_file_list in os.walk(dump_dir_str):
            for dump_dir_file in dump_dir_file_list:
                # Can't use list comprehension because os.walk is a generator.
                file_matches_filter = (
                    not self.result_name_filter
                    or self.result_name_filter in dump_dir_file
                )
                if ".jsonl" in dump_dir_file and file_matches_filter:
                    full_result_path = os.path.join(root, dump_dir_file)
                    logging.info(f"File matches filter: {full_result_path}")
                    result_files.append(str(full_result_path))

        if len(result_files) == 0:
            raise ValueError(
                f"No analysis results found in {dump_dir_str} that match filter {self.result_name_filter}."
            )

        analysis_input_dict: Dict[str, TextInclusionAnalysisInput] = {}

        for full_result_path in result_files:
            logging.info(f"Executing Attack on '{full_result_path}'...")
            result_path = full_result_path.split(dump_dir_str + "/")[-1].replace(
                "/", "."
            )

            text_inclusion_attack = TextInclusionAttack(
                llm_generation_file=full_result_path,
                num_rows=self.num_rows,
                bound_lcs=self.bound_lcs,
            )
            attack_result = text_inclusion_attack.run_attack()
            analysis_input_dict[result_path] = attack_result

        return TextInclusionAnalysisInputBatch(input_batch=analysis_input_dict)

    def run_attack(self) -> TextInclusionAnalysisInputBatch:
        """
        Pull results from mount directory and prepare input for text inclusion analysis.
        """

        analysis_input_batch = self.load_results_from_mnt()

        return analysis_input_batch
