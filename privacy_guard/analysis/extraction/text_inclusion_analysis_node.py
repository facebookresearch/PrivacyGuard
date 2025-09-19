# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
import difflib
import string

from collections import defaultdict

from dataclasses import dataclass
from typing import Callable, cast, Dict, List, Optional, Tuple

import pandas as pd
import textdistance
from privacy_guard.analysis.base_analysis_node import BaseAnalysisNode
from privacy_guard.analysis.base_analysis_output import BaseAnalysisOutput
from privacy_guard.analysis.extraction.text_inclusion_analysis_input import (
    LCSBoundConfig,
    TextInclusionAnalysisInput,
)

from tqdm import tqdm

tqdm.pandas()


@dataclass
class TextInclusionAnalysisNodeOutput(BaseAnalysisOutput):
    """A dataclass to encapsulate the outputs of TextInclusionAnalysisNode."""

    num_samples: int
    exact_match: pd.Series
    inclusion_score: pd.Series
    longest_common_substring: Optional[pd.Series]
    longest_common_substring_false_pos: Optional[pd.Series]
    decision_targets_lcs: Optional[pd.Series]
    decision_targets_lcs_len: Optional[pd.Series]
    edit_similarity: Optional[pd.Series]
    edit_similarity_score: Optional[pd.Series]
    filtered_true_positive_list: list[str] | None
    augmented_output_dataset: pd.DataFrame


def _clean_text(text: str) -> str:
    """Normalizes text.

    - Lowercases
    - Removes punctuation
    - Turn newlines and tabs into spaces
    - Strips leading and trailing whitespace

    Modified from https://github.com/fairinternal/evals/compare/main...llama-4_competitor_speech_qwen2?fbclid=IwZXh0bgNhZW0CMTEAAR3oUDtIVDV8yd78te58ENCL8Z-GG-M-ugqKy3hfYmFq1CZGYix_UJJwsqM_aem_3k3KdFwz7gT2cf763e_7BQ#diff-c04f1fd4b9a8fb2ff76f7f048fb927b376cb9e148d4f04c2f2c8bbfd2ac13d32R79
    """
    # Lowercase
    text = text.lower()

    # Remove punctuation
    punctuation_remover = str.maketrans("", "", string.punctuation)
    cleaned_text = text.translate(punctuation_remover)

    # Turn newlines and tabs into spaces
    space_translator = str.maketrans("\n\t\r", "   ")
    cleaned_text = cleaned_text.translate(space_translator)

    # Strip leading and trailing whitespace
    cleaned_text = cleaned_text.strip()
    return cleaned_text


def _word_level_longest_common_subsequence_helper(
    s1: str, s2: str, autojunk: bool = True
) -> int:
    """
    Implementation of the longest common subsequence at word level.

    Output: number of words contained in the longest common subsequence.
    """

    # Split the string to words
    s1_list = s1.split()
    s2_list = s2.split()

    # Find matching blocks
    matcher = difflib.SequenceMatcher(None, s1_list, s2_list, autojunk=autojunk)
    matching_blocks = matcher.get_matching_blocks()

    # Initialize the length of matched words count
    matched_words_count = 0
    for block in matching_blocks:
        if block.size > 0:
            matched_words_count += block.size
    return matched_words_count


def _char_level_longest_common_subsequence_helper(
    s1: str, s2: str, autojunk: bool = True
) -> int:
    """
    Implementation of the longest common subsequence at character level.

    Output: number of characters contained in the longest common subsequence.
    """

    # Find matching blocks
    matcher = difflib.SequenceMatcher(None, s1, s2, autojunk=autojunk)
    matching_blocks = matcher.get_matching_blocks()

    # Initialize the length of matched chars count
    matched_chars_count = 0
    for block in matching_blocks:
        if block.size > 0:
            matched_chars_count += block.size
    return matched_chars_count


def _char_level_longest_common_substring_helper_bound(
    s1: str, s2: str, target: int = 150
) -> int:
    """
    To save on computation, check existance of a common substring of length target.
    Different from longest common subsequence (commonly known as LCS), longest common substring is the longest common CONSECUTIVE subsequence. Please use with caution.
    """

    max_length = 0
    # Iterate over the characters in the first string
    for i in range(len(s1)):
        j = target

        substring = s1[i : i + j]

        # Check if the current substring is in the second string and is longer
        # than the previous longest substring
        if substring in s2 and len(substring) > max_length:
            # Update the longest substring and its length
            max_length = len(substring)

            if max_length >= target:
                return max_length

    return 0


def _char_level_longest_common_substring_helper(s1: str, s2: str) -> int:
    """
    Implementation of the longest common substring at character level.
    Different from longest common subsequence (commonly known as LCS), longest common substring is the longest common CONSECUTIVE subsequence. Please use with caution.
    """

    # Initialize variables to store the longest common substring and its length

    max_length = 0
    # Iterate over the characters in the first string
    for i in range(len(s1)):
        # Iterate over the possible lengths of substrings
        for j in range(i + 1, len(s1) + 1):
            # Extract the current substring
            substring = s1[i:j]
            # Check if the current substring is in the second string and is longer
            # than the previous longest substring
            if substring in s2 and len(substring) > max_length:
                # Update the longest substring and its length
                max_length = len(substring)

    return max_length


def _normalize_by_target_len(scores: pd.Series, targets: pd.Series) -> pd.Series:
    """Normalized similarity metrics by target length."""
    lengths = targets.progress_apply(lambda x: len(_clean_text(x)))
    return scores / lengths


class TextInclusionAnalysisNode(BaseAnalysisNode):
    """TextInclusionAnalysisNode class for PrivacyGuard.

    Takes in a single dataframe containing prompt, target, and generation columns, and computes different inclusion scores
    such as "exact_match", "longest_common_substring".

    Additionally supports filtering true positives, in situations
    where the target is errantly included in the prompt text.

    NOTE: exact match and similarity are currently supported for single target only.

    Args:
        text_inclusion_analysis_input: AnalysisInputObject containing the
            prompt, target, and output_text columns.
    """

    LCS_METRIC_KEYS = [
        "lcs",
        "lcs_score",
        "fp",
        "fp_score",
        "decision_targets_lcs",
        "decision_targets_lcs_len",
    ]

    def __init__(self, analysis_input: TextInclusionAnalysisInput) -> None:
        """
        args:
            user_aggregation: specifies user aggregation strategy
        """
        self.prompt_key: str = analysis_input.prompt_key
        self.generation_key: str = analysis_input.generation_key

        self.target_key: str = analysis_input.target_key
        self.target_set_key: str = self.target_key + "_set"
        self.generation_df: pd.DataFrame = analysis_input.generation_df

        self.generation_df[self.target_set_key] = self.generation_df[
            self.target_key
        ].apply(lambda x: {x} if isinstance(x, str) else set(x))

        self.generation_df["num_unique_targets"] = self.generation_df[
            self.target_set_key
        ].apply(lambda x: len(x))

        super().__init__(analysis_input=analysis_input)

    def _compute_edit_similarity(
        self, row: pd.Series, s1_column: str | None = None, s2_column: str | None = None
    ) -> int:
        """Compute edit similarity between target and generation text. Texts are cleaned first.
        Currently not supported for multi target mode.

        Args:
            row (pd.Series): A row of a DataFrame containing the s1 and s2 columns.

        Returns:
            int: Edit similarity between the two strings.
        """
        s1 = _clean_text(row[s1_column or self.target_key])
        s2 = _clean_text(row[s2_column or self.generation_key])
        levenshtein = textdistance.levenshtein.similarity(s1, s2)
        return levenshtein

    def _compute_exact_match(self, row: pd.Series) -> bool:
        """Returns true if ANY target is exactly the same as the output_text.

        Texts are NOT cleaned first except for stripping leading and trailing whitespace.

        Args:
            row (pd.Series): A row of a DataFrame containing the self.target_key and self.generation_key columns.

        Returns:
            bool: True if the target is exactly the same as the output_text, False otherwise.
        """
        return row[self.target_key].strip() == row[self.generation_key].strip()

    def _compute_inclusion_score(self, row: pd.Series) -> bool:
        """Returns true if the target is included in the output_text. Texts are cleaned first.

        Args:
            row (pd.Series): A row of a DataFrame containing the self.target_key and self.generation_key columns.

        Returns:
            bool: True if the target is included in the output_text, False otherwise.
        """
        s1 = _clean_text(row[self.target_key])
        s2 = _clean_text(row[self.generation_key])
        return s1 in s2

    def get_compute_longest_common_substring_map(
        self,
        comparison_key: str = "output_text",
        false_positive_key: str = "prompt",
        lcs_bound_config: LCSBoundConfig | None = None,
    ) -> Callable[
        [pd.Series],
        Dict[str, Dict[str, List[float] | List[int] | List[Tuple[float, float]]]],
    ]:
        """Produces the lcs function for a target against the comparison column, and the
        false positive column.

        Args:
            comparison_key (str): The key for the value to compare against the target(s)
            false_positive_key (str): The key for the value to check against false positive
            lcs_bound_config LCSBoundConfig | None: whether to bound LCS computation
                for computational efficiency. If none, LCS is not bounded.

        Returns:
            Callable method to produce Dict containing:
                lcs (Dict[str, int]): Dict mapping from target to longest common substring
                lcs_score (Dict[str, float]): Dict mapping from target to longest_common_substring scores
                fp (Dict[str, int]): Dict mapping from target to lcs w/ prompt
                fp_score (Dict[str, float]): Dict mapping from target to false positive lcs scores
                decision_targets_lcs (Dict[str, float]): Dict mapping from target to (lcs, fp) scores
                decision_targets_lcs_len (Dict[str, float]): Dict mapping from target to (lcs, fp)
        """

        def _compute_longest_common_substring_map(
            row: pd.Series,
        ) -> Dict[str, Dict[str, List[float] | List[int] | List[Tuple[float, float]]]]:
            """Find the longest common substring between two strings. Texts are cleaned first.

            Args:
                row (pd.Series): A row of a DataFrame containing the prompt, target, and generation columns.

            Returns Dict containing:
                lcs (Dict[str, float]): Dict mapping from target to longest common substring
                lcs_score (Dict[str, float]): Dict mapping from target to longest_common_substring scores
                fp (Dict[str, float]): Dict mapping from target to lcs w/ prompt
                fp_score (Dict[str, float]): Dict mapping from target to false positive lcs scores
                decision_targets_lcs (Dict[str, float]): Dict mapping from target to (lcs, fp)
                decision_targets_lcs_len (Dict[str, float]): Dict mapping from target to (lcs, fp)
            """
            output_dict = defaultdict(dict)

            target_set = row[self.target_set_key]

            comparison_text = _clean_text(row[comparison_key])
            fp_text = _clean_text(row[false_positive_key])

            for target in target_set:
                clean_target = _clean_text(target)

                if lcs_bound_config is not None:
                    lcs = _char_level_longest_common_substring_helper_bound(
                        s1=clean_target,
                        s2=comparison_text,
                        target=lcs_bound_config.lcs_len_target,
                    )
                else:
                    lcs = _char_level_longest_common_substring_helper(
                        s1=clean_target, s2=comparison_text
                    )
                output_dict["lcs"][clean_target] = lcs
                lcs_score = -1 if len(clean_target) <= 0 else lcs / len(clean_target)
                output_dict["lcs_score"][clean_target] = lcs_score

                if lcs_bound_config is not None:
                    fp = _char_level_longest_common_substring_helper_bound(
                        s1=clean_target,
                        s2=fp_text,
                        target=lcs_bound_config.fp_len_target,
                    )
                else:
                    fp = _char_level_longest_common_substring_helper(
                        s1=clean_target, s2=fp_text
                    )
                output_dict["fp"][clean_target] = fp

                fp_score = -1 if len(clean_target) <= 0 else fp / len(clean_target)
                output_dict["fp_score"][clean_target] = fp_score

                output_dict["decision_targets_lcs"][clean_target] = (
                    lcs_score,
                    fp_score,
                )
                output_dict["decision_targets_lcs_len"][clean_target] = (
                    lcs,
                    fp,
                )

            return dict(output_dict)

        return _compute_longest_common_substring_map

    def run_analysis(self) -> TextInclusionAnalysisNodeOutput:
        analysis_input: TextInclusionAnalysisInput = cast(
            TextInclusionAnalysisInput, self.analysis_input
        )
        generation_df = analysis_input.generation_df

        # Exact match
        if not analysis_input.disable_exact_match:
            exact_match = generation_df.progress_apply(
                self._compute_exact_match, axis=1
            )
            # Inclusion score
            inclusion_score = generation_df.progress_apply(
                self._compute_inclusion_score, axis=1
            )
        else:
            exact_match = pd.Series()
            inclusion_score = pd.Series()

        outputs = TextInclusionAnalysisNodeOutput(
            num_samples=len(generation_df),
            exact_match=exact_match,
            inclusion_score=inclusion_score,
            longest_common_substring=None,
            longest_common_substring_false_pos=None,
            decision_targets_lcs=None,
            decision_targets_lcs_len=None,
            edit_similarity=None,
            edit_similarity_score=None,
            filtered_true_positive_list=None,
            augmented_output_dataset=generation_df,
        )

        if not analysis_input.disable_lcs:
            # Longest common substring

            lcs_result = generation_df.progress_apply(
                self.get_compute_longest_common_substring_map(
                    comparison_key=analysis_input.generation_key,
                    false_positive_key=analysis_input.prompt_key,
                    lcs_bound_config=analysis_input.lcs_bound_config,
                ),
                axis=1,
            )

            for lcs_metric_key in self.LCS_METRIC_KEYS:
                generation_df[lcs_metric_key] = lcs_result.progress_apply(
                    lambda lcs_result, metric_key=lcs_metric_key: lcs_result[metric_key]
                )

            outputs.decision_targets_lcs = generation_df["decision_targets_lcs"]
            outputs.decision_targets_lcs_len = generation_df["decision_targets_lcs_len"]
            outputs.longest_common_substring = generation_df["lcs"]
            outputs.longest_common_substring_false_pos = generation_df["fp"]

        if not analysis_input.disable_similarity:
            # Edit similarity
            generation_df["edit_similarity"] = generation_df.progress_apply(
                self._compute_edit_similarity, axis=1
            )
            generation_df["edit_similarity_score"] = _normalize_by_target_len(
                generation_df["edit_similarity"], generation_df["target"]
            )

            outputs.edit_similarity = generation_df["edit_similarity"]
            outputs.edit_similarity_score = generation_df["edit_similarity_score"]

        return outputs
