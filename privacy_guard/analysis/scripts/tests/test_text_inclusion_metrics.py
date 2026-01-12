# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-strict
import tempfile
import unittest

import pandas as pd
from privacy_guard.analysis.scripts.text_inclusion_metrics import (
    dump_augmented_df,
    longest_common_substring_decision,
    longest_common_substring_decision_copyright,
    run_analysis_on_json_data,
    run_analysis_on_json_data_impl,
)


class TestTextInclusionScript(unittest.TestCase):
    def setUp(self) -> None:
        self.data = {
            "prompt": [
                "This is a test prompt",
                "This is another test prompt",
            ],
            "targets": [
                ["Target text 1"],
                ["Target text 2", "Another target"],
            ],
            "prediction": [
                "A success: Target text 1",
                "Failure, no match. ",
            ],
        }

        self.sft_data = {
            "prompt": [
                {
                    "type": "SampleSFT",
                    "dialog": [{"body": "This is a test prompt", "source": "user"}],
                },
                {
                    "type": "SampleSFT",
                    "dialog": [
                        {"body": "This is another test prompt", "source": "user"}
                    ],
                },
            ],
            "targets": [
                ["Target text 1"],
                ["Target text 2", "Another target"],
            ],
            "prediction": [
                "A success: Target text 1",
                "Failure, no match. ",
            ],
        }

        self.intermediate_augmented_data: str = (
            """
        {"prompt":"CHANGED FOR TEST","targets":["Target text 1"],"prediction":"A success: Target text 1","output_text":"A success:"""
            + """ Target text 1","target":"Target text 1","targets_set":["Target text 1"],"num_unique_targets":1,"lcs":{"target """
            + """text 1":13},"lcs_score":{"target text 1":1.0},"fp":{"target text 1":3},"fp_score":{"target text 1":0.2307692308},"decision_targets_"""
            + """lcs":{"target text 1":[1.0,0.2307692308]},"decision_targets_lcs_len":{"target text 1":[0,0]}}
        {"prompt":"CHANGED FOR TEST 2","targets":["Target text 2","Another target"],"prediction":"Failure, """
            + """no match. ","output_text":"Failure, no match. ","target":"Target text 2","targets_set":["Target text 2","Another target"],"num_unique_"""
            + """targets":2,"lcs":{"target text 2":1,"another target":2},"lcs_score":{"target text 2":0.0769230769,"another """
            + """target":0.1428571429},"fp":{"target text 2":3,"another target":9},"fp_score":{"target text 2":0.2307692308,"another target":0.6428571429},"decision_"""
            + """targets_lcs":{"target text 2":[0.0769230769,0.2307692308],"another target":[0.1428571429,0.6428571429]}"""
            + ""","decision_targets_lcs_len":{"target text 2":[5,5],"another target":[5,5]}}
        """
        )

        self.temp_input_file = tempfile.NamedTemporaryFile(
            prefix="test_input", suffix=".jsonl", mode="w"
        )
        self.temp_input_file_name = self.temp_input_file.name
        self.temp_input_file.write(
            pd.DataFrame(self.data).to_json(orient="records", lines=True)
        )
        self.temp_input_file.flush()
        self.temp_input_file.seek(0)

        self.temp_sft_input_file = tempfile.NamedTemporaryFile(
            prefix="test_input", suffix=".jsonl", mode="w"
        )
        self.temp_sft_input_file_name = self.temp_sft_input_file.name
        self.temp_sft_input_file.write(
            pd.DataFrame(self.sft_data).to_json(orient="records", lines=True)
        )
        self.temp_sft_input_file.flush()
        self.temp_sft_input_file.seek(0)

        self.temp_output_file = tempfile.NamedTemporaryFile(
            prefix="test_output", suffix=".jsonl", mode="w"
        )
        self.temp_output_file_name = self.temp_output_file.name

        self.final_output_file = tempfile.NamedTemporaryFile(
            prefix="test_final_output", suffix=".jsonl", mode="w"
        )
        self.final_output_file_name = self.final_output_file.name

        self.temp_intermediate_output_file = tempfile.NamedTemporaryFile(
            prefix="test_intermediate_output", suffix=".jsonl", mode="w"
        )

        self.temp_intermediate_output_file_name = (
            self.temp_intermediate_output_file.name
        )
        self.temp_intermediate_output_file.write(self.intermediate_augmented_data)
        self.temp_intermediate_output_file.flush()
        self.temp_intermediate_output_file.seek(0)

        super().setUp()

    def test_run_analysis_on_json_data_overwrite_intermediate(self) -> None:
        result_df = run_analysis_on_json_data(
            jsonl_input_path=self.temp_input_file_name,
            jsonl_output_path=self.temp_intermediate_output_file_name,
            final_output_path=self.final_output_file_name,
            recompute_augmented_df=True,
        )

        # Check that final result was computed from intermediate dataframe.
        self.assertEqual(result_df.iloc[0]["prompt"], "This is a test prompt")

    def test_run_analysis_on_json_data_sft_mode(self) -> None:
        result_df = run_analysis_on_json_data(
            jsonl_input_path=self.temp_sft_input_file_name,
            jsonl_output_path=self.temp_intermediate_output_file_name,
            final_output_path=self.final_output_file_name,
            recompute_augmented_df=True,
        )

        # Check that final result was computed from intermediate dataframe.
        self.assertEqual(result_df.iloc[0]["prompt"], "This is a test prompt")

    def test_run_analysis_on_json_data_load_from_intermediate(self) -> None:
        result_df = run_analysis_on_json_data(
            jsonl_input_path=self.temp_input_file_name,
            jsonl_output_path=self.temp_intermediate_output_file_name,
            final_output_path=self.final_output_file_name,
            recompute_augmented_df=False,
        )

        # Check that final result was computed from intermediate dataframe.
        self.assertEqual(result_df.iloc[0]["prompt"], "CHANGED FOR TEST")

    def test_run_analysis_on_json_data_impl_num_rows(self) -> None:
        results = run_analysis_on_json_data_impl(self.temp_input_file_name, num_rows=1)

        self.assertIn("augmented_output_dataset", results)
        self.assertEqual(len(results["augmented_output_dataset"]), 1)

    def test_run_analysis_on_json_data_impl(self) -> None:
        results = run_analysis_on_json_data_impl(self.temp_input_file_name)

        self.assertIn("augmented_output_dataset", results)

    def test_dump_results(self) -> None:
        results = run_analysis_on_json_data_impl(self.temp_input_file_name)

        self.assertIn("augmented_output_dataset", results)

        dump_augmented_df(
            df=results["augmented_output_dataset"],
            jsonl_output_path=self.temp_output_file_name,
        )

    def test_longest_common_substring_decision_copyright(self) -> None:
        lcs_data = pd.DataFrame(
            {
                "decision_targets_lcs_len": [
                    {"target text 1": (160, 40)},
                    {"target text 2": (170, 80)},
                ],
            }
        )
        decision_data = longest_common_substring_decision_copyright(
            augmented_df=lcs_data
        )

        self.assertEqual(decision_data.iloc[0]["decision_prompt"], True)
        self.assertEqual(decision_data.iloc[0]["is_false_positive"], False)

        self.assertEqual(decision_data.iloc[1]["decision_prompt"], False)
        self.assertEqual(decision_data.iloc[1]["is_false_positive"], True)

    def test_longest_common_substring_decision_audience_expansion(self) -> None:
        lcs_data = pd.DataFrame(
            {
                "decision_targets_lcs": [
                    {"true pos": (0.9, 0.3)},
                    {"false pos": (0.85, 0.8)},
                    {"no match": (0.3, 0.3)},
                ],
            }
        )
        decision_data = longest_common_substring_decision(augmented_df=lcs_data)

        self.assertEqual(decision_data.iloc[0]["decision_prompt"], True)
        self.assertEqual(decision_data.iloc[0]["is_false_positive"], False)

        self.assertEqual(decision_data.iloc[1]["decision_prompt"], False)
        self.assertEqual(decision_data.iloc[1]["is_false_positive"], True)

        self.assertEqual(decision_data.iloc[2]["decision_prompt"], False)
        self.assertEqual(decision_data.iloc[2]["is_false_positive"], False)
