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
import os
import tempfile
import unittest
from typing import Callable, Dict, List, Tuple

import pandas as pd
from privacy_guard.attacks.extraction.utils.data_utils import load_data, save_results


class TestDataUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.test_data: List[Dict[str, str]] = [
            {"id": "1", "text": "This is test text 1", "label": "A"},
            {"id": "2", "text": "This is test text 2", "label": "B"},
            {"id": "3", "text": "This is test text 3", "label": "C"},
        ]
        self.df = pd.DataFrame(self.test_data)

        self.temp_dir = tempfile.TemporaryDirectory()

        self.formats: List[
            Tuple[
                str,
                str,
                Callable[[pd.DataFrame, str], None],
                Callable[[str], pd.DataFrame],
            ]
        ] = [
            (
                "jsonl",
                ".jsonl",
                lambda df, path: df.to_json(path, orient="records", lines=True),
                lambda path: pd.read_json(path, lines=True),
            ),
            (
                "csv",
                ".csv",
                lambda df, path: df.to_csv(path, index=False),
                lambda path: pd.read_csv(path),
            ),
            (
                "json",
                ".json",
                lambda df, path: df.to_json(path, orient="records"),
                lambda path: pd.read_json(path),
            ),
        ]

        self.test_files = {}
        for format_name, extension, write_func, _ in self.formats:
            file_path = os.path.join(
                self.temp_dir.name, f"test_{format_name}{extension}"
            )
            write_func(self.df, file_path)
            self.test_files[format_name] = file_path

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_load_data(self) -> None:
        for format_name, file_path in self.test_files.items():
            with self.subTest(format=format_name):
                df = load_data(file_path, format=format_name)

                self.assertEqual(len(df), len(self.test_data))
                self.assertEqual(list(df.columns), ["id", "text", "label"])

                if format_name == "jsonl":
                    self.assertEqual(
                        df["text"].tolist(),
                        [
                            "This is test text 1",
                            "This is test text 2",
                            "This is test text 3",
                        ],
                    )
                    self.assertEqual(df["label"].tolist(), ["A", "B", "C"])

    def test_load_data_invalid_format(self) -> None:
        with self.assertRaises(ValueError):
            load_data(next(iter(self.test_files.values())), format="invalid")

    def test_save_results(self) -> None:
        for format_name, extension, _, read_func in self.formats:
            with self.subTest(format=format_name):
                output_path = os.path.join(
                    self.temp_dir.name, f"output_{format_name}{extension}"
                )

                save_results(self.df, output_path, format=format_name)

                self.assertTrue(os.path.exists(output_path))

                loaded_df = read_func(output_path)
                self.assertEqual(len(loaded_df), len(self.df))
                self.assertEqual(list(loaded_df.columns), list(self.df.columns))

    def test_save_results_invalid_format(self) -> None:
        output_path = os.path.join(self.temp_dir.name, "output_invalid.txt")
        with self.assertRaises(ValueError):
            save_results(self.df, output_path, format="invalid")

    # Note: We're not testing load_model_and_tokenizer here as it requires
    # actual model files and would make the test slow and dependent on external resources.
