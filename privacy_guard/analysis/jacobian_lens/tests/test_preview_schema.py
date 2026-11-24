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

import copy
import unittest
from collections.abc import Callable
from typing import Any

from privacy_guard.analysis.jacobian_lens.preview_fixture import fixture_receipt_dict
from privacy_guard.analysis.jacobian_lens.preview_schema import (
    GLIMMER_TARGET_REPO_ID,
    JLensPreviewReceipt,
    ReceiptValidationError,
)


def _first_ok_cell(data: dict[str, Any]) -> dict[str, Any]:
    return next(c for c in data["cells"] if c["status"] == "ok")


def _as_glimmer(data: dict[str, Any]) -> None:
    data["run_kind"] = "real_model"
    data["model_role"] = "glimmer_target"
    data["model_identity"]["repo_id"] = GLIMMER_TARGET_REPO_ID
    data["model_identity"]["dtype"] = "bfloat16"


def _add_activations(d: dict[str, Any]) -> None:
    d["cells"][0]["activations"] = [0.0]


def _add_vector(d: dict[str, Any]) -> None:
    d["numerical_receipt"]["probe"] = [0.5] * 6656


def _assistant_drafter(d: dict[str, Any]) -> None:
    _as_glimmer(d)
    d["model_identity"]["repo_id"] = GLIMMER_TARGET_REPO_ID + "-assistant"


def _quantized(d: dict[str, Any]) -> None:
    _as_glimmer(d)
    d["model_identity"]["dtype"] = "int4"


def _fixture_called_glimmer(d: dict[str, Any]) -> None:
    _as_glimmer(d)
    d["run_kind"] = "cpu_fixture"


def _missing_tokenizer_hash(d: dict[str, Any]) -> None:
    del d["model_identity"]["tokenizer_sha256"]


def _branch_revision(d: dict[str, Any]) -> None:
    _as_glimmer(d)
    d["model_identity"]["revision_sha"] = "main"


def _unhashed_shards(d: dict[str, Any]) -> None:
    _as_glimmer(d)
    d["model_identity"]["safetensors_sha256_index"] = {"model.safetensors": "x"}


def _skip_rank(d: dict[str, Any]) -> None:
    _first_ok_cell(d)["top"][1]["rank"] = 3


def _duplicate_cell(d: dict[str, Any]) -> None:
    d["cells"].append(copy.deepcopy(d["cells"][0]))


def _prompt_hash(d: dict[str, Any]) -> None:
    d["prompt"]["text"] += " "


def _silent_unavailable(d: dict[str, Any]) -> None:
    cell = next(c for c in d["cells"] if c["status"] == "unavailable")
    cell["unavailable_reason"] = None


def _tracked_disagrees(d: dict[str, Any]) -> None:
    cell = _first_ok_cell(d)
    top_ids = {t["token_id"]: t["rank"] for t in cell["top"]}
    entry = next(t for t in cell["tracked"] if t["token_id"] in top_ids)
    entry["rank"] = top_ids[entry["token_id"]] + 1


def _missing_tracked(d: dict[str, Any]) -> None:
    _first_ok_cell(d)["tracked"].pop()


def _layer_out_of_range(d: dict[str, Any]) -> None:
    _first_ok_cell(d)["layer"] = d["num_layers"]


def _wrong_schema(d: dict[str, Any]) -> None:
    d["schema_version"] = "privacyguard.jlens_preview/v0"


def _bos_mismatch(d: dict[str, Any]) -> None:
    d["prompt"]["bos_token_id"] = 1


def _two_primary_arms(d: dict[str, Any]) -> None:
    d["interventions"].append(dict(d["interventions"][0], name="second"))


def _primary_control(d: dict[str, Any]) -> None:
    d["interventions"][1]["primary"] = True


def _primary_unavailable_control(d: dict[str, Any]) -> None:
    d["interventions"][2]["primary"] = True


def _untracked_next_token(d: dict[str, Any]) -> None:
    d["actual_output"]["next_token_tracked"][0]["token_id"] = 12


def _duplicate_run_name(d: dict[str, Any]) -> None:
    d["interventions"][1]["name"] = d["interventions"][0]["name"]


def _reserved_run_name(d: dict[str, Any]) -> None:
    d["interventions"][1]["name"] = "actual_output"


def _unavailable_run_with_results(d: dict[str, Any]) -> None:
    d["interventions"][2]["readout_cells"] = d["interventions"][0]["readout_cells"]


def _whitespace_tracked_token(d: dict[str, Any]) -> None:
    d["tracked_tokens"][0]["text"] = " "


def _local_path_in_code(d: dict[str, Any]) -> None:
    d["code"]["run_command"] = "python -m x --out-dir /srv/someone/out"


def _local_path_as_key(d: dict[str, Any]) -> None:
    d["display"]["/srv/someone/receipts.json"] = "source"


def _local_path_in_notes(d: dict[str, Any]) -> None:
    d["interventions"][0]["notes"] = "see /srv/someone/ckpt/run3"


def _null_interventions(d: dict[str, Any]) -> None:
    d["interventions"] = None


def _scalar_source_layers(d: dict[str, Any]) -> None:
    d["lens"]["source_layers"] = 3


def _full_vocab_top_k(d: dict[str, Any]) -> None:
    d["top_k"] = 101


def _duplicate_tracked_entry(d: dict[str, Any]) -> None:
    cell = _first_ok_cell(d)
    cell["tracked"][1] = dict(cell["tracked"][0], rank=cell["tracked"][0]["rank"] + 1)


def _jlens_without_lens(d: dict[str, Any]) -> None:
    d["lens"] = None


def _jlens_unfitted_layer(d: dict[str, Any]) -> None:
    d["lens"]["source_layers"] = [2]


def _chunked_vector(d: dict[str, Any]) -> None:
    d["numerical_receipt"]["chunks"] = [[0.5] * 512 for _ in range(13)]


class PreviewSchemaTest(unittest.TestCase):
    def test_round_trip_and_hash(self) -> None:
        data = fixture_receipt_dict()
        receipt = JLensPreviewReceipt.from_dict(data)
        self.assertEqual(receipt.to_dict(), data)
        self.assertEqual(receipt.receipt_sha256, receipt.content_sha256())
        self.assertEqual(len(receipt.content_sha256()), 64)

        tampered = copy.deepcopy(data)
        tampered["actual_output"]["greedy_text"] = " 6."
        with self.assertRaisesRegex(ReceiptValidationError, "was modified"):
            JLensPreviewReceipt.from_dict(tampered)

        unhashed = copy.deepcopy(tampered)
        unhashed["receipt_sha256"] = None
        self.assertIsNone(JLensPreviewReceipt.from_dict(unhashed).receipt_sha256)

    def test_rejects_private_payloads_and_bad_identity(self) -> None:
        cases: dict[str, Callable[[dict[str, Any]], None]] = {
            "forbidden key": _add_activations,
            "raw vector": _add_vector,
            "assistant drafter": _assistant_drafter,
            "non-bf16 glimmer": _quantized,
            "fixture labeled glimmer": _fixture_called_glimmer,
            "missing tokenizer hash": _missing_tokenizer_hash,
            "branch revision": _branch_revision,
            "unhashed shards": _unhashed_shards,
        }
        for name, mutate in cases.items():
            with self.subTest(name):
                data = fixture_receipt_dict()
                data["receipt_sha256"] = None
                mutate(data)
                with self.assertRaises(ReceiptValidationError):
                    JLensPreviewReceipt.from_dict(data)

        data = fixture_receipt_dict()
        data["receipt_sha256"] = None
        _as_glimmer(data)
        self.assertEqual(
            JLensPreviewReceipt.from_dict(data).model_role, "glimmer_target"
        )

    def test_rejects_inconsistent_readouts(self) -> None:
        cases: dict[str, Callable[[dict[str, Any]], None]] = {
            "rank gap": _skip_rank,
            "duplicate cell": _duplicate_cell,
            "prompt hash": _prompt_hash,
            "unavailable without reason": _silent_unavailable,
            "tracked rank disagrees": _tracked_disagrees,
            "missing tracked": _missing_tracked,
            "layer range": _layer_out_of_range,
            "schema version": _wrong_schema,
            "bos mismatch": _bos_mismatch,
            "two primary arms": _two_primary_arms,
            "primary control": _primary_control,
            "primary unavailable control": _primary_unavailable_control,
            "untracked next token": _untracked_next_token,
            "duplicate run name": _duplicate_run_name,
            "reserved run name": _reserved_run_name,
            "unavailable run with results": _unavailable_run_with_results,
            "whitespace tracked token": _whitespace_tracked_token,
            "local path in code": _local_path_in_code,
            "local path as key": _local_path_as_key,
            "local path in notes": _local_path_in_notes,
            "null interventions": _null_interventions,
            "scalar source layers": _scalar_source_layers,
            "top_k too large": _full_vocab_top_k,
            "duplicate tracked entry": _duplicate_tracked_entry,
            "jlens without lens": _jlens_without_lens,
            "jlens unfitted layer": _jlens_unfitted_layer,
            "chunked vector": _chunked_vector,
        }
        for name, mutate in cases.items():
            with self.subTest(name):
                data = fixture_receipt_dict()
                data["receipt_sha256"] = None
                mutate(data)
                with self.assertRaises(ReceiptValidationError):
                    JLensPreviewReceipt.from_dict(data)
