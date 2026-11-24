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

from collections.abc import Mapping
from typing import Any

import pandas as pd
from privacy_guard.analysis.base_analysis_input import BaseAnalysisInput
from privacy_guard.analysis.jacobian_lens.preview_schema import JLensPreviewReceipt


TRACKED_RANK_COLUMNS: tuple[str, ...] = (
    "source",
    "method",
    "layer",
    "position",
    "concept",
    "token_id",
    "text",
    "rank",
    "score",
)


class JacobianLensPreviewInput(BaseAnalysisInput):
    """
    Input for ``JacobianLensPreviewNode``: one validated
    ``privacyguard.jlens_preview/v1`` receipt.

    ``df_train_user`` is a long table of tracked-token ranks, one row per
    (method, layer, position, tracked token) of available cells, plus rows with
    source ``intervention:<name>`` for intervention readout cells.
    ``df_test_user`` is empty; there is no train/test split in this analysis.

    args:
        receipt: a ``JLensPreviewReceipt`` or a receipt dict (validated on load)
    """

    def __init__(self, receipt: JLensPreviewReceipt | Mapping[str, Any]) -> None:
        self._receipt: JLensPreviewReceipt = (
            receipt
            if isinstance(receipt, JLensPreviewReceipt)
            else JLensPreviewReceipt.from_dict(receipt)
        )
        super().__init__(
            df_train_user=_tracked_rank_table(self._receipt),
            df_test_user=pd.DataFrame(),
        )

    @property
    def receipt(self) -> JLensPreviewReceipt:
        return self._receipt

    @property
    def tracked_ranks(self) -> pd.DataFrame:
        return self._df_train_user


def _tracked_rank_table(receipt: JLensPreviewReceipt) -> pd.DataFrame:
    concept_of = {t.token_id: t.concept for t in receipt.tracked_tokens}
    sources = [("receipt", receipt.cells)] + [
        (f"intervention:{run.name}", run.readout_cells) for run in receipt.interventions
    ]
    rows = [
        (
            source,
            cell.method,
            cell.layer,
            cell.position,
            concept_of[entry.token_id],
            entry.token_id,
            entry.text,
            entry.rank,
            entry.score,
        )
        for source, cells in sources
        for cell in cells
        if cell.status == "ok"
        for entry in cell.tracked
    ]
    return pd.DataFrame(rows, columns=list(TRACKED_RANK_COLUMNS))
