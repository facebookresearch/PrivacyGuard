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

"""
Render a self-contained J-lens preview bundle from receipt JSON files.

    python -m privacy_guard.analysis.jacobian_lens.preview_render_main \\
        --receipts receipts.json --out-dir preview/

Each receipt file holds one receipt object or a list of them. ``--fixture``
renders the labeled synthetic ``cpu_fixture`` receipt instead.
"""

import argparse
import gzip
import json
import logging
import sys
from collections.abc import Sequence

from privacy_guard.analysis.jacobian_lens.preview_analysis_node import (
    HEAD_SCORE_TOLERANCE,
)
from privacy_guard.analysis.jacobian_lens.preview_fixture import make_fixture_receipt
from privacy_guard.analysis.jacobian_lens.preview_schema import JLensPreviewReceipt
from privacy_guard.analysis.jacobian_lens.preview_trim import trim_receipt
from privacy_guard.analysis.jacobian_lens.preview_viewer import (
    DEFAULT_CONTACT_LINE,
    DEFAULT_TITLE,
    write_preview_bundle,
)


logger: logging.Logger = logging.getLogger(__name__)


def load_receipts(paths: Sequence[str]) -> list[JLensPreviewReceipt]:
    """Load and validate receipts; each file (optionally ``.gz``) holds one
    receipt or a list."""
    receipts = []
    for path in paths:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else [data]
        receipts.extend(JLensPreviewReceipt.from_dict(item) for item in items)
    return receipts


def _trim(
    receipt: JLensPreviewReceipt, args: argparse.Namespace
) -> JLensPreviewReceipt:
    keep = None
    if args.keep_readouts is not None:
        primary = {r.name for r in receipt.interventions if r.primary}
        keep = set(args.keep_readouts) - {"primary"}
        if "primary" in args.keep_readouts:
            if receipt.interventions and not primary:
                raise ValueError(
                    f"--keep-readouts primary: receipt {receipt.prompt.label!r} "
                    "has interventions but no primary arm"
                )
            keep |= primary
    top_k = receipt.top_k if args.trim_top_k is None else args.trim_top_k
    return trim_receipt(receipt, top_k, args.score_decimals, keep)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--receipts", nargs="+", help="receipt JSON files")
    source.add_argument(
        "--fixture",
        action="store_true",
        help="render the synthetic cpu_fixture receipt",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--contact-line", default=DEFAULT_CONTACT_LINE)
    parser.add_argument(
        "--trim-top-k", type=int, default=None, help="shorten every top list"
    )
    parser.add_argument(
        "--score-decimals", type=int, default=None, help="round scores when trimming"
    )
    parser.add_argument(
        "--keep-readouts",
        nargs="*",
        default=None,
        help="run names whose readout cells are kept when trimming; 'primary' "
        "names the predeclared primary arm; omit to keep all",
    )
    parser.add_argument(
        "--head-score-tolerance",
        type=float,
        default=HEAD_SCORE_TOLERANCE,
        help="max |score diff| between final-layer logit lens and actual output",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    receipts = (
        [make_fixture_receipt()] if args.fixture else load_receipts(args.receipts)
    )
    if (
        args.trim_top_k is not None
        or args.score_decimals is not None
        or args.keep_readouts is not None
    ):
        known = {run.name for r in receipts for run in r.interventions}
        unknown = set(args.keep_readouts or ()) - known - {"primary"}
        if unknown:
            raise ValueError(
                f"--keep-readouts names no run in the receipts: {sorted(unknown)}"
            )
        receipts = [_trim(r, args) for r in receipts]
    digests = write_preview_bundle(
        receipts,
        args.out_dir,
        title=args.title,
        contact_line=args.contact_line,
        head_score_tolerance=args.head_score_tolerance,
    )
    for name, digest in sorted(digests.items()):
        logger.info("%s  %s", digest, name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
