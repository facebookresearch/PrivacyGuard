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
import json
import logging
import sys
from collections.abc import Sequence

from privacy_guard.analysis.jacobian_lens.preview_analysis_node import (
    HEAD_SCORE_TOLERANCE,
)
from privacy_guard.analysis.jacobian_lens.preview_fixture import make_fixture_receipt
from privacy_guard.analysis.jacobian_lens.preview_schema import JLensPreviewReceipt
from privacy_guard.analysis.jacobian_lens.preview_viewer import (
    DEFAULT_CONTACT_LINE,
    DEFAULT_TITLE,
    write_preview_bundle,
)


logger: logging.Logger = logging.getLogger(__name__)


def load_receipts(paths: Sequence[str]) -> list[JLensPreviewReceipt]:
    """Load and validate receipts; each file holds one receipt or a list."""
    receipts = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else [data]
        receipts.extend(JLensPreviewReceipt.from_dict(item) for item in items)
    return receipts


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
