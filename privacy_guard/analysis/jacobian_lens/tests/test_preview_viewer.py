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

import dataclasses
import hashlib
import json
import os
import re
import tempfile
import unittest

from privacy_guard.analysis.jacobian_lens.preview_fixture import make_fixture_receipt
from privacy_guard.analysis.jacobian_lens.preview_schema import TokenScore
from privacy_guard.analysis.jacobian_lens.preview_viewer import (
    CHECKSUMS_FILENAME,
    DEFAULT_CONTACT_LINE,
    HTML_FILENAME,
    RECEIPTS_FILENAME,
    render_preview_html,
    SUMMARIES_FILENAME,
    write_preview_bundle,
)

SVG_NAMESPACE = "http://www.w3.org/2000/svg"


def _embedded(page: str, element_id: str) -> object:
    match = re.search(
        f'<script type="application/json" id="{element_id}">(.*?)</script>',
        page,
        re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group(1))


class PreviewViewerTest(unittest.TestCase):
    def test_page_is_self_contained(self) -> None:
        page = render_preview_html([make_fixture_receipt()])
        self.assertIn("Content-Security-Policy", page)
        self.assertIn("default-src 'none'", page)
        self.assertRegex(
            page,
            r"<h1>PrivacyGuard J-Lens preview</h1>\s*<div class=\"subtitle\">"
            + re.escape(DEFAULT_CONTACT_LINE),
        )
        for forbidden in (
            "<link",
            " src=",
            "fetch(",
            "XMLHttpRequest",
            "import(",
            "@import",
            "WebSocket",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, page)
        self.assertEqual(re.findall(r"https?://[^\"' ]+", page), [SVG_NAMESPACE])
        receipts = _embedded(page, "jlens-receipts")
        self.assertEqual(receipts, [make_fixture_receipt().to_dict()])
        summaries = _embedded(page, "jlens-summaries")
        self.assertIsInstance(summaries, list)

    def test_escapes_untrusted_token_text(self) -> None:
        receipt = make_fixture_receipt(with_hash=False)
        evil = "</script><script>alert(1)</script>__SUMMARIES__"
        cell = receipt.cells[1]
        patched_top = (dataclasses.replace(cell.top[0], text=evil),) + cell.top[1:]
        cells = (
            receipt.cells[0],
            dataclasses.replace(cell, top=patched_top),
        ) + receipt.cells[2:]
        receipt = dataclasses.replace(receipt, cells=cells).with_hash()
        page = render_preview_html([receipt], title="<b>t</b>", contact_line="a & b")
        self.assertNotIn(evil, page)
        self.assertEqual(page.count("<script>"), 1)
        self.assertIn("&lt;b&gt;t&lt;/b&gt;", page)
        self.assertIn("a &amp; b", page)
        receipts = _embedded(page, "jlens-receipts")
        self.assertIsInstance(receipts, list)
        self.assertEqual(receipts[0]["cells"][1]["top"][0]["text"], evil)
        self.assertIsInstance(receipt.cells[1].top[0], TokenScore)
        with self.assertRaises(ValueError):
            render_preview_html([])

    def test_write_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            digests = write_preview_bundle(
                [make_fixture_receipt(), make_fixture_receipt()], out_dir
            )
            self.assertEqual(
                set(os.listdir(out_dir)),
                {
                    HTML_FILENAME,
                    RECEIPTS_FILENAME,
                    SUMMARIES_FILENAME,
                    CHECKSUMS_FILENAME,
                },
            )
            for name, digest in digests.items():
                with open(os.path.join(out_dir, name), "rb") as f:
                    self.assertEqual(hashlib.sha256(f.read()).hexdigest(), digest)
            with open(os.path.join(out_dir, CHECKSUMS_FILENAME)) as f:
                self.assertEqual(len(f.read().splitlines()), 3)
            with open(os.path.join(out_dir, RECEIPTS_FILENAME)) as f:
                self.assertEqual(len(json.load(f)), 2)
