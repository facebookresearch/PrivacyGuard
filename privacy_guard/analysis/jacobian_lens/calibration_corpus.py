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
"""Calibration corpus selection and provenance for the experimental J-lens fit.

The default rule mirrors the public reference walkthrough: the first ``n``
WikiText-103 (``Salesforce/wikitext``, ``wikitext-103-raw-v1``, train) records
with at least 600 characters, tokenized and truncated to 128 tokens. Loading
the dataset is the caller's job; this module only selects, hashes, tokenizes
and checks that the calibration text is separate from evaluation prompts.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

WIKITEXT_DATASET = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-103-raw-v1"
SELECTION_RULE = (
    "first n train records with len(text.strip()) >= min_chars, in file order"
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def select_prompts(
    records: Iterable[str], n_prompts: int, *, min_chars: int = 600
) -> list[str]:
    """The first ``n_prompts`` records whose stripped length is >= ``min_chars``."""
    if n_prompts < 1:
        raise ValueError("n_prompts must be >= 1")
    selected: list[str] = []
    for text in records:
        if len(text.strip()) >= min_chars:
            selected.append(text)
            if len(selected) == n_prompts:
                return selected
    raise ValueError(
        f"only {len(selected)} records satisfy the rule; {n_prompts} requested"
    )


def _normalize(text: str) -> str:
    """Lowercase words only, so WikiText spacing such as ``legs .`` still matches."""
    return " " + " ".join(re.findall(r"[a-z0-9]+", text.lower())) + " "


def _word_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def separation_report(
    calibration: Sequence[str], eval_texts: Sequence[str], *, ngram: int = 6
) -> dict[str, Any]:
    """Raise if any evaluation prompt's word sequence occurs in the calibration
    text (case, punctuation and spacing ignored); report shared word ``ngram``s
    as a diagnostic count."""
    joined = [_normalize(t) for t in calibration]
    contained = [e for e in eval_texts if any(_normalize(e) in c for c in joined)]
    if contained:
        raise ValueError(
            f"{len(contained)} evaluation prompt(s) occur in the calibration corpus"
        )
    calib_grams: set[tuple[str, ...]] = set()
    for text in calibration:
        calib_grams |= _word_ngrams(text, ngram)
    shared = sum(len(_word_ngrams(e, ngram) & calib_grams) for e in eval_texts)
    return {
        "eval_prompts_checked": len(eval_texts),
        "eval_prompts_contained": len(contained),
        "ngram": ngram,
        "shared_eval_ngrams": shared,
    }


@dataclass
class CorpusManifest:
    """Provenance of the calibration prompts (hashes and lengths, no text)."""

    dataset: str
    config: str
    split: str
    revision: str
    source_files: dict[str, str]
    selection_rule: str
    min_chars: int
    max_seq_len: int
    tokenizer_sha256: str
    prompt_sha256: list[str] = field(default_factory=list)
    token_lengths: list[int] = field(default_factory=list)
    truncated: list[bool] = field(default_factory=list)
    separation: dict[str, Any] = field(default_factory=dict)

    def corpus_sha256(self) -> str:
        """One digest over the ordered per-prompt hashes and the token budget."""
        return sha256_text("\n".join([*self.prompt_sha256, str(self.max_seq_len)]))

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["corpus_sha256"] = self.corpus_sha256()
        return out


def tokenize_prompts(
    encode: Callable[[str], torch.Tensor],
    prompts: Sequence[str],
    *,
    max_seq_len: int = 128,
) -> tuple[list[torch.Tensor], list[int], list[bool]]:
    """Encode each prompt to ``[1, T]`` and keep at most ``max_seq_len`` tokens.

    Truncating a calibration document is the reference fitting rule; the
    returned flags record it per prompt so it is never silent.
    """
    tensors: list[torch.Tensor] = []
    lengths: list[int] = []
    truncated: list[bool] = []
    for text in prompts:
        ids = encode(text)
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("encode must return [1, T]")
        truncated.append(ids.shape[1] > max_seq_len)
        ids = ids[:, :max_seq_len]
        tensors.append(ids)
        lengths.append(int(ids.shape[1]))
    return tensors, lengths, truncated
