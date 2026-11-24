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
"""A tiny deterministic causal decoder (float64, CPU) for J-lens unit tests.

It is a fixture only: no checkpoint, tokenizer or real-model behaviour.
"""

from __future__ import annotations

import re

import torch
from torch import nn


class _Block(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, dtype=torch.float64)
        self.qkv = nn.Linear(d_model, 3 * d_model, dtype=torch.float64)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(2 * d_model, d_model, dtype=torch.float64),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=-1)
        t = x.shape[-2]
        scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
        mask = torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1)
        attn = scores.masked_fill(mask, float("-inf")).softmax(-1) @ v
        x = x + attn
        return x + self.mlp(x)


class TinyLensModel(nn.Module):
    """Implements the ``LensModel`` protocol with a soft-capped native head."""

    def __init__(
        self,
        *,
        vocab: int = 23,
        d_model: int = 5,
        n_layers: int = 4,
        softcap: float = 3.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.n_layers, self.d_model, self.vocab_size = n_layers, d_model, vocab
        self.embed = nn.Embedding(vocab, d_model, dtype=torch.float64)
        self.layers = nn.ModuleList(_Block(d_model) for _ in range(n_layers))
        self.final_norm = nn.LayerNorm(d_model, dtype=torch.float64)
        self.head = nn.Linear(d_model, vocab, bias=False, dtype=torch.float64)
        self.softcap = softcap
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.layers:
            x = block(x)
        return x

    def run_from(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        for block in self.layers[layer + 1 :]:
            x = block(x)
        return x

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        logits = self.head(self.final_norm(residual.to(torch.float64)))
        return self.softcap * torch.tanh(logits / self.softcap)

    def native_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.unembed(self.forward(input_ids))


def encode_words(text: str, vocab: int = 23) -> list[int]:
    """Deterministic word-level ids with BOS 0 (fixture tokenizer)."""
    words = re.findall(r"\w+|[^\w\s]", text)
    return [0] + [1 + sum(map(ord, w)) % (vocab - 1) for w in words]
