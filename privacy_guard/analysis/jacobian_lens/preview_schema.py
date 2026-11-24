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
Receipt schema ``privacyguard.jlens_preview/v1`` for the optional, experimental
Jacobian-lens (J-lens) layer-by-token preview.

A receipt holds only ranked vocabulary readouts (token IDs, decoded text, scores
and full-vocabulary ranks), the model's actual output, intervention summaries and
pinned provenance. It never holds activations, residual vectors, full logit
arrays, lens matrices or weights; ``validate_receipt_dict`` rejects them.
"""

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_VERSION = "privacyguard.jlens_preview/v1"
GLIMMER_TARGET_REPO_ID = "meta-models/Muse-Glimmer-30B"

RUN_KINDS: tuple[str, ...] = ("real_model", "cpu_fixture", "synthetic_demo")
MODEL_ROLES: tuple[str, ...] = ("glimmer_target", "independent_control", "fixture")
METHODS: tuple[str, ...] = ("jlens", "logit_lens")
CELL_STATUSES: tuple[str, ...] = ("ok", "unavailable")
PROMPT_RENDERINGS: tuple[str, ...] = ("raw_completion", "chat_template")
INTERVENTION_KINDS: tuple[str, ...] = (
    "baseline",
    "intervention",
    "noop",
    "matched_norm_unrelated",
    "restoration",
)
REQUIRED_MODEL_IDENTITY_KEYS: tuple[str, ...] = (
    "repo_id",
    "revision_sha",
    "safetensors_sha256_index",
    "tokenizer_sha256",
    "transformers_version",
    "torch_version",
    "dtype",
)
REQUIRED_LENS_KEYS: tuple[str, ...] = (
    "lens_id",
    "sha256",
    "source_layers",
    "target_site",
    "estimator",
)
BF16_DTYPES: tuple[str, ...] = ("bfloat16", "torch.bfloat16", "bf16")
# Name the analysis node uses for the model's own output in answer margins.
RESERVED_RUN_NAME = "actual_output"

# Keys that would carry model-internal tensors or weights. A receipt is a
# shareable preview; these never belong in it.
FORBIDDEN_KEYS: frozenset[str] = frozenset(
    {
        "activation",
        "activations",
        "embedding",
        "embeddings",
        "full_logits",
        "gradient",
        "gradients",
        "hidden_state",
        "hidden_states",
        "jacobian",
        "jacobians",
        "lens_matrix",
        "logits",
        "raw_logits",
        "residual",
        "residuals",
        "state_dict",
        "weights",
    }
)
# Longest purely numeric list allowed anywhere in a receipt. Token-id lists and
# layer lists are short; a longer numeric list is most likely a dumped vector.
MAX_NUMERIC_LIST_LEN = 512
# Free-form metadata maps (identity, code, lens manifest, numerical receipt,
# display settings, intervention directions) may hold at most this many numbers
# in total, so a vector cannot be smuggled in as many short lists.
FREE_FORM_KEYS: tuple[str, ...] = (
    "model_identity",
    "code",
    "lens",
    "numerical_receipt",
    "display",
)
MAX_FREE_FORM_NUMERIC_LEAVES = 1024


class ReceiptValidationError(ValueError):
    """Raised when a receipt does not satisfy ``privacyguard.jlens_preview/v1``."""


@dataclass(frozen=True)
class TokenScore:
    """
    One vocabulary entry of a readout.

    args:
        token_id: tokenizer ID
        text: decoded text of the single token, unmodified (whitespace kept)
        score: native-head output score (after any output scaling/softcapping)
        rank: 1-based rank over the full vocabulary; ties broken by ascending ID
        ties: number of other vocabulary entries with exactly the same score
    """

    token_id: int
    text: str
    score: float
    rank: int
    ties: int = 0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TokenScore":
        return cls(
            token_id=int(data["token_id"]),
            text=str(data["text"]),
            score=float(data["score"]),
            rank=int(data["rank"]),
            ties=int(data.get("ties", 0)),
        )


@dataclass(frozen=True)
class TrackedToken:
    """A token whose rank is reported in every cell, grouped by concept."""

    concept: str
    token_id: int
    text: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrackedToken":
        return cls(
            concept=str(data["concept"]),
            token_id=int(data["token_id"]),
            text=str(data["text"]),
        )


@dataclass(frozen=True)
class CellReadout:
    """
    Readout of one method at one (layer, token position) cell.

    ``layer`` is 0-based: layer i is the post-block residual output of block i.
    ``top`` is unfiltered: ranks 1..k of the full vocabulary.
    """

    method: str
    layer: int
    position: int
    status: str
    unavailable_reason: str | None = None
    top: tuple[TokenScore, ...] = ()
    tracked: tuple[TokenScore, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CellReadout":
        return cls(
            method=str(data["method"]),
            layer=int(data["layer"]),
            position=int(data["position"]),
            status=str(data["status"]),
            unavailable_reason=_optional_str(data.get("unavailable_reason")),
            top=_token_scores(data.get("top", ())),
            tracked=_token_scores(data.get("tracked", ())),
        )


@dataclass(frozen=True)
class PromptRecord:
    """
    The exact analyzed input and its tokenization.

    ``label`` names a predeclared prompt variant (e.g. the exact prompt versus
    an upstream verbatim item); ``note`` records any adaptation of the text.
    ``bos_token_id`` is the BOS ID prepended to ``token_ids[0]`` (None: none).
    """

    text: str
    sha256: str
    rendering: str
    token_ids: tuple[int, ...]
    token_texts: tuple[str, ...]
    label: str = ""
    note: str | None = None
    bos_token_id: int | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PromptRecord":
        return cls(
            text=str(data["text"]),
            sha256=str(data["sha256"]),
            rendering=str(data["rendering"]),
            token_ids=tuple(int(t) for t in data["token_ids"]),
            token_texts=tuple(str(t) for t in data["token_texts"]),
            label=str(data.get("label", "")),
            note=_optional_str(data.get("note")),
            bos_token_id=(
                None if data.get("bos_token_id") is None else int(data["bos_token_id"])
            ),
        )


@dataclass(frozen=True)
class ActualOutput:
    """The model's own greedy continuation and next-token scores."""

    greedy_ids: tuple[int, ...]
    greedy_text: str
    max_new_tokens: int
    next_token_top: tuple[TokenScore, ...]
    next_token_tracked: tuple[TokenScore, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActualOutput":
        return cls(
            greedy_ids=tuple(int(t) for t in data["greedy_ids"]),
            greedy_text=str(data["greedy_text"]),
            max_new_tokens=int(data["max_new_tokens"]),
            next_token_top=_token_scores(data["next_token_top"]),
            next_token_tracked=_token_scores(data.get("next_token_tracked", ())),
        )


@dataclass(frozen=True)
class InterventionRun:
    """
    Summary of one controlled run (baseline, intervention or control).

    ``method_label`` names the exact technique, e.g. ``contrast_mean_clamp``
    or ``linear_probe_swap``; neither is a J-lens readout. ``layers`` and
    ``positions`` are every edited site (a band has several layers). Exactly one
    ``kind="intervention"`` run is ``primary``: the predeclared arm. Other arms
    are exploratory. ``readout_cells`` may hold the J-lens/logit-lens readout of
    the patched forward pass, e.g. at layers downstream of the edited band.
    """

    name: str
    kind: str
    method_label: str
    status: str
    layers: tuple[int, ...] = ()
    positions: tuple[int, ...] = ()
    primary: bool = False
    strength: float | None = None
    delta_norm: float | None = None
    direction: Mapping[str, Any] | None = None
    next_token_top: tuple[TokenScore, ...] = ()
    next_token_tracked: tuple[TokenScore, ...] = ()
    greedy_text: str | None = None
    unavailable_reason: str | None = None
    notes: str | None = None
    readout_cells: tuple[CellReadout, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InterventionRun":
        direction = data.get("direction")
        return cls(
            name=str(data["name"]),
            kind=str(data["kind"]),
            method_label=str(data["method_label"]),
            status=str(data["status"]),
            layers=tuple(int(layer) for layer in data.get("layers", ())),
            positions=tuple(int(p) for p in data.get("positions", ())),
            primary=bool(data.get("primary", False)),
            strength=_optional_float(data.get("strength")),
            delta_norm=_optional_float(data.get("delta_norm")),
            direction=(
                None
                if direction is None
                else {str(k): v for k, v in dict(direction).items()}
            ),
            next_token_top=_token_scores(data.get("next_token_top", ())),
            next_token_tracked=_token_scores(data.get("next_token_tracked", ())),
            greedy_text=_optional_str(data.get("greedy_text")),
            unavailable_reason=_optional_str(data.get("unavailable_reason")),
            notes=_optional_str(data.get("notes")),
            readout_cells=tuple(
                CellReadout.from_dict(c) for c in data.get("readout_cells", ())
            ),
        )


@dataclass(frozen=True)
class UnavailableReason:
    """Why a part of the preview is missing, e.g. scope ``jlens/layer=12``."""

    scope: str
    reason: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "UnavailableReason":
        return cls(scope=str(data["scope"]), reason=str(data["reason"]))


@dataclass(frozen=True)
class JLensPreviewReceipt:
    """
    Immutable ``privacyguard.jlens_preview/v1`` receipt.

    ``receipt_sha256`` is the SHA-256 of the canonical JSON of every other field;
    ``with_hash()`` fills it and ``from_dict`` verifies it when present.
    """

    run_kind: str
    model_role: str
    created_utc: str
    prompt: PromptRecord
    model_identity: Mapping[str, Any]
    code: Mapping[str, Any]
    num_layers: int
    vocab_size: int
    top_k: int
    score_definition: str
    tracked_tokens: tuple[TrackedToken, ...]
    cells: tuple[CellReadout, ...]
    actual_output: ActualOutput | None
    lens: Mapping[str, Any] | None = None
    interventions: tuple[InterventionRun, ...] = ()
    numerical_receipt: Mapping[str, Any] | None = None
    unavailable: tuple[UnavailableReason, ...] = ()
    display: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    receipt_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))

    def content_sha256(self) -> str:
        content = self.to_dict()
        content.pop("receipt_sha256", None)
        return hashlib.sha256(canonical_json(content).encode("utf-8")).hexdigest()

    def with_hash(self) -> "JLensPreviewReceipt":
        data = self.to_dict()
        data["receipt_sha256"] = self.content_sha256()
        return JLensPreviewReceipt.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JLensPreviewReceipt":
        """Parse and validate a receipt dict; raises ``ReceiptValidationError``."""
        validate_receipt_dict(data)
        try:
            receipt = cls._parse(data)
        except (KeyError, TypeError, ValueError) as e:
            raise ReceiptValidationError(f"Malformed receipt: {e!r}") from e
        validate_receipt(receipt)
        return receipt

    @classmethod
    def _parse(cls, data: Mapping[str, Any]) -> "JLensPreviewReceipt":
        actual = data.get("actual_output")
        return cls(
            schema_version=str(data["schema_version"]),
            run_kind=str(data["run_kind"]),
            model_role=str(data["model_role"]),
            created_utc=str(data["created_utc"]),
            prompt=PromptRecord.from_dict(data["prompt"]),
            model_identity=dict(data["model_identity"]),
            code=dict(data["code"]),
            num_layers=int(data["num_layers"]),
            vocab_size=int(data["vocab_size"]),
            top_k=int(data["top_k"]),
            score_definition=str(data["score_definition"]),
            tracked_tokens=tuple(
                TrackedToken.from_dict(t) for t in data["tracked_tokens"]
            ),
            cells=tuple(CellReadout.from_dict(c) for c in data["cells"]),
            actual_output=None if actual is None else ActualOutput.from_dict(actual),
            lens=None if data.get("lens") is None else dict(data["lens"]),
            interventions=tuple(
                InterventionRun.from_dict(i) for i in data.get("interventions", ())
            ),
            numerical_receipt=(
                None
                if data.get("numerical_receipt") is None
                else dict(data["numerical_receipt"])
            ),
            unavailable=tuple(
                UnavailableReason.from_dict(u) for u in data.get("unavailable", ())
            ),
            display=dict(data.get("display", {})),
            receipt_sha256=_optional_str(data.get("receipt_sha256")),
        )


def canonical_json(data: Any) -> str:
    """Deterministic JSON used for hashing."""
    return json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_receipt_dict(data: Mapping[str, Any]) -> None:
    """Structural checks on the raw dict before parsing."""
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ReceiptValidationError(
            f"schema_version must be {SCHEMA_VERSION!r}, got {data.get('schema_version')!r}"
        )
    _reject_private_payloads(data, path="$")
    free_form = [data.get(key) for key in FREE_FORM_KEYS] + [
        run.get("direction") for run in data.get("interventions", ())
    ]
    leaves = sum(_count_numeric_leaves(value) for value in free_form)
    if leaves > MAX_FREE_FORM_NUMERIC_LEAVES:
        raise ReceiptValidationError(
            f"free-form fields hold {leaves} numbers (max {MAX_FREE_FORM_NUMERIC_LEAVES}); "
            "raw vectors are not allowed in a preview receipt"
        )


def validate_receipt(receipt: JLensPreviewReceipt) -> None:
    """Semantic checks on a parsed receipt; raises ``ReceiptValidationError``."""
    _validate_identity(receipt)
    _validate_prompt(receipt)
    tracked_ids = _validate_tracked(receipt)
    seen: set[tuple[str, int, int]] = set()
    for cell in receipt.cells:
        key = (cell.method, cell.layer, cell.position)
        if key in seen:
            raise ReceiptValidationError(f"Duplicate cell {key}")
        seen.add(key)
        _validate_cell(receipt, cell, tracked_ids)
    if receipt.actual_output is not None:
        _validate_ranked(
            receipt, receipt.actual_output.next_token_top, "actual_output", top=True
        )
        _validate_tracked_scores(
            receipt,
            receipt.actual_output.next_token_tracked,
            tracked_ids,
            "actual_output",
        )
    names = [run.name for run in receipt.interventions]
    if len(set(names)) != len(names) or RESERVED_RUN_NAME in names:
        raise ReceiptValidationError(
            f"intervention names must be unique and not {RESERVED_RUN_NAME!r}"
        )
    for run in receipt.interventions:
        _validate_intervention(receipt, run, tracked_ids)
    arms = [r for r in receipt.interventions if r.kind == "intervention"]
    if arms and sum(r.primary for r in arms) != 1:
        raise ReceiptValidationError(
            "exactly one intervention run must be the predeclared primary arm"
        )
    if (
        receipt.receipt_sha256 is not None
        and receipt.receipt_sha256 != receipt.content_sha256()
    ):
        raise ReceiptValidationError(
            "receipt_sha256 does not match content; the receipt was modified"
        )


def _validate_identity(receipt: JLensPreviewReceipt) -> None:
    _require_member(receipt.run_kind, RUN_KINDS, "run_kind")
    _require_member(receipt.model_role, MODEL_ROLES, "model_role")
    missing = [
        k for k in REQUIRED_MODEL_IDENTITY_KEYS if k not in receipt.model_identity
    ]
    if missing:
        raise ReceiptValidationError(f"model_identity is missing {missing}")
    is_fixture_run = receipt.run_kind != "real_model"
    if is_fixture_run != (receipt.model_role == "fixture"):
        raise ReceiptValidationError(
            "run_kind and model_role disagree: non-real runs must use model_role "
            "'fixture' and real runs must not"
        )
    if not is_fixture_run:
        _validate_pinned_hashes(receipt.model_identity)
    if receipt.model_role == "glimmer_target":
        identity = receipt.model_identity
        if identity["repo_id"] != GLIMMER_TARGET_REPO_ID:
            raise ReceiptValidationError(
                f"glimmer_target requires repo_id {GLIMMER_TARGET_REPO_ID!r}, "
                f"got {identity['repo_id']!r}"
            )
        if str(identity["dtype"]) not in BF16_DTYPES:
            raise ReceiptValidationError(
                f"glimmer_target requires the BF16 checkpoint, got {identity['dtype']!r}"
            )
    if receipt.lens is not None:
        missing_lens = [k for k in REQUIRED_LENS_KEYS if k not in receipt.lens]
        if missing_lens:
            raise ReceiptValidationError(f"lens is missing {missing_lens}")
    if receipt.num_layers <= 0 or receipt.vocab_size <= 0 or receipt.top_k <= 0:
        raise ReceiptValidationError("num_layers, vocab_size and top_k must be > 0")


def _validate_pinned_hashes(identity: Mapping[str, Any]) -> None:
    """Real-model runs must pin an immutable revision and content hashes."""
    if not re.fullmatch("[0-9a-f]{40}", str(identity["revision_sha"])):
        raise ReceiptValidationError(
            "revision_sha must be a 40-hex commit, not a branch"
        )
    if not re.fullmatch("[0-9a-f]{64}", str(identity["tokenizer_sha256"])):
        raise ReceiptValidationError("tokenizer_sha256 must be a SHA-256 hex digest")
    shards = identity["safetensors_sha256_index"]
    if not isinstance(shards, Mapping) or not shards:
        raise ReceiptValidationError(
            "safetensors_sha256_index must map files to SHA-256"
        )
    for name, digest in shards.items():
        if not re.fullmatch("[0-9a-f]{64}", str(digest)):
            raise ReceiptValidationError(
                f"safetensors digest for {name} is not SHA-256"
            )


def _validate_prompt(receipt: JLensPreviewReceipt) -> None:
    prompt = receipt.prompt
    _require_member(prompt.rendering, PROMPT_RENDERINGS, "prompt.rendering")
    if sha256_text(prompt.text) != prompt.sha256:
        raise ReceiptValidationError("prompt.sha256 does not match prompt.text")
    if not prompt.token_ids or len(prompt.token_ids) != len(prompt.token_texts):
        raise ReceiptValidationError(
            "prompt.token_ids must be non-empty and match prompt.token_texts"
        )
    for token_id in prompt.token_ids:
        _require_token_id(receipt, token_id, "prompt.token_ids")
    if prompt.bos_token_id is not None and prompt.token_ids[0] != prompt.bos_token_id:
        raise ReceiptValidationError("prompt.bos_token_id must equal token_ids[0]")


def _validate_tracked(receipt: JLensPreviewReceipt) -> set[int]:
    tracked_ids = [t.token_id for t in receipt.tracked_tokens]
    if len(set(tracked_ids)) != len(tracked_ids):
        raise ReceiptValidationError("tracked_tokens has duplicate token IDs")
    for token_id in tracked_ids:
        _require_token_id(receipt, token_id, "tracked_tokens")
    return set(tracked_ids)


def _validate_cell(
    receipt: JLensPreviewReceipt, cell: CellReadout, tracked_ids: set[int]
) -> None:
    where = f"cell({cell.method}, layer={cell.layer}, position={cell.position})"
    _require_member(cell.method, METHODS, f"{where}.method")
    _require_member(cell.status, CELL_STATUSES, f"{where}.status")
    if not 0 <= cell.layer < receipt.num_layers:
        raise ReceiptValidationError(f"{where}: layer out of range")
    if not 0 <= cell.position < len(receipt.prompt.token_ids):
        raise ReceiptValidationError(f"{where}: position out of range")
    if cell.status == "unavailable":
        if not cell.unavailable_reason or cell.top or cell.tracked:
            raise ReceiptValidationError(
                f"{where}: unavailable cells need a reason and no scores"
            )
        return
    if cell.method == "jlens":
        _require_fitted_layer(receipt, cell.layer, where)
    _validate_ranked(receipt, cell.top, where, top=True)
    _validate_ranked(receipt, cell.tracked, where)
    if {t.token_id for t in cell.tracked} != tracked_ids:
        raise ReceiptValidationError(
            f"{where}: tracked ranks must cover tracked_tokens"
        )
    top_ranks = {t.token_id: t.rank for t in cell.top}
    for entry in cell.tracked:
        if entry.token_id in top_ranks and top_ranks[entry.token_id] != entry.rank:
            raise ReceiptValidationError(
                f"{where}: tracked rank of {entry.token_id} disagrees with top list"
            )


def _require_fitted_layer(receipt: JLensPreviewReceipt, layer: int, where: str) -> None:
    """A J-lens readout needs a lens fitted at that source layer; at the final
    layer (the lens target) the J-lens is the identity map."""
    if receipt.lens is None:
        raise ReceiptValidationError(f"{where}: J-lens scores need a lens manifest")
    source_layers = {int(x) for x in receipt.lens["source_layers"]}
    if layer not in source_layers and layer != receipt.num_layers - 1:
        raise ReceiptValidationError(f"{where}: no lens fitted for this layer")


def _validate_ranked(
    receipt: JLensPreviewReceipt,
    entries: Sequence[TokenScore],
    where: str,
    top: bool = False,
) -> None:
    for entry in entries:
        _require_token_id(receipt, entry.token_id, where)
        if not 1 <= entry.rank <= receipt.vocab_size:
            raise ReceiptValidationError(f"{where}: rank {entry.rank} out of range")
        if not math.isfinite(entry.score) or entry.ties < 0:
            raise ReceiptValidationError(f"{where}: non-finite score or negative ties")
    if not top:
        return
    if not entries or len(entries) > receipt.top_k:
        raise ReceiptValidationError(f"{where}: top list must hold 1..top_k entries")
    if [e.rank for e in entries] != list(range(1, len(entries) + 1)):
        raise ReceiptValidationError(f"{where}: top list must be ranks 1..k in order")
    scores = [e.score for e in entries]
    if any(a < b for a, b in zip(scores, scores[1:])):
        raise ReceiptValidationError(f"{where}: top scores must be non-increasing")


def _validate_tracked_scores(
    receipt: JLensPreviewReceipt,
    entries: Sequence[TokenScore],
    tracked_ids: set[int],
    where: str,
) -> None:
    """Next-token tracked entries, when given, cover exactly ``tracked_tokens``."""
    _validate_ranked(receipt, entries, where)
    if entries and sorted(e.token_id for e in entries) != sorted(tracked_ids):
        raise ReceiptValidationError(
            f"{where}: tracked ranks must cover tracked_tokens"
        )


def _validate_intervention(
    receipt: JLensPreviewReceipt, run: InterventionRun, tracked_ids: set[int]
) -> None:
    where = f"intervention({run.name})"
    _require_member(run.kind, INTERVENTION_KINDS, f"{where}.kind")
    _require_member(run.status, CELL_STATUSES, f"{where}.status")
    if not run.method_label:
        raise ReceiptValidationError(f"{where}: method_label is required")
    if run.primary and run.kind != "intervention":
        raise ReceiptValidationError(
            f"{where}: only an intervention run can be primary"
        )
    for layer in run.layers:
        if not 0 <= layer < receipt.num_layers:
            raise ReceiptValidationError(f"{where}: layer out of range")
    for position in run.positions:
        if not 0 <= position < len(receipt.prompt.token_ids):
            raise ReceiptValidationError(f"{where}: position out of range")
    if run.status == "unavailable":
        if not run.unavailable_reason:
            raise ReceiptValidationError(f"{where}: unavailable runs need a reason")
        if run.next_token_top or run.next_token_tracked or run.readout_cells:
            raise ReceiptValidationError(f"{where}: unavailable runs carry no results")
        return
    _validate_ranked(receipt, run.next_token_top, where, top=True)
    _validate_tracked_scores(receipt, run.next_token_tracked, tracked_ids, where)
    seen: set[tuple[str, int, int]] = set()
    for cell in run.readout_cells:
        key = (cell.method, cell.layer, cell.position)
        if key in seen:
            raise ReceiptValidationError(f"{where}: duplicate readout cell {key}")
        seen.add(key)
        _validate_cell(receipt, cell, tracked_ids)


def _reject_private_payloads(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                raise ReceiptValidationError(
                    f"{path}.{key}: model-internal tensors/weights are not allowed in a preview receipt"
                )
            _reject_private_payloads(child, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        numeric = [
            v for v in value if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
        if len(numeric) > MAX_NUMERIC_LIST_LEN:
            raise ReceiptValidationError(
                f"{path}: numeric list of length {len(numeric)} exceeds "
                f"{MAX_NUMERIC_LIST_LEN}; raw vectors are not allowed in a preview receipt"
            )
        for i, child in enumerate(value):
            _reject_private_payloads(child, f"{path}[{i}]")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ReceiptValidationError(f"{path}: non-finite number")


def _count_numeric_leaves(value: Any) -> int:
    if isinstance(value, Mapping):
        return sum(_count_numeric_leaves(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_count_numeric_leaves(v) for v in value)
    return int(isinstance(value, (int, float)) and not isinstance(value, bool))


def _require_member(value: str, allowed: Iterable[str], where: str) -> None:
    allowed_values = tuple(allowed)
    if value not in allowed_values:
        raise ReceiptValidationError(
            f"{where} must be one of {allowed_values}, got {value!r}"
        )


def _require_token_id(receipt: JLensPreviewReceipt, token_id: int, where: str) -> None:
    if not 0 <= token_id < receipt.vocab_size:
        raise ReceiptValidationError(f"{where}: token ID {token_id} out of vocabulary")


def _token_scores(items: Iterable[Mapping[str, Any]]) -> tuple[TokenScore, ...]:
    return tuple(TokenScore.from_dict(i) for i in items)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value
