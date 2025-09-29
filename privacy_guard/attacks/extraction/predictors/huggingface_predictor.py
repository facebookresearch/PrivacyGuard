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

"""
HuggingFace predictor implementation for GenAI extraction attacks.
"""

import warnings
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from privacy_guard.attacks.extraction.predictors.base_predictor import BasePredictor
from privacy_guard.attacks.extraction.utils.data_utils import load_model_and_tokenizer
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer


class HuggingFacePredictor(BasePredictor):
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        model_kwargs: Dict[str, Any] | None = None,
        tokenizer_kwargs: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.model_name: str = model_name
        self.device: str = (
            device
            if device is not None
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        # Handle kwargs - if model_kwargs/tokenizer_kwargs not provided, use general kwargs for backward compatibility
        self.model_kwargs: Dict[str, Any] = model_kwargs or kwargs
        self.tokenizer_kwargs: Dict[str, Any] = tokenizer_kwargs or {}
        self.model: PreTrainedModel
        self.tokenizer: PreTrainedTokenizer
        # Model already loaded on device - now pass the kwargs
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_name,
            device,
            model_kwargs=self.model_kwargs,
            tokenizer_kwargs=self.tokenizer_kwargs,
        )

    def _prepare_tokenizer(self) -> None:
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def preprocess_batch(self, batch: List[str]) -> List[str]:
        clean_batch = []
        for item in batch:
            if not isinstance(item, str):
                raise Warning(f"Found non-string item in batch: {type(item)}")
                clean_batch.append(str(item) if item is not None else "")
            else:
                clean_batch.append(item)
        return clean_batch

    def _generate_process_batch(
        self, batch: List[str], max_new_tokens: int = 512, **generation_kwargs: Any
    ) -> List[str]:
        """Process a single batch of prompts."""
        clean_batch = self.preprocess_batch(batch)

        inputs = self.tokenizer(
            clean_batch, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        with torch.no_grad():
            # Handle both regular models and DDP-wrapped models
            if hasattr(self.model, "module"):
                outputs = self.model.module.generate(  # pyre-ignore
                    **inputs, max_new_tokens=max_new_tokens, **generation_kwargs
                )
            else:
                outputs = self.model.generate(  # pyre-ignore
                    **inputs, max_new_tokens=max_new_tokens, **generation_kwargs
                )

        batch_results = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return batch_results

    def generate(self, prompts: List[str], **generation_kwargs: Any) -> List[str]:
        """Generate text continuations for given prompts."""
        if not prompts:
            return []

        self._prepare_tokenizer()

        # Extract batch_size from generation_kwargs, default to 1
        batch_size = generation_kwargs.pop("batch_size", 1)
        max_new_tokens = generation_kwargs.pop("max_new_tokens", 512)

        results = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
            batch = prompts[i : i + batch_size]
            batch_results = self._generate_process_batch(
                batch=batch, max_new_tokens=max_new_tokens, **generation_kwargs
            )
            results.extend(batch_results)

        return results

    def apply_sampling_params(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        """
        Apply sampling parameters (temperature, top_k, top_p) to logits.

        Args:
            logits: Input logits tensor of shape (..., vocab_size)
            temperature: Temperature for scaling logits (default: 1.0)
            top_k: Keep only top k logits, set others to -inf (optional)
            top_p: Keep top p probability mass, set others to -inf (optional)

        Returns:
            Modified logits tensor with sampling parameters applied
        """
        # Warn if both top_k and top_p are specified (not typical usage)
        if top_k is not None and top_p is not None:
            warnings.warn(
                "Both top_k and top_p sampling parameters are specified. "
                "While both will be applied sequentially, this is not typical usage. "
                "Consider using only one sampling method.",
                UserWarning,
                stacklevel=2,
            )

        # Apply temperature scaling
        if temperature != 1.0:
            logits = logits / temperature

        # Apply top_k filtering if specified
        if top_k is not None and top_k > 0:
            # Keep only top_k logits, set others to -inf
            top_k_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            kth_scores = top_k_logits[..., -1].unsqueeze(-1)
            # Set logits below kth score to -inf
            logits = torch.where(
                logits < kth_scores,
                torch.full_like(logits, float("-inf")),
                logits,
            )

        # Apply top_p (nucleus) filtering if specified
        if top_p is not None and 0.0 < top_p < 1.0:
            # Sort logits in descending order
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

            # Convert to probabilities
            sorted_probs = F.softmax(sorted_logits, dim=-1)

            # Calculate cumulative probabilities
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Create mask for tokens to keep (cumulative probability <= top_p)
            # We keep the first token that exceeds top_p to ensure we always have at least one token
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift right to keep the first token that exceeds top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = False

            # Convert back to original indices
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)

            # Set removed indices to -inf
            logits = torch.where(
                indices_to_remove, torch.full_like(logits, float("-inf")), logits
            )

        return logits

    def get_logits(
        self, prompts: List[str], targets: List[str], batch_size: int = 1
    ) -> List[torch.Tensor]:
        """
        Compute logits for target sequences given prompts using batched processing.

        Args:
            prompts: List of input prompts
            targets: List of target sequences to compute logits for
            batch_size: Number of sequences to process in each batch

        Returns:
            List of tensors, each with shape (target_length, vocab_size) for the
            corresponding prompt-target pair
        """
        if not prompts or not targets:
            return []

        if len(prompts) != len(targets):
            raise ValueError(
                f"Number of prompts ({len(prompts)}) must equal number of targets ({len(targets)})"
            )

        self._prepare_tokenizer()

        all_logits = []

        # Process in batches
        for i in tqdm(range(0, len(prompts), batch_size), desc="Computing logits"):
            batch_prompts = prompts[i : i + batch_size]
            batch_targets = targets[i : i + batch_size]

            batch_logits = self._get_logits_batch(batch_prompts, batch_targets)
            all_logits.extend(batch_logits)

        return all_logits

    def _get_logits_batch(
        self, prompts: List[str], targets: List[str]
    ) -> List[torch.Tensor]:
        """
        Process a single batch of prompts and targets to compute logits.

        Args:
            prompts: Batch of input prompts
            targets: Batch of target sequences

        Returns:
            List of tensors with logits for each prompt-target pair in the batch
        """
        # Combine prompts and targets for full sequences
        full_sequences = [prompt + target for prompt, target in zip(prompts, targets)]

        # Tokenize all sequences in the batch
        full_tokens = self.tokenizer(
            full_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
        ).to(self.device)

        # Tokenize prompts separately to find prompt lengths
        prompt_tokens = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
        ).to(self.device)

        # Get prompt lengths for each sequence (accounting for padding)
        prompt_lengths = []
        for i, _prompt in enumerate(prompts):
            # Count non-pad tokens in this prompt
            prompt_ids = prompt_tokens.input_ids[i]
            if self.tokenizer.pad_token_id is not None:
                # Find the first pad token (if any)
                pad_mask = prompt_ids == self.tokenizer.pad_token_id
                if pad_mask.any():
                    prompt_length = pad_mask.nonzero()[0].item()
                else:
                    prompt_length = len(prompt_ids)
            else:
                prompt_length = len(prompt_ids)
            prompt_lengths.append(prompt_length)

        with torch.no_grad():
            # Get model outputs for the entire batch
            outputs = (
                self.model.module  # pyre-ignore
                if hasattr(self.model, "module")
                else self.model
            )(**full_tokens)

            batch_logits = []

            # Extract target logits for each sequence in the batch
            for i, (prompt_length, target) in enumerate(zip(prompt_lengths, targets)):
                # Get the sequence logits for this batch item
                sequence_logits = outputs.logits[i]  # Shape: (seq_len, vocab_size)

                # Tokenize the target separately to get its length
                target_tokens = self.tokenizer(
                    target,
                    return_tensors="pt",
                    truncation=True,
                    add_special_tokens=False,
                ).to(self.device)
                target_length = target_tokens.input_ids.shape[1]

                # Extract logits for target positions
                # We want logits at positions [prompt_length-1 : prompt_length-1+target_length]
                # These predict the target tokens
                start_pos = prompt_length - 1
                end_pos = start_pos + target_length

                # Ensure we don't go beyond sequence length
                end_pos = min(end_pos, sequence_logits.shape[0])

                if start_pos < sequence_logits.shape[0] and start_pos < end_pos:
                    target_logits = sequence_logits[start_pos:end_pos, :]
                    batch_logits.append(target_logits)
                else:
                    # Handle edge case where target is too long or prompt too short
                    # Return empty tensor with correct vocab size
                    vocab_size = sequence_logits.shape[-1]
                    empty_logits = torch.empty(
                        (0, vocab_size), device=self.device, dtype=sequence_logits.dtype
                    )
                    batch_logits.append(empty_logits)

        return batch_logits

    def get_logprobs(
        self, prompts: List[str], targets: List[str], **generation_kwargs: Any
    ) -> List[torch.Tensor]:
        """
        Compute log probabilities for target sequences given prompts.

        Args:
            prompts: List of input prompts
            targets: List of target sequences to compute log probabilities for
            **generation_kwargs: Generation parameters including:
                - batch_size: Number of sequences to process in each batch (default: 1)
                - temperature: Temperature for scaling logits (default: 1.0)
                - top_k: Top-k filtering for logits (optional)
                - top_p: Top-p (nucleus) sampling for logits (optional)

        Returns:
            List of tensors, each containing log probabilities for the corresponding
            prompt-target pair
        """
        if not prompts or not targets:
            return []

        # Extract generation parameters
        batch_size = generation_kwargs.get("batch_size", 1)
        temperature = generation_kwargs.get("temperature", 1.0)
        top_k = generation_kwargs.get("top_k", None)
        top_p = generation_kwargs.get("top_p", None)

        # Get logits using the existing get_logits function
        logits_list = self.get_logits(prompts, targets, batch_size=batch_size)

        all_logprobs = []

        # Process each logits tensor to compute log probabilities
        for _i, (target, logits) in enumerate(zip(targets, logits_list)):
            # Tokenize target to get token IDs
            target_tokens = self.tokenizer(
                target, return_tensors="pt", truncation=True, add_special_tokens=False
            ).to(self.device)
            target_token_ids = target_tokens.input_ids[0]  # Remove batch dimension

            # Apply sampling parameters (temperature, top_k, top_p)
            modified_logits = self.apply_sampling_params(
                logits, temperature=temperature, top_k=top_k, top_p=top_p
            )

            # Convert logits to log probabilities
            target_logprobs = torch.log_softmax(modified_logits, dim=-1)

            # Get log probabilities for each target token
            token_logprobs = []
            for j, token_id in enumerate(target_token_ids):
                if j < target_logprobs.shape[0]:  # Ensure we don't go out of bounds
                    token_logprob = target_logprobs[j, token_id]
                    token_logprobs.append(token_logprob)

            if token_logprobs:
                sequence_logprobs = torch.stack(token_logprobs)
                all_logprobs.append(sequence_logprobs)

        return all_logprobs
