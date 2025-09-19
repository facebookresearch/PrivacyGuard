# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Model Inference Module

This module provides functions for processing texts through language models.
"""

from typing import Any, Dict, List, Literal, Optional

import pandas as pd

import torch

from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer


TaskType = Literal["pretrain", "instruct"]

# NOTE: Custom task configs not yet implemented.
TASK_CONFIG = {
    "pretrain": {"prompt: {prompt}"},
    "instruct": {"prompt": 'Complete the following passage: "{prompt}"'},
}


def process_batch_with_llm(
    index: int,
    batch: List[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: TaskType,
    batch_size: int = 8,
    max_new_tokens: int = 512,
    device: str = "cuda",
    **generation_kwargs: Dict[str, Any],
) -> List[str]:
    # Ensure all items in the batch are strings
    clean_batch = []
    results = []
    for item in batch:
        if not isinstance(item, str):
            print(f"Warning: Found non-string item in batch: {type(item)}")
            try:
                clean_batch.append(str(item))
            except Exception as e:
                print(f"Error converting item to string: {e}")
                clean_batch.append("")
        else:
            clean_batch.append(item)

    try:
        inputs = tokenizer(
            clean_batch, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        with torch.no_grad():
            # Handle both regular models and DDP-wrapped models
            if hasattr(model, "module"):
                outputs = model.module.generate(  # pyre-ignore Undefined attribute [16]: `transformers.utils.dummy_pt_objects.PreTrainedModel` has no attribute `module`.
                    **inputs, max_new_tokens=max_new_tokens, **generation_kwargs
                )
            else:
                outputs = model.generate(  # pyre-ignore Undefined attribute [16]: `transformers.utils.dummy_pt_objects.PreTrainedModel` has no attribute `generate`.
                    **inputs, max_new_tokens=max_new_tokens, **generation_kwargs
                )

        batch_results: List[str] = tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )

        results.extend(batch_results)
    except Exception as e:
        print(f"Error processing batch starting at index {index}: {e}")
        # Add empty results for this batch
        results.extend([""] * len(clean_batch))

    return results


def process_with_llm(
    texts: List[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: TaskType,
    batch_size: int = 8,
    max_new_tokens: int = 512,
    device: str = "cuda",
    **generation_kwargs: Dict[str, Any],
) -> List[str]:
    """
    Process a list of texts through a language model.

    Args:
        texts: List of texts to process
        model: The language model to use
        tokenizer: The tokenizer for the model
        task: Task type ('instruct, pretrain')
        batch_size: Number of texts to process at once
        max_new_tokens: Maximum number of new tokens to generate
        device: Device to use for processing ('cuda', 'cpu', etc.)
        generation_kwargs: Additional keyword arguments for model.generate()

    Returns:
        List of model outputs
    """

    # Process in batches for better performance
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch = texts[i : i + batch_size]
        new_results = process_batch_with_llm(
            index=i,
            batch=batch,
            model=model,
            tokenizer=tokenizer,
            task=task,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            device=device,
            generation_kwargs=generation_kwargs,
        )

        results.extend(new_results)

    return results


def process_dataframe(
    df: pd.DataFrame,
    input_column: str,
    output_column: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: TaskType = "pretrain",
    device: Optional[str] = None,
    batch_size: int = 8,
    max_new_tokens: int = 512,
    **generation_kwargs: Dict[str, Any],
) -> pd.DataFrame:
    """
    Process a DataFrame through the entire pipeline: format, process with LLM, and postprocess.

    Args:
        df: DataFrame to process
        input_column: Name of the column containing input text
        output_column: Name of the column to store results
        model: The language model to use
        tokenizer: The tokenizer for the model
        task: Task type ('instruct, pretrain')
        batch_size: Number of texts to process at once
        max_new_tokens: Maximum number of new tokens to generate
        generation_kwargs: Additional keyword arguments for model.generate() ("temperature", "top_p")
            Note that unused arguments will cause generation failures.

    Returns:
        DataFrame with processed results, with generations present under "output_column"
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if hasattr(model, "device") and str(model.device) != device:  # pyre-ignore
        model = model.to(device)  # pyre-ignore

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_outputs = process_with_llm(
        df[input_column].tolist(),
        model,
        tokenizer,
        task=task,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        device=device,
        **generation_kwargs,
    )
    df["raw_model_output"] = model_outputs

    # strip prompt from model output
    generations = []
    for input_text, raw_model_output in zip(df[input_column], model_outputs):
        generations.append(raw_model_output[len(input_text) :])
    df[output_column] = generations

    return df
