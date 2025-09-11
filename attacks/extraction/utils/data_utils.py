# pyre-strict

"""
Data Utilities Module

This module provides functions for loading and saving data, as well as loading models and tokenizers.
"""

import json
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def load_data(file_path: str, format: str = "jsonl") -> pd.DataFrame:
    """
    Load data from a file into a pandas DataFrame.

    Args:
        file_path: Path to the data file
        format: Format of the data file ('jsonl', 'csv', 'json')

    Returns:
        DataFrame containing the loaded data
    """
    if format.lower() == "jsonl":
        with open(file_path, "r") as file:
            data = [json.loads(line) for line in file]
        return pd.DataFrame(data)
    elif format.lower() == "csv":
        return pd.read_csv(file_path)
    elif format.lower() == "json":
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def save_results(df: pd.DataFrame, output_path: str, format: str = "jsonl") -> None:
    """
    Save results to a file.

    Args:
        df: DataFrame to save
        output_path: Path to save the results to
        format: Format to save the results in ('jsonl', 'csv', 'json')
    """
    if format.lower() == "jsonl":
        df.to_json(output_path, orient="records", lines=True)
    elif format.lower() == "csv":
        df.to_csv(output_path, index=False)
    elif format.lower() == "json":
        df.to_json(output_path, orient="records")
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_model(
    model_name_or_path: str,
    device: Optional[str] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> PreTrainedModel:
    """
    Load a model only.

    Args:
        model_name_or_path: Name or path of the model to load
        device: Device to load the model on ('cuda', 'cpu', etc.)
        model_kwargs: Additional kwargs to pass to AutoModelForCausalLM.from_pretrained()

    Returns:
        Loaded model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_kwargs = model_kwargs or {}

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs).to(
        device
    )
    return model


def load_tokenizer(
    model_name_or_path: str,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
) -> PreTrainedTokenizer:
    """
    Load a tokenizer only.

    Args:
        model_name_or_path: Name or path of the model to load
        tokenizer_kwargs: Additional kwargs to pass to AutoTokenizer.from_pretrained()

    Returns:
        Loaded tokenizer
    """
    tokenizer_kwargs = tokenizer_kwargs or {}
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
    # Ensure tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_and_tokenizer(
    model_name_or_path: str,
    device: Optional[str] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a model and tokenizer.

    Args:
        model_name_or_path: Name or path of the model to load
        device: Device to load the model on ('cuda', 'cpu', etc.)
        model_kwargs: Additional kwargs to pass to AutoModelForCausalLM.from_pretrained()
        tokenizer_kwargs: Additional kwargs to pass to AutoTokenizer.from_pretrained()

    Returns:
        Tuple of (model, tokenizer)
    """
    model = load_model(model_name_or_path, device, model_kwargs)
    tokenizer = load_tokenizer(model_name_or_path, tokenizer_kwargs)

    return model, tokenizer
