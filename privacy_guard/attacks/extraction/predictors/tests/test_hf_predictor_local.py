# pyre-strict

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
Test script for HuggingFacePredictor generation capabilities.

This script demonstrates how to use the HuggingFacePredictor class to generate text
continuations from a list of prompts. It serves as both a test and example usage.
"""

import argparse
import logging
from typing import List

from privacy_guard.attacks.extraction.predictors.huggingface_predictor import (
    HuggingFacePredictor,
)


def setup_logger() -> logging.Logger:
    """Set up the logger for the script."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Remove any existing handlers to avoid duplicates
    logger.handlers.clear()

    logger.propagate = False

    # Create console handler and set level
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


logger: logging.Logger = setup_logger()


def test_model_generation(
    model_name: str,
    prompts: List[str],
    device: str = "cuda",
    batch_size: int = 2,
    max_new_tokens: int = 100,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> None:
    """
    Test the HuggingFacePredictor generation capabilities.

    Args:
        model_name: HuggingFace model name or path
        prompts: List of prompts to test generation with
        device: Device to use for inference
        batch_size: Batch size for generation
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Temperature for sampling
        top_p: Top-p value for nucleus sampling
    """
    logger.info(f"Initializing HuggingFacePredictor with {model_name}")

    try:
        # Initialize the predictor
        predictor = HuggingFacePredictor(model_name=model_name, device=device)
        logger.info("Predictor initialized successfully")

        # Test generation
        logger.info(f"Testing generation with {len(prompts)} prompts")
        logger.info(
            f"Generation parameters: batch_size={batch_size}, max_new_tokens={max_new_tokens}, temperature={temperature}, top_p={top_p}, top_k={top_k}"
        )

        generated_texts = predictor.generate(
            prompts=prompts,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True if temperature or top_p else False,
            top_k=top_k,
        )

        # Display results
        logger.info("Generation completed. Results:")
        print("\n" + "=" * 80)
        for i, (prompt, generated) in enumerate(zip(prompts, generated_texts)):
            print(f"\nPrompt {i+1}:")
            print(f"Input:  {prompt}")
            print(f"Output: {generated}")
            print("-" * 40)
        print("=" * 80 + "\n")

        logger.info("Test completed successfully")

    except Exception as e:
        logger.error(f"Error during model testing: {str(e)}")
        raise


def test_logits_extraction(
    model_name: str,
    prompts: List[str],
    targets: List[str],
    device: str = "cuda",
    batch_size: int = 2,
) -> None:
    """
    Test the HuggingFacePredictor get_logits capabilities.

    Args:
        model_name: HuggingFace model name or path
        prompts: List of prompts to test logits extraction with
        targets: List of target sequences to extract logits for
        device: Device to use for inference
        batch_size: Batch size for processing
    """
    logger.info(f"Testing logits extraction with {model_name}")

    try:
        # Initialize the predictor
        predictor = HuggingFacePredictor(model_name=model_name, device=device)
        logger.info("Predictor initialized successfully for logits testing")

        # Test logits extraction
        logger.info(
            f"Testing logits extraction with {len(prompts)} prompt-target pairs"
        )
        logger.info(f"Using batch size: {batch_size}")

        logits_list = predictor.get_logits(
            prompts=prompts, targets=targets, batch_size=batch_size
        )

        # Get vocabulary size for expected shape calculation
        vocab_size = predictor.model.config.vocab_size  # pyre-ignore

        logger.info("Logits extraction completed. Results:")
        print("\n" + "=" * 80)
        print(f"Number of logits tensors returned: {len(logits_list)}")

        # Print expected shapes for each example
        for i, (prompt, target) in enumerate(zip(prompts, targets)):
            target_tokens = predictor.tokenizer(target, return_tensors="pt")
            target_length = target_tokens.input_ids.shape[1] - 1
            # Remove the start token from the target length
            expected_shape = (target_length, vocab_size)

            print(f"\nExample {i+1}:")
            print(f"Prompt: {prompt}")
            print(f"Target: {target}")
            print(f"Target tokens: {target_tokens}")
            print(f"Target token length: {target_length}")
            print(f"Expected logits shape: {expected_shape}")
            if i < len(logits_list):
                actual_shape = logits_list[i].shape
                print(f"Actual logits shape: {actual_shape}")
                # Verify the shape matches expectations
                shape_match = actual_shape == tuple(expected_shape)
                print(f"Shape matches expected: {'✓' if shape_match else '✗'}")
            print("-" * 40)

        print("=" * 80 + "\n")

        logger.info("Logits extraction test completed successfully")

    except Exception as e:
        logger.error(f"Error during logits testing: {str(e)}")
        raise


def test_logprobs_extraction(
    model_name: str,
    prompts: List[str],
    targets: List[str],
    device: str = "cuda",
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    batch_size: int = 2,
) -> None:
    """
    Test the HuggingFacePredictor get_logprobs capabilities.

    Args:
        model_name: HuggingFace model name or path
        prompts: List of prompts to test logprobs extraction with
        targets: List of target sequences to extract logprobs for
        device: Device to use for inference
        temperature: Temperature for scaling logits
        top_k: Top-k filtering parameter
        top_p: Top-p (nucleus) sampling parameter
    """
    logger.info(f"Testing log probabilities extraction with {model_name}")

    try:
        # Initialize the predictor
        predictor = HuggingFacePredictor(model_name=model_name, device=device)
        logger.info("Predictor initialized successfully for logprobs testing")

        # Test logprobs extraction
        logger.info(
            f"Testing logprobs extraction with {len(prompts)} prompt-target pairs"
        )
        logger.info(
            f"Generation parameters: temperature={temperature}, top_k={top_k}, top_p={top_p}"
        )

        logprobs_list = predictor.get_logprobs(
            prompts=prompts,
            targets=targets,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            batch_size=batch_size,
        )

        logger.info("Log probabilities extraction completed. Results:")
        print("\n" + "=" * 80)
        print(f"Number of logprobs tensors returned: {len(logprobs_list)}")

        # Print information for each example
        for i, (prompt, target) in enumerate(zip(prompts, targets)):
            target_tokens = predictor.tokenizer(target, return_tensors="pt")
            target_length = target_tokens.input_ids.shape[1] - 1
            # Remove the start token from the target length

            print(f"\nExample {i+1}:")
            print(f"Prompt: {prompt}")
            print(f"Target: {target}")
            print(f"Target token length: {target_length}")
            if i < len(logprobs_list):
                logprobs = logprobs_list[i]
                print(f"Logprobs tensor shape: {logprobs.shape}")
                print(f"Logprobs values: {logprobs.tolist()}")
                print(f"Sum of logprobs (sequence score): {logprobs.sum().item():.4f}")
                print(f"Mean logprob per token: {logprobs.mean().item():.4f}")

                # Validate that all logprobs are <= 0 (mathematical requirement)
                max_logprob = logprobs.max().item()
                all_valid = max_logprob <= 0.0
                print(f"Max logprob value: {max_logprob:.6f}")
                print(f"All logprobs <= 0: {'✓' if all_valid else '✗'}")
                if not all_valid:
                    print(
                        "ERROR: Found logprob > 0! This violates mathematical constraints."
                    )

                # Convert to probabilities for better interpretation
                probs = logprobs.exp()
                print(f"Token probabilities: {probs.tolist()}")
            print("-" * 40)

        print("=" * 80 + "\n")

        logger.info("Log probabilities extraction test completed successfully")

    except Exception as e:
        logger.error(f"Error during logprobs testing: {str(e)}")
        raise


def get_default_prompts() -> List[str]:
    """Get a list of default test prompts."""
    return [
        "The capital of France is",
        "In a world where artificial intelligence",
        "Once upon a time, in a distant galaxy",
        "The benefits of renewable energy include",
        "The recipe for chocolate chip cookies requires",
    ]


def get_default_targets() -> List[str]:
    """Get a list of default test targets."""
    return [
        " Paris",
        " is becoming more",
        " far, far away, there was",
        " reduced carbon emissions",
        " flour, sugar, and butter",
    ]


def main() -> None:
    """Main function to run the generation and logits tests."""
    parser = argparse.ArgumentParser(
        description="Test HuggingFacePredictor generation and logits capabilities"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="/home/knchadha/models/Llama-3.2-1B",
        help="HuggingFace model name or path (default: gpt2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cuda", "cpu"],
        help="Device to use for inference (default: cuda)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for generation (default: 2)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate (default: 100)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling (default: 0.7)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p value for nucleus sampling (default: 0.9)",
    )
    parser.add_argument(
        "--prompts",
        nargs="*",
        help="Custom prompts to test (if not provided, uses default prompts)",
    )
    parser.add_argument(
        "--test_generations",
        action="store_true",
        help="Run text generation test",
    )
    parser.add_argument(
        "--test_logits",
        action="store_true",
        help="Run logits extraction test",
    )
    parser.add_argument(
        "--test_logprobs",
        action="store_true",
        help="Run log probabilities extraction test",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k filtering for logprobs test (default: None)",
    )

    args = parser.parse_args()

    # Use custom prompts if provided, otherwise use defaults
    test_prompts = args.prompts if args.prompts else get_default_prompts()

    # Determine which tests to run
    run_generations = args.test_generations
    run_logits = args.test_logits
    run_logprobs = args.test_logprobs

    # If no flags are specified, run all tests for backward compatibility
    if not run_generations and not run_logits and not run_logprobs:
        run_generations = True
        run_logits = True
        run_logprobs = True

    logger.info("Starting HuggingFacePredictor tests")
    logger.info(f"Using {len(test_prompts)} test prompts")

    tests_to_run = []
    if run_generations:
        tests_to_run.append("text generation")
    if run_logits:
        tests_to_run.append("logits extraction")
    if run_logprobs:
        tests_to_run.append("log probabilities extraction")

    logger.info(f"Tests to run: {', '.join(tests_to_run)}")

    # Run generation test if requested
    if run_generations:
        test_model_generation(
            model_name=args.model_name,
            prompts=test_prompts,
            device=args.device,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )

    # Run logits test if requested
    if run_logits:
        test_targets = get_default_targets()
        test_logits_extraction(
            model_name=args.model_name,
            prompts=test_prompts,
            targets=test_targets,
            device=args.device,
            batch_size=args.batch_size,
        )

    # Run logprobs test if requested
    if run_logprobs:
        test_targets = get_default_targets()
        test_logprobs_extraction(
            model_name=args.model_name,
            prompts=test_prompts,
            targets=test_targets,
            device=args.device,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
