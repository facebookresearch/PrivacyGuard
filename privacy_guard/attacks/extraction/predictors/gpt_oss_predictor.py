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
GPT OSS predictor implementation for openai extraction attacks.
"""

from typing import Any, Dict, List

import transformers.utils.import_utils

from privacy_guard.attacks.extraction.predictors.huggingface_predictor import (
    HuggingFacePredictor,
)
from transformers.utils.import_utils import (
    _is_package_available,
    is_accelerate_available,
)


class GPTOSSPredictor(HuggingFacePredictor):
    """
    Inherits from HuggingFacePredictor and updates the generation logic to match
    GPT OSS expectation.

    Use this predictor for models like "gpt-oss-20b" and "gpt-oss-120b"

    Note: HuggingFacePredictor "get_logits" and "get_logprobs" behavior is
    not yet tested w/ GPTOSSPredictor
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        accelerate_available = self.accelerate_available_workaround()
        if not accelerate_available:
            raise ImportError(
                'Required library "accelerate"  for GPT OSS not available'
            )

        super().__init__(
            *args,
            **kwargs,
        )

    def accelerate_available_workaround(self) -> bool:
        """
        In old transformers versions, availability for the required 'accelerate' package
        is checked once at import time and the result is saved for all future checks.

        For Meta internal packaging this check returns as false at import time even when
        the package is available at runtime.

        This is a workaround which updates the saved values in transformers
        when this class is initialized.

        See the following link to the old transformers code pointer.
        https://github.com/huggingface/transformers/blob/
        e95441bdb586a7c3c9b4f61a41e99178c1becf54/src/transformers/utils/import_utils.py#L126
        """
        if is_accelerate_available():
            return True

        _accelerate_available, _accelerate_version = (  # pyre-ignore
            _is_package_available("accelerate", return_version=True)
        )

        if _accelerate_available:
            transformers.utils.import_utils._accelerate_available = (
                _accelerate_available
            )
            transformers.utils.import_utils._accelerate_version = _accelerate_version

            return is_accelerate_available()

        return False

    def preprocess_batch_messages(self, batch: List[str]) -> List[Dict[str, str]]:
        """
        Prepare a batch of messages for prediction.

        Differs than parent HuggingfacePredictor in that it returns a list of Dict
        instead of str, and includes "role" user field.
        """
        clean_batch = []
        for item in batch:
            if not isinstance(item, str):
                raise Warning(f"Found non-string item in batch: {type(item)}")
                clean_batch.append(str(item) if item is not None else "")
            else:
                clean_batch.append({"role": "user", "content": item})
        return clean_batch

    # Override
    def _generate_process_batch(
        self, batch: List[str], max_new_tokens: int = 512, **generation_kwargs: Any
    ) -> List[str]:
        """Process a single batch of prompts.
        apply_chat_template is used to apply the harmony response format, required for
        gpt models to work properly.
        """
        clean_batch: List[Dict[str, str]] = self.preprocess_batch_messages(batch)

        # Different than parent HuggingfacePredictor class
        add_generation_prompt = (
            True
            if "add_generation_prompt" not in generation_kwargs
            else generation_kwargs.pop("add_generation_prompt")
        )
        reasoning_effort = (
            "medium"
            if "reasoning_effort" not in generation_kwargs
            else generation_kwargs.pop("reasoning_effort")
        )
        inputs = self.tokenizer.apply_chat_template(  # pyre-ignore
            clean_batch,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            reasoning_effort=reasoning_effort,
        ).to(self.device)

        return self._generate_decode_logic(
            inputs=inputs, max_new_tokens=max_new_tokens, **generation_kwargs
        )
