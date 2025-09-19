# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class BaseAnalysisOutput:
    """
    Base dataclass to encapsulate the outputs of analysis nodes.
    """

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the analysis output to a dictionary.
        """
        return asdict(self)
