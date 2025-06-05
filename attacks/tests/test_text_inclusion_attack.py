# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import pandas as pd
from later.unittest import TestCase

from privacy_guard.attacks.text_inclusion_attack import TextInclusionAttack


class TestTextInclusionAttack(TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_exactly_one_input(self) -> None:
        with self.assertRaises(ValueError):
            _ = TextInclusionAttack(llm_generation_file="test", data=pd.DataFrame())
        with self.assertRaises(ValueError):
            _ = TextInclusionAttack(llm_generation_file=None, data=None)

        _ = TextInclusionAttack(llm_generation_file=None, data=pd.DataFrame())

    def test_sft_mode_assertion_not_user(self) -> None:
        sft_data_no_user = {
            "prompt": [
                {
                    "type": "SampleSFT",
                    "dialog": [{"body": "This is a test prompt", "source": "user"}],
                },
                {
                    "type": "SampleSFT",
                    "dialog": [
                        {"body": "This is another test prompt", "source": "agent"}
                    ],
                },
            ],
            "targets": [
                ["Target text 1"],
                ["Target text 2", "Another target"],
            ],
            "prediction": [
                "A success: Target text 1",
                "Failure, no match. ",
            ],
        }
        attack_node = TextInclusionAttack(data=pd.DataFrame(sft_data_no_user))

        with self.assertRaises(ValueError):
            _ = attack_node.preprocess_data()

    def test_sft_mode_assertion_multi_turn(self) -> None:
        sft_data_multi_turn = {
            "prompt": [
                {
                    "type": "SampleSFT",
                    "dialog": [
                        {"body": "This is a test prompt", "source": "user"},
                        {"body": "This a second turn.", "source": "agent"},
                    ],
                },
                {
                    "type": "SampleSFT",
                    "dialog": [
                        {"body": "This is another test prompt", "source": "user"}
                    ],
                },
            ],
            "targets": [
                ["Target text 1"],
                ["Target text 2", "Another target"],
            ],
            "prediction": [
                "A success: Target text 1",
                "Failure, no match. ",
            ],
        }

        attack_node = TextInclusionAttack(data=pd.DataFrame(sft_data_multi_turn))

        with self.assertRaises(NotImplementedError):
            _ = attack_node.preprocess_data()
