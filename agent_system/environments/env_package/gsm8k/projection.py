# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.

from typing import List, Tuple


def gsm8k_projection(text_actions: List[str]) -> Tuple[List[str], List[int]]:
    """Pass model responses through; GSM8K scoring happens in the env step."""
    valids = [1] * len(text_actions)
    return list(text_actions), valids
