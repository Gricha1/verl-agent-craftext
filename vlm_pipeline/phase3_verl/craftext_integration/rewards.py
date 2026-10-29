"""Dense CrafText reward placement for a concatenated multi-turn response."""

from __future__ import annotations

from collections.abc import Sequence


def route_turn_rewards(
    response_mask: Sequence[int], turn_end_indices: Sequence[int], turn_rewards: Sequence[float]
) -> list[float]:
    """Place every environment reward at its own turn's final trainable token.

    Modern veRL's generic agent loop assigns one scalar reward to the final
    response token.  CrafText produces one environment transition per assistant
    turn, so that convention would collapse all dense ``r_t`` into the terminal
    answer.  This helper is the contract used by the custom CrafText loop.
    """

    if len(turn_end_indices) != len(turn_rewards):
        raise ValueError("turn_end_indices and turn_rewards must have equal length")
    routed = [0.0] * len(response_mask)
    previous = -1
    for index, reward in zip(turn_end_indices, turn_rewards, strict=True):
        if index <= previous or index < 0 or index >= len(response_mask):
            raise ValueError("turn end indices must be strictly increasing response positions")
        if int(response_mask[index]) != 1:
            raise ValueError("a CrafText reward must target a trainable assistant token")
        routed[index] = float(reward)
        previous = index
    return routed
