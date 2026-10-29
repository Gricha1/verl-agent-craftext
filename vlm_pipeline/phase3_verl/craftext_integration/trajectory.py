"""Per-turn trajectory bookkeeping used by the CrafText agent loop."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .rewards import route_turn_rewards


@dataclass
class TurnTrajectory:
    """Records where each environment reward belongs in a generated sequence."""

    turn_end_indices: list[int] = field(default_factory=list)
    turn_rewards: list[float] = field(default_factory=list)

    def append(self, *, response_mask: list[int], reward: float) -> None:
        trainable = [index for index, value in enumerate(response_mask) if value]
        if not trainable:
            raise ValueError("A CrafText turn produced no trainable assistant tokens")
        end_index = trainable[-1]
        if self.turn_end_indices and end_index <= self.turn_end_indices[-1]:
            raise ValueError("Turn boundaries must strictly advance")
        self.turn_end_indices.append(end_index)
        self.turn_rewards.append(float(reward))

    def dense_scores(self, response_mask: Sequence[int]) -> list[float]:
        return route_turn_rewards(response_mask, self.turn_end_indices, self.turn_rewards)

    def as_extra_fields(self) -> dict[str, list[float] | list[int]]:
        return {
            "craftext_turn_end_indices": self.turn_end_indices.copy(),
            "craftext_turn_rewards": self.turn_rewards.copy(),
        }
