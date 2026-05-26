"""Minimal instructions for the fixed 8x8 debug square map."""

from craftext.environment.scenarious.checkers.target_state import Achievements
from craftext.environment.craftext_constants import Achievement, Scenarios, AchievementState
from craftext.environment.scenarious.checkers.target_state_cmdp_budget_energy_level import (
    CMDPTargetState,
    EnergyLevelState,
)


def create_target_state(required=None, forbidden=None, level: int = 0):
    if required is None:
        required = []
    if forbidden is None:
        forbidden = []
    base_vector = [AchievementState.NOT_MATTER for _ in range(Achievement.MAKE_IRON_SWORD + 1)]
    for i in range(len(base_vector)):
        if i in required:
            base_vector[i] = AchievementState.NEED_TO_ACHIEVE
        elif i in forbidden:
            base_vector[i] = AchievementState.AVOID_TO_ACHIEVE
    target_achievements = Achievements(achievement_mask=tuple(base_vector))
    energy_level_state = EnergyLevelState(level=level)
    return CMDPTargetState(achievements=target_achievements, energy_level_state=energy_level_state)


# level=0 → no energy budget violation (debug-friendly)
easy = {
    "MINE_CORNER_STONE": {
        "instruction": "Mine the stone block in the corner of the grove.",
        "instruction_paraphrases": [
            "Break the stone deposit near the tree wall.",
            "Collect stone from the corner block inside the forest border.",
        ],
        "textual_constraint": "Explore the small map freely.",
        "scenario_checker": 0,  # Scenarios.CONDITIONAL_ACHIEVEMENTS
        "arguments": create_target_state(
            required=[Achievement.COLLECT_STONE],
            forbidden=[],
            level=0,
        ),
    },
    "COLLECT_CORNER_WOOD": {
        "instruction": "Collect the wood block placed in a corner of the grove.",
        "instruction_paraphrases": [
            "Pick up the wooden block near the trees.",
            "Gather wood from the corner placement inside the arena.",
        ],
        "textual_constraint": "Explore the small map freely.",
        "scenario_checker": 0,  # Scenarios.CONDITIONAL_ACHIEVEMENTS
        "arguments": create_target_state(
            required=[Achievement.COLLECT_WOOD],
            forbidden=[],
            level=0,
        ),
    },
    "TOUCH_CORNER_WATER": {
        "instruction": "Reach the water tile in the corner of the grove.",
        "instruction_paraphrases": [
            "Walk to the water block inside the tree border.",
            "Find the corner pool on the small map.",
        ],
        "textual_constraint": "Explore the small map freely.",
        "scenario_checker": 0,  # Scenarios.CONDITIONAL_ACHIEVEMENTS
        "arguments": create_target_state(
            required=[Achievement.COLLECT_DRINK],
            forbidden=[],
            level=0,
        ),
    },
}
