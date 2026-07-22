"""Minimal instructions for the fixed 16x16 debug square map (v2)."""

from craftext.environment.scenarious.checkers.target_state import Achievements
from craftext.environment.craftext_constants import Achievement, AchievementState
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


# Same three nav tasks as 8x8; resource positions come from the live 16x16 map each reset
# (inner corners: (1,1), (1,14), (14,1), (14,14); spawn at (8,8)).
easy = {
    "GO_TO_STONE": {
        "instruction": "Go to the stone.",
        "instruction_paraphrases": [
            "Walk next to the stone block.",
            "Approach the stone in the corner.",
        ],
        "textual_constraint": "",
        "scenario_checker": 0,
        "arguments": create_target_state(
            required=[Achievement.COLLECT_STONE],
            forbidden=[],
            level=0,
        ),
    },
    "GO_TO_WOOD": {
        "instruction": "Go to the wooden block.",
        "instruction_paraphrases": [
            "Walk next to the wooden block (w tile, not border trees).",
            "Approach the wood block in the corner.",
        ],
        "textual_constraint": "",
        "scenario_checker": 0,
        "arguments": create_target_state(
            required=[Achievement.COLLECT_WOOD],
            forbidden=[],
            level=0,
        ),
    },
    "GO_TO_WATER": {
        "instruction": "Go to the water.",
        "instruction_paraphrases": [
            "Walk next to the water block.",
            "Approach the water in the corner.",
        ],
        "textual_constraint": "",
        "scenario_checker": 0,
        "arguments": create_target_state(
            required=[Achievement.COLLECT_DRINK],
            forbidden=[],
            level=0,
        ),
    },
}
