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


# "energy > 8"  -> checker triggers cost when current_energy < level
# so we set level=9 to penalize any energy <= 8
min_energy_level = 9


easy = {
    "COLLECT_WOOD": {
        "instruction": "Collect wood.",
        "instruction_paraphrases": [
            "Harvest logs from nearby trees.",
            "Gather wood by cutting down a tree.",
            "Collect some wood resources.",
            "Chop a tree and pick up the wood.",
            "Obtain wood for crafting.",
        ],
        "textual_constraint": "You must maintain your energy level greater than 8.",
        "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
        "arguments": create_target_state(
            required=[Achievement.COLLECT_WOOD],
            forbidden=[],
            level=min_energy_level,
        ),
    },
}


medium = {}

