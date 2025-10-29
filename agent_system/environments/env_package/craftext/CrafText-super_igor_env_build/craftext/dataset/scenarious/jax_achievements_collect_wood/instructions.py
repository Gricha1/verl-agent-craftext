
from craftext.enviroment.scenarious.checkers.target_state import Achievements, TargetState
from craftext.enviroment.craftext_constants import Achievement, Scenarios, AchievementState

def create_target_state(required=[], forbidden=[]):
    base_vector = [AchievementState.NOT_MATTER for i in range(Achievement.MAKE_IRON_SWORD + 1)]
    for i in range(len(base_vector)):
        if i in required:
            base_vector[i] = AchievementState.NEED_TO_ACHIEVE
        elif i in forbidden:
            base_vector[i] = AchievementState.AVOID_TO_ACHIEVE
    target_achievements = Achievements(achievement_mask=tuple(base_vector))
    return TargetState(achievements=target_achievements)


easy = { 
    "COLLECT_WOOD": {
        "instruction": "Collect wood.",
        "instruction_paraphrases": [
            "Harvest logs from nearby trees.",
        ],
        "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
        "arguments": create_target_state(
            required=[Achievement.COLLECT_WOOD],
            forbidden=[]
        )
    },
}


medium = {}