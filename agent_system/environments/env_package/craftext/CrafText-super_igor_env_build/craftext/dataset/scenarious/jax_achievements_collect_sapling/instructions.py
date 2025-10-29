
from craftext.enviroment.craftext_constants import Achievement, AchievementState, Scenarios
from craftext.enviroment.scenarious.checkers.target_state import Achievements, TargetState


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
    "COLLECT_SAPLING": {
      "instruction": "Gather a sapling.",
      "instruction_paraphrases": [
          "Pick up a small tree shoot from the ground.",
          "Retrieve a sapling to plant elsewhere.",
          "Harvest a sprouting tree seedling.",
          "Find and collect a young tree sprout.",
          "Gather a tree offspring ready for planting."
      ],
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.COLLECT_SAPLING],
          forbidden=[]
      )
  },
}


medium = {}