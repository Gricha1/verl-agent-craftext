
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
  "EAT_COW": {
      "instruction": "Eat a cow.",
      "instruction_paraphrases": [
          "Consume beef from a butchered cow.",
          "Devour meat obtained from a bovine animal.",
          "Savor a meal made from cow flesh.",
          "Enjoy a dish prepared with beef.",
          "Ingest cow meat for nourishment."
      ],
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.EAT_COW],
          forbidden=[]
      )
  },
  "DEFEAT_ZOMBIE": {
      "instruction": "Defeat a zombie.",
      "instruction_paraphrases": [
          "Eliminate an undead creature lurking nearby.",
          "Vanquish a wandering zombie in combat.",
          "Destroy a rotting foe roaming the area.",
          "Take down a zombie using any weapon.",
          "Overcome a night-stalking undead being."
      ],
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.DEFEAT_ZOMBIE],
          forbidden=[]
      )
  },
  "DEFEAT_SKELETON": {
      "instruction": "Defeat a skeleton.",
      "instruction_paraphrases": [
          "Destroy a bony adversary roaming nearby.",
          "Vanquish a skeletal warrior in combat.",
          "Eliminate a skeleton using your weapon.",
          "Overpower a bone-clad enemy in battle.",
          "Take down a skeletal creature in the area."
      ],
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.DEFEAT_SKELETON],
          forbidden=[]
      )
  },
}


medium = {}