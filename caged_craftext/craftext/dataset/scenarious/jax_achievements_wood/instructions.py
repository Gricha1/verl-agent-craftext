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
            "Chop down some timber to gather wood.",
            "Cut a tree to obtain wooden resources.",
            "Retrieve lumber from a fallen tree.",
            "Procure wood by felling trees in the area."
        ],
        "textual_constraint": "You must maintain your energy level greater than 8.",
        "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
        "arguments": create_target_state(
            required=[Achievement.COLLECT_WOOD],
            forbidden=[],
            level=min_energy_level,
        )
    },
    "PLACE_TABLE": {
        "instruction": "Place a crafting table.",
        "instruction_paraphrases": [
            "Set down a crafting bench in the area.",
            "Install a crafting workstation at your location.",
            "Position a workbench for crafting nearby.",
            "Drop a table designed for crafting tasks.",
            "Arrange a crafting station in a suitable spot."
        ],
        "textual_constraint": "You must maintain your energy level greater than 8.",
        "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
        "arguments": create_target_state(
            required=[Achievement.PLACE_TABLE],
            forbidden=[],
            level=min_energy_level,
        )
    },
  "MAKE_WOOD_PICKAXE": {
      "instruction": "Craft a wooden pickaxe.",
      "instruction_paraphrases": [
          "Assemble a mining tool made of wood.",
          "Construct a wooden pickaxe for digging.",
          "Fashion a pickaxe out of wooden parts.",
          "Carve and build a wooden mining tool.",
          "Forge a lightweight pickaxe from wood."
      ],
      "textual_constraint": "You must maintain your energy level greater than 8.",
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.MAKE_WOOD_PICKAXE],
          forbidden=[],
          level=min_energy_level,
      )
  },
  "MAKE_WOOD_SWORD": {
      "instruction": "Craft a wooden sword.",
      "instruction_paraphrases": [
          "Forge a blade made from wooden materials.",
          "Carve a wooden sword for protection.",
          "Construct a simple sword using wood.",
          "Create a weapon crafted from timber.",
          "Build a wooden blade for self-defense."
      ],
      "textual_constraint": "You must maintain your energy level greater than 8.",
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.MAKE_WOOD_SWORD],
          forbidden=[],
          level=min_energy_level,
      )
  }
}


medium = {}
