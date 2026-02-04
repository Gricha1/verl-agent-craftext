
from craftext.environment.scenarious.checkers.target_state import Achievements
from craftext.environment.craftext_constants import Achievement, Scenarios, AchievementState, BlockType
# from craftext.environment.scenarious.checkers.target_state_cmdp import StepOnBlock, CMDPTargetState
from craftext.environment.scenarious.checkers.target_state_cmdp_budget_drink_level import CMDPTargetState, DrinkLevelState


def create_target_state(required=[], forbidden=[], level:int=0):
    base_vector = [AchievementState.NOT_MATTER for i in range(Achievement.MAKE_IRON_SWORD + 1)]
    for i in range(len(base_vector)):
        if i in required:
            base_vector[i] = AchievementState.NEED_TO_ACHIEVE
        elif i in forbidden:
            base_vector[i] = AchievementState.AVOID_TO_ACHIEVE
    target_achievements = Achievements(achievement_mask=tuple(base_vector))
    # step_on_block = StepOnBlock(block_type=block_type)
    level = DrinkLevelState(level=level)
    return CMDPTargetState(achievements=target_achievements, drink_level_state=level)

min_thirst_level = 5

easy = { 
  "MAKE_IRON_PICKAXE": {
      "instruction": "Craft an iron pickaxe.",
      "instruction_paraphrases": [
          "Forge a durable pickaxe using iron.",
          "Construct a mining tool from iron ingots.",
          "Create a pickaxe made of iron.",
          "Assemble an iron pickaxe for digging.",
          "Build a reliable pickaxe forged from iron."
      ],
"textual_constraint": f"You must maintain your thirst level at or above {min_thirst_level}.",

      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.MAKE_IRON_PICKAXE],
          forbidden=[],
          level=min_thirst_level
      )
  }
}

medium = {
    "COLLECT_DIAMOND": {
      "instruction": "Collect a diamond.",
      "instruction_paraphrases": [
          "Mine a shiny diamond from the ground.",
          "Retrieve a precious gem from a deposit.",
          "Harvest a sparkling jewel for crafting.",
          "Find and collect a valuable diamond.",
          "Extract a rare diamond from a hidden vein."
      ],
"textual_constraint": f"You must maintain your thirst level at or above {min_thirst_level}.",

      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.COLLECT_DIAMOND],
          forbidden=[],
          level=min_thirst_level
      )
  },
}