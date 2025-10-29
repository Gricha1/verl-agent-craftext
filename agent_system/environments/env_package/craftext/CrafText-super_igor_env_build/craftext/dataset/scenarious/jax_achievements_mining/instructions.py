
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
    "PLACE_TABLE": {
        "instruction": "Place a crafting table.",
        "instruction_paraphrases": [
            "Install a crafting workstation at your location.",
        ],
        "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
        "arguments": create_target_state(
            required=[Achievement.PLACE_TABLE],
            forbidden=[]
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
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.MAKE_WOOD_PICKAXE],
          forbidden=[]
      )
  },
  "COLLECT_STONE": {
      "instruction": "Collect stone.",
      "instruction_paraphrases": [
          "Mine stone blocks from a rocky surface.",
          "Break apart rocks to gather stone.",
          "Harvest stone materials from the ground.",
          "Retrieve stone fragments from nearby boulders.",
          "Extract useful stone for crafting purposes."
      ],
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.COLLECT_STONE],
          forbidden=[]
      )
  },
  "MAKE_STONE_PICKAXE": {
      "instruction": "Craft a stone pickaxe.",
      "instruction_paraphrases": [
          "Forge a sturdy pickaxe from stone.",
          "Construct a durable mining tool using rocks.",
          "Create a pickaxe built from stone materials.",
          "Carve a reliable pickaxe from stone.",
          "Assemble a heavy-duty stone pickaxe."
      ],
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.MAKE_STONE_PICKAXE],
          forbidden=[]
      )
  },
  "PLACE_FURNACE": {
      "instruction": "Place a furnace.",
      "instruction_paraphrases": [
          "Set up a furnace for smelting.",
          "Install a furnace at your location.",
          "Position a smelter in the area.",
          "Drop a furnace for crafting needs.",
          "Put a furnace on the ground to use."
      ],
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.PLACE_FURNACE],
          forbidden=[]
      )
  },
  "COLLECT_COAL": {
      "instruction": "Collect coal.",
      "instruction_paraphrases": [
          "Mine coal from the ground.",
          "Harvest coal for smelting purposes.",
          "Retrieve black ore to use as fuel.",
          "Extract coal from a nearby deposit.",
          "Gather some coal for crafting."
      ],
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.COLLECT_COAL],
          forbidden=[]
      )
  },
  "COLLECT_IRON": {
      "instruction": "Collect iron.",
      "instruction_paraphrases": [
          "Mine iron ore from the earth.",
          "Gather raw iron for forging.",
          "Extract iron from a nearby vein.",
          "Retrieve iron ore for crafting tools.",
          "Harvest metallic resources for use."
      ],
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.COLLECT_IRON],
          forbidden=[]
      )
  },
  "COLLECT_DIAMOND": {
      "instruction": "Collect a diamond.",
      "instruction_paraphrases": [
          "Mine a shiny diamond from the ground.",
          "Retrieve a precious gem from a deposit.",
          "Harvest a sparkling jewel for crafting.",
          "Find and collect a valuable diamond.",
          "Extract a rare diamond from a hidden vein."
      ],
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.COLLECT_DIAMOND],
          forbidden=[]
      )
  },
  "MAKE_IRON_PICKAXE": {
      "instruction": "Craft an iron pickaxe.",
      "instruction_paraphrases": [
          "Forge a durable pickaxe using iron.",
          "Construct a mining tool from iron ingots.",
          "Create a pickaxe made of iron.",
          "Assemble an iron pickaxe for digging.",
          "Build a reliable pickaxe forged from iron."
      ],
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.MAKE_IRON_PICKAXE],
          forbidden=[]
      )
  },
}


medium = {}