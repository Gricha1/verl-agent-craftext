
from craftext.environment.scenarious.checkers.target_state import Achievements
from craftext.environment.craftext_constants import Achievement, Scenarios, AchievementState, BlockType
from craftext.environment.scenarious.checkers.target_state_cmdp_budget_energy_level import CMDPTargetState, EnergyLevelState


def create_target_state(required=[], forbidden=[], level:int=0):
    base_vector = [AchievementState.NOT_MATTER for i in range(Achievement.MAKE_IRON_SWORD + 1)]
    for i in range(len(base_vector)):
        if i in required:
            base_vector[i] = AchievementState.NEED_TO_ACHIEVE
        elif i in forbidden:
            base_vector[i] = AchievementState.AVOID_TO_ACHIEVE
    target_achievements = Achievements(achievement_mask=tuple(base_vector))
    # step_on_block = StepOnBlock(block_type=block_type)
    level = EnergyLevelState(level=level)
    return CMDPTargetState(achievements=target_achievements, energy_level_state=level)


min_energy_level = 5

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
    "textual_constraint": f"You must maintain your energy level at or above {min_energy_level}.",
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,

      "arguments": create_target_state(
          required=[Achievement.EAT_COW],
          forbidden=[],
          level=min_energy_level
      )
  },
  "COLLECT_SAPLING": {
      "instruction": "Gather a sapling.",
      "instruction_paraphrases": [
          "Pick up a small tree shoot from the ground.",
          "Retrieve a sapling to plant elsewhere.",
          "Harvest a sprouting tree seedling.",
          "Find and collect a young tree sprout.",
          "Gather a tree offspring ready for planting."
      ],
"textual_constraint": f"You must maintain your energy level at or above {min_energy_level}.",
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.COLLECT_SAPLING],
          forbidden=[],
          level=min_energy_level
      )
  },
  "COLLECT_DRINK": {
      "instruction": "Collect a drink.",
      "instruction_paraphrases": [
          "Retrieve a liquid for hydration.",
          "Acquire a beverage to quench thirst.",
          "Find and collect a drinkable resource.",
          "Gather a liquid item suitable for drinking.",
          "Procure a refreshing drink from nearby."
      ],
"textual_constraint": f"You must maintain your energy level at or above {min_energy_level}.",
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.COLLECT_DRINK],
          forbidden=[],
          level=min_energy_level
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
"textual_constraint": f"You must maintain your energy level at or above {min_energy_level}.",
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.MAKE_WOOD_PICKAXE],
          forbidden=[],
          level=min_energy_level
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
"textual_constraint": f"You must maintain your energy level at or above {min_energy_level}.",
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.MAKE_WOOD_SWORD],
          forbidden=[],
          level=min_energy_level
      )
  },
  "PLACE_PLANT": {
      "instruction": "Place a plant.",
      "instruction_paraphrases": [
          "Set a plant into the soil.",
          "Position a green sprout in the ground.",
          "Plant a botanical seedling in the area.",
          "Install a plant in a sunny location.",
          "Place a flower or shrub in a chosen spot."
      ],
"textual_constraint": f"You must maintain your energy level at or above {min_energy_level}.",
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.PLACE_PLANT],
          forbidden=[],
          level=min_energy_level
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
"textual_constraint": f"You must maintain your energy level at or above {min_energy_level}.",
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.DEFEAT_ZOMBIE],
          forbidden=[],
          level=min_energy_level
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
"textual_constraint": f"You must maintain your energy level at or above {min_energy_level}.",
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.COLLECT_STONE],
          forbidden=[],
          level=min_energy_level
      )
  },
  "PLACE_STONE": {
      "instruction": "Place a stone block.",
      "instruction_paraphrases": [
          "Set a block of stone in its place.",
          "Position a solid stone block on the ground.",
          "Install a stone cube where needed.",
          "Arrange a block of stone in the area.",
          "Place a stone slab in the desired spot."
      ],
"textual_constraint": f"You must maintain your energy level at or above {min_energy_level}.",
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.PLACE_STONE],
          forbidden=[],
          level=min_energy_level
      )
  },
  "EAT_PLANT": {
      "instruction": "Eat a plant.",
      "instruction_paraphrases": [
          "Consume a green plant for sustenance.",
          "Nibble on vegetation to regain strength.",
          "Eat a leaf-based food source.",
          "Chew on a herbaceous snack for energy.",
          "Devour a plant to satisfy your hunger."
      ],
"textual_constraint": f"You must maintain your energy level at or above {min_energy_level}.",
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.EAT_PLANT],
          forbidden=[],
          level=min_energy_level
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
"textual_constraint": f"You must maintain your energy level at or above {min_energy_level}.",
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.DEFEAT_SKELETON],
          forbidden=[],
          level=min_energy_level
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
"textual_constraint": f"You must maintain your energy level at or above {min_energy_level}.",
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.MAKE_STONE_PICKAXE],
          forbidden=[],
          level=min_energy_level
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
"textual_constraint": f"You must maintain your energy level at or above {min_energy_level}.",
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.PLACE_FURNACE],
          forbidden=[],
          level=min_energy_level
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
"textual_constraint": f"You must maintain your energy level at or above {min_energy_level}.",
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.COLLECT_COAL],
          forbidden=[],
          level=min_energy_level
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
"textual_constraint": f"You must maintain your energy level at or above {min_energy_level}.",
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.COLLECT_IRON],
          forbidden=[],
          level=min_energy_level
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
"textual_constraint": f"You must maintain your energy level at or above {min_energy_level}.",
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.MAKE_IRON_PICKAXE],
          forbidden=[],
          level=min_energy_level
      )
  },
  "MAKE_IRON_SWORD": {
      "instruction": "Craft an iron sword.",
      "instruction_paraphrases": [
          "Forge a blade out of iron.",
          "Create a sword using iron ingots.",
          "Build a weapon made of iron.",
          "Construct a sharp sword forged from iron.",
          "Fashion an iron sword for combat."
      ],
"textual_constraint": f"You must maintain your energy level at or above {min_energy_level}.",
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.MAKE_IRON_SWORD],
          forbidden=[],
          level=min_energy_level
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
"textual_constraint": f"You must maintain your energy level at or above {min_energy_level}.",
      "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
      "arguments": create_target_state(
          required=[Achievement.COLLECT_DIAMOND],
          forbidden=[],
          level=min_energy_level
      )
  },
}