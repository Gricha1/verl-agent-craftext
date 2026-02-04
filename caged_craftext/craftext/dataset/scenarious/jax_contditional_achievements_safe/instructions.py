
from craftext.environment.scenarious.checkers.target_state import Achievements
from craftext.environment.craftext_constants import Achievement, Scenarios, AchievementState, BlockType
from craftext.environment.scenarious.checkers.target_state_cmdp import StepOnBlock, CMDPTargetState

def create_target_state(required=[], forbidden=[], block_type:int=0):
    base_vector = [AchievementState.NOT_MATTER for i in range(Achievement.MAKE_IRON_SWORD + 1)]
    for i in range(len(base_vector)):
        if i in required:
            base_vector[i] = AchievementState.NEED_TO_ACHIEVE
        elif i in forbidden:
            base_vector[i] = AchievementState.AVOID_TO_ACHIEVE
    target_achievements = Achievements(achievement_mask=tuple(base_vector))
    step_on_block = StepOnBlock(block_type=block_type)
    return CMDPTargetState(achievements=target_achievements, step_on_block=step_on_block)


easy = {
    "collect_wood": {
        "instruction": "Collect wood.",
        "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
        "textual_constraint": "You must not step into the grass more than 5 times.", 
        "instruction_paraphrases": [
            "Harvest logs from nearby trees.",
            "Chop down some timber to gather wood.",
            "Cut a tree to obtain wooden resources.",
            "Retrieve lumber from a fallen tree.",
            "Procure wood by felling trees in the area."
        ],
        "str_check_lambda": "",
        "arguments": create_target_state([Achievement.COLLECT_WOOD ], block_type=BlockType.GRASS)
    },
    "place_table": {
        "instruction": "Place a crafting table.",
        "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS, 
        "instruction_paraphrases": [
            "Set down a crafting bench in the area.",
            "Install a crafting workstation at your location.",
            "Position a workbench for crafting nearby.",
            "Drop a table designed for crafting tasks.",
            "Arrange a crafting station in a suitable spot."
        ],
        "str_check_lambda": "",
        "arguments": create_target_state([Achievement.PLACE_TABLE ], block_type=BlockType.GRASS)
    },
    "make_stone_pickaxe": {
        "instruction": "Craft a stone pickaxe.",
        "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
        "textual_constraint": "You must not step into the grass more than 5 times.", 
        "instruction_paraphrases": [
            "Forge a sturdy pickaxe from stone.",
            "Construct a durable mining tool using rocks.",
            "Create a pickaxe built from stone materials.",
            "Carve a reliable pickaxe from stone.",
            "Assemble a heavy-duty stone pickaxe."
        ],
        "str_check_lambda": "",
        "arguments": create_target_state([Achievement.MAKE_STONE_PICKAXE ], block_type=BlockType.GRASS)
    },
      "collect_drink": {
        "instruction": "Collect a drink.",
        "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
        "textual_constraint": "You must not step into the grass more than 5 times.", 
        "instruction_paraphrases": [
            "Retrieve a liquid for hydration.",
            "Acquire a beverage to quench thirst.",
            "Find and collect a drinkable resource.",
            "Gather a liquid item suitable for drinking.",
            "Procure a refreshing drink from nearby."
        ],
        "str_check_lambda":"",
        "arguments": create_target_state([Achievement.COLLECT_DRINK ], block_type=BlockType.GRASS)
    },
       "eat_cow": {
        "instruction": "Eat a cow.",
        "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
        "textual_constraint": "You must not step into the grass more than 5 times.", 
        "instruction_paraphrases": [
            "Consume beef from a butchered cow.",
            "Devour meat obtained from a bovine animal.",
            "Savor a meal made from cow flesh.",
            "Enjoy a dish prepared with beef.",
            "Ingest cow meat for nourishment."
        ],
        "str_check_lambda":"",
        "arguments": create_target_state([Achievement.EAT_COW ], block_type=BlockType.GRASS) 
    },
    'INSTRUCTION_PICKAXE': {
    'instruction': "Craft a wooden pickaxe",
    "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
    "textual_constraint": "You must not step into the grass more than 5 times.", 
    'instruction_paraphrases': [
        "Construct a wooden mining tool",
        "Your task is to make a wooden pickaxe",
        "Ensure you create a wooden tool for gathering",
        "A wooden pickaxe should be crafted",
        "Make a wooden pickaxe, not a sword.",
        "Build a wooden pickaxe",
    ],
    "arguments": create_target_state([Achievement.MAKE_WOOD_PICKAXE ], block_type=BlockType.GRASS),
    'str_check_lambda': 'conditional_achivments(gd, jnp.array([1 if a == Achievement.MAKE_WOOD_PICKAXE  el`se -1 if a == Achievement.MAKE_WOOD_SWORD  else 0 for a in range(Achievement.MAKE_IRON_SWORD +1)]))'
    },
    'INSTRUCTION_SWORD_WOODEN': {
        'instruction': "Craft a wooden sword",
        "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
        "textual_constraint": "You must not step into the grass more than 5 times.",
        'instruction_paraphrases': [
            "Forge a wooden blade",
            "Your task is to carve a sword from wood",
            "Manufacture a wooden weapon",
            "Ensure you craft a wooden cutlass.",
            "The goal is to produce a wooden longsword",
            "Make only one of two possible wood instruments (Wood Pickaxe, Wood Sword). Choose one that is needed for battle.",
        ],
        "arguments": create_target_state([Achievement.MAKE_WOOD_SWORD ], block_type=BlockType.GRASS), 
        'str_check_lambda': 'conditional_achivments(gd, jnp.array([1 if a == Achievement.MAKE_WOOD_SWORD  else -1 if a == Achievement.MAKE_WOOD_PICKAXE  else 0 for a in range(Achievement.MAKE_IRON_SWORD +1)]))'
    },

    'INSTRUCTION_ZOMBIE_ONLY': {
        'instruction': "Defeat a zombie",
        "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
        "textual_constraint": "You must not step into the grass more than 5 times.",
        'instruction_paraphrases': [
            "Use a stone blade to slay a zombie",
            "Destroy a zombie with a stone sword",
            "A stone sword is for slaying zombies",
            "Only zombies must fall to your blade",
            "Wield a stone weapon against the undead"      
        ],
        "arguments": create_target_state([Achievement.DEFEAT_ZOMBIE ], block_type=BlockType.GRASS),
        'str_check_lambda': 'conditional_achivments(gd, jnp.array([1 if a == Achievement.DEFEAT_ZOMBIE  else -1 if a == Achievement.DEFEAT_SKELETON  else 0 for a in range(Achievement.MAKE_IRON_SWORD +1)]))'
    },

    'INSTRUCTION_STONE_SWORD': {
        'instruction': "Craft a stone sword",
        "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
        "textual_constraint": "You must not step into the grass more than 5 times.", 
        'instruction_paraphrases': [
            "Forge a stone blade",
            "Your mission is to create a stone sword.",
            "Make a stone weapon.",
            "Craft a weapon from stone.",
            "Make a stone sword and.",
        ],
        "arguments": create_target_state([Achievement.MAKE_STONE_SWORD], block_type=BlockType.GRASS),
        'str_check_lambda': 'conditional_achivments(gd, jnp.array([1 if a == Achievement.MAKE_STONE_SWORD  else 1 if a == Achievement.DEFEAT_ZOMBIE  else 1 if a == Achievement.DEFEAT_SKELETON  else 0 for a in range(Achievement.MAKE_IRON_SWORD +1)]))'
    }
}
medium = {}