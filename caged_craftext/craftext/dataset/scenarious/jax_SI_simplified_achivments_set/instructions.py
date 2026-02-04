from functools import partial

from enum import Enum

from craftext.environment.scenarious.checkers.achivments import conditional_achivments
from craftext.environment.scenarious.checkers.target_state import Achievements, TargetState
from craftext.environment.craftext_constants import Achievement, Scenarios, AchievementState
import jax
import jax.numpy as jnp
from jax import lax
from enum import Enum
from jax.tree_util import Partial  # Используем jax.tree_util.Partial
import re

def extract_achievements_and_numbers(input_string):
    # Регулярное выражение для поиска Achievement.*?\  и чисел
    pattern = re.compile(r'Achievement\.\w+\ |[-]?\d+')
    
    # Найдем все совпадения
    matches = pattern.findall(input_string)
    
    # Объединим результат в строку через пробел
    return " ".join(matches)

def split_achievements(matches):
    first_part = []
    second_part = []
    temp_part = []
    
    state = "first"
    for item in matches:
        if item == "-1":
            state = "temp"
            continue
        elif item == "0":
            state = "second"
            continue
        
        if item.startswith("Achievement."):
            if state == "first":
                first_part.append(item)
            elif state == "temp":
                temp_part.append(item)
            elif state == "second":
                second_part.append(item)
    
    return first_part, temp_part

def format_to_target_state(input_str):
  mathces = split_achievements(extract_achievements_and_numbers(input_str).split())
 # print(mathces)
  str_matchses = str(mathces).replace("'", "")
  arguments =  f'create_target_state{str_matchses}'
  return arguments
  
@jax.jit
def _check_func(gd, ix, achievement_mask):
    return conditional_achivments(gd, achievement_mask)

def create_check_lambda(achievements, penalties=None):

    if penalties is None:
        penalties = set()

    achievement_mask = jnp.array([
        1 if a in achievements else
        -1 if a in penalties else 0
        for a in range(Achievement.MAKE_IRON_SWORD + 1)
    ])
    return Partial(_check_func, achievement_mask=achievement_mask)

def create_target_state(required=[], forbidden=[]):
    base_vector = [AchievementState.NOT_MATTER for i in range(Achievement.MAKE_IRON_SWORD + 1)]
    for i in range(len(base_vector)):
        if i in required:
            base_vector[i] = AchievementState.NEED_TO_ACHIEVE
        elif i in forbidden:
            base_vector[i] = AchievementState.AVOID_TO_ACHIEVE
    target_achievements = Achievements(tuple(base_vector))
    return TargetState(achievements=target_achievements)

add =  [{
        "instruction": "Collect wood.",
        "scenario_checker": Scenarios.CONDITIONAL_ACIEVEMENTS.value, 
        "instruction_paraphrases": [
            "Harvest logs from nearby trees.",
            "Chop down some timber to gather wood.",
            "Cut a tree to obtain wooden resources.",
            "Retrieve lumber from a fallen tree.",
            "Procure wood by felling trees in the area."
        ],
        "str_check_lambda": "",
        "arguments": create_target_state([Achievement.COLLECT_WOOD.value])
    },
   {
        "instruction": "Chop some wood and install a workbench.",
        "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS, 
        "instruction_paraphrases": [
            "Gather wood and set down a crafting table.",
            "Harvest timber and position a crafting bench nearby.",
            "Chop some wood and install a workbench.",
            "Retrieve logs and put a crafting table on the ground.",
            "Procure wood and place a crafting table in the area."
        ],
        "str_check_lambda": "",
        "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS, 
        "arguments": create_target_state(required=[Achievement.COLLECT_WOOD , Achievement.PLACE_TABLE ]),
      #  "arguments": create_target_state([Achievement.COLLECT_WOOD , Achievement.PLACE_TABLE ])
    }

# one = {
#     "1. collect_wood_place_table": {
#         "instruction": "Chop some wood and install a workbench.",
# "scenario_checker": Scenarios.CONDITIONAL_ACIEVEMENTS ,         
# "instruction_paraphrases": [
#             "Gather wood and set down a crafting table.",
#             "Harvest timber and position a crafting bench nearby.",
#             "Chop some wood and install a workbench.",
#             "Retrieve logs and put a crafting table on the ground.",
#             "Procure wood and place a crafting table in the area."
#         ],
#         "str_check_lambda": "",
#         "scenario_checker": Scenarios.CONDITIONAL_ACIEVEMENTS , 
#         "arguments": create_target_state(required=[Achievement.COLLECT_WOOD , Achievement.PLACE_TABLE ]),
#       #  "arguments": create_target_state([Achievement.COLLECT_WOOD , Achievement.PLACE_TABLE ])
#     },
#     "2. collect_wood_place_table": {
#         "instruction": "Harvest timber and position a crafting bench nearby.",
# "scenario_checker": Scenarios.CONDITIONAL_ACIEVEMENTS ,         
# "instruction_paraphrases": [
#             "Gather wood and set down a crafting table.",
#             "Harvest timber and position a crafting bench nearby.",
#             "Chop some wood and install a workbench.",
#             "Retrieve logs and put a crafting table on the ground.",
#             "Procure wood and place a crafting table in the area."
#         ],
#         "str_check_lambda": "",
#         "scenario_checker": Scenarios.CONDITIONAL_ACIEVEMENTS , 
#         "arguments": create_target_state(required=[Achievement.COLLECT_WOOD , Achievement.PLACE_TABLE ]),
#       #  "arguments": create_target_state([Achievement.COLLECT_WOOD , Achievement.PLACE_TABLE ])
#     },
# }

one = {str(i):add[i] for i in range(3)}

        
easy = {
    "collect_wood": {
        "instruction": "Collect wood.",
        "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS, 
        "instruction_paraphrases": [
            "Harvest logs from nearby trees.",
            "Chop down some timber to gather wood.",
            "Cut a tree to obtain wooden resources.",
            "Retrieve lumber from a fallen tree.",
            "Procure wood by felling trees in the area."
        ],
        "str_check_lambda": "",
        "arguments": create_target_state([Achievement.COLLECT_WOOD ])
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
        "arguments": create_target_state([Achievement.PLACE_TABLE ])
    },
    "make_stone_pickaxe": {
        "instruction": "Craft a stone pickaxe.",
        "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS, 
        "instruction_paraphrases": [
            "Forge a sturdy pickaxe from stone.",
            "Construct a durable mining tool using rocks.",
            "Create a pickaxe built from stone materials.",
            "Carve a reliable pickaxe from stone.",
            "Assemble a heavy-duty stone pickaxe."
        ],
        "str_check_lambda": "",
        "arguments": create_target_state([Achievement.MAKE_STONE_PICKAXE ])
    },
      "collect_drink": {
        "instruction": "Collect a drink.",
        "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS, 
        "instruction_paraphrases": [
            "Retrieve a liquid for hydration.",
            "Acquire a beverage to quench thirst.",
            "Find and collect a drinkable resource.",
            "Gather a liquid item suitable for drinking.",
            "Procure a refreshing drink from nearby."
        ],
        "str_check_lambda":"",
        "arguments": create_target_state([Achievement.COLLECT_DRINK ])
    },
       "eat_cow": {
        "instruction": "Eat a cow.",
        "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS, 
        "instruction_paraphrases": [
            "Consume beef from a butchered cow.",
            "Devour meat obtained from a bovine animal.",
            "Savor a meal made from cow flesh.",
            "Enjoy a dish prepared with beef.",
            "Ingest cow meat for nourishment."
        ],
        "str_check_lambda":"",
        "arguments": create_target_state([Achievement.EAT_COW ]) 
    },
    'INSTRUCTION_PICKAXE': {
    'instruction': "Craft a wooden pickaxe but avoid making a wooden sword.",
    "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS, 
    'instruction_paraphrases': [
        "Construct a wooden mining tool, but do not forge a wooden blade.",
        "Your task is to make a wooden pickaxe—do not craft a sword.",
        "Ensure you create a wooden tool for gathering, yet refrain from shaping a weapon.",
        "A wooden pickaxe should be crafted, a wooden sword should not.",
        "Make a wooden pickaxe, not a sword.",
        "Do not make a wood sword, it is unnecessary. But craft a wood pickaxe.",
        "Build a wooden pickaxe; there is no need for a sword.",
        "A wood pickaxe is essential for gathering, a sword is not.",
        "You need a wood pickaxe. Forget about the sword.",
        "A wooden mining tool must be crafted, but avoid making a blade."
    ],
    "arguments": create_target_state([Achievement.MAKE_WOOD_PICKAXE ], [Achievement.MAKE_WOOD_SWORD ]),
    'str_check_lambda': 'conditional_achivments(gd, jnp.array([1 if a == Achievement.MAKE_WOOD_PICKAXE  el`se -1 if a == Achievement.MAKE_WOOD_SWORD  else 0 for a in range(Achievement.MAKE_IRON_SWORD +1)]))'
},
'INSTRUCTION_SWORD': {
    'instruction': "Craft a wooden sword but avoid making a wooden pickaxe.",
    "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS, 
    'instruction_paraphrases': [
        "Forge a wooden blade, but make sure not to construct a timber pickaxe.",
        "Your task is to carve a sword from wood, yet you must abstain from fashioning a wooden mining tool.",
        "Manufacture a wooden weapon, but under no circumstances should you assemble a lumber-based pickaxe.",
        "Ensure you craft a wooden cutlass while strictly avoiding the creation of a wooden digging tool.",
        "The goal is to produce a wooden longsword, but be cautious not to craft a wood-sourced pickaxe.",
        "Make only one of two possible wood instruments (Wood Pickaxe, Wood Sword). Choose one that is needed for battle.",
        "Create a single wooden tool: either a sword or a pickaxe. Pick the one required for combat.",
        "Choose one wooden tool to make: a pickaxe or a sword. The right choice is what helps in combat.",
        "You will have a fight: craft with wood a weapon or a mining tool—only one is allowed.",
        "Craft a wooden sword, as a pickaxe is not required."
    ],
    "arguments": create_target_state([Achievement.MAKE_WOOD_SWORD ], [Achievement.MAKE_WOOD_PICKAXE ]), 
    'str_check_lambda': 'conditional_achivments(gd, jnp.array([1 if a == Achievement.MAKE_WOOD_SWORD  else -1 if a == Achievement.MAKE_WOOD_PICKAXE  else 0 for a in range(Achievement.MAKE_IRON_SWORD +1)]))'
},

'INSTRUCTION_WOOD_SWORD_ONLY': {
    'instruction': "Craft a wooden sword but avoid making a stone sword.",
    "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS, 
    'instruction_paraphrases': [
        "Forge a blade from wood, but do not shape one from stone.",
        "You must create a wooden sword—do not craft a stone one.",
        "Make a wooden sword, not a stone sword.",
        "A sword of timber must be forged, while a stone blade is forbidden.",
        "Construct a wooden weapon, but refrain from working with stone.",
        "Do not create a stone sword, it is unnecessary. But craft a wooden one.",
        "Wood is the only acceptable material for your blade; avoid stone.",
        "Choose your weapon carefully—wood is allowed, stone is not.",
        "Craft a sword from logs, but do not touch the stone.",
        "If you need a weapon, use wood. Do not shape stone into a blade."
    ],
    "arguments": create_target_state([Achievement.MAKE_WOOD_SWORD ], [Achievement.MAKE_STONE_SWORD ]), 
    'str_check_lambda': 'conditional_achivments(gd, jnp.array([1 if a == Achievement.MAKE_WOOD_SWORD  else -1 if a == Achievement.MAKE_STONE_SWORD  else 0 for a in range(Achievement.MAKE_IRON_SWORD +1)]))'
},
'INSTRUCTION_WOOD_PICKAXE_ONLY': {
    'instruction': "Craft a wooden pickaxe but avoid making a stone pickaxe.",
    "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS, 
    'instruction_paraphrases': [
        "Construct a mining tool from wood, but do not carve one from stone.",
        "Your task is to make a wooden pickaxe—do not fashion a stone one.",
        "Make a wooden pickaxe, not a stone pickaxe.",
        "Only a wooden tool is permitted; stone is off-limits.",
        "Do not make a stone pickaxe—it’s unnecessary. But craft a wooden one.",
        "Gather materials using wood, avoid stone-based tools.",
        "You need a wooden pickaxe, not a stone one.",
        "The only allowed tool for mining is made of wood.",
        "Create a pickaxe from planks, leave the stone untouched.",
        "A wooden pickaxe is required, while a stone pickaxe is forbidden."
    ],
    "arguments": create_target_state([Achievement.MAKE_WOOD_PICKAXE ], [Achievement.MAKE_STONE_PICKAXE ]), 
    'str_check_lambda': 'conditional_achivments(gd, jnp.array([1 if a == Achievement.MAKE_WOOD_PICKAXE  else -1 if a == Achievement.MAKE_STONE_PICKAXE  else 0 for a in range(Achievement.MAKE_IRON_SWORD +1)]))'
},
'INSTRUCTION_SWORD_ZOMBIE_ONLY': {
    'instruction': "Craft a wooden sword and defeat a zombie, but do not attack a skeleton.",
    "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS, 
    'instruction_paraphrases': [
        "Forge a wooden blade and slay a zombie, but avoid fighting skeletons.",
        "Your task is to craft a wooden sword and take down a zombie, not a skeleton.",
        "Make a wooden sword and hunt zombies, but leave skeletons alone.",
        "Only zombies are your target—skeletons must remain untouched.",
        "Defeat a zombie using a wooden sword, but do not engage a skeleton.",
        "Make a sword from wood and fight zombies, but do not challenge a skeleton.",
        "Slay the undead, but only the ones that rot—skeletons are not to be harmed.",
        "Craft a wooden weapon and battle zombies, not bone warriors.",
        "The wooden sword is for zombies only, not for skeletons.",
        "Destroy the walking dead, but ignore the ones without flesh."
    ],
    "arguments": create_target_state([Achievement.MAKE_WOOD_SWORD , Achievement.DEFEAT_ZOMBIE ], [Achievement.DEFEAT_SKELETON ]),
    'str_check_lambda': 'conditional_achivments(gd, jnp.array([1 if a == Achievement.MAKE_WOOD_SWORD  else 1 if a == Achievement.DEFEAT_ZOMBIE  else -1 if a == Achievement.DEFEAT_SKELETON  else 0 for a in range(Achievement.MAKE_IRON_SWORD +1)]))'
},
'INSTRUCTION_SWORD_SKELETON_ONLY': {
    'instruction': "Craft a wooden sword and defeat a skeleton, but do not attack a zombie.",
    "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS, 
    'instruction_paraphrases': [
        "Forge a wooden blade and slay a skeleton, but do not fight zombies.",
        "Your goal is to craft a wooden sword and eliminate a skeleton, not a zombie.",
        "Make a wooden sword and engage a skeleton, but avoid harming zombies.",
        "Skeletons are your target, zombies must be left alone.",
        "Defeat a skeleton using a wooden sword, but do not engage a zombie.",
        "Craft a wooden weapon and attack skeletons, but let zombies be.",
        "Destroy skeleton warriors, but do not touch the undead corpses.",
        "Make a wooden blade and strike down skeletons, not zombies.",
        "Skeletons fall to wooden swords; zombies are not your concern.",
        "Only bones should break today—leave the zombies alone."
    ],
    "arguments": create_target_state([Achievement.MAKE_WOOD_SWORD , Achievement.DEFEAT_SKELETON ], [Achievement.DEFEAT_ZOMBIE ]),
    'str_check_lambda': 'conditional_achivments(gd, jnp.array([1 if a == Achievement.MAKE_WOOD_SWORD  else 1 if a == Achievement.DEFEAT_SKELETON  else -1 if a == Achievement.DEFEAT_ZOMBIE  else 0 for a in range(Achievement.MAKE_IRON_SWORD +1)]))'
},
'INSTRUCTION_STONE_SWORD_ZOMBIE_ONLY': {
    'instruction': "Defeat a zombie using a stone sword, but do not attack skeletons.",
    "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS, 
    'instruction_paraphrases': [
        "Use a stone blade to slay a zombie, but do not fight skeletons.",
        "Destroy a zombie with a stone sword, but leave skeletons alone.",
        "A stone sword is for slaying zombies—do not touch skeletons.",
        "Only zombies must fall to your blade; skeletons are off-limits.",
        "Wield a stone weapon against the undead, but not against skeletons.",
        "Defeat a zombie, but do not battle the ones made of bones.",
        "Take down a zombie with stone, but let skeletons roam.",
        "Your mission: kill zombies with stone, spare skeletons.",
        "The dead must be slain with stone, but skeletons are exempt.",
        "A zombie must perish by stone, but a skeleton is not your enemy."
    ],
    "arguments": create_target_state([Achievement.DEFEAT_ZOMBIE ], [Achievement.DEFEAT_SKELETON ]),
    'str_check_lambda': 'conditional_achivments(gd, jnp.array([1 if a == Achievement.DEFEAT_ZOMBIE  else -1 if a == Achievement.DEFEAT_SKELETON  else 0 for a in range(Achievement.MAKE_IRON_SWORD +1)]))'
},
'INSTRUCTION_STONE_SWORD_NO_SKELETON': {
    'instruction': "Craft a stone sword but do not kill any skeletons.",
    "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS, 
    'instruction_paraphrases': [
        "Forge a stone weapon, but keep skeletons unharmed.",
        "You must craft a stone blade—however, do not strike skeletons.",
        "Make a stone sword, but do not use it on skeletons.",
        "A stone sword is required, yet skeletons shall remain untouched.",
        "Craft a stone blade, but refrain from fighting skeleton warriors.",
        "Do not attack skeletons, but you must forge a stone sword.",
        "Skeletons must not fall to your blade, though a stone sword is needed.",
        "Create a sword from stone, but do not engage skeletons in battle.",
        "Forge your weapon, yet let the bones remain unbroken.",
        "Stone weapons are necessary, but skeletons are not your target."
    ],
    "arguments": create_target_state([Achievement.MAKE_STONE_SWORD ], [Achievement.DEFEAT_SKELETON ]),
    'str_check_lambda': 'conditional_achivments(gd, jnp.array([1 if a == Achievement.MAKE_STONE_SWORD  else -1 if a == Achievement.DEFEAT_SKELETON  else 0 for a in range(Achievement.MAKE_IRON_SWORD +1)]))'
},
'INSTRUCTION_STONE_SWORD_NO_ZOMBIE': {
    'instruction': "Craft a stone sword but do not kill any zombies.",
    "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS, 
    'instruction_paraphrases': [
        "Forge a stone blade, but avoid striking down zombies.",
        "Your task is to craft a stone sword—just don’t use it on zombies.",
        "A stone sword must be made, but it shall not slay the undead.",
        "Create a weapon from stone, yet refrain from harming zombies.",
        "Make a stone sword, but do not attack the walking dead.",
        "Build a stone weapon—zombies are not your enemies.",
        "You must forge a stone sword, but do not let it touch a zombie.",
        "A sword of stone must be crafted, yet zombies shall be spared.",
        "Craft your weapon, but remember: zombies are off-limits.",
        "A stone sword is required, but zombies must remain untouched."
    ],
    "arguments": create_target_state([Achievement.MAKE_STONE_SWORD ], [Achievement.DEFEAT_ZOMBIE ]),
    'str_check_lambda': 'conditional_achivments(gd, jnp.array([1 if a == Achievement.MAKE_STONE_SWORD  else -1 if a == Achievement.DEFEAT_ZOMBIE  else 0 for a in range(Achievement.MAKE_IRON_SWORD +1)]))'
},
'INSTRUCTION_STONE_SWORD_ALL_UNDEAD': {
    'instruction': "Craft a stone sword and eliminate all undead.",
    "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS, 
    'instruction_paraphrases': [
        "Forge a stone blade and exterminate every undead creature.",
        "Your mission is to create a stone sword and defeat all undead enemies.",
        "Make a stone weapon and vanquish all creatures of darkness.",
        "A stone sword must be forged—wipe out all undead threats.",
        "Craft a weapon from stone and rid the land of the undead.",
        "All zombies and skeletons must fall before your stone blade.",
        "The undead shall be no more—strike them down with your sword.",
        "Make a stone sword and destroy every unnatural being.",
        "From cursed souls to walking corpses, none shall survive.",
        "Raise a sword of stone and purge the land of undead horrors."
    ],
    "arguments": create_target_state([Achievement.MAKE_STONE_SWORD , Achievement.DEFEAT_ZOMBIE , Achievement.DEFEAT_SKELETON ]),
    'str_check_lambda': 'conditional_achivments(gd, jnp.array([1 if a == Achievement.MAKE_STONE_SWORD  else 1 if a == Achievement.DEFEAT_ZOMBIE  else 1 if a == Achievement.DEFEAT_SKELETON  else 0 for a in range(Achievement.MAKE_IRON_SWORD +1)]))'
}

# 'INSTRUCTION_STONE_SWORD_NO_KILLS': {
#     'instruction': "Craft a stone sword but do not kill anyone.",
#     "scenario_checker": Scenarios.CONDITIONAL_ACIEVEMENTS , 
#     'instruction_paraphrases': [
#         "Forge a stone blade, yet shed no blood—not of zombies, skeletons, or cows.",
#         "Your task is to make a stone sword—just don’t use it to harm zombies, skeletons, or animals.",
#         "A stone weapon must be crafted, but it shall not be wielded against zombies, skeletons, or cows.",
#         "Make a sword from stone, but do not attack zombies, skeletons, or any living creature.",
#         "A weapon of stone is required, yet no lives—zombie, skeleton, or cow—shall be taken.",
#         "You may craft a stone sword, but harming others is strictly forbidden.",
#         "Construct a blade, but do not raise it against any foe, living or undead.",
#         "Make a sword, but do not slay any undead or animals.",
#         "Create a stone sword, but let both the living and the undead remain untouched.",
#         "A stone weapon is necessary, but this is a time of peace—no creatures shall fall by your hand."
#     ],
#     'check_lambda': 
#         create_check_lambda([Achievement.MAKE_STONE_SWORD ],[Achievement.DEFEAT_ZOMBIE , Achievement.DEFEAT_SKELETON ]),
#     'str_check_lambda': 'conditional_achivments(gd, jnp.array([1 if a == Achievement.MAKE_STONE_SWORD  else -1 if a in [Achievement.DEFEAT_ZOMBIE , Achievement.DEFEAT_SKELETON , Achievement.EAT_COW ] else 0 for a in range(Achievement.MAKE_IRON_SWORD +1)]))'
# },

}

from craftext.scenarios.parce_dataset import update_previous_dict
from craftext.scenarios.constants import base_path
import os

easy = update_previous_dict(
    easy, 
    os.path.join(base_path, "jax_conditional_achivments/instructions/train/easy"), 
    "achivments"
)

for key in easy:
    if 'arguments' not in easy[key]:
     #   print(easy[key])
        arguments = format_to_target_state(easy[key]['str_check_lambda'])
        
        print(arguments)
        
        easy[key]['arguments'] = eval(arguments)
        easy[key]['scenario_checker'] = Scenarios.CONDITIONAL_ACHIEVEMENTS
        


if __name__ == "__main__":
    import json
    instructions = []
    for key in easy.keys():
        instructions.append(easy[key]['instruction'])
        instructions += easy[key]['instruction_paraphrases']
    with open("instructions_achivments_easy_train.json", "w", encoding="utf-8") as json_file:
        json.dump(instructions, json_file, ensure_ascii=False, indent=4)