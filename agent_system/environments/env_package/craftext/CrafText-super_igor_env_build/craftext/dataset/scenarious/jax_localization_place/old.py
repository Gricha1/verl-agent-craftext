from craftext.checkers.relevant import place_object_relevant_to
from craftext.scenarios.constants import BlockType


one = {"one_1": {
        'instruction': "Put a crafting table 1 step above the tree.",
        'instruction_paraphrases': [
            "Place a crafting table one tile above the tree.",
            "Position a crafting table exactly one block up from the tree.",
            "Set up a crafting table one unit away from the tree, above it.",
            "Arrange a crafting table one step above the tree.",
            "Put a crafting table one tile away from the tree, on top."
        ],
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
            game_data, BlockType.CRAFTING_TABLE, BlockType.TREE, 2, 1
        ),
        'complexity': "easy-pease"
    },
       "one_2": {
        'instruction': "Put a crafting table 1 step above the tree.",
        'instruction_paraphrases': [
            "Place a crafting table one tile above the tree.",
            "Position a crafting table exactly one block up from the tree.",
            "Set up a crafting table one unit away from the tree, above it.",
            "Arrange a crafting table one step above the tree.",
            "Put a crafting table one tile away from the tree, on top."
        ],
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
            game_data, BlockType.CRAFTING_TABLE, BlockType.TREE, 2, 1
        ),
        'complexity': "easy-pease"
    },
      }
easy = {
    "place_easy_1": {
        'instruction': "Place a crafting table next to the water, 1 step away to the right.",
        'instruction_paraphrases': [
            "Set a crafting table one tile to the right of the water.",
            "Place a crafting table exactly one block away from the water, on its right.",
            "Put a crafting table one unit to the right of the water.",
            "Arrange a crafting table one step away from the water, on the right side.",
            "Position a crafting table just one tile away from the water, on the right."
        ],
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
            game_data, BlockType.CRAFTING_TABLE, BlockType.WATER, 0, 1
        ),
        'str_check_lambda': "place_object_relevant_to(game_data, BlockType.CRAFTING_TABLE, BlockType.WATER, 0, 1)",
        'complexity': "easy-pease"
    },

    "place_easy_2": {
        'instruction': "Place a furnace below the stone, 2 steps away.",
        'instruction_paraphrases': [
            "Put a furnace two tiles below the stone block.",
            "Place a furnace exactly two blocks under the stone.",
            "Set a furnace two units away from the stone, below it.",
            "Arrange a furnace two steps south of the stone block.",
            "Position a furnace two tiles away from the stone, downwards."
        ],
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
            game_data,  BlockType.FURNACE, BlockType.STONE, 3, 2
        ),
        'str_check_lambda': "place_object_relevant_to(game_data, BlockType.FURNACE, BlockType.STONE, 3, 2)",
        'complexity': "easy-pease"
    },

    "place_easy_3": {
        'instruction': "Put a crafting table 1 step above the tree.",
        'instruction_paraphrases': [
            "Place a crafting table one tile above the tree.",
            "Position a crafting table exactly one block up from the tree.",
            "Set up a crafting table one unit away from the tree, above it.",
            "Arrange a crafting table one step above the tree.",
            "Put a crafting table one tile away from the tree, on top."
        ],
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
            game_data, BlockType.CRAFTING_TABLE, BlockType.TREE, 2, 1
        ),
        'str_check_lambda': "place_object_relevant_to(game_data, BlockType.CRAFTING_TABLE, BlockType.TREE, 2, 1)",
        'complexity': "easy-pease"
    },

    "place_easy_4": {
        'instruction': "Place a furnace 3 steps to the left of the path.",
        'instruction_paraphrases': [
            "Set a furnace three tiles left of the path.",
            "Place a furnace exactly three spaces to the left of the path.",
            "Position a furnace block three units to the left of the path.",
            "Arrange a furnace three steps left of the path.",
            "Put a furnace block three tiles away from the path, on the left."
        ],
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
            game_data, BlockType.FURNACE, BlockType.PATH, 1, 3
        ),
        'str_check_lambda': "place_object_relevant_to(game_data, BlockType.FURNACE, BlockType.PATH, 1, 3)",
        'complexity': "easy"
    },

    "place_easy_5": {
        'instruction': "Place a crafting table 2 steps above the coal block.",
        'instruction_paraphrases': [
            "Put a crafting table two tiles above the coal block.",
            "Position a crafting table exactly two spaces above the coal.",
            "Set a crafting table two units away from the coal, upwards.",
            "Arrange a crafting table two steps north of the coal block.",
            "Place a crafting table two tiles away from the coal, above it."
        ],
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
            game_data, BlockType.CRAFTING_TABLE, BlockType.COAL, 2, 2
        ),
        'str_check_lambda': "place_object_relevant_to(game_data, BlockType.CRAFTING_TABLE, BlockType.COAL, 2, 2)",
        'complexity': "medium"
    },

    "place_easy_6": {
        'instruction': "Put a furnace 1 step to the right of the iron block.",
        'instruction_paraphrases': [
            "Place a furnace one tile to the right of the iron block.",
            "Position a furnace exactly one space right of the iron block.",
            "Set a furnace one unit away from the iron, on its right.",
            "Arrange a furnace one step to the right of the iron block.",
            "Put a furnace block one tile away from the iron block, to the right."
        ],
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
            game_data, BlockType.FURNACE, BlockType.IRON, 0, 1
        ),
        'str_check_lambda': "place_object_relevant_to(game_data, BlockType.FURNACE, BlockType.IRON, 0, 1)",
        'complexity': "medium"
    }
}

medium = {
    "place_medium_1": {
        'instruction': "Place a crafting table to the right of the plant, 2 steps away.",
        'instruction_paraphrases': [
            "Set up a crafting table two tiles to the right of the plant.",
            "Put a crafting table exactly two blocks away from the plant, on the right.",
            "Position a crafting table 2 spaces to the right of the plant.",
            "Place a crafting table to the right of the plant, 2 units away.",
            "Arrange a crafting table two steps to the right of the plant."
        ],
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
            game_data,  BlockType.CRAFTING_TABLE, BlockType.PLANT, 0, 2
        ),
        'complexity': "easy-pease"
    },

    "place_medium_2": {
        'instruction': "Put a plant below the crafting table, 1 step away.",
        'instruction_paraphrases': [
            "Place a plant one tile beneath the crafting table.",
            "Position a plant exactly one step below the crafting table.",
            "Set a plant block just one unit down from the crafting table.",
            "Put a plant block one tile away from the crafting table, below it.",
            "Arrange a plant block right under the crafting table, 1 step away."
        ],
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
            game_data,  BlockType.PLANT, BlockType.CRAFTING_TABLE, 3, 1
        ),
        'complexity': "easy-pease"
    },

    "place_medium_3": {
        'instruction': "Place a furnace to the left of the stone block, 3 steps away.",
        'instruction_paraphrases': [
            "Set a furnace three tiles to the left of the stone.",
            "Put a furnace exactly three spaces left of the stone block.",
            "Position a furnace block three units to the left of the stone.",
            "Place a furnace to the left of the stone block, 3 tiles away.",
            "Arrange a furnace three steps to the left of the stone."
        ],
        'check_lambda': lambda game_data, ix,: place_object_relevant_to(
            game_data, BlockType.FURNACE, BlockType.STONE, 1, 3
        ),
        'complexity': "easy"
    },

    "place_medium_4": {
        'instruction': "Place a stone to the right of the furnace, 2 steps away.",
        'instruction_paraphrases': [
            "Put a stone block two tiles to the right of the furnace.",
            "Position a stone exactly two blocks to the right of the furnace.",
            "Set a stone two units away from the furnace, on the right.",
            "Place a stone block two steps to the right of the furnace.",
            "Arrange a stone block two tiles away from the furnace on its right."
        ],
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
            game_data, BlockType.STONE, BlockType.FURNACE, 0, 2
        ),
        'complexity': "easy"
    },

    "place_medium_5": {
        'instruction': "Position a chest above the fountain, 4 steps away.",
        'instruction_paraphrases': [
            "Set a chest four tiles above the fountain.",
            "Place a chest block exactly four blocks up from the fountain.",
            "Position a chest four units away from the fountain, above it.",
            "Arrange a chest four tiles north of the fountain.",
            "Place a chest block four steps up from the fountain."
        ],
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
            game_data, BlockType.CHEST, BlockType.FOUNTAIN, 2, 4
        ),
        'complexity': "hard"
    },

    "place_medium_6": {
        'instruction': "Place an enchantment table (fire) to the right of the plant, 5 steps away.",
        'instruction_paraphrases': [
            "Put an enchantment table (fire) five tiles to the right of the plant.",
            "Position an enchantment table of fire type exactly five blocks to the right of the plant.",
            "Set an enchantment table (fire) five units away from the plant, on the right.",
            "Place an enchantment table (fire) five steps to the right of the plant.",
            "Arrange an enchantment table (fire) five tiles away from the plant on its right."
        ],
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
            game_data, BlockType.ENCHANTMENT_TABLE_FIRE, BlockType.PLANT, 0, 5
        ),
        'complexity': "hard"
    }
}
from craftext.scenarios.parce_dataset import update_previous_dict

easy = update_previous_dict(easy, "jax_localization_place/instructions/train/easy", "localization_place")
medium = update_previous_dict(medium, "jax_localization_place/instructions/train/medium", "localization_place")
