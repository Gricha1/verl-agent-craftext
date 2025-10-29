from craftext.checkers.relevant import place_object_relevant_to
from craftext.scenarios.constants import BlockType

easy_test_parafrased = {
    "place_test_parafrased_easy_1": {
        'instruction': "Position a crafting table just one tile to the right of the water source.",
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

    "place_test_parafrased_easy_2": {
        'instruction': "Arrange a furnace two blocks directly below the stone structure.",
        'instruction_paraphrases': [
            "Put a furnace two tiles below the stone block.",
            "Place a furnace exactly two blocks under the stone.",
            "Set a furnace two units away from the stone, below it.",
            "Arrange a furnace two steps south of the stone block.",
            "Position a furnace two tiles away from the stone, downwards."
        ],
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
            game_data, BlockType.FURNACE, BlockType.STONE, 3, 2
        ),
        'str_check_lambda': "place_object_relevant_to(game_data, BlockType.FURNACE, BlockType.STONE, 3, 2)",
        'complexity': "easy-pease"
    },

    "place_test_parafrased_easy_3": {
        'instruction': "Put the crafting table one unit on top of the tree.",
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
    }
}


easy_test_other_paramets = {
    "place_easy_test_other_paramets_easy_4": {
        'instruction': "Place a furnace 1 step to the left of the lake.",
        'instruction_paraphrases': [
            "Set a furnace one tile left of the lake.",
            "Place a furnace exactly one space to the left of the lake.",
            "Position a furnace block one unit to the left of the lake.",
            "Arrange a furnace one step left of the lake.",
            "Put a furnace block one tile away from the lake, on the left."
        ],
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
            game_data, BlockType.FURNACE, BlockType.WATER, 1, 1
        ),
        'str_check_lambda': "place_object_relevant_to(game_data, BlockType.FURNACE, BlockType.WATER, 1, 1)",
        'complexity': "easy"
    },

    "place_easy_test_other_paramets_easy_5": {
        'instruction': "Place a crafting table 3 steps above the coal block.",
        'instruction_paraphrases': [
            "Put a crafting table three tiles above the coal block.",
            "Position a crafting table exactly three spaces above the coal.",
            "Set a crafting table three units away from the coal, upwards.",
            "Arrange a crafting table three steps north of the coal block.",
            "Place a crafting table three tiles away from the coal, above it."
        ],
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
            game_data, BlockType.CRAFTING_TABLE, BlockType.COAL, 2, 3
        ),
        'str_check_lambda': "place_object_relevant_to(game_data, BlockType.CRAFTING_TABLE, BlockType.COAL, 2, 3)",
        'complexity': "medium"
    },

    "place_easy_test_other_paramets_easy_6": {
        'instruction': "Put a furnace 2 steps to the right of the tree.",
        'instruction_paraphrases': [
            "Place a furnace two tiles to the right of the tree.",
            "Position a furnace exactly two spaces right of the tree.",
            "Set a furnace two units away from the tree, on its right.",
            "Arrange a furnace two steps to the right of the tree.",
            "Put a furnace block two tiles away from the tree, to the right."
        ],
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
            game_data, BlockType.FURNACE, BlockType.TREE, 0, 2
        ),
        'str_check_lambda': "place_object_relevant_to(game_data, BlockType.FURNACE, BlockType.TREE, 0, 2)",
        'complexity': "medium"
    }
}



medium_test_parafrased = {
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
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
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

### TODO: check the complexity
medium_test_other_paramets = {
    "place_medium_1": {
        'instruction': "Place a crafting table to the right of the plant, 2 steps away.",
        'instruction_paraphrases': [
            "Set up a workbench two tiles to the right of the plant.",
            "Put a crafting station exactly two blocks away from the plant, on the right.",
            "Position a crafting bench 2 spaces to the right of the plant.",
            "Place a crafting station to the right of the vegetation, 2 units away.",
            "Arrange a workbench two steps to the right of the plant."
        ],
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
            game_data,  BlockType.ENCHANTMENT_TABLE_ICE, BlockType.PLANT, 0, 2  # Changed to Enchantment Table (Ice)
        ),
        'complexity': "easy-pease"
    },
    "place_medium_2": {
        'instruction': "Put a plant below the crafting table, 1 step away.",
        'instruction_paraphrases': [
            "Place a tree one tile beneath the crafting table.",
            "Position a plant exactly one step below the workbench.",
            "Set a bush block just one unit down from the crafting station.",
            "Put vegetation one tile away from the crafting bench, below it.",
            "Arrange a plant right under the crafting table, 1 step away."
        ],
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
            game_data,  BlockType.FOUNTAIN, BlockType.STONE, 3, 1  # Changed block to Fountain
        ),
        'complexity': "easy-pease"
    },
    "place_medium_3": {
        'instruction': "Place a furnace to the left of the stone block, 3 steps away.",
        'instruction_paraphrases': [
            "Set a kiln three tiles to the left of the rock.",
            "Put a furnace exactly three spaces left of the stone block.",
            "Position an oven three units to the left of the stone.",
            "Place a furnace to the left of the stone, 3 tiles away.",
            "Arrange a stove three steps to the left of the rock."
        ],
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
            game_data, BlockType.ENCHANTMENT_TABLE_FIRE, BlockType.PLANT, 1, 3  # Changed block to Enchantment Table (Fire)
        ),
        'complexity': "easy"
    },
    "place_medium_4": {
        'instruction': "Place a stone to the right of the furnace, 2 steps away.",
        'instruction_paraphrases': [
            "Put a rock two tiles to the right of the kiln.",
            "Position a stone exactly two blocks to the right of the stove.",
            "Set a stone two units away from the furnace, on the right.",
            "Place a boulder two steps to the right of the oven.",
            "Arrange a stone block two tiles away from the furnace on its right."
        ],
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
            game_data, BlockType.FOUNTAIN, BlockType.CHEST, 0, 2  # Changed block to Fountain
        ),
        'complexity': "easy"
    },
    "place_medium_5": {
        'instruction': "Position a chest above the fountain, 4 steps away.",
        'instruction_paraphrases': [
            "Set a storage box four tiles above the water feature.",
            "Place a trunk exactly four blocks up from the fountain.",
            "Position a container four units away from the fountain, above it.",
            "Arrange a chest four tiles north of the water source.",
            "Place a storage unit four steps up from the fountain."
        ],
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
            game_data, BlockType.ENCHANTMENT_TABLE_ICE, BlockType.FOUNTAIN, 2, 3  # Changed to Enchantment Table (Ice)
        ),
        'complexity': "hard"
    },
    "place_medium_6": {
        'instruction': "Place an enchantment table (fire) to the right of the plant, 5 steps away.",
        'instruction_paraphrases': [
            "Put a flame spellbinding table five tiles to the right of the plant.",
            "Position a fire enchantment table exactly five blocks to the right of the vegetation.",
            "Set a fire magic table five units away from the plant, on the right.",
            "Place a flame enchantment table five steps to the right of the tree.",
            "Arrange a fire spellbinding table five tiles away from the plant on its right."
        ],
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
            game_data, BlockType.ENCHANTMENT_TABLE_FIRE, BlockType.FOUNTAIN, 0, 5  # Changed to Enchantment Table (Fire)
        ),
        'complexity': "hard"
    }
}

from craftext.scenarios.parce_dataset import update_previous_dict

easy_test_other_paramets = update_previous_dict(easy_test_other_paramets, "jax_localization_place/instructions/test/easy/other_params", "localization_place_test_op")
medium_test_other_paramets = update_previous_dict(medium_test_other_paramets, "jax_localization_place/instructions/test/medium/other_params", "localization_place_test_op")
