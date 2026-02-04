from craftext.checkers.conditional import conditional_placing
from craftext.scenarios.constants import InventoryItems, BlockType

easy_test_parafrased = {
    'simple_conditional_placing_001': {
        'instruction': "Place a furnace on the ground after collecting 3 pieces of wood",
        'instruction_paraphrases': [
            "After gathering three logs, place a kiln on the ground",
            "Collect three wooden blocks and position a heater on the ground",
            "Once you've gathered 3 logs, set up an oven on the terrain",
            "Accumulate three wooden units and put a stove on the ground",
            "Gather three pieces of timber and install a smelter at the designated spot"
        ],
        'check_lambda': lambda gd, ix: conditional_placing(gd, InventoryItems.WOOD, BlockType.FURNACE, 3, 2),
        'str_check_lambda': "conditional_placing(gd, InventoryItems.WOOD, BlockType.FURNACE, 3, 2)"
    },
    'simple_conditional_placing_002': {
        'instruction': "Collect 2 pieces of wood and place a crafting table",
        'instruction_paraphrases': [
            "Gather two logs and set down a workbench",
            "Acquire two wooden blocks and position a crafting station",
            "After collecting two pieces of timber, place a tool table",
            "Obtain two wooden units and place a worktable",
            "Amass two logs and arrange a crafting surface at the designated location"
        ],
        'check_lambda': lambda gd, ix: conditional_placing(gd, InventoryItems.WOOD, BlockType.CRAFTING_TABLE, 2, 3),
        'str_check_lambda': "conditional_placing(gd, InventoryItems.WOOD, BlockType.CRAFTING_TABLE, 2, 3)"
    },
    'simple_conditional_placing_003': {
        'instruction': "Place a plant after collecting 2 saplings",
        'instruction_paraphrases': [
            "After gathering two seedlings, plant a shrub",
            "Collect two saplings and position some greenery",
            "Once you've collected two small plants, set up a tree",
            "Accumulate two young plants and put a bush on the terrain",
            "Gather two saplings and install vegetation at the specified spot"
        ],
        'check_lambda': lambda gd, ix: conditional_placing(gd, InventoryItems.SAPLING, BlockType.PLANT, 2, 2),
        'str_check_lambda': "conditional_placing(gd, InventoryItems.SAPLING, BlockType.PLANT, 2, 2)"
    },
    'simple_conditional_placing_004': {
        'instruction': "Collect 1 piece of wood and place a furnace",
        'instruction_paraphrases': [
            "After gathering one log, place a kiln",
            "Collect one wooden block and set down a heater",
            "Once you've obtained a piece of timber, position an oven on the terrain",
            "Accumulate one wooden unit and put a stove on the ground",
            "Collect one log and install a smelter at the designated location"
        ],
        'check_lambda': lambda gd, ix: conditional_placing(gd, InventoryItems.WOOD, BlockType.FURNACE, 1, 2),
        'str_check_lambda': "conditional_placing(gd, InventoryItems.WOOD, BlockType.FURNACE, 1, 2)"
    }
}


# TODO: need to check how many synonyms are used

easy_test_other_paramets = {
    'simple_conditional_placing_001': {
        'instruction': "Collect 1 piece of wood and place 2 plant blocks.",
        'instruction_paraphrases': [
            "Gather one log and position two plant blocks on the ground.",
            "After collecting one wood piece, set down two plant blocks.",
            "Collect a wooden block and place two plants on the terrain.",
            "Once you've gathered a piece of timber, put two plant blocks.",
            "Collect one unit of wood and install two plant blocks."
        ],
        'check_lambda': lambda gd, ix: conditional_placing(gd, InventoryItems.WOOD, BlockType.PLANT, 1, 2),
        'str_check_lambda': "conditional_placing(gd, InventoryItems.WOOD, BlockType.PLANT, 1, 2)"
    },
    'simple_conditional_placing_002': {
        'instruction': "Collect 2 saplings and place 2 furnace blocks.",
        'instruction_paraphrases': [
            "Gather two saplings and set two furnaces on the ground.",
            "After collecting two saplings, position two furnace blocks.",
            "Collect a pair of saplings and place two furnaces on the terrain.",
            "Once you've gathered two saplings, set down two furnace blocks.",
            "Collect two saplings and install two furnaces."
        ],
        'check_lambda': lambda gd, ix: conditional_placing(gd, InventoryItems.SAPLING, BlockType.FURNACE, 2, 2),
        'str_check_lambda': "conditional_placing(gd, InventoryItems.SAPLING, BlockType.FURNACE, 2, 2)"
    },
    'simple_conditional_placing_003': {
        'instruction': "Collect 1 wood pickaxe and place 2 crafting tables.",
        'instruction_paraphrases': [
            "Gather one wooden pickaxe and position two crafting tables.",
            "After collecting one wood pickaxe, set two crafting tables.",
            "Collect one wooden pickaxe and place two crafting tables on the terrain.",
            "Once you've gathered one wood pickaxe, put down two crafting tables.",
            "Accumulate one wooden pickaxe and install two crafting tables."
        ],
        'check_lambda': lambda gd, ix: conditional_placing(gd, InventoryItems.WOOD_PICKAXE, BlockType.CRAFTING_TABLE, 1, 2),
        'str_check_lambda': "conditional_placing(gd, InventoryItems.WOOD_PICKAXE, BlockType.CRAFTING_TABLE, 1, 2)"
    },
    'simple_conditional_placing_004': {
        'instruction': "Collect 3 pieces of iron and place 2 crafting tables.",
        'instruction_paraphrases': [
            "Gather three iron pieces and set two crafting tables.",
            "After collecting three pieces of iron, position two crafting tables.",
            "Collect three iron units and place two crafting tables on the terrain.",
            "Once you've gathered three iron pieces, put down two crafting tables.",
            "Accumulate three iron blocks and install two crafting tables."
        ],
        'check_lambda': lambda gd, ix: conditional_placing(gd, InventoryItems.IRON, BlockType.CRAFTING_TABLE, 3, 2),
        'str_check_lambda': "conditional_placing(gd, InventoryItems.IRON, BlockType.CRAFTING_TABLE, 3, 2)"
    }
}

medium_test_parafrased = {
    'medium_conditional_placing_001': {
        'instruction': "Collect 4 pieces of iron and place a chest",
        'instruction_paraphrases': [
            "Once you've gathered four iron ingots, set up a storage chest",
            "Gather four metal units and place a chest on the map",
            "After collecting 4 iron, position a storage box at the required location",
            "Accumulate four pieces of iron and put a chest on the terrain",
            "Collect four iron units and install a chest at the designated spot"
        ],
        'check_lambda': lambda  gd, ix: conditional_placing(gd, InventoryItems.IRON, BlockType.CHEST, 4, 1)
    },
    'medium_conditional_placing_002': {
        'instruction': "Collect 3 pieces of coal and place a fountain",
        'instruction_paraphrases': [
            "Gather three coal pieces and place a water feature at the specified location",
            "After gathering 3 coal lumps, set up a water fountain on the terrain",
            "Once you've collected three units of coal, position a water fountain",
            "Collect three coal items and install a fountain on the map",
            "Accumulate three coal units and put a water source at the designated spot"
        ],
        'check_lambda': lambda  gd, ix: conditional_placing(gd, InventoryItems.COAL, BlockType.FOUNTAIN, 3, 2)
    },
    'medium_conditional_placing_003': {
        'instruction': "Collect 2 pieces of iron and place an enchantment table",
        'instruction_paraphrases': [
            "After collecting two iron ingots, place a magical table on the terrain",
            "Once you've gathered two iron pieces, install an enchantment table",
            "Gather two metal units and position a magic table on the map",
            "Accumulate two iron items and set up an enchantment table",
            "Collect two iron units and put a magical table at the specified spot"
        ],
        'check_lambda': lambda  gd, ix: conditional_placing(gd, InventoryItems.IRON, BlockType.ENCHANTMENT_TABLE_FIRE, 2, 2)
    },
    'medium_conditional_placing_004': {
        'instruction': "Collect 4 diamonds and place two enchantment tables (ice)",
        'instruction_paraphrases': [
            "Once you've gathered four diamonds, position two ice enchantment tables on the terrain",
            "Collect four diamonds and place two frost magic tables",
            "Gather four gemstones and install two ice tables at the required location",
            "Accumulate four diamond units and put two ice spellbinding tables",
            "After collecting four diamonds, set up two frozen magic tables"
        ],
        'check_lambda': lambda  gd, ix: conditional_placing(gd, InventoryItems.DIAMOND, BlockType.ENCHANTMENT_TABLE_ICE, 4, 2)
    },
    'medium_conditional_placing_005': {
        'instruction': "Collect 3 pieces of iron and place an enchantment table (fire)",
        'instruction_paraphrases': [
            "After gathering three iron ingots, set up a fire magic table",
            "Once you've collected three pieces of iron, place a flame enchantment table",
            "Gather 3 iron units and install a fire spellbinding table",
            "Collect three iron pieces and position a fire enchantment table on the map",
            "Accumulate three metal pieces and put a fire magic table at the designated spot"
        ],
        'check_lambda': lambda  gd, ix: conditional_placing(gd, InventoryItems.IRON, BlockType.ENCHANTMENT_TABLE_FIRE, 3, 2)
    }
}
medium_test_other_paramets = {
    'medium_conditional_placing_001': {
        'instruction': "Collect 4 pieces of iron and place a chest",
        'instruction_paraphrases': [
            "Once you've gathered four iron ingots, set up a storage chest",
            "Gather four metal units and place a chest on the map",
            "After collecting 4 iron, position a storage box at the required location",
            "Accumulate four pieces of iron and put a chest on the terrain",
            "Collect four iron units and install a chest at the designated spot"
        ],
        'check_lambda': lambda  gd, ix: conditional_placing(gd, InventoryItems.COAL, BlockType.CHEST, 4, 1)  # Changed to Coal
    },
    'medium_conditional_placing_002': {
        'instruction': "Collect 3 pieces of coal and place a fountain",
        'instruction_paraphrases': [
            "Gather three coal pieces and place a water feature at the specified location",
            "After gathering 3 coal lumps, set up a water fountain on the terrain",
            "Once you've collected three units of coal, position a water fountain",
            "Collect three coal items and install a fountain on the map",
            "Accumulate three coal units and put a water source at the designated spot"
        ],
        'check_lambda': lambda  gd, ix: conditional_placing(gd, InventoryItems.COAL, BlockType.CHEST, 3, 2)  # Changed to Chest
    },
    'medium_conditional_placing_003': {
        'instruction': "Collect 2 pieces of iron and place an enchantment table",
        'instruction_paraphrases': [
            "After collecting two iron ingots, place a magical table on the terrain",
            "Once you've gathered two iron pieces, install an enchantment table",
            "Gather two metal units and position a magic table on the map",
            "Accumulate two iron items and set up an enchantment table",
            "Collect two iron units and put a magical table at the specified spot"
        ],
        'check_lambda': lambda  gd, ix: conditional_placing(gd, InventoryItems.IRON, BlockType.CHEST, 2, 1)  # Changed block to Chest
    },
    'medium_conditional_placing_004': {
        'instruction': "Collect 4 diamonds and place two enchantment tables (ice)",
        'instruction_paraphrases': [
            "Once you've gathered four diamonds, position two ice enchantment tables on the terrain",
            "Collect four diamonds and place two frost magic tables",
            "Gather four gemstones and install two ice tables at the required location",
            "Accumulate four diamond units and put two ice spellbinding tables",
            "After collecting four diamonds, set up two frozen magic tables"
        ],
        'check_lambda': lambda  gd, ix: conditional_placing(gd, InventoryItems.DIAMOND, BlockType.FOUNTAIN, 4, 2)  # Changed block to Fountain
    },
    'medium_conditional_placing_005': {
        'instruction': "Collect 3 pieces of iron and place an enchantment table (fire)",
        'instruction_paraphrases': [
            "After gathering three iron ingots, set up a fire magic table",
            "Once you've collected three pieces of iron, place a flame enchantment table",
            "Gather 3 iron units and install a fire spellbinding table",
            "Collect three iron pieces and position a fire enchantment table on the map",
            "Accumulate three metal pieces and put a fire magic table at the designated spot"
        ],
        'check_lambda': lambda  gd, ix: conditional_placing(gd, InventoryItems.COAL, BlockType.ENCHANTMENT_TABLE_FIRE, 3, 2)  # Changed item to Coal
    }
}

from craftext.scenarios.parce_dataset import update_previous_dict

easy_test_other_paramets = update_previous_dict(easy_test_other_paramets, "jax_conditional_placing/instructions/test/easy/other_params", "jax_conditional_placing_test_op")
medium_test_other_paramets = update_previous_dict(medium_test_other_paramets, "jax_conditional_placing/instructions/test/medium/other_params", "jax_conditional_placing_test_op")
