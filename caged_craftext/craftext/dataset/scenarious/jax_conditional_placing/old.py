from craftext.checkers.conditional import conditional_placing
from craftext.scenarios.constants import InventoryItems, BlockType, MediumInventoryItems


one = {
    'simple_conditional_placing_001': {
        'instruction': "Place a furnace on the ground after collecting 3 pieces of wood",
        'instruction_paraphrases': [
            "After gathering three logs, place a kiln on the ground",
            "Collect three wooden blocks and then position a heater on the ground",
            "Once you've collected 3 logs, set up an oven on the terrain",
            "Accumulate three wooden units and put a stove on the ground",
            "Gather three pieces of timber and install a smelter at the designated spot"
        ],
        'check_lambda': lambda gd, ix: conditional_placing(gd, InventoryItems.WOOD, BlockType.FURNACE, 1, 1)
    },
    'simple_conditional_placing_002': {
        'instruction': "Place a furnace on the ground after collecting 3 pieces of wood",
        'instruction_paraphrases': [
            "After gathering three logs, place a kiln on the ground",
            "Collect three wooden blocks and then position a heater on the ground",
            "Once you've collected 3 logs, set up an oven on the terrain",
            "Accumulate three wooden units and put a stove on the ground",
            "Gather three pieces of timber and install a smelter at the designated spot"
        ],
        'check_lambda': lambda gd, ix: conditional_placing(gd, InventoryItems.WOOD, BlockType.FURNACE, 1, 1)
    },
}
    
easy = {
    'simple_conditional_placing_001': {
        'instruction': "Place a furnace on the ground after collecting 3 pieces of wood",
        'instruction_paraphrases': [
            "After gathering three logs, place a kiln on the ground",
            "Collect three wooden blocks and then position a heater on the ground",
            "Once you've collected 3 logs, set up an oven on the terrain",
            "Accumulate three wooden units and put a stove on the ground",
            "Gather three pieces of timber and install a smelter at the designated spot"
        ],
        'check_lambda': lambda gd, ix: conditional_placing(gd, InventoryItems.WOOD, BlockType.FURNACE, 3, 2),
        'str_check_lambda': "conditional_placing(gd, InventoryItems.WOOD, BlockType.FURNACE, 3, 2)"
    },
    'simple_conditional_placing_002': {
        'instruction': "Collect 2 pieces of wood and place a crafting table",
        'instruction_paraphrases': [
            "Gather two logs and then set down a workbench",
            "Acquire two wooden blocks and position a crafting station",
            "After collecting two pieces of timber, place a tool table",
            "Obtain two wooden units and then place a worktable",
            "Amass two logs and arrange a crafting surface at the specified location"
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
            "Gather two saplings and install vegetation at the designated location"
        ],
        'check_lambda': lambda gd, ix: conditional_placing(gd, InventoryItems.SAPLING, BlockType.PLANT, 2, 2),
        'str_check_lambda': "conditional_placing(gd, InventoryItems.SAPLING, BlockType.PLANT, 2, 2)"
    },
    'simple_conditional_placing_004': {
        'instruction': "Collect 1 piece of wood and place a furnace",
        'instruction_paraphrases': [
            "After gathering one log, place a kiln",
            "Collect one wooden block and then set down a heater",
            "Once you've obtained a piece of timber, position an oven on the terrain",
            "Accumulate one wooden unit and put a stove on the ground",
            "Collect one log and install a smelter at the designated location"
        ],
        'check_lambda': lambda gd, ix: conditional_placing(gd, InventoryItems.WOOD, BlockType.FURNACE, 1, 2),
        'str_check_lambda': "conditional_placing(gd, InventoryItems.WOOD, BlockType.FURNACE, 1, 2)"
    },
    'simple_conditional_placing_005': {
        'instruction': "Collect 3 pieces of wood and place a crafting table",
        'instruction_paraphrases': [
            "After gathering three logs, place a workbench",
            "Collect three wooden blocks and then position a crafting station",
            "Once you've gathered three pieces of timber, set up a tool table",
            "Accumulate three wooden units and place a worktable",
            "Gather three logs and install a crafting surface at the specified spot"
        ],
        'check_lambda': lambda gd, ix: conditional_placing(gd, InventoryItems.WOOD, BlockType.CRAFTING_TABLE, 3, 2),
        'str_check_lambda': "conditional_placing(gd, InventoryItems.WOOD, BlockType.CRAFTING_TABLE, 3, 2)"
    },
    'simple_conditional_placing_006': {
        'instruction': "Collect 4 pieces of wood and place a furnace",
        'instruction_paraphrases': [
            "After gathering four logs, place a kiln on the ground",
            "Collect four wooden blocks and then position a heater",
            "Once you've collected four pieces of timber, set down an oven on the map",
            "Accumulate four wooden units and put a stove on the terrain",
            "Gather four logs and install a smelter at the specified spot"
        ],
        'check_lambda': lambda gd, ix: conditional_placing(gd, InventoryItems.WOOD, BlockType.FURNACE, 4, 1),
        'str_check_lambda': "conditional_placing(gd, InventoryItems.WOOD, BlockType.FURNACE, 4, 1)"
    },
    'simple_conditional_placing_007': {
        'instruction': "Collect 2 pieces of wood and place two crafting tables",
        'instruction_paraphrases': [
            "After gathering two logs, set down two workbenches",
            "Collect two wooden units and then position two crafting stations",
            "Once you've collected two pieces of timber, set up two tool tables",
            "Accumulate two wooden units and install two worktables",
            "Gather two logs and arrange two crafting surfaces at the required spot"
        ],
        'check_lambda': lambda gd, ix: conditional_placing(gd, InventoryItems.WOOD, BlockType.CRAFTING_TABLE, 2, 2),
        'str_check_lambda': "conditional_placing(gd, InventoryItems.WOOD, BlockType.CRAFTING_TABLE, 2, 2)"
    },
    'simple_conditional_placing_008': {
        'instruction': "Collect 3 saplings and place two plants",
        'instruction_paraphrases': [
            "After gathering three seedlings, plant two shrubs",
            "Collect three saplings and then set up two patches of greenery",
            "Once you've collected three small plants, install two trees",
            "Accumulate three young plants and position two bushes on the terrain",
            "Gather three saplings and place two vegetation spots at the specified location"
        ],
        'check_lambda': lambda gd, ix: conditional_placing(gd, InventoryItems.SAPLING, BlockType.PLANT, 3, 2),
        'str_check_lambda': "conditional_placing(gd, InventoryItems.SAPLING, BlockType.PLANT, 3, 2)"
    },
    'simple_conditional_placing_009': {
        'instruction': "Collect 1 piece of wood and place three furnaces",
        'instruction_paraphrases': [
            "After gathering one log, place three kilns",
            "Collect one wooden block and then set up three heaters on the terrain",
            "Once you've gathered one piece of timber, put three ovens on the map",
            "Accumulate one wooden unit and install three stoves",
            "Collect one log and arrange three smelters at the specified spot"
        ],
        'check_lambda': lambda gd, ix: conditional_placing(gd, InventoryItems.WOOD, BlockType.FURNACE, 1, 3),
        'str_check_lambda': "conditional_placing(gd, InventoryItems.WOOD, BlockType.FURNACE, 1, 3)"
    },
    'simple_conditional_placing_010': {
        'instruction': "Collect 2 saplings and place three plants",
        'instruction_paraphrases': [
            "After gathering two seedlings, set up three shrubs",
            "Collect two saplings and then plant three patches of greenery",
            "Once you've collected two small plants, install three trees at the designated spot",
            "Accumulate two young plants and position three bushes on the terrain",
            "Gather two saplings and place three vegetation spots at the specified location"
        ],
        'check_lambda': lambda gd, ix: conditional_placing(gd, InventoryItems.SAPLING, BlockType.PLANT, 2, 3),
        'str_check_lambda': "conditional_placing(gd, InventoryItems.SAPLING, BlockType.PLANT, 2, 3)"
    }
}

medium = {
    'medium_conditional_placing_001': {
        'instruction': "Collect 4 pieces of iron and place a chest",
        'instruction_paraphrases': [
            "After collecting four iron ingots, place a storage box",
            "Collect four metal units and then position a container",
            "Once you've gathered four iron pieces, set up a trunk on the map",
            "Accumulate four metal ingots and put a chest on the terrain",
            "Gather four iron units and install the storage chest at the designated spot"
        ],
        'check_lambda': lambda  gd, ix: conditional_placing(gd, MediumInventoryItems.IRON, BlockType.CHEST, 4, 1)
    },
    'medium_conditional_placing_002': {
        'instruction': "Collect 3 pieces of coal and place a fountain",
        'instruction_paraphrases': [
            "After gathering three lumps of coal, position the water feature",
            "Collect three coal units and then place a water fountain",
            "Once you've gathered three coal items, set up a water source",
            "Accumulate three coal pieces and put a fountain on the map",
            "Gather three coal pieces and install the fountain at the required spot"
        ],
        'check_lambda': lambda  gd, ix: conditional_placing(gd, MediumInventoryItems.COAL, BlockType.FOUNTAIN, 3, 2)
    },
    'medium_conditional_placing_003': {
        'instruction': "Collect 2 pieces of iron and place an enchantment table",
        'instruction_paraphrases': [
            "After collecting two iron ingots, place a magic table",
            "Collect two metal units and then position a spellbinding table",
            "Once you've gathered two iron pieces, set up a magical table on the map",
            "Accumulate two metal ingots and put an enchantment table on the terrain",
            "Gather two iron units and install the enchantment table at the specified spot"
        ],
        'check_lambda': lambda  gd, ix: conditional_placing(gd, MediumInventoryItems.IRON, BlockType.ENCHANTMENT_TABLE_FIRE, 2, 2)
    },
    'medium_conditional_placing_004': {
        'instruction': "Collect 4 diamonds and place two enchantment tables (ice)",
        'instruction_paraphrases': [
            "After collecting four gems, position two ice spellbinding tables",
            "Collect four diamond units and then place two frost enchantment tables",
            "Once you've gathered four diamond pieces, set up two frozen enchantment tables",
            "Accumulate four gemstones and put two ice magic tables on the terrain",
            "Gather four diamonds and install the two frost enchantment tables at the designated spot"
        ],
        'check_lambda': lambda  gd, ix: conditional_placing(gd, MediumInventoryItems.DIAMOND, BlockType.ENCHANTMENT_TABLE_ICE, 4, 2)
    },
    'medium_conditional_placing_005': {
        'instruction': "Collect 3 pieces of iron and place an enchantment table (fire)",
        'instruction_paraphrases': [
            "After collecting three iron ingots, position a fire magic table",
            "Collect three metal units and then place a flame enchantment table",
            "Once you've gathered three iron pieces, set up a fire spellbinding table on the map",
            "Accumulate three metal pieces and put a flame magic table on the terrain",
            "Gather three iron units and install the fire enchantment table at the designated spot"
        ],
        'check_lambda': lambda  gd, ix: conditional_placing(gd, MediumInventoryItems.IRON, BlockType.ENCHANTMENT_TABLE_FIRE, 3, 2)
    },
    'medium_conditional_placing_006': {
        'instruction': "Collect 2 pieces of coal and place three fountains",
        'instruction_paraphrases': [
            "After gathering two lumps of coal, set up three water features",
            "Collect two coal units and then position three fountains on the map",
            "Once you've gathered two coal items, install three fountains at the designated spot",
            "Accumulate two coal pieces and put three water sources on the terrain",
            "Gather two coal pieces and arrange three fountains at the required location"
        ],
        'check_lambda': lambda  gd, ix: conditional_placing(gd, MediumInventoryItems.COAL, BlockType.FOUNTAIN, 2, 3)
    },
    'medium_conditional_placing_007': {
        'instruction': "Collect 3 pieces of iron and place two chests",
        'instruction_paraphrases': [
            "After gathering three iron ingots, place two storage boxes",
            "Collect three metal units and then set up two containers on the terrain",
            "Once you've gathered three iron pieces, install two trunks",
            "Accumulate three metal pieces and position two storage chests on the map",
            "Gather three iron units and arrange two containers at the specified spot"
        ],
        'check_lambda': lambda  gd, ix: conditional_placing(gd, MediumInventoryItems.IRON, BlockType.CHEST, 3, 2)
    },
    'medium_conditional_placing_008': {
        'instruction': "Collect 4 pieces of coal and place two enchantment tables (fire)",
        'instruction_paraphrases': [
            "After collecting four lumps of coal, set up two flame spellbinding tables",
            "Gather four coal units and then position two fire enchantment tables on the map",
            "Once you've gathered four coal items, install two fire magic tables at the designated spot",
            "Accumulate four coal pieces and put two flame enchantment tables on the terrain",
            "Gather four coal pieces and arrange two fire spellbinding tables at the required location"
        ],
        'check_lambda': lambda  gd, ix: conditional_placing(gd, MediumInventoryItems.COAL, BlockType.ENCHANTMENT_TABLE_FIRE, 4, 2)
    },
    'medium_conditional_placing_009': {
        'instruction': "Collect 2 diamonds and place three enchantment tables (ice)",
        'instruction_paraphrases': [
            "After collecting two gems, position three frost magic tables",
            "Gather two diamond units and then place three ice spellbinding tables on the map",
            "Once you've gathered two diamond pieces, set up three frozen enchantment tables",
            "Accumulate two gemstones and put three frost magic tables on the terrain",
            "Gather two diamonds and install the three ice enchantment tables at the designated spot"
        ],
        'check_lambda': lambda  gd, ix: conditional_placing(gd, MediumInventoryItems.DIAMOND, BlockType.ENCHANTMENT_TABLE_ICE, 2, 3)
    },
    'medium_conditional_placing_010': {
        'instruction': "Collect 3 pieces of iron and place four chests",
        'instruction_paraphrases': [
            "After gathering three iron ingots, set up four storage boxes",
            "Collect three metal units and then position four containers on the map",
            "Once you've gathered three iron pieces, install four trunks at the designated spot",
            "Accumulate three metal pieces and put four storage chests on the terrain",
            "Gather three iron units and arrange four containers at the required location"
        ],
        'check_lambda': lambda  gd, ix: conditional_placing(gd, MediumInventoryItems.IRON, BlockType.CHEST, 3, 4)
    }
}

from craftext.scenarios.parce_dataset import update_previous_dict
easy = update_previous_dict(easy, "jax_conditional_placing/instructions/train/easy", "conditional_placing")
medium = update_previous_dict(medium, "jax_conditional_placing/instructions/train/medium", "conditional_placing")
