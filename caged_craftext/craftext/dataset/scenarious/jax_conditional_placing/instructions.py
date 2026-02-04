from craftext.environment.scenarious.checkers.target_state import TargetState, ConditionalPlacingState
from craftext.environment.craftext_constants import InventoryItems, BlockType, MediumInventoryItems, Scenarios
# one = {
#     'simple_conditional_placing_001': {
#         'instruction': "Place a furnace on the ground after collecting 3 pieces of wood",
#         'instruction_paraphrases': [
#             "After gathering three logs, place a kiln on the ground",
#             "Collect three wooden blocks and then position a heater on the ground",
#             "Once you've collected 3 logs, set up an oven on the terrain",
#             "Accumulate three wooden units and put a stove on the ground",
#             "Gather three pieces of timber and install a smelter at the designated spot"
#         ],
#         'check_lambda': lambda gd, ix: conditional_placing(gd, InventoryItems.WOOD, BlockType.FURNACE, 1, 1)
#     },
#     'simple_conditional_placing_002': {
#         'instruction': "Place a furnace on the ground after collecting 3 pieces of wood",
#         'instruction_paraphrases': [
#             "After gathering three logs, place a kiln on the ground",
#             "Collect three wooden blocks and then position a heater on the ground",
#             "Once you've collected 3 logs, set up an oven on the terrain",
#             "Accumulate three wooden units and put a stove on the ground",
#             "Gather three pieces of timber and install a smelter at the designated spot"
#         ],
#         'check_lambda': lambda gd, ix: conditional_placing(gd, InventoryItems.WOOD, BlockType.FURNACE, 1, 1)
#     },
# }

def create_target_state(object_inventory_enum, object_to_place, count_to_collect, count_to_stand):
    target_achievements = ConditionalPlacingState(object_inventory_enum, object_to_place, count_to_collect, count_to_stand)
    return TargetState(conditional_placing=target_achievements)

easy = {
    "INSTRUCTION_PLACE_IRON_FURNACE_4_3": {
        "instruction": "Collect 4 iron and place 3 furnace",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "Gather four pieces of iron and set down three furnaces",
            "Acquire and hold onto four irons, then create three heating places.",
            "Find four iron ores and afterwards put down three pieces of furnace.",
            "Procure a quantity of four irons and subsequently arrange three units of furnace.",
            "Accumulate four iron units in your possession and thereafter install three furnace structures"
        ],
        "arguments": create_target_state(InventoryItems.IRON, BlockType.FURNACE, 4, 3),
        "str_check_lambda": "conditional_placing(gd, ix)"
    },
    "INSTRUCTION_PLACE_DIAMOND_STONE_3_1": {
        "instruction": "Collect three diamonds and place one rock on the map.",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "Procure a trio of diamonds and subsequently install a single stone on the terrain.",
            "Accumulate enough diamonds to number three and deposit a lone boulder on the landscape.",
            "Acquire three gems of the diamond variety and then position a solitary piece of rock within the game map.",
            "Amass an aggregate of three diamonds and place one instance of stone upon the mapped terrain.",
            "Secure three diamonds in your inventory and plant one fragment of rock on our strategic planning map."
        ],
        "arguments": create_target_state(InventoryItems.DIAMOND, BlockType.STONE, 3, 1),
        "str_check_lambda": "conditional_placing(gd, ix)"
    },
    "INSTRUCTION_PLACE_WOOD_CRAFTING_TABLE_3_1": {
        "instruction": "Collect 3 pieces of wood and then place 1 crafting table.",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "Gather three logs and place a single workbench.",
            "Procure a trio of timber and set down one builder's desk.",
            "Amass three units of lumber and establish a single artisan table.",
            "Accumulate three fragments of wood and then construct one crafting station.",
            "Assemble a collection of three wooden pieces and situate one carpentry table."
        ],
        "arguments": create_target_state(InventoryItems.WOOD, BlockType.CRAFTING_TABLE, 3, 1),
        "str_check_lambda": "conditional_placing(gd, ix)"
    },
    "INSTRUCTION_PLACE_STONE_CRAFTING_TABLE_2_5": {
        "instruction": "Collect 2 stones and place 5 crafting tables.",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "Acquire two rocks and set up five workstations.",
            "Obtain a duo of pebbles and establish quintuple functioning desks.",
            "Secure two fragments of stone and position five craft stations.",
            "Procure a pair of boulders and install five engineering surfaces.",
            "Get hold of two cobblestones and lay down five fabrication benches."
        ],
        "arguments": create_target_state(InventoryItems.STONE, BlockType.CRAFTING_TABLE, 2, 5),
        "str_check_lambda": "conditional_placing(gd, ix)"
    },
    "INSTRUCTION_PLACE_COAL_FURNACE_4_3": {
        "instruction": "Pick up 4 pieces of coal and place down 3 furnaces",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "Grab 4 coal blocks and deploy 3 heating furnaces",
            "Collect 4 amounts of coal and put down 3 of those furnaces",
            "Get your hands on 4 fragments of coal and install 3 furnace units",
            "You have to harvest 4 coals and lay down 3 cooking furnaces",
            "Ensure to grab 4 pieces of coal and firmly place 3 units of the furnace"
        ],
        "arguments": create_target_state(InventoryItems.COAL, BlockType.FURNACE, 4, 3),
        "str_check_lambda": "conditional_placing(gd, ix)"
    },
    "INSTRUCTION_PLACE_IRON_STONE_4_3": {
        "instruction": "Gather four pieces of iron and place three stones in your environment",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "Please collect four iron objects and then put three stones in place",
            "Secure four units of iron and lay down three units of stone",
            "Could you obtain, please, quartet of iron resources and deploy a trio of stones?",
            "Your task is to acquire a collection of four iron items and properly locate three rock objects",
            "You should seek to accumulate an amount of iron which totals to four and subsequently install three parts of rock within the game"
        ],
        "arguments": create_target_state(InventoryItems.IRON, BlockType.STONE, 4, 3),
        "str_check_lambda": "conditional_placing(gd, ix)"
    },
    "INSTRUCTION_PLACE_STONE_STONE_1_4": {
        "instruction": "Collect one piece of stone and then place four of them.",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "Gather a single rock and then position four rocks.",
            "Scavenge for a single piece of stone, afterwards, lay four of them.",
            "Acquire one piece of rock, then proceed to arrange four accordingly.",
            "Discover one rock, followed by setting up four of them.",
            "Obtain one lump of stone, succeeding which, position four such pieces."
        ],
        "arguments": create_target_state(InventoryItems.STONE, BlockType.STONE, 1, 4),
        "str_check_lambda": "conditional_placing(gd, ix)"
    },
    "INSTRUCTION_PLACE_STONE_FURNACE_1_4": {
        "instruction": "Collect one rock and place four furnaces.",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "Get a hold of one stone and put down four ovens.",
            "Accumulate a single piece of rock and establish four furnaces.",
            "Gather one piece of stone and install a quartet of ovens.",
            "Obtain a unit of a stone and locate four pieces of heating equipment. Secure for yourself a solitary rock and then, in turn, position on the map four furnaces."
        ],
        "arguments": create_target_state(InventoryItems.STONE, BlockType.FURNACE, 1, 4),
        "str_check_lambda": "conditional_placing(gd, ix)"
    },
    "INSTRUCTION_PLACE_WOOD_PLANT_5_1": {
        "instruction": "Collect 5 pieces of wood and place 1 plant",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "Gather 5 logs and put down 1 vegetation",
            "You need to collect five wooden logs and then place a single shrub",
            "Accumulate a total of 5 wood items and proceed to situate one greenery",
            "The task requires the procurement of quintuple pieces of timber followed by the installation of a solitary undergrowth",
            "Your mission is to amass a quintet of lumber elements and subsequently establish one form of flora"
        ],
        "arguments": create_target_state(InventoryItems.WOOD, BlockType.PLANT, 5, 1),
        "str_check_lambda": "conditional_placing(gd, ix)"
    },
    "INSTRUCTION_PLACE_COAL_FURNACE_2_1": {
        "instruction": "Collect 2 pieces of coal and place 1 furnace.",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "Find and pick up 2 coals then put down 1 furnace.",
            "Acquire a couple of coal then put a furnace in place.",
            "Gather two pieces of black stone and build one stove.",
            "Retrieve two chunks of fossil fuel and position one oven.",
            "Accumulate two units of carbon-rich rock and construct a single forge."
        ],
        "arguments": create_target_state(InventoryItems.COAL, BlockType.FURNACE, 2, 1),
        "str_check_lambda": "conditional_placing(gd, ix)"
    },
    "INSTRUCTION_PLACE_DIAMOND_STONE_5_1": {
        "instruction": "Collect 5 diamonds and place 1 stone",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "Gather a group of five diamonds and position a stone",
            "Amass five sparklers and set one rock",
            "Grab five treasures of the earth and situate a single boulder",
            "Procure a handful of these precious gems, exactly five diamonds and establish one solid element, a stone to be exact",
            "Your primary mission is to acquire an exact quantity of five of the most desirable gems known as diamonds, and following that, position one fundamental element on the field which refers to as 'stone'"
        ],
        "arguments": create_target_state(InventoryItems.DIAMOND, BlockType.STONE, 5, 1),
        "str_check_lambda": "conditional_placing(gd, ix)"
    },
    "INSTRUCTION_PLACE_COAL_PLANT_4_4": {
        "instruction": "Collect 4 pieces of coal and place 4 plants down.",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "Gather 4 units of coal and deploy 4 shrubs.",
            "Acquire four lumps of coal and carefully position four flora.",
            "Procure a quartet of pieces of coal and settle four greenery in suitable spots.",
            "Your mission is to hoard four chunks of this black mineral we call coal and subsequently make four vegetation items find their roots in the soil.",
            "The expectations are upon you to amass an exact four count of the dark, carbon-rich mineral, much precious for the realm, and to follow it up by establishing, steadily, four saplings in the earth."
        ],
        "arguments": create_target_state(InventoryItems.COAL, BlockType.PLANT, 4, 4),
        "str_check_lambda": "conditional_placing(gd, ix)"
    },
    "INSTRUCTION_PLACE_COAL_CRAFTING_TABLE_3_3": {
        "instruction": "Gather 3 pieces of coal and set up 3 crafting tables.",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "Collect 3 units of coal and place 3 workbenches.",
            "Pick up a trio of coals and establish 3 craft stations.",
            "Accumulate three pieces of carbon and install 3 stations for crafting.",
            "Amass 3 chunks of coal and erect three tables for artisanal tasks.",
            "Procure triple units of dark matter and build a trio of surfaces dedicated to crafting."
        ],
        "arguments": create_target_state(InventoryItems.COAL, BlockType.CRAFTING_TABLE, 3, 3),
        "str_check_lambda": "conditional_placing(gd, ix)"
    },
    "INSTRUCTION_PLACE_DIAMOND_PLANT_5_3": {
        "instruction": "Collect five diamonds and plant three saplings.",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "Gather five pieces of diamond and put three saplings into the ground.",
            "Go and find five diamonds, then ensure you plant three seedlings.",
            "Your task is to accumulate a stock of five diamonds and establish three young trees in the soil.",
            "Retrieve five precious diamonds from your surroundings, after which you are to embed three offspring of a tree into the earth.",
            "Procure a quintet of shimmering diamonds, subsequent to which you should ensure that a trio of tree progenies is securely situated in the ground."
        ],
        "arguments": create_target_state(InventoryItems.DIAMOND, BlockType.PLANT, 5, 3),
        "str_check_lambda": "conditional_placing(gd, ix)"
    },
    "INSTRUCTION_PLACE_SAPLING_PLANT_3_1": {
        "instruction": "Collect 3 saplings and place 1 plant.",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "Pick up three small trees and then place a plant.",
            "Gather three saplings and place one vegetation.",
            "You need to secure three young trees and position a piece of greenery.",
            "Your task is to accumulate three seedlings and then situate a single flora.",
            "It is required of you to amass a trio of sprouts and subsequently localize one botanical entity."
        ],
        "arguments": create_target_state(InventoryItems.SAPLING, BlockType.PLANT, 3, 1),
        "str_check_lambda": "conditional_placing(gd, ix)"
    }
}


medium = {
    "INSTRUCTION_PLACE_SAPPHIRE_FURNACE_3_5": {
        "instruction": "[MEDIUM] COLLECT 3 SAPPHIRE AND PLACE 5 FURNACE",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "[MEDIUM] Gather 3 gems and setup 5 heaters",
            "[MEDIUM] Need you to pick up 3 jewels and put down 5 ovens",
            "Stash 3 jades, at medium level, and set 5 cooktops in place",
            "Grab 3 precious stones over a medium difficulty and station 5 stoves",
            "Recover 3 glistening treasures and erect 5 warming units, bear in mind this isn't easy"
        ],
        "arguments": create_target_state(MediumInventoryItems.SAPPHIRE, BlockType.FURNACE, 3, 5),
        "str_check_lambda": "conditional_placing(gd, ix)"
    },
    "INSTRUCTION_PLACE_IRON_ENCHANTMENT_TABLE_ICE_5_1": {
        "instruction": "Collect 5 iron ores and place down 1 Ice Enchantment Table.",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "Gather five pieces of iron, then situate a Frost Conjuration Workbench.",
            "Pick up five iron minerals and install one Freezing Spell Desk.",
            "Procure 5 Iron and set up an Icy Charm Table.",
            "Accumulate a quintet of iron and position a single Frostbound Enchantment Stand.",
            "Obtain handful of five iron resources and establish singular Glacial Magic Crafting Station."
        ],
        "arguments": create_target_state(MediumInventoryItems.IRON, BlockType.ENCHANTMENT_TABLE_ICE, 5, 1),
        "str_check_lambda": "conditional_placing(gd, ix)"
    },
    "INSTRUCTION_PLACE_RUBY_ENCHANTMENT_TABLE_ICE_1_2": {
        "instruction": "Collect one ruby and place two ice enchantment tables",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "Gather one of rubies and put down two ice mage's desks",
            "Acquire a single precious stone and establish a pair of frost wizard's workbenches",
            "Obtain a single gemstone and install two tables of icy sorcery",
            "Accumulate one jewel and arrange two icy magic altars",
            "Amass an individual gem and position a duo of frosty enchantment stations"
        ],
        "arguments": create_target_state(MediumInventoryItems.RUBY, BlockType.ENCHANTMENT_TABLE_ICE, 1, 2),
        "str_check_lambda": "conditional_placing(gd, ix)"
    },
    "INSTRUCTION_PLACE_SAPLING_TORCHES_5_5": {
        "instruction": "Collect five saplings and place five torches.",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "Gather five saplings and set up five torches.",
            "You need to obtain a quantity of five saplings and establish five torches.",
            "Your task is to accrue five saplings and emplace five torches.",
            "It is your duty to procure five saplings and mount five torches.",
            "The endeavor requires you to amass five saplings and affix five torches."
        ],
        "arguments": create_target_state(MediumInventoryItems.SAPLING, MediumInventoryItems.TORCHES, 5, 5),
        "str_check_lambda": "conditional_placing(gd, ix)"
    },
    "INSTRUCTION_PLACE_RUBY_STONE_5_2": {
        "instruction": "Collect 5 rubies from the inventory and then place 2 stones on the map.",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "From your inventory, gather 5 gems and subsequently position 2 rocks on the map.",
            "Secure 5 red stones from your collection and then arrange 2 pebbles on the terrain.",
            "Acquire 5 pieces of ruby from your existing resources and then settle 2 boulders on the landscape.",
            "Procure a handful of five precious rubies and then install a couple of stones on the designated map zones.",
            "Amass quintuple ruby crystals from your reserves and thereafter, anchor duo of stones into mapped demarcation."
        ],
        "arguments": create_target_state(MediumInventoryItems.RUBY, BlockType.STONE, 5, 2),
        "str_check_lambda": "conditional_placing(gd, ix)"
    },
    "INSTRUCTION_PLACE_DIAMOND_STONE_3_1": {
        "instruction": "Collect three diamonds and place one stone block.",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "Gather a total of three diamond gems and position a single block of stone.",
            "Accumulate three glittering diamonds and set one solid stone into place.",
            "Pick up trio of diamonds and carefully lay down just one stone brick.",
            "Ensure you have three shiny diamonds in your possession and position a single chunk of rock.",
            "Secure three sparkling diamond gems and meticulously place one piece of stone."
        ],
        "arguments": create_target_state(MediumInventoryItems.DIAMOND, BlockType.STONE, 3, 1),
        "str_check_lambda": "conditional_placing(gd, ix)"
    },
    "INSTRUCTION_PLACE_STONE_ENCHANTMENT_TABLE_ICE_2_4": {
        "instruction": "Collect two stones and place four Ice Tables.",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "Pick up two chunks of rock and then set down four Enchantment Icy Workbenches.",
            "Secure two pieces of stone and then position four Magic Icy Tables.",
            "Gather a pair of stones and then put in place four Enchantment Tables of Ice.",
            "Acquire two units of stone and then establish four Alchemy Ice Tables.",
            "Obtain a couple of rocks and then lay down four Icy Enchantment Desks."
        ],
        "arguments": create_target_state(MediumInventoryItems.STONE, BlockType.ENCHANTMENT_TABLE_ICE, 2, 4),
        "str_check_lambda": "conditional_placing(gd, ix)"
    },
    "INSTRUCTION_PLACE_WOOD_ENCHANTMENT_TABLE_ICE_4_2": {
        "instruction": "Collect 4 pieces of wood and then place 2 ice enchantment tables.",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "Gather four logs and subsequently set down two tables of ice enchantment.",
            "Pick up quartet of wood logs for the purpose to erect duo of ice-enchanted desks afterwards.",
            "Grab a total of four lumber pieces and then proceed to lay down two icy wizardry stations.",
            "Securely obtain an accumulation of four timber items and on achieving that, meticulously arrange a pair of enchantment stations made of ice.",
            "Accumulate a count of four timber units, and when that's done, ensure to station two tables endowed with ice enchantment."
        ],
        "arguments": create_target_state(MediumInventoryItems.WOOD, BlockType.ENCHANTMENT_TABLE_ICE, 4, 2),
        "str_check_lambda": "conditional_placing(gd, ix)"
    },
    "INSTRUCTION_PLACE_SAPPHIRE_FURNACE_5_4": {
        "instruction": "Collect 5 sapphires and place 4 furnaces.",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "Please gather 5 of sapphires and then place down 4 furnaces.",
            "I would like you to acquire 5 sapphires and then, subsequently, install 4 heating systems.",
            "Accumulate 5 sapphires and deposit 4 heating equipment.",
            "Procure 5 of blue gemstones and erect 4 stove-like apparatus.",
            "Can you please generate a pile of 5 sapphires and establish a placement of 4 smelting devices?"
        ],
        "arguments": create_target_state(MediumInventoryItems.SAPPHIRE, BlockType.FURNACE, 5, 4),
        "str_check_lambda": "conditional_placing(gd, ix)"
    },
    "INSTRUCTION_PLACE_RUBY_PLANT_2_1": {
        "instruction": "For your task, gather two rubies and plant a seed.",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "I need you to gather a pair of rubies and sow a seed.",
            "Could you please collect a couple of rubies and place a sprout?",
            "Your task is to gather two sparkling rubies and cultivate a sapling.",
            "Kindly obtain two precious red stones and grow a young plant.",
            "Your assignment is to capture two gleaming gemstones and propagate a vegetation."
        ],
        "arguments": create_target_state(MediumInventoryItems.RUBY, BlockType.PLANT, 2, 1),
        "str_check_lambda": "conditional_placing(gd, ix)"
    },
    "INSTRUCTION_PLACE_COAL_TORCHES_4_3": {
        "instruction": "Collect four coals and then place three torches.",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "The first task for you is to gather four coals, and afterward, put down three lights.",
            "Begin by picking up four pieces of coal, and once you've done that, place three torches.",
            "Your mission is to first amass a quartet of coals, following which, you are to put three torches.",
            "Task yourself with accumulating four chunks of carbon, then fulfil the role of lighting up three torches.",
            "We would require you to obtain a handful of four coals, post which your duty would be to position three torches strategically."
        ],
        "arguments": create_target_state(MediumInventoryItems.COAL, MediumInventoryItems.TORCHES, 4, 3),
        "str_check_lambda": "conditional_placing(gd, ix)"
    },
    "INSTRUCTION_PLACE_WOOD_FURNACE_3_5": {
        "instruction": "Collect 3 wood and place 5 furnaces.",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "Gather a total of three logs at your disposal and litter the field with five of our best burners.",
            "Amass three portions of timber then establish five heating systems.",
            "Accumulate three units of lumber and position five ovens.",
            "Build your inventory with three materials from the trees and situate five stoves in the playing area.",
            "I need you to gather three wooden items and successfully set down five kilns."
        ],
        "arguments": create_target_state(MediumInventoryItems.WOOD, BlockType.FURNACE, 3, 5),
        "str_check_lambda": "conditional_placing(gd, ix)"
    },
    "INSTRUCTION_PLACE_DIAMOND_CRAFTING_TABLE_1_2": {
        "instruction": "Collect 1 diamond and place down 2 crafting tables",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "Gather 1 gem and set up 2 workbenches",
            "Procure a single gemstone and install a pair of construction platforms",
            "Acquire a diamond and establish two artisan tables",
            "Obtain one shiny stone and organize dual crafting surfaces",
            "Secure an individual jewel and station two fabrication boards"
        ],
        "arguments": create_target_state(MediumInventoryItems.DIAMOND, BlockType.CRAFTING_TABLE, 1, 2),
        "str_check_lambda": "conditional_placing(gd, ix)"
    },
    "INSTRUCTION_PLACE_COAL_STONE_2_1": {
        "instruction": "Collect 2 pieces of coal from the inventory and place 1 stone on the map.",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "Fetch two coal items from your storage and put a piece of rock on the map.",
            "Get a couple of coal items from your inventory and position a single stone in the game map.",
            "Retrieve two units of coal from your inventory and deposit a rock on the map.",
            "Obtain two pieces of coal from your storage, then place one stone within the game area.",
            "From your stockpile, secure two quantities of coal, then locate a rock on the game map."
        ],
        "arguments": create_target_state(MediumInventoryItems.COAL, BlockType.STONE, 2, 1),
        "str_check_lambda": "conditional_placing(gd, ix)"
    },
    "INSTRUCTION_PLACE_SAPPHIRE_ENCHANTMENT_TABLE_FIRE_4_2": {
        "instruction": "Collect 4 sapphires and place 2 enchantment table fire",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "Gather four sapphires and put up two fire enchantment tables",
            "Retrieve four pieces of sapphire and establish two tables with fire enchantments",
            "Accumulate four blue jewels and erect two enchanting tables aflame",
            "Amass a quartet of sapphires and install a pair of enchantment platforms on fire",
            "Secure four sapphire gems and construct two consecration tables with fire elements"
        ],
        "arguments": create_target_state(MediumInventoryItems.SAPPHIRE, BlockType.ENCHANTMENT_TABLE_FIRE, 4, 2),
        "str_check_lambda": "conditional_placing(gd, ix)"
    },
    "INSTRUCTION_PLACE_DIAMOND_ENCHANTMENT_TABLE_FIRE_4_2": {
        "instruction": "Collect 4 diamonds and place 2 fire enchantment tables.",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "Gather 4 gems and set up 2 fire imbuing stations.",
            "Amass a total of four diamonds and install a couple of enchantment tables of fire.",
            "Retrieve four pieces of diamond and lay down two fire enchantment workbenches.",
            "Accumulate four crystals then establish two fire magical counters.",
            "Hoard four diamond stones and develop two fiery enchanting platforms."
        ],
        "arguments": create_target_state(MediumInventoryItems.DIAMOND, BlockType.ENCHANTMENT_TABLE_FIRE, 4, 2),
        "str_check_lambda": "conditional_placing(gd, ix)"
    },
    "INSTRUCTION_PLACE_SAPLING_PLANT_3_5": {
        "instruction": "Collect 3 saplings and place 5 plants.",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "You need to gather three saplings and then put down five plants.",
            "Make sure to amass a total of three small trees and subsequently situate a handful of five green plants.",
            "Accumulate a trio of little tree sprouts, followed by the placement of five pieces of foliage.",
            "Your task is to obtain three young trees and afterwards establish a setting of five botanical organisms.",
            "Procure three samples of saplings and later distribute a quintet of photosynthetic living entities."
        ],
        "arguments": create_target_state(MediumInventoryItems.SAPLING, BlockType.PLANT, 3, 5),
        "str_check_lambda": "conditional_placing(gd, ix)"
    },
    "INSTRUCTION_PLACE_COAL_ENCHANTMENT_TABLE_FIRE_3_3": {
        "instruction": "Collect 3 coal and place 3 fire enchantment tables.",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "Obtain three pieces of coal and set down three fire enchanting tables.",
            "Harvest trio of coals and establish a trio of enchantment tables of fire.",
            "Gather three coal items and deploy three fire enchantment workbenches.",
            "Accumulate three coals and position three fire wizardry tables.",
            "Amass a set of three coals and erect three flame sorcery stations."
        ],
        "arguments": create_target_state(MediumInventoryItems.COAL, BlockType.ENCHANTMENT_TABLE_FIRE, 3, 3),
        "str_check_lambda": "conditional_placing(gd, ix)"
    },
    "INSTRUCTION_PLACE_COAL_FURNACE_2_1": {
        "instruction": "Collect 2 coal pieces and place 1 furnace",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "Prepare 2 units of coal and establish 1 hearth",
            "Retreive two chunks of coal and put down a single furnace",
            "Assemble 2 lumps of coal then position 1 stove",
            "Gather a pair of coals and then install one kiln",
            "Procure two coals and then set up a solitary furnace"
        ],
        "arguments": create_target_state(MediumInventoryItems.COAL, BlockType.FURNACE, 2, 1),
        "str_check_lambda": "conditional_placing(gd, ix)"
    },
    "INSTRUCTION_PLACE_RUBY_ENCHANTMENT_TABLE_FIRE_3_4": {
        "instruction": "Collect 3 rubies and place 4 enchantment fire tables.",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "Gather 3 set of rubies and place four fire enchantment stations.",
            "Accumulate three red gems and install four enchantment desks with fire enchantment.",
            "Obtain a total of trio of rubies and position quartet of fire enchanting benches.",
            "Secure three red-colored gemstones and add four workstations with fire-based enchantment.",
            "Pick up three units of gleaming rubies and set down four magical fire tables."
        ],
        "arguments": create_target_state(MediumInventoryItems.RUBY, BlockType.ENCHANTMENT_TABLE_FIRE, 3, 4),
        "str_check_lambda": "conditional_placing(gd, ix)"
    },
    "INSTRUCTION_PLACE_COAL_FURNACE_1_2": {
        "instruction": "Gather 1 Coal and then place 2 furnaces.",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "Pick 1 lump of coal and then build 2 heating furnaces.",
            "Procure 1 piece of coal and afterwards lay down a pair of furnaces.",
            "You need to collect 1 bit of coal and subsequently put up two ovens.",
            "Secure a single coal, following that erect 2 stoves.",
            "Obtain one unit of coal first, succeedingly assemble a duo of heating apparatus."
        ],
        "arguments": create_target_state(MediumInventoryItems.COAL, BlockType.FURNACE, 1, 2),
        "str_check_lambda": "conditional_placing(gd, ix)"
    },
    "INSTRUCTION_PLACE_SAPLING_ENCHANTMENT_TABLE_FIRE_1_1": {
        "instruction": "Collect a sapling and place an enchantment table of fire.",
        "scenario_checker": Scenarios.CONDITIONAL_PLACING,
        "instruction_paraphrases": [
            "Find a sapling and then position a fire enchantment table.",
            "Firstly, acquire a sapling and afterwards, establish a table for fire enchantment.",
            "Pick up one sapling and then set down an enchantment table with fire abilities.",
            "Procure one sapling and subsequently position an arcane fire table.",
            "First, appropriate a sapling and then deposit a table of pyro-enchantment."
        ],
        "arguments": create_target_state(MediumInventoryItems.SAPLING, BlockType.ENCHANTMENT_TABLE_FIRE, 1, 1),
        "str_check_lambda": "conditional_placing(gd, ix)"
    }
}


