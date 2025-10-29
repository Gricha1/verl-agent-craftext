from flax.struct import dataclass
from craftext.enviroment.scenarious.checkers.target_state import TargetState, BuildStarState
from craftext.enviroment.craftext_constants import Scenarios, BlockType, CrossType

def create_target_state(block_type:int, size:int, cross_type:int):
    target_achievements = BuildStarState(block_type=block_type, size=size, radius=10, cross_type=cross_type)
    return TargetState(building_star=target_achievements)


easy = {
    "INSTRUCTION_PLANT_5": {
        "instruction": "Form a cross with Plant elements, each side should be 5 units long and in diagonal shape.",
        "scenario_checker": Scenarios.BUILD_STAR,
        "instruction_paraphrases": [
            "Use Plant elements to construct a diagonal cross with each side consisting of 5 units.",
            "Make a cross that is diagonally oriented using Plant items, ensuring each arm of the cross is 5 units in length.",
            "Create a diagonal cross with 5 units on each side, using various forms of Plant elements for construction.",
            "With a Plant item, form a cross that has a side length of 5 units and is oriented diagonally.",
            "Assemble a 5-unit each side diagonal cross using various Plant components."
        ],
        "arguments": create_target_state(block_type=BlockType.PLANT, size=5,  cross_type=CrossType.DIAGONAL),
        "str_check_lambda": "lambda gd, ix: is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_STONE_5": {
        "instruction": "Create a cross of stones with arms of length 5 in any direction.",
        "scenario_checker": Scenarios.BUILD_STAR,
        "instruction_paraphrases": [
            "Form a cross using rocks, ensuring each arm extends 5 units long, and you can orient it in any way.",
            "Configure a collection of pebbles into a cruciform shape, with the length of each limb equally measuring 5 units; the alignment of the cross is up to you.",
            "Using any form of alignment, assemble a cross, consisting of boulders, where each arm stretches out to 5 spaces.",
            "In any given direction, erect an equilateral crucifix shaped configuration from chunks of rock, each arm of which should span 5 units.",
            "Fashion a cross of stones of any orientation where each arm's length equals 5 units."
        ],
        "arguments": create_target_state(block_type=BlockType.STONE, size=5,  cross_type=CrossType.STRAIGHT),
        "str_check_lambda": "lambda gd, ix: is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_PLANT_3": {
        "instruction": "Make a cross shape using plants with a side length of 3 units.",
        "scenario_checker": Scenarios.BUILD_STAR,
        "instruction_paraphrases": [
            "Create a plant-based cross with each side being 3 units long.",
            "Construct a cross figure using flora with each arm spanning 3 units",
            "Use vegetation to form a cross, each of which measures 3 units.",
            "Design a cruciform structure using plants, with each side having a length of 3 units.",
            "Fabricate a cross-shaped figure with greenery having each side of 3 units."
        ],
        "arguments": create_target_state(block_type=BlockType.PLANT, size=3,  cross_type=CrossType.STRAIGHT),
        "str_check_lambda": "lambda gd, ix: is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_STONE_5_DIAGONAL": {
        "instruction": "Create a cross of stones with side size equal to 5 arranged in a diagonal pattern.",
        "scenario_checker": Scenarios.BUILD_STAR,
        "instruction_paraphrases": [
            "Construct a stone cross where each side length is 5 and in diagonal formation.",
            "I would like you to form a cross using rocks with five units on each side in a diagonal orientation.",
            "Build a diagonal cross with pebbles, make sure each side is made up of a total of five units.",
            "Design a boulder-formed cross utilizing a diagonal layout with each arm having a length of 5.",
            "Using rubble, sketch out a cross in a slanted arrangement where each line is composed of five units."
        ],
        "arguments": create_target_state(block_type=BlockType.STONE, size=5,  cross_type=CrossType.DIAGONAL),
        "str_check_lambda": "lambda gd, ix: is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_STONE_7": {
        "instruction": "Construct a cross of stone of side size 7 and with a diagonal shape.",
        "scenario_checker": Scenarios.BUILD_STAR,
        "instruction_paraphrases": [
            "Create a diagonal cross using rock, with each arm being seven units long.",
            "Use pebbles to form a cross diagonally, ensuring each side measures seven units.",
            "Make a diagonal cross by using cobblestones, each arm should be seven units long.",
            "Shape a rock into a cross with seven units on each side, ensuring it's laid out diagonally.",
            "Craft a cross-shaped design using small stones diagonally, and it should span seven units on each side."
        ],
        "arguments": create_target_state(block_type=BlockType.STONE, size=7,  cross_type=CrossType.DIAGONAL),
        "str_check_lambda": "lambda gd, ix: is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_STONE_3": {
        "instruction": "Create a direct cross made of stone with each side measuring 3 blocks.",
        "scenario_checker": Scenarios.BUILD_STAR,
        "instruction_paraphrases": [
            "Craft a straightforward cross shape out of rock with each arm being 3 blocks long.",
            "Please form a rock cross with each of its sides being three units long in a direct shape.",
            "Would you mind making a cross from stone blocks each side of which measures up to three units, and make sure it's not diagonal or combined?",
            "I need you to construct a direct cross using stone material, make sure that the length of each of it sides is exactly three blocks.",
            "Make a formation in a direct cross pattern using stone blocks where each branch of the cross is three units long."
        ],
        "arguments": create_target_state(block_type=BlockType.STONE, size=3,  cross_type=CrossType.STRAIGHT),
        "str_check_lambda": "lambda gd, ix: is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_FIRE_TREE_5": {
        "instruction": "Form a cross of torchlights each side having 5 units and in combined form.",
        "scenario_checker": Scenarios.BUILD_STAR,
        "instruction_paraphrases": [
            "Build a blended cross using torches that is five units long on each side.",
            "Use your luminary torches to construct a cross where each side measures five units and the shape is amalgamated.",
            "Create a fusion design of a cross using illumination torches with each arm spanning five units long.",
            "Five unit long on each side, design a merged cross symbol, using your light torch.",
            "With a combination approach, construct a five-unit long cross structure using fire sticks."
        ],
        "arguments": create_target_state(block_type=BlockType.FIRE_TREE, size=5,  cross_type=CrossType.COMBINED),
        "str_check_lambda": "lambda gd, ix: is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_ENCHANTMENT_TABLE_FIRE_3": {
        "instruction": "Make sure you have an enchantment table fire arranged in the form of a cross with a size of 3.",
        "scenario_checker": Scenarios.BUILD_STAR,
        "instruction_paraphrases": [
            "Construct a cross using the fire enchantment table, it should have a length of three.",
            "Place the enchantment table fire units in a direct-shape cross configuration with a size of three.",
            "Can you arrange three of our enchantment table fire units in a cross format?",
            "I want you to form a cross with a side length of three using the fire enchantment table.",
            "With a size of three, form a cross from the units of enchantment table fire."
        ],
        "arguments": create_target_state(block_type=BlockType.ENCHANTMENT_TABLE_FIRE, size=3,  cross_type=CrossType.STRAIGHT),
        "str_check_lambda": "lambda gd, ix: is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_PLANT_7_COMBINED": {
        "instruction": "Make a cross of plant blocks that is 7 blocks on each side and combined shape around you.",
        "scenario_checker": Scenarios.BUILD_STAR,
        "instruction_paraphrases": [
            "Create a cross shape using the plant blocks with each arm having 7 blocks in a combined formation.",
            "Use plant blocks to form a combined cross shape, with each side made up of 7 blocks.",
            "Shape a 7 block per side cross using plant blocks and make sure the shape is combined.",
            "I want you to form a combined cross structure using plant blocks, each side should consists of 7 blocks.",
            "Fabricate a combined cross structure using seven blocks of plant on each side."
        ],
        "arguments": create_target_state(block_type=BlockType.PLANT, size=7,  cross_type=CrossType.COMBINED),
        "str_check_lambda": "lambda gd, ix: is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_STONE_3_DIAGONAL": {
        "instruction": "Form a cross made of stone with a side size of 3 in a diagonal shape.",
        "scenario_checker": Scenarios.BUILD_STAR,
        "instruction_paraphrases": [
            "Build a stone cross that has 3 units on each side and is arranged diagonally.",
            "Create a cross using stones, ensure it has 3 blocks on each side and is designed diagonally.",
            "I'd like you to construct a cross, diagonally. The material should be stone and the side size should measure three.",
            "Construct a cross with a side length of three using stones and align it in a diagonal direction.",
            "Arrange a cross made from rocks with each side measuring three units, make sure it's set diagonally."
        ],
        "arguments": create_target_state(block_type=BlockType.STONE, size=3,  cross_type=CrossType.DIAGONAL),
        "str_check_lambda": "lambda gd, ix: is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_FURNACE_5": {
        "instruction": "Arrange the blocks in a cross form of furnaces with a size of 5 blocks in diagonal way in the game",
        "scenario_checker": Scenarios.BUILD_STAR,
        "instruction_paraphrases": [
            "I need you to form a cross using furnace blocks, the cross should be diagonal and be made of 5 blocks.",
            "Shape a diagonal cross using 5 blocks, and all blocks should be furnace blocks.",
            "Make a cross within the game using furnaces, ensuring its a diagonal shape and include a total of five units",
            "In the game, create a diagonal cross composed entirely of furnaces, it should have five blocks",
            "Can you arrange furnace blocks in the form of a cross of 5 units in a diagonal form in the game?"
        ],
        "arguments": create_target_state(block_type=BlockType.FURNACE, size=5,  cross_type=CrossType.DIAGONAL),
        "str_check_lambda": "lambda gd, ix: is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_FURNACE_3": {
        "instruction": "Form a cross with a furnace having a side size of three and in a direct shape.",
        "scenario_checker": Scenarios.BUILD_STAR,
        "instruction_paraphrases": [
            "Create a three-sided direct shape cross using the furnace.",
            "With the help of a furnace, could you make a direct cross that has three units on each side?",
            "Construct a cross that is directly shaped and has a side length of three units, utilizing the kiln.",
            "A direct cross of a three on each side needs to be formed, can you create this using the oven?",
            "Centrally arrange the smelter to create a three-unit direct shaped cross."
        ],
        "arguments": create_target_state(block_type=BlockType.FURNACE, size=3,  cross_type=CrossType.STRAIGHT),
        "str_check_lambda": "lambda gd, ix: is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_PLANT_7_DIRECT": {
        "instruction": "Make a cross out of plants with a side length of 7",
        "scenario_checker": Scenarios.BUILD_STAR,
        "instruction_paraphrases": [
            "Form a plant cross of seven units on every side.",
            "Construct a cross using plants with each arm measuring seven units long.",
            "Could you create a flora cross with a total of seven units for every side?",
            "Please generate a vegetation cross with each branch having seven units in length.",
            "I would like to see you create a diagonal cross from greenery with the length of each segment equal to seven units."
        ],
        "arguments": create_target_state(block_type=BlockType.PLANT, size=7,  cross_type=CrossType.STRAIGHT),
        "str_check_lambda": "lambda gd, ix: is_cross_formed(gd, ix)"
    }
}

medium = {
    "INSTRUCTION_PLANT3": {
        "instruction": "Can you verify if there is a cross made up of plants with a side length of 3 and forms a diagonal shape?",
        "scenario_checker": Scenarios.BUILD_STAR,
        "instruction_paraphrases": [
            "Confirm if you can find a plant cross of side length 3 arranged diagonally",
            "Do see a diagonal cross structure made up of flora, each side having 3 units?",
            "Look around to see if there is a cross pattern with plant blocks of 3 units along each side, placed diagonally",
            "Is there a diagonal cross made from botanical units, each of its arms measuring 3 lengths?",
            "Check to see if a diagonal cross-form of vegetation exists, with each side spanning 3 units in length"
        ],
        "arguments": create_target_state(BlockType.PLANT, size=3,  cross_type=CrossType.STRAIGHT),
        "str_check_lambda": "is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_ENCHANTMENT_TABLE_ICE5": {
        "instruction": "Arrange a cross of Enchantment Table Ice blocks with a side size of 5 in a combined shape.",
        "scenario_checker": Scenarios.BUILD_STAR,
        "instruction_paraphrases": [
            "In a combined layout, fashion a cross with Enchantment Table Ice blocks, each side being five units long",
            "Create a combined cross from Enchantment Table Ice cubes with all sides measuring five units",
            "Shape Enchantment Table Ice blocks into a combined shape of a cross where every side has a length of five",
            "Five units on each side, form a combined type cross with blocks of Enchantment Table Ice",
            "Using five units for each arm, construct a combined cross with blocks of Frost Magic Table"
        ],
        "arguments": "create_target_state(BlockType.ENCHANTMENT_TABLE_ICE, size=5,  crossType.STRAIGHT)",
        "str_check_lambda": "is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_STONE5_1": {
        "instruction": "Form a cross with the stone, where the side is of size five and the total shape itself stands combined.",
        "scenario_checker": Scenarios.BUILD_STAR,
        "instruction_paraphrases": [
            "Can you devise a cross constructed of rock, where each side has five units and the complete sketch amalgamates?",
            "Could you fashion a cross with a pebble ensuring that every side has a total of five units and the general design is comprehensive?",
            "Can you shape a cross by employing the cobblestones, having each side measuring five units and the complete pattern is unified?",
            "Please structure a cross composed of boulders, maintaining that every arm expands to five units and the holistic structure is conjoined?",
            "Would you be able to generate a cross using the gravel, where each arm is five units in length and the whole configuration is integrated?"
        ],
        "arguments": "create_target_state(BlockType.STONE, size=5,  crossType.STRAIGHT)",
        "str_check_lambda": "is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_CRAFTING_TABLE3": {
        "instruction": "Form a cross of crafting tables, each side should be 3 blocks long and it should be directly shaped",
        "scenario_checker": Scenarios.BUILD_STAR,
        "instruction_paraphrases": [
            "Make a straight cross using crafting tables, each arm should be 3 blocks long",
            "Create a cross shape directly with crafting benches, each one containing 3 blocks",
            "Build a directly shaped cross with a length of 3 units per side using workbenches",
            "Craft a direct cross with worktables, ensuring each side measures 3 blocks",
            "Assemble a direct-design cross using 3-block long sides made of workstations"
        ],
        "arguments": "create_target_state(BlockType.CRAFTING_TABLE, size=3,  crossType.STRAIGHT)",
        "str_check_lambda": "is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_FURNACE3_1": {
        "instruction": "Make a cross of furnace blocks, each side should contain 3 blocks.",
        "scenario_checker": Scenarios.BUILD_STAR,
        "instruction_paraphrases": [
            "Create a cross-structure using forge-like blocks, with each stem having a length of 3 units.",
            "Construct a cross form, where each side has 3 units, using blocks of furnace.",
            "Using heating system blocks in the game, arrange them in the shape of a cross with every arm having an extent of 3 units.",
            "In a four-pronged configuration shaped like a cross, place three stove blocks along each prong.",
            "Form an evenly balanced cross, where each branch is represented by three blocks of the kiln."
        ],
        "arguments": "create_target_state(BlockType.FURNACE, size=3,  crossType.STRAIGHT)",
        "str_check_lambda": "is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_STONE5_2": {
        "instruction": "Form a cross shape with rocks, each side should have a length of 5",
        "scenario_checker": Scenarios.BUILD_STAR,
        "instruction_paraphrases": [
            "Create a diagonal cross using stones, each arm of the cross should have 5 units length",
            "Shape an arrangement in the form of a cross made up of boulders, with every side being 5 units long",
            "Create a cross design with pebbles ensuring that each side has 5 units",
            "Construct a cross figure using stones, every side of the cross should have a length of 5 units",
            "Using rocks, fashion out a cross shape where every side measures up to 5 units in length"
        ],
        "arguments": "create_target_state(BlockType.STONE, size=5,  crossType.STRAIGHT)",
        "str_check_lambda": "is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_ENCHANTMENT_TABLE_FIRE5": {
        "instruction": "Create a cross composed of Fire Enchantment Tables with a side length of 5.",
        "scenario_checker": Scenarios.BUILD_STAR,
        "instruction_paraphrases": [
            "Build a cross shape using Fire Enchant Spruce Boards, and ensure each arm of the cross is 5 units in length.",
            "Start crafting a Fire Magic Desk in a cross pattern where each branch of the cross measures 5 units long.",
            "Using The Fiery Sorcery Platform, construct a cross where each segment is a total of 5 units in distance.",
            "Craft a combined shape resembling a cross with Ignite Spell Tables, making sure each line extends up to 5 units.",
            "I want you to shape a Fire Wizard's Bench into a cross structure with each side measuring a total of five units."
        ],
        "arguments": "create_target_state(BlockType.ENCHANTMENT_TABLE_FIRE, size=5,  crossType.STRAIGHT)",
        "str_check_lambda": "is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_ENCHANTMENT_TABLE_FIRE7": {
        "instruction": "Create a cross structure on the map using the Enchantment of Fire table, with each side containing 7 units.",
        "scenario_checker": Scenarios.BUILD_STAR,
        "instruction_paraphrases": [
            "Make a cross pattern using fire enchantment tables, with each arm consisting of seven units.",
            "I need you to form a cross with Fire Enchanting Tables, each arm should contain a total of seven pieces.",
            "Can you arrange seven Fire Enchantment Tables on each arm in a cross shape on the map?",
            "Please assemble a seven-unit long arm cross shape using the tables of Fire Enchantment on our game map.",
            "On the game map, erect a cross, each arm of which is crafted from seven Fire Enchantment Tables."
        ],
        "arguments": "create_target_state(BlockType.ENCHANTMENT_TABLE_FIRE, size=7,  crossType.STRAIGHT)",
        "str_check_lambda": "is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_TORCH7": {
        "instruction": "Form a combined cross shape with a torch, sides should be 7 units long.",
        "scenario_checker": Scenarios.BUILD_STAR,
        "instruction_paraphrases": [
            "Make a torch pattern in the shape of a 7 unit per side combined cross.",
            "Create an aggregate cross pattern using torch, each side should be seven units.",
            "Utilize a command to position the torch so that it forms an integrated cross pattern, each arm should be 7 units long.",
            "Shape a torch in such a way that it ends up looking like a mixed cross with each side stretching up to seven units.",
            "Configure a total of seven units on each side of the torch to produce a compound cross shape."
        ],
        "arguments": "create_target_state(BlockType.TORCH, size=7,  crossType.STRAIGHT)",
        "str_check_lambda": "is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_ENCHANTMENT_TABLE_ICE7": {
        "instruction": "Create a cross design using enchanted ice tables. The cross should have 7 blocks on each limb and it should be a combined shape within a radius of 5 blocks from my position",
        "scenario_checker": Scenarios.BUILD_STAR,
        "instruction_paraphrases": [
            "Fashion a combined shape cross with enchantment ice tables, each arm should have seven blocks and it should be within a five block distance of me",
            "I want you to arrange seven enchanted frost workbenches in a cross formation around me, five blocks as the furthest distance within the radius",
            "In a 5 block radius around me, arrange the frost enchantment desks so it forms a cross shape with each line of the cross containing seven blocks",
            "Within my immediate vicinity of five blocks, can you create a combined layout in a cross configuration using seven of the enchanted ice tabletops",
            "Please create an intricate pattern in the shape of a cross around me using the frost charm stations, with each limb of the cross spanning seven blocks and every station positioned within a radius of five blocks from where I am"
        ],
        "arguments": "create_target_state(BlockType.ENCHANTMENT_TABLE_ICE, size=7,  crossType.STRAIGHT)",
        "str_check_lambda": "is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_FIRE_GRASS3": {
        "instruction": "Form a cross with fire grass, that is 3 blocks in size.",
        "scenario_checker": Scenarios.BUILD_STAR,
        "instruction_paraphrases": [
            "Make a cross shape with fire grass of a total of 3 units.",
            "Create a flame grass cross 3 blocks big.",
            "I need a cross made out of fire grass that spans 3 blocks.",
            "Craft a 3 block large cross using fire grass."
        ],
        "arguments": "create_target_state(BlockType.FIRE_GRASS, size=3,  crossType.STRAIGHT)",
        "str_check_lambda": "is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_PLANT3_2": {
        "instruction": "Construct a cross out of vegetation with a side size of 3 in a diagonal pattern.",
        "scenario_checker": Scenarios.BUILD_STAR,
        "instruction_paraphrases": [
            "Create a cross shape from plants, having each of its side measure 3 units, arranged diagonally.",
            "Assemble a diagonal plant cross whose sides are 3 blocks long.",
            "I want you to form a diagonal pattern using plants to create a cross with sides that are 3 units long.",
            "Utilizing plants, design a cross shape where each side consists of an arrangement of 3 units in a diagonal pattern.",
            "Form a diagonal cross using a collection of plants which should have a side size of three units."
        ],
        "arguments": "create_target_state(BlockType.PLANT, size=3,  crossType.STRAIGHT)",
        "str_check_lambda": "is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_STONE7": {
        "instruction": "Create a direct cross made of stone blocks, each side should have a length of seven.",
        "scenario_checker": Scenarios.BUILD_STAR,
        "instruction_paraphrases": [
            "Build a cross of rock, make sure each side has a total of seven units.",
            "Can you form a seven units long cross with boulders directly?",
            "I need you to make a direct cross with pebbles, each side should have seven blocks.",
            "Using cobblestones, construct a cross shape directly with each side consisting of seven units.",
            "Form a direct cross design using stone blocks, make certain that each side measures seven units in length."
        ],
        "arguments": "create_target_state(BlockType.STONE, size=7,  crossType.STRAIGHT)",
        "str_check_lambda": "is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_FURNACE5": {
        "instruction": "Make a combined cross pattern of 5x5 blocks using a furnace",
        "scenario_checker": Scenarios.BUILD_STAR,
        "instruction_paraphrases": [
            "Using furnaces, assemble a merged cross configuration that spans five blocks on each side.",
            "Create a 5x5 cruciform pattern using the furnace object.",
            "I want you to build a combined cross shape with a size of five blocks in each direction using only furnace blocks.",
            "Your task is to arrange furnaces in a combined cross shape, with each arm of the cross spanning five blocks.",
            "Use blocks of the furnace type to fabricate a cross shape design where each prong measures five blocks in length."
        ],
        "arguments": "create_target_state(BlockType.FURNACE, size=5,  crossType.STRAIGHT)",
        "str_check_lambda": "is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_PLANT7": {
        "instruction": "Create a cross of plant blocks with a side size of 7 in the given region.",
        "scenario_checker": Scenarios.BUILD_STAR,
        "instruction_paraphrases": [
            "Arrange some plants in a cross pattern, making sure each side is 7 units long.",
            "I want you to construct a cross using vegetation, where each arm extends 7 units.",
            "Make a cross with the plants, each extending seven units from center.",
            "Can you build a cross comprised of plants? Each side should span a total of seven units.",
            "Could you please construct a combined cross with flora extending seven units in every direction?"
        ],
        "arguments": "create_target_state(BlockType.PLANT, size=7,  crossType.STRAIGHT)",
        "str_check_lambda": "is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_FURNACE3": {
        "instruction": "Create a cross shape using the furnace, the arms of the cross should have a length of 3 blocks.",
        "scenario_checker": Scenarios.BUILD_STAR,
        "instruction_paraphrases": [
            "Construct a cross figure using furnace blocks; each arm should be 3 blocks long.",
            "I need you to make a cross using furnaces, each arm extending three blocks.",
            "Please form a cross using furnace blocks. Each arm should extend by three blocks.",
            "With furnace blocks, I'd like you to create a cross. Ensure the length of each arm is 3 blocks.",
            "Could you arrange the furnace blocks into a cross pattern? Each of the arms should be three blocks in length."
        ],
        "arguments": "create_target_state(BlockType.FURNACE, size=3,  crossType.STRAIGHT)",
        "str_check_lambda": "is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_STONE3": {
        "instruction": "Create a cross made of stone blocks with a side size of three and in direct shape.",
        "scenario_checker": Scenarios.BUILD_STAR,
        "instruction_paraphrases": [
            "Construct a cross using stones, each side should be three blocks long in a straight line.",
            "I need you to form a cross using rocks, with each of the four sides measuring three units in length, placed upright.",
            "Can you compose a vertically aligned cross using cobblestone with three blocks in length on each arm?",
            "Using boulders, can you arrange them on the ground in the shape of an upright cross with each limb extending to three blocks?",
            "With pebbles at your disposal, could you arrange them in a shape that mimics a standing cross, with a total of three units forming each arm?"
        ],
        "arguments": "create_target_state(BlockType.STONE, size=3,  crossType.STRAIGHT)",
        "str_check_lambda": "is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_PLANT3_3": {
        "instruction": "Form a cross with the plant having each side of size 3 and it should be in a direct shape.",
        "scenario_checker": Scenarios.BUILD_STAR,
        "instruction_paraphrases": [
            "Construct a cross using plant-based materials. Make sure each side of the cross is 3 units and it's directly positioned.",
            "Using the plant, create a cross with each side measuring 3 units and ensure it maintains a direct formation.",
            "Erect a direct shape of a cross using plant materials. Ensure the cross has a side length of 3 units.",
            "Go ahead and construct a cross from plant elements. Directly position the cross and make each side measure 3 units.",
            "Craft a direct shape of the cross, utilizing plant materials as the building block. Each side of this formation should be three units in length."
        ],
        "arguments": "create_target_state(BlockType.PLANT, size=3,  crossType.STRAIGHT)",
        "str_check_lambda": "check_lambda(gd, ix)"
    },
    "INSTRUCTION_ENCHANTMENT_TABLE_FIRE3": {
        "instruction": "Create a direct cross of enchantment fire tables with a side length of 3.",
        "scenario_checker": Scenarios.BUILD_STAR,
        "instruction_paraphrases": [
            "Form a cross with enchantment fire workbenches, each arm should have a length of 3.",
            "Build a symmetrical cross with a side size of 3 using enchantment fire tables.",
            "I want you to lay out enchantment fire desks in a cross pattern where each arm of the cross is 3 units long.",
            "Form a straight cross made of enchantment fire tables, with each branch of the cross being 3 units in length.",
            "Construct a symmetrically direct cross using enchantment fire workbenches with each arm of the cross measuring a length of 3 units."
        ],
        "arguments": "create_target_state(BlockType.ENCHANTMENT_TABLE_FIRE, size=3,  crossType.STRAIGHT)",
        "str_check_lambda": "is_cross_formed(gd, ix)"
    }
}