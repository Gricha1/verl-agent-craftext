from craftext.environment.scenarious.checkers.target_state import TargetState, BuildSquareState
from craftext.environment.craftext_constants import Scenarios, BlockType

def create_target_state(block_type:int, size:int):
    target_achievements = BuildSquareState(block_type=block_type, size=size, radius=10)
    return TargetState(building_square=target_achievements)



easy = {
    "INSTRUCTION_STONE_4": {
        "instruction": "Create a square out of stone, each side of size 4",
        "scenario_checker": Scenarios.BUILD_SQUARE,
        "instruction_paraphrases": [
            "Please form a square using rocks, ensuring each side is of length 4",
            "Using stone blocks, assemble a square structure with side length 4",
            "Could you construct a square with boulders? Each side should be 4 units long",
            "I'd like you to put together a square with a side of 4 units using pebbles",
            "It'd be nice if you could fashion a 4-unit side square using the stones"
        ],
        "arguments": create_target_state(BlockType.STONE, size=4 ),
        "str_check_lambda": "is_square_formed(gd, ix)"
    },
    "INSTRUCTION_CRAFTING_TABLE_2": {
        "instruction": "Form a square using crafting tables of side length 2.",
        "scenario_checker": Scenarios.BUILD_SQUARE,
        "instruction_paraphrases": [
            "Make a square with side length 2 using crafting tables.",
            "You need to arrange crafting tables in a square shape with each side being 2 blocks long.",
            "Build a square from crafting tables, each side should contain 2 blocks.",
            "Crafting tables should be placed in a way to form a square with each side made of 2 blocks.",
            "Set crafting tables in a position to produce a square shape having 2 as the length of its each side."
        ],
        "arguments": create_target_state(BlockType.CRAFTING_TABLE, size=2 ),
        "str_check_lambda": "is_square_formed(gd, ix)))"
    },
    "INSTRUCTION_CRAFTING_TABLE_4": {
        "instruction": "Construct a square using crafting tables, with each side of the square being 4 blocks long",
        "scenario_checker": Scenarios.BUILD_SQUARE,
        "instruction_paraphrases": [
            "Create a square with crafting benches in each corner, and each side sizing up to 4 blocks",
            "Please make a geometrical square using workstations, and let the length of each side be 4 blocks",
            "I need you to set up a square. Each side should have a length of 4 blocks and it should be formed completely out of crafting tables",
            "With the utilization of crafting desks, arrange them in a way to form a square structure where the length of each side measures 4 blocks",
            "Using crafting tables as the building material, construct a square pattern, ensuring that each side measures exactly 4 blocks long"
        ],
        "arguments": create_target_state(BlockType.CRAFTING_TABLE, size=4 ),
        "str_check_lambda": "is_square_formed(gd, ix)"
    },
    "INSTRUCTION_PLANT_4": {
        "instruction": "Check for a square of plants with a side size of 4.",
        "scenario_checker": Scenarios.BUILD_SQUARE,
        "instruction_paraphrases": [
            "Make sure there's a 4x4 square of vegetation.",
            "Look out for a square that consists of plants and measures 4 blocks on each side.",
            "Identify a region in the shape of a square with a side length of 4, entirely composed of greenery.",
            "Probe for the presence of a quadrilateral of botanical elements with each side spanning 4 blocks.",
            "I need you to verify the existence of a 2D square-shaped arrangement of flora measuring 4 units on all sides."
        ],
        "arguments": create_target_state(BlockType.PLANT, size=4 ),
        "str_check_lambda": "is_square_formed(gd, ix)))"
    },
    "INSTRUCTION_STONE_3": {
        "instruction": "Form a square with sides of 3 blocks using stones",
        "scenario_checker": Scenarios.BUILD_SQUARE,
        "instruction_paraphrases": [
            "Create a square block using 3 stones on each side",
            "I want you to assemble a square from rocks, size=3 units long on each side",
            "Can you build a 3 by 3 block square using some cobblestones?",
            "Utilizing stones, shape out a square block with each side being 3 blocks long",
            "With the use of rocks, I'd like you to construct a square formation where each side is composed of 3 blocks",
            "Configure a square with a 3-unit side length using stones.",
            "Create a square using stone blocks, each side should contain 3 blocks.",
            "With stones, develop a square of three blocks on a side.",
            "Utilize stones to shape a square that each side contains 3 blocks.",
            "Make a square configuration using rocks, having three blocks on all sides"
        ],
        "arguments": create_target_state(BlockType.STONE, size=3 ),
        "str_check_lambda": "is_square_formed(gd, ix)"
    }
}

medium = {
    "INSTRUCTION_CRAFTING_TABLE_3": {
        "instruction": "Form a square of crafting tables with each side having a length of 3",
        "scenario_checker": Scenarios.BUILD_SQUARE,
        "instruction_paraphrases": [
            "Create a square shape by using work benches where the number of work benches on one side is 3",
            "Layout 3 blocks of builder’s table in a square shape.",
            "Make sure to arrange the building blocks in a square shape with each side having 3 of them",
            "Kindly arrange three Crafting platforms on each side to form a square configuration.",
            "I want you to position the construction desks in such a way that they form a square structure with each side containing three desks"
        ],
        "arguments": create_target_state(BlockType.CRAFTING_TABLE, size=3 ),
        "str_check_lambda": "is_square_formed(gd, ix)))"
    },
    "INSTRUCTION_FURNACE_6": {
        "instruction": "Verify if there is a square formed of furnace blocks with a side size of 6.",
        "scenario_checker": Scenarios.BUILD_SQUARE,
        "instruction_paraphrases": [
            "Can you confirm if a square with side-length of 6 is made up of furnace blocks?",
            "Tell me, do we have a furnace blocks square with each side 6 blocks long?",
            "Check for a 6 blocks sided square made entirely out of furnace blocks.",
            "Do assess if we have a square, each side 6 blocks long, made completely from furnace blocks.",
            "Inspect and confirm whether there is a square structure constituted of furnace blocks, six blocks long per side."
        ],
        "arguments": create_target_state(BlockType.FURNACE, size=6 ),
        "str_check_lambda": "is_square_formed(gd, ix)))"
    },
    "INSTRUCTION_FURNACE_5": {
        "instruction": "Check if there is a square made of furnace blocks with side length of 5.",
        "scenario_checker": Scenarios.BUILD_SQUARE,
        "instruction_paraphrases": [
            "Investigate for a square formation of 5x5 using heater blocks.",
            "Look for a quadrant constructed with stove blocks, each side measuring 5 units.",
            "Can you find a geometric square with a side size of 5, constructed from kiln blocks?",
            "Make sure a form of square having dimensions 5 by 5, built using forge blocks is in position?",
            "Inspect for any presence of a geometric configuration resembling a square with side length of 5, created using smelter blocks."
        ],
        "arguments": create_target_state(BlockType.FURNACE, size=5 ),
        "str_check_lambda": "is_square_formed(gd, ix)))"
    },
    "INSTRUCTION_CRAFTING_TABLE_7": {
        "instruction": "Check for a distinct square made out of crafting tables with each side having a length of 7 blocks.",
        "scenario_checker": Scenarios.BUILD_SQUARE,
        "instruction_paraphrases": [
            "Look for a 7x7 crafting block square",
            "Search for a square formation composed of workbenches, with each side consisting of 7 blocks",
            "Confirm if there is a square of crafting tables existing, where each side is equivalent to 7 blocks",
            "Verify the presence of a 7-blocks-wide square of crafting station",
            "Ensure the existence of a perfect square shape made up of 7 blocks per side of crafting tables"
        ],
        "arguments": create_target_state(BlockType.CRAFTING_TABLE, size=7 ),
        "str_check_lambda": "is_square_formed(gd, ix)))"
    },
    "INSTRUCTION_ENCHANTMENT_TABLE_ICE_3": {
        "instruction": "Check for a square formation of Enchantment Table Ice with a side of size 3.",
        "scenario_checker": Scenarios.BUILD_SQUARE,
        "instruction_paraphrases": [
            "Can you see if there's a square configuration of Ice Enchantment Table each side measuring 3 blocks?",
            "Verify if there's a 3x3 square arrangement of the Ice Magic Desk.",
            "Determine if you have a 9-block square formation of the Frosty Wizard's Stand.",
            "Could you look for a square pattern of Ice Sorcerer's Bench? Each side should have 3 blocks.",
            "Confirm if there exists a square structure of three units on each side of the Cryo Spell Table."
        ],
        "arguments": create_target_state(BlockType.ENCHANTMENT_TABLE_ICE, size=3 ),
        "str_check_lambda": "is_square_formed(gd, ix)))"
    },
    "INSTRUCTION_ENCHANTMENT_TABLE_ICE_7": {
        "instruction": "Create an enchantment table of ice shaped into a square with each side size 7.",
        "scenario_checker": Scenarios.BUILD_SQUARE,
        "instruction_paraphrases": [
            "Form a square using an ice enchantment table, each side should be of length 7.",
            "Make a square with sides of 7 units using a table enchanted with ice.",
            "Utilize an enchanted ice table to fabricate a square having side length of seven.",
            "Shape an ice enchantment table into a square with a side length of 7.",
            "With the ice enchantment table, assemble a square where each side measures 7 units."
        ],
        "arguments": create_target_state(BlockType.ENCHANTMENT_TABLE_ICE, size=7 ),
        "str_check_lambda": "is_square_formed(gd, ix)))"
    },
    "INSTRUCTION_FURNACE_7": {
        "instruction": "Build a square using furnaces with each side being 7 blocks long.",
        "scenario_checker": Scenarios.BUILD_SQUARE,
        "instruction_paraphrases": [
            "Craft a square pattern with smelters that has 7 blocks as the dimension.",
            "Create a geometric square shape using heating devices where each side is 7 blocks long.",
            "Erect a quadrilateral with furnaces with each of its sides being made of 7 blocks.",
            "Form a furnace square that each edge has a length of 7 blocks.",
            "Construct a four-sided figure using 7 furnaces on each side."
        ],
        "arguments": create_target_state(BlockType.FURNACE, size=7 ),
        "str_check_lambda": "is_square_formed(gd, ix)))"
    }
}

