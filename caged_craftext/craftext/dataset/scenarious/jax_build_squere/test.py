from craftext.environment.scenarious.checkers.target_state import TargetState, BuildSquareState
from craftext.environment.craftext_constants import Scenarios, BlockType

def create_target_state(block_type:int, size:int):
    target_achievements = BuildSquareState(block_type=block_type, size=size, radius=10)
    return TargetState(building_square=target_achievements)

easy_test_parafrased = {
"INSTRUCTION_STONE_4_SQUARE": {
        "instruction": "Create a square out of stone, each side of size 4",
        "scenario_checker": Scenarios.BUILD_SQUARE,
        "instruction_paraphrases": [
            "Form a cube from rock with each boundary being of magnitude 4",
            "Taking stone as the material, establish a geometrical figure that is square in shape in such a way that each of its sides measures 4",
            "Fabricate, using the medium of rock, a geometric entity that remains quadrate while all its borders maintain a dimension of 4",
            "Carve ye a square from the bedrock, every flank of which hath the breadth of four, mirroring the steadfastness of the earth",
            "Shape from the rough, unyielding stone, a square, its sides measuring four each, echoing the world's generous symmetry and laws of form."
        ],
        "arguments": create_target_state(BlockType.STONE, 4),
        "str_check_lambda": "is_square_formed(game_data, ix, BlockType.STONE, 4)"
    },
    "INSTRUCTION_CRAFTING_TABLE_2_SQUARE": {
        "instruction": "Form a square using crafting tables of side length 2.",
        "scenario_checker": Scenarios.BUILD_SQUARE,
        "instruction_paraphrases": [
            "Construct a quadrilateral with crafting benches of side span 2",
            "Using crafting tables that have a side length of 2, form a geometric square",
            "Create a four-sided shape using crafting platforms that have a side measurement of 2",
            "Using crafting tables of old, build thee a square with sides of even measure",
            "Construct a precise cube-space with crafting surfaces, each one no more, no less, than two units in its side length"
        ],
        "arguments": create_target_state(BlockType.CRAFTING_TABLE, 2),
        "str_check_lambda": "is_square_formed(game_data, ix, BlockType.CRAFTING_TABLE, 2)"
    },
    "INSTRUCTION_CRAFTING_TABLE_4_SQUARE": {
        "instruction": "Construct a square using crafting tables, with each side of the square being 4 blocks long",
        "scenario_checker": Scenarios.BUILD_SQUARE,
        "instruction_paraphrases": [
            "Build a square using assembly platforms, each side measuring 4 units in length",
            "With each side being 4 blocks in length, you should use crafting tables to put together a square",
            "Employ assembly platforms to establish a square, a geometric entity wherein each side measures exactly 4 units",
            "In the hands of a master craftsman, lay the foundations of a square, forged from the unyielding surfaces of crafting tables, each side a testament to symmetry, embodied in four steadfast blocks",
            "As if creating a world within a world, use your crafting tables to shape a square, each side a steady four blocks, mirroring the constancy of the four cardinal directions amidst the infinite realm of possibilities"
        ],
        "arguments": create_target_state(BlockType.CRAFTING_TABLE, 4),
        "str_check_lambda": "is_square_formed(game_data, ix, BlockType.CRAFTING_TABLE, 4)"
    },
    "INSTRUCTION_PLANT_4_SQUARE": {
        "instruction": "Check for a square of plants with a side size of 4.",
        "scenario_checker": Scenarios.BUILD_SQUARE,
        "instruction_paraphrases": [
            "Inspect a plot of vegetation with a boundary length of 4.",
            "Verify if there exists a quadrate formation of flora whose dimensions on one side equals 4.",
            "Scrutinize if there is a parcel of foliage whose edge measure matches 4, shaping a square pattern.",
            "In the kingdom of green foliage, cast thine eyes upon a checkered battlefield, four strides' length each way forming its boundary.",
            "In the whispers of the green world, seek the plant-formed square whose sides, four measures long, sing a quiet geometry song."
        ],
        "arguments": create_target_state(BlockType.PLANT, 4),
        "str_check_lambda": "is_square_formed(game_data, ix, BlockType.PLANT, 4)"
    },
    "INSTRUCTION_STONE_3_SQUARE": {
        "instruction": "Form a square with sides of 3 blocks using stones",
        "scenario_checker": Scenarios.BUILD_SQUARE,
        "instruction_paraphrases": [
            "Develop a quadrilateral utilizing stones whose sides measure 3 block units",
            "Using stones to structure, there must be a creation of a square which has sides of a length equivalent to 3 blocks",
            "Construct a polygon with stones, ensuring that each side stretches to a span of 3 block units",
            "Assemble thou a four-sided figure akin to a chessboard, using the sturdy fabric of earth, carved and chiseled into stones, each side bearing the length of three uniform blocks, revealing the ancient symmetry of the square",
            "With stones, forge a space that defies the shapelessness of nature, crafting a square honorably holding a steadiness, with each side grounding the concept of balance and measuring the length of three worthy blocks."
        ],
        "arguments": create_target_state(BlockType.STONE, 3),
        "str_check_lambda": "is_square_formed(game_data, ix, BlockType.STONE, 3)"
    }
}


# TODO: check complexity
easy_test_other_paramets = {
    "INSTRUCTION_STONE_2_SQUARE": {
        "instruction": "Verify if there is a square formed with rock of side length 2 in the game.",
        "scenario_checker": Scenarios.BUILD_SQUARE,
        "instruction_paraphrases": [
            "Check if a 2x2 square made of rock exists in the game.",
            "Does a block of stone formed as a square with side length of two exist in the game?",
            "In the game, is there a square shaped block of stone with a size of two present?",
            "I need confirmation if there's a stone square with a side length of 2 within the game.",
            "Can you find a square formation of rock with each side of two units in the game environment?"
        ],
        "arguments": create_target_state(BlockType.STONE, 2),
        "str_check_lambda": "is_square_formed(game_data, ix, BlockType.STONE, 2)"
    },
    "INSTRUCTION_FURNACE_4_SQUARE": {
        "instruction": "Check if there is a square of furnaces that is 4 blocks wide.",
        "scenario_checker": Scenarios.BUILD_SQUARE,
        "instruction_paraphrases": [
            "Can you see if there exists a grid with a side length of 4 made entirely of furnace blocks?",
            "Could you verify if a furnace square having a side of 4 blocks exists?",
            "You need to find if there's a oven square of 4x4 present.",
            "Locate whether there is a square composed of stoves with a dimension of 4x4 blocks.",
            "Hinting towards the presence of a square arrangement of 4x4 firebox units in the game area."
        ],
        "arguments": create_target_state(BlockType.FURNACE, 4),
        "str_check_lambda": "is_square_formed(game_data, ix, BlockType.FURNACE, 4)"
    },
    "INSTRUCTION_PLANT_3_SQUARE": {
        "instruction": "Can you check if there is a square of plants with a side length of 3?",
        "scenario_checker": Scenarios.BUILD_SQUARE,
        "instruction_paraphrases": [
            "I need you to verify the presence of a plant patch shaped in a 3x3 square.",
            "Could you see if there is a square form made of plants measuring 3 by 3 in size?",
            "You need to ascertain the existence of a square formation of botany, which is 3 units in length and width.",
            "I require you to confirm the existence of a square-shaped arrangement of flora with a size measuring 3 on each side.",
            "Is it possible to authenticate the presence of a quadrilateral configuration comprised of vegetation with a dimension of three by three?"
        ],
        "arguments": create_target_state(BlockType.PLANT, 3),
        "str_check_lambda": "is_square_formed(game_data, ix, BlockType.PLANT, 3)"
    },
    "INSTRUCTION_FURNACE_3_5_SQUARE": {
        "instruction": "Check for a square made of furnace blocks with a side length of 3.",
        "scenario_checker": Scenarios.BUILD_SQUARE,
        "instruction_paraphrases": [
            "Verify the presence of a 3x3 square constructed from furnace blocks.",
            "Scan the area for a three-sided square structure built exclusively from furnace blocks.",
            "Examine the vicinity for a furnace block arrangement in the shape of a square with a size of 3.",
            "Find a configuration designed as a square, that is formed using furnace blocks and measuring a length of three blocks on each side.",
            "Confirm if there's an arrangement of furnace blocks in a perfect square that measures three units on each of its side."
        ],
        "arguments": create_target_state(BlockType.FURNACE, 3, 5),
        "str_check_lambda": "is_square_formed(game_data, ix, BlockType.FURNACE, 3, 5)"
    },
    "INSTRUCTION_PLANT_2_SQUARE": {
        "instruction": "Make a square of Plants with side size 2.",
        "scenario_checker": Scenarios.BUILD_SQUARE,
        "instruction_paraphrases": [
            "Arrange the Flora into a square formation where each side is two blocks long.",
            "Organize the Greenery into a 2 by 2 square shape.",
            "Form a square using the Vegetation with the side length equal to two.",
            "Create a square block arrangement of Botanical items making sure each side of the square is two units long.",
            "Construct a square with a 2-block length on each side using the Herbal elements."
        ],
        "arguments": create_target_state(BlockType.PLANT, 2),
        "str_check_lambda": "is_square_formed(game_data, ix, BlockType.PLANT, 2)"
    }
}



medium_test_parafrased = {}


medium_test_other_paramets = {}

