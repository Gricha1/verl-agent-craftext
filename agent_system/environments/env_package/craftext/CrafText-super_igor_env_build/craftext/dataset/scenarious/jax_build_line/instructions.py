from craftext.enviroment.craftext_constants import BlockType
from craftext.enviroment.scenarious.checkers.target_state import BuildLineState as AchievmentClass
from craftext.enviroment.scenarious.checkers.target_state import TargetState
from craftext.enviroment.craftext_constants import Scenarios
# one = {
#     'line_one_1': {
#         'instruction': "Make a line of 2 blocks using table.",
#         'instruction_paraphrases': [
#             "Construct a row of 2 pieces with the crafting station.",
#             "Place 2 units in a straight row using the workbench.",
#             "Use the crafting table to form a row of 2 items.",
#             "Arrange a sequence of 2 blocks with the crafting platform.",
#             create a straight formation of 2 blocks with the crafting table."
#         ],
#         'check_lambda': lambda game_data, ix: is_line_formed(game_data,ix, BlockType.CRAFTING_TABLE, 2, is_diagonal=False),
#         'str_check_lambda': "is_line_formed(game_data,ix, BlockType.CRAFTING_TABLE, 2, is_diagonal=False)"
#     },
    
#     'line_one_2': {
#         'instruction': "Make a line of 2 blocks using table.",
#         'instruction_paraphrases': [
#             "Construct a row of 2 pieces with the crafting station.",
#             "Place 2 units in a straight row using the workbench.",
#             "Use the crafting table to form a row of 2 items.",
#             "Arrange a sequence of 2 blocks with the crafting platform.",
#             create a straight formation of 2 blocks with the crafting table."
#         ],
#         'check_lambda': lambda game_data, ix: is_line_formed(game_data,ix, BlockType.CRAFTING_TABLE, 2, is_diagonal=False),
#         'str_check_lambda': "is_line_formed(game_data,ix, BlockType.CRAFTING_TABLE, 2, is_diagonal=False)"
#     },
# }

def create_target_state(block_type:int, size:int, is_diagonal:bool):
    target_achievements = AchievmentClass(block_type=block_type, size=size, is_diagonal=is_diagonal, radius=10)
    return TargetState(building_line=target_achievements)

easy = {
    "INSTRUCTION_CRAFTING_TABLE_2": {
        "instruction": "Make a line of 2 blocks using table.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Construct a row of 2 pieces with the crafting station.",
            "Place 2 units in a straight row using the workbench.",
            "Use the crafting table to form a row of 2 items.",
            "Arrange a sequence of 2 blocks with the crafting platform.",
            "Create a straight formation of 2 blocks with the crafting table."
        ],
        "arguments": create_target_state(BlockType.CRAFTING_TABLE, 2, is_diagonal=False),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.CRAFTING_TABLE, 2, check_diagonal=False)"
    },
    "INSTRUCTION_CRAFTING_TABLE_3_DIAGONAL": {
        "instruction": "Make a diagonal line of 3 blocks using table.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Construct a slanted row of 3 items with the crafting station.",
            "Place 3 blocks in a diagonal sequence using the workbench.",
            "Use the crafting platform to arrange a diagonal row of 3 pieces.",
            "Arrange a sloped line of 3 units using the crafting bench.",
            "Create a diagonal sequence of 3 items with the crafting table."
        ],
        "arguments": create_target_state(BlockType.CRAFTING_TABLE, 3, is_diagonal=True),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.CRAFTING_TABLE, 3, check_diagonal=True)"
    },
    "INSTRUCTION_STONE_2_DIAGONAL": {
        "instruction": "Make a diagonal line of 2 blocks using stone.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Construct a slanted row of 2 stones.",
            "Place 2 stone blocks in a diagonal sequence.",
            "Use stones to arrange a diagonal line of 2 pieces.",
            "Arrange a sloped row of 2 stone units.",
            "Create a diagonal sequence of 2 stone blocks."
        ],
        "arguments": create_target_state(BlockType.STONE, 2, is_diagonal=True),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.STONE, 2, check_diagonal=True)"
    },
    "INSTRUCTION_FURNACE_2": {
        "instruction": "Make a line of 2 blocks using furnace.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Construct a straight row of 2 units with the furnace.",
            "Place 2 blocks in a line using the heating station.",
            "Use the furnace to form a sequence of 2 items.",
            "Arrange a straight line of 2 pieces with the smelter.",
            "Create a row of 2 blocks using the furnace."
        ],
        "arguments": create_target_state(BlockType.FURNACE, 2, is_diagonal=False),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.FURNACE, 2, check_diagonal=False)"
    },
    "INSTRUCTION_STONE_4": {
        "instruction": "Make a horizontal line of 4 blocks using stone.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Construct a straight line of 4 stone blocks.",
            "Place 4 stone units in a horizontal row.",
            "Use stones to form a line of 4 blocks in a straight path.",
            "Arrange 4 stones in a straight sequence.",
            "Create a horizontal formation of 4 stone blocks."
        ],
        "arguments": create_target_state(BlockType.STONE, 4, is_diagonal=False),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.STONE, 4, check_diagonal=False)"
    },
    "INSTRUCTION_STONE_4": {
        "instruction": "Make a line of stone with four blocks.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Create a line using four rocks.",
            "Construct a line that consists of four stone blocks.",
            "Put together a straight line that contains four distinct stone blocks.",
            "With the use of four stone blocks, form a straight line.",
            "Assemble a linear pattern where each of four distinct spots is occupied by a stone block."
        ],
        "arguments": create_target_state(BlockType.STONE, 4, is_diagonal=False),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.STONE, 4, check_diagonal=False)"
    },
    "INSTRUCTION_FURNACE_3_DIAGONAL": {
        "instruction": "Check if there is a diagonal line of furnace with a size of three on the map.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Verify whether a line with a length of three tiles, composed of furnace units, is present diagonally on the game map.",
            "Inspect the playing field to see if there is a three-tile-long furnace line in a diagonal orientation.",
            "Can you see a line of three furnaces arranged diagonally on the map?",
            "Please confirm whether there's a contiguous, diagonal arrangement of three furnaces on our game map.",
            "Validate on the map if furnaces have been positioned in a line diagonally stretching to three blocks."
        ],
        "arguments": create_target_state(BlockType.FURNACE, 3, is_diagonal=True),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.FURNACE, 3, check_diagonal=True)"
    },
    "INSTRUCTION_PLANT_4": {
        "instruction": "Check if there's any line of Plants arranged in four.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Confirm if you can find a row of Plants that is four long.",
            "Can you verify whether there's a line of Vegetation that spans four squares?",
            "Please assess if there is a sequence of Plants that covers four blocks in a row.",
            "I need you to determine if there exists a linear arrangement of Flora that stretches across four squares.",
            "Can you authenticate the presence of a succession of Vegetation that extends to four units in a straight line?"
        ],
        "arguments": create_target_state(BlockType.PLANT, 4, is_diagonal=False),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.PLANT, 4, check_diagonal=False)"
    },
    "INSTRUCTION_PLANT_3": {
        "instruction": "Check for a line of plants with a length of three.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Verify if there's a row of three flora in the game.",
            "Inspect the game map for a straight line formation of vegetation that consists of three blocks.",
            "Ascertain if there's a series of three plant blocks in a row on the playing field.",
            "Investigate if a linear arrangement of three flora blocks is present in the current game state.",
            "Can you confirm the existence of a straight sequence of three vegetation units in line on the game?"
        ],
        "arguments": create_target_state(BlockType.PLANT, 3, is_diagonal=False),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.PLANT, 3, check_diagonal=False)"
    },
    "INSTRUCTION_STONE_3": {
        "instruction": "Form a line of stones of length 3.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Create a straight line using 3 pieces of rock.",
            "Set up three stones in a straight fashion.",
            "Irrespective of anywhere, place three rocks in a linear order.",
            "Keep three pieces of stone in a manner that they form a straight line.",
            "Regardless of the location, position three stones in such a way that they are perfectly aligned to create a line."
        ],
        "arguments": create_target_state(BlockType.STONE, 3, is_diagonal=False),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.STONE, 3, check_diagonal=False)"
    },
    "INSTRUCTION_PLANT_2": {
        "instruction": "Check if there is a line of Plant blocks of size 2.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Inspect for a sequence of Plant blocks with a length of 2.",
            "Keep an eye out for a line of 2 Plant type blocks.",
            "Can you verify if there's a row of 2 blocks of Plant type out there?",
            "Could you take a look and see if there's a continuous line of two blocks belonging to the Plant type?",
            "I want you to see if there is a straight, uninterrupted sequence of two blocks, with each block being categorized as Plant type."
        ],
        "arguments": create_target_state(BlockType.PLANT, 2, is_diagonal=False),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.PLANT, 2, check_diagonal=False)"
    },
    "INSTRUCTION_CRAFTING_TABLE_3_DIAGONAL": {
        "instruction": "Create a diagonal line of crafting tables, each one block apart for a size of three.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Build a slanted line using three crafting benches, with each one separated by a block.",
            "Establish a diagonal string of workbenches. Ensure there's one block space between them, and that you've only used three benches.",
            "With a size of three blocks, construct a diagonal line of crafting tables.",
            "Fashion a diagonal line using three crafting tables such that each is one block apart from the next.",
            "Create a diagonal row with three crafting stations, keeping a one block break in-between."
        ],
        "arguments": create_target_state(BlockType.CRAFTING_TABLE, 3, is_diagonal=True),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.CRAFTING_TABLE, 3, check_diagonal=True)"
    },
    "INSTRUCTION_CRAFTING_TABLE_2_DIAGONAL": {
        "instruction": "Check on the game map for a diagonal line of Crafting Table blocks that are of length 2.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Verify if there exists a two-block long Crafting Table arrangement in a diagonal line on the map.",
            "On the game map, confirm if a Crafting Table line in a diagonal direction and which consists of 2 blocks is present.",
            "Can you spot a Crafting Table line that stretches diagonally across two squares on our game map?",
            "Please confirm whether there is a diagonal configuration of two Crafting Table blocks on the gaming platform.",
            "On the gaming battleground, is there a positioning of two Crafting Tables that form a slant line?"
        ],
        "arguments": create_target_state(BlockType.CRAFTING_TABLE, 2, is_diagonal=True),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.CRAFTING_TABLE, 2, is_diagonal=True)"
    }
}


medium = {
    "INSTRUCTION_STONE_2": {
        "instruction": "create a diagonal line of stones of size two.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Can you make a diagonal line with two rocks?",
            "Arrange two stones in a diagonal manner",
            "Put two stones in a line, but make sure it's slanted",
            "I want to see two rocks positioned in a slanting line",
            "Could you please arrange a pair of stones diagonally to form a line?"
        ],
        "arguments":create_target_state(BlockType.STONE, 2, is_diagonal=True),
        "str_check_lambda": "str_check_lambda: is_line_formed(gd, ix)"
    },
    "INSTRUCTION_CRAFTING_TABLE_2": {
        "instruction": "Check if there is a line of Crafting Tables, at least two in size.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Verify if a sequence of at least two Crafting Tables has formed.",
            "Could you see if there's a chain of two or more Crafting Tables?",
            "I need you to ascertain if there is a line up of no less than two Crafting Tables.",
            "Is there a succession of Crafting Tables in at least a duet formation?",
            "Would it be possible to identify a progression of Crafting Tables having a minimum length of two?"
        ],
        "arguments":create_target_state(BlockType.CRAFTING_TABLE, 2, is_diagonal=False),
        "str_check_lambda": "str_check_lambda: is_line_formed(gd, ix)"
    },
    "INSTRUCTION_FURNACE_3": {
        "instruction": "Form a diagonal line of furnaces of length 3.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Arrange furnaces in a slant line with 3 in total.",
            "Assemble a line of 3 furnaces angled diagonally.",
            "Construct a three-length diagonal row of furnaces.",
            "Diagonally, set up a sequence of three furnaces.",
            "Set about diagonally placing three furnaces in a linear pattern."
        ],
        "arguments":create_target_state(BlockType.FURNACE, 3, is_diagonal=True),
        "str_check_lambda": "str_check_lambda: is_line_formed(gd, ix)"
    },
    "INSTRUCTION_ENCHANTMENT_TABLE_ICE_2": {
        "instruction": "Form a diagonal line of enchantment ice tables of length 2.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "create a line of two enchantment ice tables slanted diagonally.",
            "Arrange two ice enchantment tables in a sloped arrangement.",
            "Design a diagonal configuration using two ice enchantment tables.",
            "Establish a slanted sequence of two enchantment tables made of ice.",
            "Place two ice enchantment tables in an angular line."
        ],
        "arguments": create_target_state(BlockType.ENCHANTMENT_TABLE_ICE, 2, is_diagonal=True),
        "str_check_lambda": "str_check_lambda: is_line_formed(gd, ix)"
    },
    "INSTRUCTION_ENCHANTMENT_TABLE_FIRE_7": {
        "instruction": "Check a diagonal line of enchantment table fire blocks of size 7",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Make sure there's a diagonal line of 7 fiery enchantment tables.",
            "Verify the existence of an inclined row consisting of seven blocks of enchantment table on fire.",
            "Could you check if a continuous diagonal sequence of seven fire enchantment tables is present?",
            "I need you to ascertain if a slanting linear arrangement of seven fire enchantment tables exists in the given location.",
            "Confirm whether there exists an unbroken diagonal chain comprising of seven blocks, each comprised of an enchantment table engulfed in flame."
        ],
        "arguments": create_target_state(BlockType.ENCHANTMENT_TABLE_FIRE, 7, is_diagonal=True),
        "str_check_lambda": "str_check_lambda: is_line_formed(gd, ix)"
    },
    "INSTRUCTION_FURNACE_2": {
        "instruction": "Check if there is a line of two furnaces",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Inspect if there are two furnaces in row",
            "Verify if two furnaces were established in a line",
            "Guarantee that a line is formed with a couple of furnaces",
            "Affirm the existence of a linear arrangement of two furnaces",
            "Ascertain the alignment of pair of furnaces into a straight pattern"
        ],
        "arguments": create_target_state(BlockType.FURNACE, 2, is_diagonal=False),
        "str_check_lambda": "str_check_lambda: is_line_formed(gd, ix)"
    },
    "INSTRUCTION_PLANT_7": {
        "instruction": "create a diagonal line of plants that is seven units long.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Please, make a seven unit long diagonal line with flora.",
            "Can you construct a slanted line using plants that measures seven units?",
            "I need you to form a diagonal row of vegetation that is seven units in length.",
            "You need to arrange a span of seven plants in a diagonal layout.",
            "Could you design a sequence of vegetation displayed diagonally that extends for seven units?"
        ],
        "arguments": create_target_state(BlockType.PLANT, 7, is_diagonal=True),
        "str_check_lambda": "str_check_lambda: is_line_formed(gd, ix)"
    },
    "INSTRUCTION_FURNACE_7": {
        "instruction": "create a diagonal line of Furnace with 7 blocks.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Construct a diagonal row of seven Furnaces.",
            "Put together a sequence of Furnace blocks in a bent line, make sure this sequence is 7 blocks long.",
            "Build a series of Furnaces diagonally, and it should consist of 7 Furnaces.",
            "In a diagonal manner, set up a line of 7 Furnace blocks.",
            "Establish a line at an angle using Furnace blocks and ensure this line includes exactly 7 blocks."
        ],
        "arguments": create_target_state(BlockType.FURNACE, 7, is_diagonal=True),
        "str_check_lambda": "str_check_lambda: is_line_formed(gd, ix)"
    },
    "INSTRUCTION_STONE_2_2": {
        "instruction": "Verify a 2-block line of stone arranged linearly in the game space.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Confirm if there is a linear alignment of stone blocks, each of size two, in the current game area.",
            "Check the game for a straight line configuration of two blocks of rocks.",
            "Establish the existence of a two-piece series of stones positioned in a line within the game environment.",
            "Authenticate if a pair of stone blocks are found in a direct line sequence in the game domain.",
            "Can you substantiate whether a duo of boulders exist in a line formation within the gaming zone?"
        ],
        "arguments": create_target_state(BlockType.STONE, 2, is_diagonal=False),
        "str_check_lambda": "str_check_lambda: is_line_formed(gd, ix)"
    },
    "INSTRUCTION_ENCHANTMENT_TABLE_FIRE_6": {
        "instruction": "Form a line containing six Enchantment Fire Tables arranged diagonally.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Compose a diagonal sequence incorporating six Enchantment Fire Tables.",
            "Creatively arrange six Enchantment Fire Tables in a straight line at an angle.",
            "Set up a diagonal pattern of six Enchantment Fire Tables.",
            "Organize six Enchantment Fire Tables into a diagonal row.",
            "Construct a diagonal line by using six Fire Enchantment Tables."
        ],
        "arguments": create_target_state(BlockType.ENCHANTMENT_TABLE_FIRE, 6, is_diagonal=True),
        "str_check_lambda": "str_check_lambda: is_line_formed(gd, ix)"
    },
    "INSTRUCTION_STONE_6": {
        "instruction": "Form a diagonal line of stone blocks with length 6.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Make a diagonal sequence of 6 rock blocks.",
            "I need you to construct a line, on the diagonal axis, using 6 blocks of stone.",
            "create a slanted line using 6 stone blocks.",
            "Place 6 blocks of rock in a diagonal direction, forming a straight line.",
            "Arrange a line with 6 stones diagonally."
        ],
        "arguments": create_target_state(BlockType.STONE, 6, is_diagonal=True),
        "str_check_lambda": "str_check_lambda: is_line_formed(gd, ix)"
    },
    "INSTRUCTION_ENCHANTMENT_TABLE_ICE_4": {
        "instruction": "Make a diagonal line of enchantment table ice blocks with a size of four.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Can you create a line of enchantment table ice with four blocks diagonally?",
            "Can you place four enchantment table ice blocks in a diagonal arrangement?",
            "Arrange four blocks of enchantment table ice in a diagonal line.",
            "Could you diagonally organize four blocks of enchantment table ice in a row?",
            "Craft a diagonal chain using four pieces of ice enchantment table."
        ],
        "arguments": create_target_state(BlockType.ENCHANTMENT_TABLE_ICE, 4, is_diagonal=True),
        "str_check_lambda": "str_check_lambda: is_line_formed(gd, ix)"
    },
    "INSTRUCTION_FURNACE_3_2": {
        "instruction": "Form a line of furnace blocks, 3 blocks in length.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Arrange three furnace blocks in a straight line.",
            "Position a series of three furnace blocks linearly.",
            "I need you to line up three furnace blocks, one after the other.",
            "Sequentially place three furnace units in a row.",
            "Erect a horizontally straight sequence of three furnace blocks."
        ],
        "arguments": create_target_state(BlockType.FURNACE, 3, is_diagonal=False),
        "str_check_lambda": "str_check_lambda: is_line_formed(gd, ix)"
    },
    "INSTRUCTION_PLANT_7_2": {
        "instruction": "create a line of plants with a length of 7 blocks.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Form a sequence of vegetation with a span of seven units.",
            "Make a linear progression of foliage that spans over seven blocks.",
            "Put together a row of greenery extending to seven sections.",
            "Establish a succession of botanical items that provides a stretch of seven blocks.",
            "Set up an arrangement of flora, creating a steady line of seven constituents."
        ],
        "arguments": create_target_state(BlockType.PLANT, 7, is_diagonal=False),
        "str_check_lambda": "str_check_lambda: is_line_formed(gd, ix)"
    },
    "INSTRUCTION_STONE_4": {
        "instruction": "Form a diagonal line of four rocks.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Draw a line of four stones at an angle.",
            "llace four stones so that they make a diagonal line.",
            "Produce a slanted line using four rocks.",
            "Construct a diagonal arrangement of four stones.",
            "Fabricate a tilted line composed of four rocks."
        ],
        "arguments": create_target_state(BlockType.STONE, 4, is_diagonal=True),
        "str_check_lambda": "str_check_lambda: is_line_formed(gd, ix)"
    },
    "INSTRUCTION_CRAFTING_TABLE_7": {
        "instruction": "Check if there is a diagonal line of Crafting Tables of size 7.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Verify the presence of a diagonal sequence of Workbenches of length 7.",
            "Investigate if there exists a diagonal row of Crafting Stations that is 7 units long.",
            "Can you find a diagonal line of BlockType.CRAFTING_TABLE with a size of 7?",
            "Confirm if a diagonal line made up of the BlockType.CRAFTING_TABLE and with a length of 7 units can be found.",
            "Check for the existence of a Crafting Desk diagonal series with seven items in its sequence."
        ],
        "arguments": create_target_state(BlockType.CRAFTING_TABLE, 7, is_diagonal=True),
        "str_check_lambda": "str_check_lambda: is_line_formed(gd, ix)"
    }
}

