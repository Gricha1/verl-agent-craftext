from craftext.checkers.building_line import is_line_formed
from craftext.scenarios.constants import BlockType
from craftext.scenarios.parce_dataset import update_previous_dict
one = {
    'line_one_1': {
        'instruction': "Make a line of 2 blocks using table.",
        'instruction_paraphrases': [
            "Construct a row of 2 pieces with the crafting station.",
            "Place 2 units in a straight row using the workbench.",
            "Use the crafting table to form a row of 2 items.",
            "Arrange a sequence of 2 blocks with the crafting platform.",
            "Create a straight formation of 2 blocks with the crafting table."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data,ix, BlockType.CRAFTING_TABLE, 2, check_diagonal=False),
        'str_check_lambda': "is_line_formed(game_data,ix, BlockType.CRAFTING_TABLE, 2, check_diagonal=False)"
    },
    
    'line_one_2': {
        'instruction': "Make a line of 2 blocks using table.",
        'instruction_paraphrases': [
            "Construct a row of 2 pieces with the crafting station.",
            "Place 2 units in a straight row using the workbench.",
            "Use the crafting table to form a row of 2 items.",
            "Arrange a sequence of 2 blocks with the crafting platform.",
            "Create a straight formation of 2 blocks with the crafting table."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data,ix, BlockType.CRAFTING_TABLE, 2, check_diagonal=False),
        'str_check_lambda': "is_line_formed(game_data,ix, BlockType.CRAFTING_TABLE, 2, check_diagonal=False)"
    },
}
easy = {
    'line_easy_1': {
        'instruction': "Make a line of 2 blocks using table.",
        'instruction_paraphrases': [
            "Construct a row of 2 pieces with the crafting station.",
            "Place 2 units in a straight row using the workbench.",
            "Use the crafting table to form a row of 2 items.",
            "Arrange a sequence of 2 blocks with the crafting platform.",
            "Create a straight formation of 2 blocks with the crafting table."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data, ix, BlockType.CRAFTING_TABLE, 2, check_diagonal=False),
        'str_check_lambda': "is_line_formed(game_data, ix, BlockType.CRAFTING_TABLE, 2, check_diagonal=False)"
    },
    'line_easy_2': {
        'instruction': "Make a diagonal line of 3 blocks using table.",
        'instruction_paraphrases': [
            "Construct a slanted row of 3 items with the crafting station.",
            "Place 3 blocks in a diagonal sequence using the workbench.",
            "Use the crafting platform to arrange a diagonal row of 3 pieces.",
            "Arrange a sloped line of 3 units using the crafting bench.",
            "Create a diagonal sequence of 3 items with the crafting table."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data, ix, BlockType.CRAFTING_TABLE, 3, check_diagonal=True),
        'str_check_lambda': "is_line_formed(game_data, ix, BlockType.CRAFTING_TABLE, 3, check_diagonal=True)"
    },
    'line_easy_3': {
        'instruction': "Make a diagonal line of 2 blocks using stone.",
        'instruction_paraphrases': [
            "Construct a slanted row of 2 stones.",
            "Place 2 stone blocks in a diagonal sequence.",
            "Use stones to arrange a diagonal line of 2 pieces.",
            "Arrange a sloped row of 2 stone units.",
            "Create a diagonal sequence of 2 stone blocks."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data, ix, BlockType.STONE, 2, check_diagonal=True),
        'str_check_lambda': "is_line_formed(game_data, ix, BlockType.STONE, 2, check_diagonal=True)"
    },
    'line_easy_4': {
        'instruction': "Make a line of 2 blocks using furnace.",
        'instruction_paraphrases': [
            "Construct a straight row of 2 units with the furnace.",
            "Place 2 blocks in a line using the heating station.",
            "Use the furnace to form a sequence of 2 items.",
            "Arrange a straight line of 2 pieces with the smelter.",
            "Create a row of 2 blocks using the furnace."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data, ix, BlockType.FURNACE, 2, check_diagonal=False),
        'str_check_lambda': "is_line_formed(game_data, ix, BlockType.FURNACE, 2, check_diagonal=False)"
    },
    'line_easy_5': {
        'instruction': "Make a horizontal line of 4 blocks using stone.",
        'instruction_paraphrases': [
            "Construct a straight line of 4 stone blocks.",
            "Place 4 stone units in a horizontal row.",
            "Use stones to form a line of 4 blocks in a straight path.",
            "Arrange 4 stones in a straight sequence.",
            "Create a horizontal formation of 4 stone blocks."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data, ix, BlockType.STONE, 4, check_diagonal=False),
        'str_check_lambda': "is_line_formed(game_data, ix, BlockType.STONE, 4, check_diagonal=False)"
    }
}


medium = {
    "line_medium_1": {
        'instruction': "Align three crafting tables in a straight horizontal line.",
        'instruction_paraphrases': [
            "Place three crafting tables in a single horizontal row.",
            "Arrange three crafting tables side by side in a straight line.",
            "Set up three crafting tables in a horizontal line, each one next to the other.",
            "Create a straight horizontal sequence of three crafting tables.",
            "Position three crafting tables to form a continuous horizontal line."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data,ix,BlockType.CRAFTING_TABLE, 3, check_diagonal=False)
    },
    "line_medium_2": {
        'instruction': "Position four furnaces in a vertical line.",
        'instruction_paraphrases': [
            "Arrange four furnaces in a single vertical column.",
            "Place four furnaces one on top of the other in a vertical line.",
            "Set up four furnaces in a straight vertical line.",
            "Create a vertical alignment of four furnaces, stacking them in a column.",
            "Form a vertical line by placing four furnaces one above the other."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data,ix,BlockType.FURNACE, 4, check_diagonal=False)
    },
    "line_medium_3": {
        'instruction': "Arrange five chests in a diagonal line, from top-left to bottom-right.",
        'instruction_paraphrases': [
            "Position five chests diagonally from the top-left corner to the bottom-right.",
            "Create a diagonal line of five chests starting from the top-left.",
            "Set up five chests in a diagonal pattern from the top-left to the bottom-right.",
            "Form a diagonal sequence with five chests, extending from the upper left to the lower right.",
            "Align five chests in a diagonal line, running from the top-left to the bottom-right."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data,ix, BlockType.CHEST, 5, check_diagonal=True)
    },
    "line_medium_4": {
        'instruction': "Place six fountains in a continuous line, either horizontally or vertically.",
        'instruction_paraphrases': [
            "Arrange six fountains in a straight line, either in a row or a column.",
            "Form a line of six fountains, oriented either horizontally or vertically.",
            "Position six fountains in a single straight line, regardless of direction.",
            "Set up a straight line of six fountains, aligned either horizontally or vertically.",
            "Create a continuous sequence of six fountains, in either a horizontal or vertical direction."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data,ix, BlockType.FOUNTAIN, 6, check_diagonal=False)
    },
    "line_medium_5": {
        'instruction': "Design a line of seven plants, ensuring they connect diagonally.",
        'instruction_paraphrases': [
            "Align seven plants in a diagonal line, making sure each plant is connected.",
            "Set up seven plants in a continuous diagonal line.",
            "Create a diagonal pattern with seven plants, ensuring they form a straight line.",
            "Arrange seven plants in a diagonal sequence, each one connected to the next.",
            "Position seven plants in a straight diagonal line, making sure they are aligned."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data,ix, BlockType.PLANT, 7, check_diagonal=True)
    }
}


easy = update_previous_dict(easy, "jax_build_line/instructions/train/easy", "build_line")
medium = update_previous_dict(medium, "jax_build_line/instructions/train/medium", "build_line")
