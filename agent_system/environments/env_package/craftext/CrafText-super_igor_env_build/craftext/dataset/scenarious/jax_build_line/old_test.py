from craftext.checkers.building_line import is_line_formed, is_square_formed
from craftext.scenarios.constants import BlockType

easy_test_parafrased = {
    'line_easy_1': {
        'instruction': "Make a line of 2 blocks using table.",
        'instruction_paraphrases': [
            "Construct a row of 2 elements with the crafting station.",
            "Place 2 blocks in a straight path using the workbench.",
            "Use the crafting table to arrange a line of 2 blocks.",
            "Form a sequence of 2 items with the crafting platform.",
            "Set up a line of 2 pieces using the crafting table."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data, ix, BlockType.CRAFTING_TABLE, 2, check_diagonal=False),
        'str_check_lambda': "is_line_formed(game_data, ix, BlockType.CRAFTING_TABLE, 2, check_diagonal=False)"
    },
    'line_easy_2': {
        'instruction': "Make a diagonal line of 3 blocks using table.",
        'instruction_paraphrases': [
            "Create a diagonal arrangement of 3 units with the crafting bench.",
            "Set up 3 blocks diagonally using the workbench.",
            "Use the crafting station to arrange 3 blocks in a diagonal formation.",
            "Form a slanted row of 3 items with the crafting table.",
            "Arrange 3 blocks in a diagonal line using the crafting table."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data, ix, BlockType.CRAFTING_TABLE, 3, check_diagonal=True),
        'str_check_lambda': "is_line_formed(game_data, ix, BlockType.CRAFTING_TABLE, 3, check_diagonal=True)"
    },
    'line_easy_4': {
        'instruction': "Make a line of 2 blocks using furnace.",
        'instruction_paraphrases': [
            "Create a row of 2 blocks with the furnace.",
            "Arrange 2 blocks in a line using the smelter.",
            "Use the furnace to position 2 blocks in a straight line.",
            "Place 2 items in a sequence with the furnace.",
            "Form a straight row of 2 blocks using the heating station."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data, ix, BlockType.FURNACE, 2, check_diagonal=False),
        'str_check_lambda': "is_line_formed(game_data, ix, BlockType.FURNACE, 2, check_diagonal=False)"
    }
}

easy_test_other_paramets = {
    'line_medium_1': {
        'instruction': "Make a vertical line of 4 blocks using stone.",
        'instruction_paraphrases': [
            "Set up a vertical row of 4 stone blocks.",
            "Place 4 stones in a straight vertical line.",
            "Use stone to arrange 4 blocks in a vertical column.",
            "Create a vertical line of 4 stone blocks.",
            "Arrange 4 stone pieces in a vertical formation."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data, ix, BlockType.STONE, 4, check_diagonal=False),
        'str_check_lambda': "is_line_formed(game_data, ix, BlockType.STONE, 4, check_diagonal=False)"
    },
    'line_medium_2': {
        'instruction': "Make a square of 4 blocks using crafting table.",
        'instruction_paraphrases': [
            "Construct a square formation of 4 blocks with the crafting table.",
            "Arrange 4 blocks in a square using the crafting station.",
            "Use the crafting table to form a square of 4 units.",
            "Place 4 items in a square shape with the crafting table.",
            "Set up a square of 4 blocks using the crafting platform."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data, ix, BlockType.CRAFTING_TABLE, 4),
        'str_check_lambda': "is_line_formed(game_data, ix, BlockType.CRAFTING_TABLE, 4)"
    },
    'line_medium_3': {
        'instruction': "Make a horizontal line of 5 blocks using furnace.",
        'instruction_paraphrases': [
            "Create a horizontal line of 5 blocks with the furnace.",
            "Arrange 5 blocks in a straight horizontal line using the furnace.",
            "Use the furnace to place 5 blocks in a row.",
            "Form a sequence of 5 items horizontally using the smelter.",
            "Set up a straight row of 5 blocks with the furnace."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data, ix, BlockType.FURNACE, 5, check_diagonal=False),
        'str_check_lambda': "is_line_formed(game_data, ix, BlockType.FURNACE, 5, check_diagonal=False)"
    }
}



# TODO: check complexity
medium_test_parafrased = {
    "line_medium_1": {
        'instruction': "Align three crafting tables in a straight horizontal line.",
        'instruction_paraphrases': [
            "Place three workbenches in a single horizontal row.",
            "Arrange three crafting stations side by side in a straight line.",
            "Set up three crafting benches in a horizontal line, each one next to the other.",
            "Create a straight horizontal sequence of three workbenches.",
            "Position three crafting stations to form a continuous horizontal line."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data,ix, BlockType.CRAFTING_TABLE, 3, check_diagonal=False)
    },
    "line_medium_2": {
        'instruction': "Position four furnaces in a vertical line.",
        'instruction_paraphrases': [
            "Arrange four stoves in a single vertical column.",
            "Place four kilns one on top of the other in a vertical line.",
            "Set up four ovens in a straight vertical line.",
            "Create a vertical alignment of four stoves, stacking them in a column.",
            "Form a vertical line by placing four kilns one above the other."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data,ix, BlockType.FURNACE, 4, check_diagonal=False)
    },
    "line_medium_3": {
        'instruction': "Arrange five chests in a diagonal line, from top-left to bottom-right.",
        'instruction_paraphrases': [
            "Position five storage boxes diagonally from the top-left corner to the bottom-right.",
            "Create a diagonal line of five trunks starting from the top-left.",
            "Set up five containers in a diagonal pattern from the top-left to the bottom-right.",
            "Form a diagonal sequence with five storage units, extending from the upper left to the lower right.",
            "Align five trunks in a diagonal line, running from the top-left to the bottom-right."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data,ix, BlockType.CHEST, 5, check_diagonal=True)
    },
    "line_medium_4": {
        'instruction': "Place six fountains in a continuous line, either horizontally or vertically.",
        'instruction_paraphrases': [
            "Arrange six water features in a straight line, either in a row or a column.",
            "Form a line of six water sources, oriented either horizontally or vertically.",
            "Position six fountains in a single straight line, regardless of direction.",
            "Set up a straight line of six water features, aligned either horizontally or vertically.",
            "Create a continuous sequence of six fountains, in either a horizontal or vertical direction."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data,ix, BlockType.FOUNTAIN, 6, check_diagonal=False)
    },
    "line_medium_5": {
        'instruction': "Design a line of seven plants, ensuring they connect diagonally.",
        'instruction_paraphrases': [
            "Align seven bushes in a diagonal line, making sure each one is connected.",
            "Set up seven trees in a continuous diagonal line.",
            "Create a diagonal pattern with seven vegetation units, ensuring they form a straight line.",
            "Arrange seven plants in a diagonal sequence, each one connected to the next.",
            "Position seven bushes in a straight diagonal line, making sure they are aligned."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data,ix, BlockType.PLANT, 7, check_diagonal=True)
    }
}


medium_test_other_paramets = {
    "line_medium_1": {
        'instruction': "Align three crafting tables in a straight horizontal line.",
        'instruction_paraphrases': [
            "Place three workbenches in a single horizontal row.",
            "Arrange three crafting stations side by side in a straight line.",
            "Set up three crafting benches in a horizontal line, each one next to the other.",
            "Create a straight horizontal sequence of three workbenches.",
            "Position three crafting stations to form a continuous horizontal line."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data,ix, BlockType.FOUNTAIN, 3, check_diagonal=False)  # Changed to Fountain
    },
    "line_medium_2": {
        'instruction': "Position four furnaces in a vertical line.",
        'instruction_paraphrases': [
            "Arrange four stoves in a single vertical column.",
            "Place four kilns one on top of the other in a vertical line.",
            "Set up four ovens in a straight vertical line.",
            "Create a vertical alignment of four stoves, stacking them in a column.",
            "Form a vertical line by placing four kilns one above the other."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data,ix, BlockType.ENCHANTMENT_TABLE_FIRE, 4, check_diagonal=False)  # Changed to Enchantment Table (Fire)
    },
    "line_medium_3": {
        'instruction': "Arrange five chests in a diagonal line, from top-left to bottom-right.",
        'instruction_paraphrases': [
            "Position five storage boxes diagonally from the top-left corner to the bottom-right.",
            "Create a diagonal line of five trunks starting from the top-left.",
            "Set up five containers in a diagonal pattern from the top-left to the bottom-right.",
            "Form a diagonal sequence with five storage units, extending from the upper left to the lower right.",
            "Align five trunks in a diagonal line, running from the top-left to the bottom-right."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data,ix, BlockType.FOUNTAIN, 5, check_diagonal=True)  # Changed to Fountain
    },
    "line_medium_4": {
        'instruction': "Place six fountains in a continuous line, either horizontally or vertically.",
        'instruction_paraphrases': [
            "Arrange six water features in a straight line, either in a row or a column.",
            "Form a line of six water sources, oriented either horizontally or vertically.",
            "Position six fountains in a single straight line, regardless of direction.",
            "Set up a straight line of six water features, aligned either horizontally or vertically.",
            "Create a continuous sequence of six fountains, in either a horizontal or vertical direction."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data,ix, BlockType.CHEST, 6, check_diagonal=False)  # Changed to Chest
    },
    "line_medium_5": {
        'instruction': "Design a line of seven plants, ensuring they connect diagonally.",
        'instruction_paraphrases': [
            "Align seven bushes in a diagonal line, making sure each one is connected.",
            "Set up seven trees in a continuous diagonal line.",
            "Create a diagonal pattern with seven vegetation units, ensuring they form a straight line.",
            "Arrange seven plants in a diagonal sequence, each one connected to the next.",
            "Position seven bushes in a straight diagonal line, making sure they are aligned."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data,ix, BlockType.ENCHANTMENT_TABLE_ICE, 7, check_diagonal=True)  # Changed to Enchantment Table (Ice)
    }
}


from craftext.scenarios.parce_dataset import update_previous_dict

easy_test_other_paramets = update_previous_dict(easy_test_other_paramets, "jax_build_line/instructions/test/easy/other_params", "jax_build_line_test_op")
medium_test_other_paramets = update_previous_dict(medium_test_other_paramets, "jax_build_line/instructions/test/medium/other_params", "jax_build_line_test_op")
