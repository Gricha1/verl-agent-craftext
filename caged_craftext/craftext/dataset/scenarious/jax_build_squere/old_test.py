from craftext.checkers.building_line import is_square_formed
from craftext.scenarios.constants import BlockType

easy_test_parafrased = {
    "squere_parafrased_1": {
        'check_lambda': lambda game_data, ix: is_square_formed(game_data, ix, BlockType.CRAFTING_TABLE, 2),
        'str_check_lambda': "is_square_formed(game_data, ix, BlockType.CRAFTING_TABLE, 2)",
        'instruction_paraphrases': [
            "Build a small crafting table structure in a 2x2 layout.",
            "Arrange four crafting tables into a square with 2 units per side.",
            "Form a 2 by 2 grid using crafting tables to create a crafting area.",
            "Set up crafting tables to form a compact 2x2 shape.",
            "Create a crafting block by placing 4 tables in a square."
        ]
    },
    "squere_parafrased_2": {
        'check_lambda': lambda game_data, ix: is_square_formed(game_data, ix, BlockType.STONE, 2),
        'str_check_lambda': "is_square_formed(game_data, ix, BlockType.STONE, 2)",
        'instruction_paraphrases': [
            "Arrange stones to create a sturdy 2x2 formation.",
            "Build a small square of stones, with each side made of 2 blocks.",
            "Place 4 stone blocks in a neat 2 by 2 pattern.",
            "Create a stone area by arranging blocks in a 2x2 grid.",
            "Form a compact square of stones using 4 blocks."
        ]
    },
    "squere_parafrased_3": {
        'check_lambda': lambda game_data, ix: is_square_formed(game_data, ix, BlockType.STONE, 3),
        'str_check_lambda': "is_square_formed(game_data, ix, BlockType.STONE, 3)",
        'instruction_paraphrases': [
            "Create a large stone square, 3 units long on each side.",
            "Form a strong 3x3 structure using stone blocks.",
            "Build a wide 3 by 3 grid using stones to create a solid foundation.",
            "Arrange stones in a 3x3 formation, forming a large square.",
            "Set up a stone square where each side consists of 3 blocks."
        ]
    },
    "squere_parafrased_4": {
        'check_lambda': lambda game_data, ix: is_square_formed(game_data, ix, BlockType.PLANT, 3),
        'str_check_lambda': "is_square_formed(game_data, ix, BlockType.PLANT, 3)",
        'instruction_paraphrases': [
            "Arrange plants to create a 3x3 square garden area.",
            "Build a large green patch by placing plants in a 3 by 3 pattern.",
            "Form a garden area using plants, arranged in a 3x3 grid.",
            "Create a 3x3 plant formation to complete the garden.",
            "Set up a large square of plants, with each side consisting of 3 plants."
        ]
    }
}


# TODO: check complexity
easy_test_other_paramets = {
    "squere_other_1": {
        'check_lambda': lambda game_data, ix: is_square_formed(game_data, ix, BlockType.CRAFTING_TABLE, 3),
        'str_check_lambda': "is_square_formed(game_data, ix, BlockType.CRAFTING_TABLE, 3)",
        'instruction_paraphrases': [
            "Build a 3x3 square of crafting tables.",
            "Create a larger crafting area by arranging crafting tables in a 3x3 shape.",
            "Form a 3x3 grid using crafting tables for an extended workspace.",
            "Set up a big crafting table block, arranging them into a 3 by 3 formation.",
            "Construct a 3x3 square with crafting tables, expanding the crafting zone."
        ]
    },
    "squere_other_2": {
        'check_lambda': lambda game_data, ix: is_square_formed(game_data, ix, BlockType.FURNACE, 2),
        'str_check_lambda': "is_square_formed(game_data, ix, BlockType.FURNACE, 2)",
        'instruction_paraphrases': [
            "Construct a furnace block by arranging 4 furnaces in a 2x2 shape.",
            "Build a small square using furnaces, with 2 units per side.",
            "Create a furnace layout in a neat 2 by 2 formation.",
            "Set up furnaces to form a 2x2 block, creating a heating area.",
            "Form a square of furnaces, with each side consisting of 2 furnaces."
        ]
    },
    "squere_other_3": {
        'check_lambda': lambda game_data, ix: is_square_formed(game_data, ix, BlockType.PLANT, 4),
        'str_check_lambda': "is_square_formed(game_data, ix, BlockType.PLANT, 4)",
        'instruction_paraphrases': [
            "Build a 4x4 green patch by arranging plants in a grid.",
            "Form a large 4 by 4 garden area using plants.",
            "Create a 4x4 grid of plants for a spacious garden.",
            "Arrange plants in a 4x4 layout to create a large green zone.",
            "Set up a 4x4 square of plants to grow a bigger garden patch."
        ]
    },
    "squere_other_4": {
        'check_lambda': lambda game_data, ix: is_square_formed(game_data, ix, BlockType.PLANT, 2),
        'str_check_lambda': "is_square_formed(game_data, ix, BlockType.PLANT, 2)",
        'instruction_paraphrases': [
            "Build a green zone by arranging plants in a 2x2 layout.",
            "Set up a square of plants, with each side measuring 2 units.",
            "Form a compact garden area using plants arranged in a 2 by 2 pattern.",
            "Create a plant square, placing 2 plants on each side.",
            "Arrange 4 plants in a tight 2x2 grid to form a small garden patch."
        ]
    }
}



medium_test_parafrased = {
    "52": {
        'instruction': "Arrange four crafting tables into a compact square shape.",
        'instruction_paraphrases': [
            "Place four workbenches in the form of a small square.",
            "Organize the crafting stations into a tight square formation.",
            "Create a square by positioning four crafting benches next to each other.",
            "Form a neat square using four workbenches arranged side by side.",
            "Establish a square layout by placing four crafting stations in a symmetrical pattern."
        ],
        'check_lambda': lambda game_data, ix: is_square_formed(game_data,ix,BlockType.CRAFTING_TABLE, 2)
    },
    "53": {
        'instruction': "Create a square grid using nine furnaces, ensuring each furnace connects to its neighbor.",
        'instruction_paraphrases': [
            "Set up a 3x3 square by placing nine stoves in a grid.",
            "Arrange the kilns into a connected square formation with three on each side.",
            "Construct a square grid using nine ovens, aligned in three rows and three columns.",
            "Form a square pattern by placing nine furnaces so that each connects to another.",
            "Design a square-shaped grid with nine kilns, ensuring they're all adjacent."
        ],
        'check_lambda': lambda game_data, ix: is_square_formed(game_data,ix, BlockType.FURNACE, 3)
    },
    "54": {
        'instruction': "Organize sixteen chests into four rows and four columns, forming a square.",
        'instruction_paraphrases': [
            "Place sixteen storage boxes in a 4x4 square arrangement.",
            "Form a larger square by arranging sixteen trunks into four lines of four.",
            "Create a square pattern with sixteen containers, positioning them in rows and columns.",
            "Construct a square grid using sixteen storage units, with four in each row and column.",
            "Design a square formation by setting up sixteen trunks in a grid pattern."
        ],
        'check_lambda':  lambda game_data, ix: is_square_formed(game_data,ix, BlockType.CHEST, 4)
    },
    "55": {
        'instruction': "Use twenty-five fountains to create a large square pattern.",
        'instruction_paraphrases': [
            "Arrange twenty-five water features into a big square.",
            "Construct a spacious square grid with twenty-five water sources, ensuring they're evenly spaced.",
            "Form a square shape using twenty-five fountains, organized into five rows and five columns.",
            "Create a sizable square by placing twenty-five water features in a grid formation.",
            "Design a large square layout with twenty-five water sources, connecting them in a 5x5 pattern."
        ],
        'check_lambda': lambda game_data, ix: is_square_formed(game_data,ix, BlockType.FOUNTAIN, 5)
    },
    "56": {
        'instruction': "Establish a pattern where thirty-six plants form a balanced square.",
        'instruction_paraphrases': [
            "Place thirty-six trees in a symmetrical square formation.",
            "Arrange thirty-six bushes to form a square pattern, with each plant aligned with its neighbors.",
            "Create a square using thirty-six vegetation units, organized into a structured grid.",
            "Design a balanced square layout with thirty-six plants, ensuring equal spacing.",
            "Form a large square by positioning thirty-six bushes into six rows and six columns."
        ],
        'check_lambda': lambda game_data, ix: is_square_formed(game_data,ix, BlockType.PLANT, 6)
    }
}


medium_test_other_paramets = {
    "52": {
        'instruction': "Arrange four crafting tables into a compact square shape.",
        'instruction_paraphrases': [
            "Place four workbenches in the form of a small square.",
            "Organize the crafting stations into a tight square formation.",
            "Create a square by positioning four crafting benches next to each other.",
            "Form a neat square using four workbenches arranged side by side.",
            "Establish a square layout by placing four crafting stations in a symmetrical pattern."
        ],
        'check_lambda': lambda game_data, ix: is_square_formed(game_data,ix, BlockType.FOUNTAIN, 2)  # Changed block to Fountain
    },
    "53": {
        'instruction': "Create a square grid using nine furnaces, ensuring each furnace connects to its neighbor.",
        'instruction_paraphrases': [
            "Set up a 3x3 square by placing nine stoves in a grid.",
            "Arrange the kilns into a connected square formation with three on each side.",
            "Construct a square grid using nine ovens, aligned in three rows and three columns.",
            "Form a square pattern by placing nine furnaces so that each connects to another.",
            "Design a square-shaped grid with nine kilns, ensuring they're all adjacent."
        ],
        'check_lambda': lambda game_data, ix: is_square_formed(game_data,ix, BlockType.ENCHANTMENT_TABLE_FIRE, 3)  # Changed block to Enchantment Table (Fire)
    },
    "54": {
        'instruction': "Organize sixteen chests into four rows and four columns, forming a square.",
        'instruction_paraphrases': [
            "Place sixteen storage boxes in a 4x4 square arrangement.",
            "Form a larger square by arranging sixteen trunks into four lines of four.",
            "Create a square pattern with sixteen containers, positioning them in rows and columns.",
            "Construct a square grid using sixteen storage units, with four in each row and column.",
            "Design a square formation by setting up sixteen trunks in a grid pattern."
        ],
        'check_lambda': lambda game_data, ix: is_square_formed(game_data,ix, BlockType.FOUNTAIN, 4)  # Changed block to Fountain
    },
    "55": {
        'instruction': "Use twenty-five fountains to create a large square pattern.",
        'instruction_paraphrases': [
            "Arrange twenty-five water features into a big square.",
            "Construct a spacious square grid with twenty-five water sources, ensuring they're evenly spaced.",
            "Form a square shape using twenty-five fountains, organized into five rows and five columns.",
            "Create a sizable square by placing twenty-five water features in a grid formation.",
            "Design a large square layout with twenty-five water sources, connecting them in a 5x5 pattern."
        ],
        'check_lambda': lambda game_data, ix: is_square_formed(game_data,ix, BlockType.ENCHANTMENT_TABLE_ICE, 5)  # Changed block to Enchantment Table (Ice)
    },
    "56": {
        'instruction': "Establish a pattern where thirty-six plants form a balanced square.",
        'instruction_paraphrases': [
            "Place thirty-six trees in a symmetrical square formation.",
            "Arrange thirty-six bushes to form a square pattern, with each plant aligned with its neighbors.",
            "Create a square using thirty-six vegetation units, organized into a structured grid.",
            "Design a balanced square layout with thirty-six plants, ensuring equal spacing.",
            "Form a large square by positioning thirty-six bushes into six rows and six columns."
        ],
        'check_lambda': lambda game_data, ix: is_square_formed(game_data,ix, BlockType.CHEST, 6)  # Changed block to Chest
    }
}


from craftext.scenarios.parce_dataset import update_previous_dict

easy_test_other_paramets = update_previous_dict(easy_test_other_paramets, "jax_build_squere/instructions/test/easy/other_params", "jax_build_squere_test_op")
medium_test_other_paramets = update_previous_dict(medium_test_other_paramets, "jax_build_squere/instructions/test/medium/other_params", "jax_build_squere_test_op")
