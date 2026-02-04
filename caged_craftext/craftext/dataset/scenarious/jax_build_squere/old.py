from craftext.checkers.building_line import is_square_formed
from craftext.scenarios.constants import BlockType
from craftext.scenarios.parce_dataset import update_previous_dict

one = {
    "squere_one": {
        'check_lambda': lambda game_data, ix: is_square_formed(game_data, ix, BlockType.STONE, 2),
        'str_check_lambda': "is_square_formed(game_data, ix, BlockType.STONE, 2)",
        'instruction': "Build a 2x2 square of stone blocks.",
        'instruction_paraphrases': [
            "Form a sturdy stone square, 2 blocks long on each side.",
            "Construct a small plaza of stones, arranged in a 2 by 2 block formation.",
            "Lay out stone blocks to create a strong 2x2 enclosed area.",
            "Place stones to build a firm square structure, with 2 blocks making up each side.",
            "Create a 2x2 stone foundation, ensuring that each corner of the square is defined."
        ]
    }, 
    "squere_one_1": {
        'check_lambda': lambda game_data, ix: is_square_formed(game_data,ix, BlockType.STONE, 2),
        'str_check_lambda': "is_square_formed(game_data, ix, BlockType.STONE, 2)",
        'instruction': "Build a 2x2 square of stone blocks.",
        'instruction_paraphrases': [
            "Form a sturdy stone square, 2 blocks long on each side.",
            "Construct a small plaza of stones, arranged in a 2 by 2 block formation.",
            "Lay out stone blocks to create a strong 2x2 enclosed area.",
            "Place stones to build a firm square structure, with 2 blocks making up each side.",
            "Create a 2x2 stone foundation, ensuring that each corner of the square is defined."
        ]
    }
}
easy = {}
#     "squere_easy_1": {
#         'check_lambda': lambda game_data, ix: is_square_formed(game_data, ix, BlockType.CRAFTING_TABLE, 2),
#         'str_check_lambda': "is_square_formed(game_data, ix, BlockType.CRAFTING_TABLE, 2)",
#         'instruction': "Build a 2x2 square of crafting tables.",
#         'instruction_paraphrases': [
#             "Form a crafting station by placing tables in a square, 2 blocks on each side.",
#             "Create a small crafting grid where each side of the square is made of 2 tables.",
#             "Construct a 2-block by 2-block crafting area using tables.",
#             "Arrange crafting tables to form a box-shaped structure with equal sides.",
#             "Set up a compact 2x2 crafting workspace, with each corner marked by a table."
#         ]
#     },
#     "squere_easy_2": {
#         'check_lambda': lambda game_data, ix: is_square_formed(game_data, ix, BlockType.STONE, 2),
#         'str_check_lambda': "is_square_formed(game_data, ix, BlockType.STONE, 2)",
#         'instruction': "Build a 2x2 square of stone blocks.",
#         'instruction_paraphrases': [
#             "Form a sturdy stone square, 2 blocks long on each side.",
#             "Construct a small plaza of stones, arranged in a 2 by 2 block formation.",
#             "Lay out stone blocks to create a strong 2x2 enclosed area.",
#             "Place stones to build a firm square structure, with 2 blocks making up each side.",
#             "Create a 2x2 stone foundation, ensuring that each corner of the square is defined."
#         ]
#     },
#     "squere_easy_3": {
#         'check_lambda': lambda game_data, ix: is_square_formed(game_data, ix, BlockType.PLANT, 2),
#         'str_check_lambda': "is_square_formed(game_data, ix, BlockType.PLANT, 2)",
#         'instruction': "Build a 2x2 square of plants.",
#         'instruction_paraphrases': [
#             "Arrange plants in a compact 2x2 green patch.",
#             "Create a small garden area by positioning plants in a 2x2 layout.",
#             "Plant greenery in a neat square shape, with each side consisting of 2 plants.",
#             "Set up a green zone by placing plants in a square pattern, 2 on each side.",
#             "Form a mini garden where plants are arranged in a 2x2 grid formation."
#         ]
#     },
#     "squere_easy_4": {
#         'check_lambda': lambda game_data, ix: is_square_formed(game_data, ix, BlockType.STONE, 3),
#         'str_check_lambda': "is_square_formed(game_data, ix, BlockType.STONE, 3)",
#         'instruction': "Build a 3x3 square of stone blocks.",
#         'instruction_paraphrases': [
#             "Create a solid 3x3 stone block structure, with 3 blocks on each side.",
#             "Form a large square area using stones, arranged in a 3x3 pattern.",
#             "Lay out stones in a square formation, ensuring 3 blocks per side.",
#             "Build a wide stone platform, with each side of the square made up of 3 blocks.",
#             "Position 9 stone blocks in a 3x3 layout to create a sturdy foundation."
#         ]
#     },
#     "squere_easy_5": {
#         'check_lambda': lambda game_data, ix: is_square_formed(game_data, ix, BlockType.PLANT, 3),
#         'str_check_lambda': "is_square_formed(game_data, ix, BlockType.PLANT, 3)",
#         'instruction': "Build a 3x3 square of plants.",
#         'instruction_paraphrases': [
#             "Set up a garden bed by arranging plants in a 3x3 square.",
#             "Create a plant grid where each side of the square consists of 3 plants.",
#             "Form a large green area using plants, laid out in a 3 by 3 structure.",
#             "Build a lush square garden by placing plants in a neat 3x3 grid.",
#             "Arrange plants in a symmetrical 3x3 pattern, creating a balanced green patch."
#         ]
#     },
#     "squere_easy_6": {  # Updated name to avoid duplicate keys
#         'check_lambda': lambda game_data, ix: is_square_formed(game_data, ix, BlockType.FURNACE, 2),
#         'str_check_lambda': "is_square_formed(game_data, ix, BlockType.FURNACE, 2)",
#         'instruction': "Build a 2x2 square of furnaces.",
#         'instruction_paraphrases': [
#             "Construct a 2x2 square using furnaces.",
#             "Place two rows of furnaces, one following the other, each consisting of 2 furnaces.",
#             "Arrange two lines of furnaces, back to back, with 2 furnaces per row.",
#             "Form a furnace square with a side of 2, ensuring a compact arrangement.",
#             "Set up a 2x2 square using 4 furnaces to create a heating zone."
#         ]
#     }
# }

        
        
medium = {}
#     "52": {
#         'instruction': "Arrange four crafting tables into a compact square shape.",
#         'instruction_paraphrases': [
#             "Place four crafting tables in the form of a small square.",
#             "Organize the crafting tables into a tight square formation.",
#             "Create a square by positioning four crafting tables next to each other.",
#             "Form a neat square using four crafting tables arranged side by side.",
#             "Establish a square layout by placing four crafting tables in a symmetrical pattern."
#         ],
#         'check_lambda': lambda game_data, ix: is_square_formed(game_data,ix,BlockType.CRAFTING_TABLE, 2)
#     },
#     "53": {
#         'instruction': "Create a square grid using nine furnaces, ensuring each furnace connects to its neighbor.",
#         'instruction_paraphrases': [
#             "Set up a 3x3 square by placing nine furnaces in a grid.",
#             "Arrange the furnaces into a connected square formation with three on each side.",
#             "Construct a square grid using nine furnaces, aligned in three rows and three columns.",
#             "Form a square pattern by placing nine furnaces so that each connects to another.",
#             "Design a square-shaped grid with nine furnaces, ensuring they're all adjacent."
#         ],
#         'check_lambda': lambda game_data, ix: is_square_formed(game_data,ix, BlockType.FURNACE, 3)
#     },
#     "54": {
#         'instruction': "Organize sixteen chests into four rows and four columns, forming a square.",
#         'instruction_paraphrases': [
#             "Place sixteen chests in a 4x4 square arrangement.",
#             "Form a larger square by arranging sixteen chests into four lines of four.",
#             "Create a square pattern with sixteen chests, positioning them in rows and columns.",
#             "Construct a square grid using sixteen chests, with four in each row and column.",
#             "Design a square formation by setting up sixteen chests in a grid pattern."
#         ],
#         'check_lambda':  lambda game_data, ix: is_square_formed(game_data,ix, BlockType.CHEST, 4)
#     },
#     "55": {
#         'instruction': "Use twenty-five fountains to create a large square pattern.",
#         'instruction_paraphrases': [
#             "Arrange twenty-five fountains into a big square.",
#             "Construct a spacious square grid with twenty-five fountains, ensuring they're evenly spaced.",
#             "Form a square shape using twenty-five fountains, organized into five rows and five columns.",
#             "Create a sizable square by placing twenty-five fountains in a grid formation.",
#             "Design a large square layout with twenty-five fountains, connecting them in a 5x5 pattern."
#         ],
#         'check_lambda': lambda game_data, ix: is_square_formed(game_data,ix, BlockType.FOUNTAIN, 5)
#     },
#     "56": {
#         'instruction': "Establish a pattern where thirty-six plants form a balanced square.",
#         'instruction_paraphrases': [
#             "Place thirty-six plants in a symmetrical square formation.",
#             "Arrange thirty-six plants to form a square pattern, with each plant aligned with its neighbors.",
#             "Create a square using thirty-six plants, organized into a structured grid.",
#             "Design a balanced square layout with thirty-six plants, ensuring equal spacing.",
#             "Form a large square by positioning thirty-six plants into six rows and six columns."
#         ],
#         'check_lambda': lambda game_data, ix: is_square_formed(game_data,ix, BlockType.PLANT, 6)
#     }
# }

easy = update_previous_dict(easy, "jax_build_squere/instructions/train/easy", "build_squere")
medium = update_previous_dict(medium, "jax_build_squere/instructions/train/medium", "build_squere")
