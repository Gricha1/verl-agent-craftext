from craftext.dataset.scenarious.jax_build_line import instructions as build_line_instructions
from craftext.dataset.scenarious.jax_build_squere import instructions as build_squere_instructions
from craftext.dataset.scenarious.jax_localization_place import instructions as localization_place_instructions
from craftext.dataset.scenarious.jax_conditional_placing import instructions as conditional_place_instructions
from craftext.dataset.scenarious.jax_conditional_achievements import instructions as conditional_achievements
from craftext.dataset.scenarious.jax_build_star import instructions as build_star_instructions
from craftext.dataset.scenarious.jax_time_constrained_placment import instructions as time_constrained_placment

# Merging 'easy' dictionaries
easy = { **build_line_instructions.easy, # + 2200 SPS + 0.015 - 0.045
         **build_squere_instructions.easy,
         **localization_place_instructions.easy,
         **conditional_achievements.easy,
        #**build_star_instructions.easy,
        #**time_constrained_placment.easy,
        **conditional_place_instructions.easy
        }# **conditional_place_instructions.easy}

# Merging 'medium' dictionaries
medium = {\
        **build_line_instructions.medium, 
        **build_squere_instructions.medium, 
        **localization_place_instructions.medium,
        **conditional_achievements.medium,
        # **build_star_instructions.medium,
        # **time_constrained_placment.medium,
        **conditional_place_instructions.medium
        #    **conditional_achievements.medium
          }
easy_without_building = {
        # **build_line_instructions.easy, 
        # **build_squere_instructions.easy,
        # **localization_place_instructions.easy,
        # **conditional_achievements.easy,
        # **build_star_instructions.easy,
        # **time_constrained_placment.easy,
        # **conditional_place_instructions.easy
        }# **conditional_place_instructions.easy}
easy_only_building = {
        # **build_line_instructions.easy, 
        # **build_squere_instructions.easy,
        # **localization_place_instructions.easy,
        # **conditional_achievements.easy,
        **build_star_instructions.easy
        # **time_constrained_placment.easy,
        # **conditional_place_instructions.easy
}

# medium_ = {
#     'make_line_2': {
#         'instruction': "Make a line of two blocks using table.",
#         'items_name': ["table"],
#         'instruction_paraphrases': [
#             "Create a sequence of two units using the crafting surface.",
#             "Use the workstation to align 2 units in a straight pattern.",
#             "Place two pieces in a row with the help of the crafting station.",
#             "Form a sequence of two units on the crafting surface.",
#             "Utilize the workstation to arrange two units in a straight line.",
#             "Position two pieces consecutively with the assistance of the crafting station."
#         ],
#         'check_lambda': lambda game_data, ix: is_line_formed(game_data, ix, 11, 2, check_diagonal=False)
#     },
#     'small_train_line_2': {
#         'instruction': "Arrange four plants into a compact square shape.",
#         'instruction_paraphrases': [
#             "Place four plants in the form of a small square.",
#             "Organize the plants into a tight square formation.",
#             "Create a square by positioning four plants next to each other.",
#             "Form a neat square using four plants arranged side by side.",
#             "Establish a square layout by placing four plants in a symmetrical pattern."
#         ],
#         'check_lambda': lambda game_data, ix: is_square_formed(game_data, ix, 15, 2)
#     },
#     'small_train_line_6': {
#         'instruction': "Arrange four stones into a compact square shape.",
#         'instruction_paraphrases': [
#             "Arrange four stones to form a square shape.",
#             "Group the stones into a tight square configuration of 4 blocks.",
#             "Position 2x2 pebbles to create a square structure.",
#             "Use four stones to shape a tidy square by placing them adjacent to one another.",
#             "Set up a square pattern by organizing four rocks symmetrically."
#         ],
#         'check_lambda': lambda game_data, ix: is_square_formed(game_data, ix, 4, 2)
#     },
    
#     'small_train_line_4': {
#         'instruction': "Arrange nine stones into a compact square shape of 9 blocks.",
#         'instruction_paraphrases': [
#             "Arrange nine stones to form a square shape.",
#             "Group the stones into a tight square configuration of 18/2 blocks.",
#             "Position 3x3 pebbles to create a square structure.",
#             "Use nine stones to shape a tidy square by placing them adjacent to one another.",
#             "Set up a square pattern by organizing nine rocks symmetrically."
#         ],
#         'check_lambda': lambda game_data, ix: is_square_formed(game_data, ix, 4, 3)
#     },
#     'small_train_line_5': {
#         'instruction': "Make a line of 2 blocks using furnace.",
#         'items_name': ["furnace"],
#         'instruction_paraphrases': [
#             "Arrange 2 units in a line using the heat source.",
#             "Use the heating device to set two units.",
#             "Use the heating element to align two pieces in a specific pattern.",
#             "Organize two pieces into a layout with the help of the heating element.",
#             "Position two pieces in a structured arrangement alongside the heating element."
#         ],
#         'check_lambda': lambda game_data, ix: is_line_formed(game_data, ix, 12, 2, check_diagonal=False)
#     }
# }


# hard_ = {
#     'small_train_line_1': {
#         'instruction': "Make a line of five blocks using table.",
#         'items_name': ["table"],
#         'instruction_paraphrases': [
#             "Create a sequence of five units using the crafting surface.",
#             "Use the workstation to align 5 units in a straight pattern.",
#             "Place 5 pieces in a row with the help of the crafting station.",
#             "Form a sequence of five units on the crafting surface.",
#             "Utilize the workstation to arrange five units in a straight line.",
#             "Position five pieces consecutively with the assistance of the crafting station."
#         ],
#         'check_lambda': lambda game_data, ix: is_line_formed(game_data, ix, 11, 5, check_diagonal=False)
#     },
#     'small_train_line_2': {
#         'instruction': "Arrange four plants into a compact square shape.",
#         'instruction_paraphrases': [
#             "Place four plants in the form of a small square.",
#             "Organize the plants into a tight square formation.",
#             "Create a square by positioning four plants next to each other.",
#             "Form a neat square using four plants arranged side by side.",
#             "Establish a square layout by placing four plants in a symmetrical pattern."
#         ],
#         'check_lambda': lambda game_data, ix: is_square_formed(game_data, ix, 15, 2)
#     },
#    "small_train_line_3": {
#         'instruction': "Arrange two chests in a diagonal line, from top-left to bottom-right.",
#         'instruction_paraphrases': [
#             "Position two chests diagonally from the top-left corner to the bottom-right.",
#             "Create a diagonal line of two chests starting from the top-left.",
#             "Set up two chests in a diagonal pattern from the top-left to the bottom-right.",
#             "Form a diagonal sequence with two chests, extending from the upper left to the lower right.",
#             "Align two chests in a diagonal line, running from the top-left to the bottom-right."
#         ],
#         'check_lambda': lambda game_data, ix: is_line_formed(game_data, ix, 23, 2, check_diagonal=True)
#     },
#     'small_train_line_4': {
#         'instruction': "Arrange nine stones into a compact square shape of 9 blocks.",
#         'instruction_paraphrases': [
#             "Arrange nine stones to form a square shape.",
#             "Group the stones into a tight, square configuration of 18/2 blocks.",
#             "Position 3x3 pebbles to create a square structure.",
#             "Use nine stones to shape a tidy square by placing them adjacent to one another.",
#             "Set up a square pattern by organizing nine rocks symmetrically."
#         ],
#         'check_lambda': lambda game_data, ix: is_square_formed(game_data, ix, 4, 3)
#     },
#     'small_train_line_5': {
#         'instruction': "Make a line of 3 blocks using furnace.",
#         'items_name': ["furnace"],
#         'instruction_paraphrases': [
#             "Arrange 3 units in a line using the heat source.",
#             "Use the heating device to set three units.",
#             "Position 3 pieces in a pattern with the heating element.",
#             "Arrange 3 pieces in a design with the heating element.",
#             "Place 3 pieces in a formation using the heating element.",
#             "Use the heating element to align 3 pieces in a specific pattern.",
#             "Organize 3 pieces into a layout with the help of the heating element.",
#             "Position 3 pieces in a structured arrangement alongside the heating element."
#         ],
#         'check_lambda': lambda game_data, ix: is_line_formed(game_data, ix, 12, 3, check_diagonal=False)
#     }
# }
