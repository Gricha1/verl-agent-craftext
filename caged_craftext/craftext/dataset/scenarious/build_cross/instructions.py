from craftext.environment.scenarious.checkers.building_line import is_cross_formed

medium = {
    "62": {
        'instruction': "Place the crafting table in the center with furnaces on the left, right, above, and below.",
        'instruction_paraphrases': [
            "Position the crafting table at the center, surrounded by furnaces on all four sides.",
            "Put a crafting table in the middle, and place furnaces to its left, right, top, and bottom.",
            "Set the crafting table centrally, and arrange furnaces around it in a cross formation.",
            "Arrange the crafting table in the center, ensuring furnaces are on every adjacent side.",
            "Place the crafting table in the middle with furnaces surrounding it in a plus sign pattern."
        ],
        'check_lambda': lambda game_data: is_cross_formed(game_data, "CRAFTING_TABLE")
    },
    "63": {
        'instruction': "Place a chest in the center, with fountains on its four adjacent sides.",
        'instruction_paraphrases': [
            "Position the chest in the middle and place fountains around it on all sides.",
            "Put a chest in the center, and place fountains above, below, to the left, and to the right of it.",
            "Set a chest at the core, surrounded by fountains in a cross-like layout.",
            "Arrange the chest in the center and place fountains at every adjoining position.",
            "Place the chest in the middle with fountains aligned in a cross shape around it."
        ],
        'check_lambda': lambda game_data: is_cross_formed(game_data, "CHEST")
    },
    "64": {
        'instruction': "Position the ice enchantment table in the center with crafting tables on all four sides.",
        'instruction_paraphrases': [
            "Set the ice enchantment table at the center, surrounded by crafting tables.",
            "Place the ice enchantment table in the middle and surround it with crafting tables.",
            "Put the ice enchantment table at the center and place crafting tables on the left, right, above, and below.",
            "Arrange the ice enchantment table centrally with crafting tables on each adjacent side.",
            "Position the ice enchantment table in the center, flanked by crafting tables in a cross formation."
        ],
        'check_lambda': lambda game_data: is_cross_formed(game_data, "ENCHANTMENT_TABLE_ICE")
    },
    "65": {
        'instruction': "Place the fire enchantment table in the center and surround it with chests.",
        'instruction_paraphrases': [
            "Position the fire enchantment table centrally, with chests on every side.",
            "Put the fire enchantment table in the center and arrange chests around it.",
            "Set the fire enchantment table in the middle and place chests to its left, right, above, and below.",
            "Arrange the fire enchantment table at the center with chests surrounding it.",
            "Place the fire enchantment table centrally, ensuring chests are on all four sides."
        ],
        'check_lambda': lambda game_data: is_cross_formed(game_data, "ENCHANTMENT_TABLE_FIRE")
    },
    "66": {
        'instruction': "Set a plant in the middle, surrounded by crafting tables on each adjacent side.",
        'instruction_paraphrases': [
            "Position the plant centrally and place crafting tables around it.",
            "Place the plant in the center, surrounded by crafting tables.",
            "Put the plant in the middle with crafting tables on its left, right, top, and bottom.",
            "Arrange the plant at the center with crafting tables forming a cross around it.",
            "Set the plant centrally, flanked by crafting tables on all four sides."
        ],
        'check_lambda': lambda game_data: is_cross_formed(game_data, "PLANT")
    }
}
