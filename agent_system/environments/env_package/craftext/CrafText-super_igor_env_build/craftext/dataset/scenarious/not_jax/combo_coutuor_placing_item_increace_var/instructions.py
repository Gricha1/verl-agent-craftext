from craftext.checkers.scenarius import did_placing_item_increase_variable, is_item_in_closed_contour

instructions = {
    '77': {
        'instruction': "Place a torch inside a stone boundary to increase the player's health.",
        'items_name': ["torch", "stone"],
        'instruction_paraphrases': [
            "Set a lantern within a rock enclosure to boost the player's vitality.",
            "Install a light source inside a stone perimeter to enhance the player's well-being.",
            "Position an illuminator within the stone boundary to raise the player's life points.",
            "Place a beacon inside the rocky contour to increase the player's health."
        ],
        'check_lambda': lambda gd: is_item_in_closed_contour(gd, "torch", "stone") and did_placing_item_increase_variable(gd, "torch", "player_health")
    },
    '78': {
        'instruction': "Place a table inside a furnace boundary to increase the player's experience points.",
        'items_name': ["table", "furnace"],
        'instruction_paraphrases': [
            "Set up a desk within an oven enclosure to gain player XP.",
            "Position a workbench inside a kiln perimeter to enhance player experience.",
            "Place a dining table within the furnace boundary to boost player XP.",
            "Locate a tabletop within the limits of the furnace to increase player experience."
        ],
        'check_lambda': lambda gd: is_item_in_closed_contour(gd, "table", "furnace") and did_placing_item_increase_variable(gd, "table", "player_xp")
    },
    '79': {
        'instruction': "Place a furnace inside a plant boundary to boost the player's energy.",
        'items_name': ["furnace", "plant"],
        'instruction_paraphrases': [
            "Set up a kiln within a shrub enclosure to increase player stamina.",
            "Position a heater inside a flower perimeter to enhance player vigor.",
            "Place a forge within the limits of the plant boundary to improve player energy.",
            "Locate a furnace within the plant's outline to raise the player's endurance."
        ],
        'check_lambda': lambda gd: is_item_in_closed_contour(gd, "furnace", "plant") and did_placing_item_increase_variable(gd, "furnace", "player_energy")
    },
    '80': {
        'instruction': "Place a plant inside a table boundary to increase the player's mana.",
        'items_name': ["plant", "table"],
        'instruction_paraphrases': [
            "Set up a shrub within a desk enclosure to boost player magic.",
            "Position a flower inside a counter perimeter to enhance player's spell power.",
            "Place a greenery within the table boundary to improve player mana.",
            "Locate foliage within the limits of the table to increase player's mana."
        ],
        'check_lambda': lambda gd: is_item_in_closed_contour(gd, "plant", "table") and did_placing_item_increase_variable(gd, "plant", "player_mana")
    },
    '81': {
        'instruction': "Place a stone inside a torch boundary to increase the player's strength.",
        'items_name': ["stone", "torch"],
        'instruction_paraphrases': [
            "Set up a rock within a lantern enclosure to boost player power.",
            "Position a boulder inside a light perimeter to enhance player's muscle strength.",
            "Place a pebble within the torch boundary to improve player might.",
            "Locate a gravel piece within the limits of the torch to increase player's strength."
        ],
        'check_lambda': lambda gd: is_item_in_closed_contour(gd, "stone", "torch") and did_placing_item_increase_variable(gd, "stone", "player_strength")
    }
}
