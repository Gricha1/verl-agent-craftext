from craftext.checkers.scenarius import did_placing_item_increase_variable

instructions = {
    '11': {
        'instruction': "Place a torch to increase the player's health.",
        'items_name': ["torch"],
        'instruction_paraphrases': [
            "Put a lantern to boost the player's vitality.",
            "Set a light source to improve the player's health.",
            "Position an illuminator to enhance the player's well-being.",
            "Install a beacon to raise the player's life points.",
            "Place a flame holder so that the player's health increases."
        ],
        'check_lambda': lambda gt: did_placing_item_increase_variable(gt, "torch", "player_health")
    },
    '12': {
        'instruction': "Place a table to increase the player's experience points.",
        'items_name': ["table"],
        'instruction_paraphrases': [
            "Set up a desk to gain player XP.",
            "Put down a workbench to raise player experience.",
            "Install a dining table to boost player experience points.",
            "Position a furniture piece to enhance player XP.",
            "Place a tabletop so that the player's experience increases."
        ],
        'check_lambda': lambda gt: did_placing_item_increase_variable(gt, "table", "player_xp")
    },
    '13': {
        'instruction': "Place a furnace to boost player energy.",
        'items_name': ["furnace"],
        'instruction_paraphrases': [
            "Set up a kiln to increase player stamina.",
            "Put down a heater to enhance player vigor.",
            "Install a forge to improve player energy levels.",
            "Position an oven to raise the player's endurance.",
            "Place a smelter so that the player's energy boosts."
        ],
        'check_lambda': lambda gt: did_placing_item_increase_variable(gt, "furnace", "player_energy")
    },
    '14': {
        'instruction': "Place a plant to increase the player's mana.",
        'items_name': ["plant"],
        'instruction_paraphrases': [
            "Set up a shrub to boost player magic.",
            "Put down a flower to enhance player's spell power.",
            "Install a greenery to improve player mana.",
            "Position a botanical item to raise the player's magical energy.",
            "Place foliage so that the player's mana increases."
        ],
        'check_lambda': lambda gt: did_placing_item_increase_variable(gt, "plant", "player_mana")
    },
    '15': {
        'instruction': "Place a stone to increase the player's strength.",
        'items_name': ["stone"],
        'instruction_paraphrases': [
            "Set up a rock to boost player power.",
            "Put down a boulder to enhance player's muscle strength.",
            "Install a pebble to improve player might.",
            "Position a mineral to raise the player's physical power.",
            "Place a gravel piece so that the player's strength increases."
        ],
        'check_lambda': lambda gt: did_placing_item_increase_variable(gt, "stone", "player_strength")
    }
}

