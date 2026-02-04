from craftext.checkers.scenarius import did_placing_item_increase_variable, was_item_placed_near_another

instructions = {
    '67': {
        'instruction': "Place a torch near the table to increase the player's health.",
        'items_name': ["torch", "table"],
        'instruction_paraphrases': [
            "Put a lantern next to the desk to boost the player's vitality.",
            "Set a light source near the counter to improve the player's health.",
            "Position an illuminator beside the bench to enhance the player's well-being.",
            "Install a beacon adjacent to the worktable to raise the player's life points.",
            "Place a flame holder near the dining table so that the player's health increases."
        ],
        'check_lambda': lambda game_data: was_item_placed_near_another(game_data, "torch", "table") and did_placing_item_increase_variable(game_data, "torch", "player_health")
    },
    '68': {
        'instruction': "Place a plant near the furnace to increase the player's mana.",
        'items_name': ["plant", "furnace"],
        'instruction_paraphrases': [
            "Set up a shrub next to the stove to boost player magic.",
            "Put down a flower near the heater to enhance player's spell power.",
            "Install greenery beside the kiln to improve player mana.",
            "Position a botanical item adjacent to the oven to raise the player's magical energy.",
            "Place foliage near the fireplace so that the player's mana increases."
        ],
        'check_lambda': lambda game_data: was_item_placed_near_another(game_data, "plant", "furnace") and did_placing_item_increase_variable(game_data, "plant", "player_mana")
    },
    '69': {
        'instruction': "Place a stone close to the torch to increase the player's strength.",
        'items_name': ["stone", "torch"],
        'instruction_paraphrases': [
            "Put the rock next to the lamp to boost player power.",
            "Position the boulder near the light to enhance player's muscle strength.",
            "Set the pebble beside the lantern to improve player might.",
            "Arrange the mineral adjacent to the beacon to raise the player's physical power.",
            "Locate the cobblestone near the flashlight so that the player's strength increases."
        ],
        'check_lambda': lambda game_data: was_item_placed_near_another(game_data, "stone", "torch") and did_placing_item_increase_variable(game_data, "stone", "player_strength")
    },
    '70': {
        'instruction': "Arrange the furnace near the stone to boost player energy.",
        'items_name': ["furnace", "stone"],
        'instruction_paraphrases': [
            "Place the stove next to the rock to increase player stamina.",
            "Set the heater beside the boulder to enhance player vigor.",
            "Position the kiln near the pebble to improve player energy levels.",
            "Put the oven close to the mineral to raise the player's endurance.",
            "Locate the fireplace adjacent to the cobblestone so that the player's energy boosts."
        ],
        'check_lambda': lambda game_data: was_item_placed_near_another(game_data, "furnace", "stone") and did_placing_item_increase_variable(game_data, "furnace", "player_energy")
    },
    '71': {
        'instruction': "Place the table beside the plant to increase the player's experience points.",
        'items_name': ["table", "plant"],
        'instruction_paraphrases': [
            "Place the desk next to the shrub to gain player XP.",
            "Position the counter near the greenery to raise player experience.",
            "Set the bench beside the flower to boost player experience points.",
            "Arrange the worktable adjacent to the bush to enhance player XP.",
            "Locate the dining table near the vegetation so that the player's experience increases."
        ],
        'check_lambda': lambda game_data: was_item_placed_near_another(game_data, "table", "plant") and did_placing_item_increase_variable(game_data, "table", "player_xp")
    }
}
