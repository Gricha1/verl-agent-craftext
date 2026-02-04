from craftext.checkers.scenarius import was_item_placed_near_another

instructions = {
    '1': {
        'instruction': "Place the torch near the table.",
        'items_name': ["torch", "table"],
        'instruction_paraphrases': [
            "Put the lamp next to the desk.",
            "Position the light close to the counter.",
            "Set the lantern beside the bench.",
            "Arrange the beacon adjacent to the worktable.",
            "Locate the flashlight near the dining table."
        ],
        'check_lambda': lambda game_data: was_item_placed_near_another(game_data, "torch", "table")
    },
    '2': {
        'instruction': "Put the plant near the furnace.",
        'items_name': ["plant", "furnace"],
        'instruction_paraphrases': [
            "Place the shrub next to the stove.",
            "Position the greenery close to the heater.",
            "Set the flower beside the kiln.",
            "Arrange the bush adjacent to the oven.",
            "Locate the vegetation near the fireplace."
        ],
        'check_lambda': lambda game_data: was_item_placed_near_another(game_data, "plant", "furnace")
    },
    '3': {
        'instruction': "Place the stone close to the torch.",
        'items_name': ["stone", "torch"],
        'instruction_paraphrases': [
            "Put the rock next to the lamp.",
            "Position the boulder near the light.",
            "Set the pebble beside the lantern.",
            "Arrange the mineral adjacent to the beacon.",
            "Locate the cobblestone near the flashlight."
        ],
        'check_lambda': lambda game_data: was_item_placed_near_another(game_data, "stone", "torch")
    },
    '4': {
        'instruction': "Put the table beside the plant.",
        'items_name': ["table", "plant"],
        'instruction_paraphrases': [
            "Place the desk next to the shrub.",
            "Position the counter near the greenery.",
            "Set the bench beside the flower.",
            "Arrange the worktable adjacent to the bush.",
            "Locate the dining table near the vegetation."
        ],
        'check_lambda': lambda game_data: was_item_placed_near_another(game_data, "table", "plant")
    },
    '5': {
        'instruction': "Arrange the furnace near the stone.",
        'items_name': ["furnace", "stone"],
        'instruction_paraphrases': [
            "Place the stove next to the rock.",
            "Set the heater beside the boulder.",
            "Position the kiln near the pebble.",
            "Put the oven close to the mineral.",
            "Locate the fireplace adjacent to the cobblestone."
        ],
        'check_lambda': lambda game_data: was_item_placed_near_another(game_data, "furnace", "stone")
    }
}
