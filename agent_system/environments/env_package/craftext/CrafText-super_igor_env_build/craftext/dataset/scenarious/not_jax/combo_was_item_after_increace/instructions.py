from craftext.checkers.scenarius import did_placing_item_increase_variable, was_item_collected_after_another

instructions = {
    '17': {
        'instruction': "Collect the wood and then place a torch to increase the player's health.",
        'items_name': ["wood", "torch"],
        'instruction_paraphrases': [
            "Gather the timber first and then set a lantern to boost the player's vitality.",
            "Pick up the wood and then put down a light source to enhance the player's health.",
            "Collect the wooden material and position an illuminator to improve the player's well-being.",
            "Obtain the wood and install a beacon to raise the player's life points.",
            "Retrieve the timber and place a flame holder so that the player's health increases."
        ],
        'check_lambda': lambda gt: was_item_collected_after_another(gt, "wood", "torch") and did_placing_item_increase_variable(gt, "torch", "player_health")
    },
    '18': {
        'instruction': "Obtain the iron and place a furnace to boost the player's energy.",
        'items_name': ["iron", "furnace"],
        'instruction_paraphrases': [
            "Retrieve the metal first and then set up a kiln to enhance player vigor.",
            "Get the iron and then put down a heater to improve player energy levels.",
            "Collect the iron and position a forge to raise the player's endurance.",
            "Acquire the metal and install a stove to boost the player's stamina.",
            "Secure the iron and place an oven so that the player's energy increases."
        ],
        'check_lambda': lambda gt: was_item_collected_after_another(gt, "iron", "furnace") and did_placing_item_increase_variable(gt, "furnace", "player_energy")
    },
    '19': {
        'instruction': "Secure the coal, then place a plant to increase the player's mana.",
        'items_name': ["coal", "plant"],
        'instruction_paraphrases': [
            "Gather the coal first and then position a shrub to improve player mana.",
            "Collect the carbon and then put down a flower to enhance player's spell power.",
            "Obtain the coal and set up greenery to raise the player's magical energy.",
            "Retrieve the charcoal and place a botanical item to boost the player's mana.",
            "Pick up the coal and install foliage so that the player's mana increases."
        ],
        'check_lambda': lambda gt: was_item_collected_after_another(gt, "coal", "plant") and did_placing_item_increase_variable(gt, "plant", "player_mana")
    },
    '20': {
        'instruction': "Gather the wood and then place a stone to increase the player's strength.",
        'items_name': ["wood", "stone"],
        'instruction_paraphrases': [
            "Collect the timber first and then set a rock to boost player power.",
            "Pick up the wood and then put down a boulder to enhance player's muscle strength.",
            "Obtain the timber and position a pebble to raise the player's physical power.",
            "Retrieve the wood and install a mineral to improve player's strength.",
            "Acquire the wood and place a gravel piece so that the player's strength increases."
        ],
        'check_lambda': lambda gt: was_item_collected_after_another(gt, "wood", "stone") and did_placing_item_increase_variable(gt, "stone", "player_strength")
    },
    '21': {
        'instruction': "Obtain the sword and then place a table to increase the player's experience points.",
        'items_name': ["sword", "table"],
        'instruction_paraphrases': [
            "Retrieve the sword first and then set up a desk to gain player XP.",
            "Pick up the sword and then put down a workbench to raise player experience.",
            "Collect the sword and position a dining table to boost player experience points.",
            "Acquire the sword and place a furniture piece to enhance player XP.",
            "Get the sword and install a tabletop so that the player's experience increases."
        ],
        'check_lambda': lambda gt: was_item_collected_after_another(gt, "sword", "table") and did_placing_item_increase_variable(gt, "table", "player_xp")
    }
}
