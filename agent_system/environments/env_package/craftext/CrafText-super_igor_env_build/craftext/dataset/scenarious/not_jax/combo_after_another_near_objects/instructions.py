from craftext.checkers.scenarius  import was_item_collected_after_another, was_item_placed_near_another

instructions = {
    '82': {
        'instruction': "Collect the wood after you have collected the stone, then place the torch near the table.",
        'items_name': ["stone", "wood", "torch", "table"],
        'instruction_paraphrases': [
            "Gather the stone first, then collect the wood, and finally position the torch beside the table.",
            "First obtain the stone, then retrieve the wood, and after that, place the torch close to the table.",
            "Pick up the stone before gathering the wood, and afterward, set the torch next to the table.",
            "Acquire the stone, followed by the wood, and finish by arranging the torch near the table.",
            "Get the stone, then the wood, and lastly place the torch adjacent to the table."
        ],
        'check_lambda': lambda gd: was_item_collected_after_another(gd, "stone", "wood") and was_item_placed_near_another(gd, "torch", "table")
    },
    '83': {
        'instruction': "Collect the coal after you have gathered the wood, then put the plant near the furnace.",
        'items_name': ["wood", "coal", "plant", "furnace"],
        'instruction_paraphrases': [
            "First, gather the wood, then collect the coal, and finally place the plant beside the furnace.",
            "Retrieve the wood before securing the coal, and afterward, position the plant next to the furnace.",
            "Pick up the wood, then the coal, and lastly, set the plant close to the furnace.",
            "Collect the wood, followed by the coal, and finish by arranging the plant near the furnace.",
            "Gather the wood, then the coal, and finally locate the plant adjacent to the furnace."
        ],
        'check_lambda': lambda gd: was_item_collected_after_another(gd, "wood", "coal") and was_item_placed_near_another(gd, "plant", "furnace")
    },
    '84': {
        'instruction': "Obtain the sword after collecting the iron, then place the stone close to the torch.",
        'items_name': ["iron", "sword", "stone", "torch"],
        'instruction_paraphrases': [
            "Gather the iron first, then collect the sword, and afterward, position the stone near the torch.",
            "First secure the iron, then retrieve the sword, and finish by placing the stone beside the torch.",
            "Pick up the iron before gathering the sword, and then set the stone close to the torch.",
            "Collect the iron, followed by the sword, and finally arrange the stone near the torch.",
            "Get the iron, then the sword, and lastly locate the stone adjacent to the torch."
        ],
        'check_lambda': lambda gd: was_item_collected_after_another(gd, "iron", "sword") and was_item_placed_near_another(gd, "stone", "torch")
    },
    '85': {
        'instruction': "Secure the potion after collecting the coal, then place the table beside the plant.",
        'items_name': ["coal", "potion", "table", "plant"],
        'instruction_paraphrases': [
            "First collect the coal, then gather the potion, and finish by placing the table next to the plant.",
            "Retrieve the coal before securing the potion, and afterward, position the table close to the plant.",
            "Pick up the coal, then the potion, and lastly, set the table beside the plant.",
            "Collect the coal, followed by the potion, and finish by arranging the table near the plant.",
            "Gather the coal, then the potion, and finally locate the table adjacent to the plant."
        ],
        'check_lambda': lambda gd: was_item_collected_after_another(gd, "coal", "potion") and was_item_placed_near_another(gd, "table", "plant")
    },
    '86': {
        'instruction': "Collect the armor after obtaining the wood, then arrange the furnace near the stone.",
        'items_name': ["wood", "armor", "furnace", "stone"],
        'instruction_paraphrases': [
            "Gather the wood first, then collect the armor, and afterward, position the furnace near the stone.",
            "First secure the wood, then retrieve the armor, and finish by placing the furnace beside the stone.",
            "Pick up the wood before gathering the armor, and then set the furnace close to the stone.",
            "Collect the wood, followed by the armor, and finally arrange the furnace near the stone.",
            "Get the wood, then the armor, and lastly locate the furnace adjacent to the stone."
        ],
        'check_lambda': lambda gd: was_item_collected_after_another(gd, "wood", "armor") and was_item_placed_near_another(gd, "furnace", "stone")
    }
}
