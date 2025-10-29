from craftext.checkers.scenarius import was_item_collected_after_another, is_item_in_closed_contour

instructions = {
    '72': {
        'instruction': "Place the pickaxe inside the table boundary after collecting the wood.",
        'items_name': ["wood", "pickaxe", "table"],
        'instruction_paraphrases': [
            "Position the pickaxe within the table's enclosure after gathering the wood.",
            "Place the pickaxe in the table boundary once you have collected the wood.",
            "Set the pickaxe inside the table perimeter after securing the wood.",
            "Arrange the pickaxe inside the table outline after obtaining the wood.",
            "Put the pickaxe in the table boundary after retrieving the wood."
        ],
        'check_lambda': lambda gd: was_item_collected_after_another(gd, "wood", "pickaxe") and is_item_in_closed_contour(gd, "pickaxe", "table")
    },
    '73': {
        'instruction': "Place the torch inside the furnace boundary after collecting the table.",
        'items_name': ["table", "torch", "furnace"],
        'instruction_paraphrases': [
            "Put the torch within the furnace's enclosure after collecting the table.",
            "Set the torch inside the furnace boundary once you have secured the table.",
            "Position the torch within the furnace perimeter after picking up the table.",
            "Arrange the torch inside the furnace outline after obtaining the table.",
            "Locate the torch in the furnace boundary after you have collected the table."
        ],
        'check_lambda': lambda gd: was_item_collected_after_another(gd, "table", "torch") and is_item_in_closed_contour(gd, "torch", "furnace")
    },
    '74': {
        'instruction': "Collect the iron after placing the torch inside the stone boundary.",
        'items_name': ["torch", "stone", "iron"],
        'instruction_paraphrases': [
            "Retrieve the iron after positioning the torch within the stone's boundary.",
            "Get the iron once the torch is placed inside the stone perimeter.",
            "Pick up the iron following the placement of the torch in the stone enclosure.",
            "Acquire the iron after arranging the torch inside the stone boundary.",
            "Collect the iron after setting the torch within the limits of the stone wall."
        ],
        'check_lambda': lambda gd: is_item_in_closed_contour(gd, "torch", "stone") and was_item_collected_after_another(gd, "stone", "iron")
    },
    '75': {
        'instruction': "Place the plant inside the table boundary after gathering the coal.",
        'items_name': ["coal", "plant", "table"],
        'instruction_paraphrases': [
            "Position the plant within the table's enclosure after collecting the coal.",
            "Put the plant inside the table boundary after obtaining the coal.",
            "Set the plant in the table perimeter once you've gathered the coal.",
            "Arrange the plant within the limits of the table wall following the collection of coal.",
            "Place the plant inside the table boundary after retrieving the coal."
        ],
        'check_lambda': lambda gd: was_item_collected_after_another(gd, "coal", "plant") and is_item_in_closed_contour(gd, "plant", "table")
    },
    '76': {
        'instruction': "Collect the potion after placing the stone inside the torch boundary.",
        'items_name': ["stone", "torch", "potion"],
        'instruction_paraphrases': [
            "Retrieve the potion after positioning the stone within the torch's boundary.",
            "Get the potion once the stone is placed inside the torch perimeter.",
            "Pick up the potion following the placement of the stone in the torch enclosure.",
            "Acquire the potion after arranging the stone inside the torch boundary.",
            "Collect the potion after setting the stone within the limits of the torch wall."
        ],
        'check_lambda': lambda gd: is_item_in_closed_contour(gd, "stone", "torch") and was_item_collected_after_another(gd, "stone", "potion")
    }
}
