from craftext.checkers.scenarius import is_item_in_closed_contour

instructions = {
    '22': {
        'instruction': "Place the table inside a stone boundary.",
        'items_name': ["table", "stone"],
        'instruction_paraphrases': [
            "Put the desk within a rock enclosure.",
            "Set the bench inside a boulder perimeter.",
            "Arrange the table within the limits of the stone wall.",
            "Position the counter inside the stone outline.",
            "Locate the table within the rocky boundary."
        ],
        'check_lambda': lambda gd: is_item_in_closed_contour(gd, "table", "stone")
    },
    '23': {
        'instruction': "Place the torch inside a furnace boundary.",
        'items_name': ["torch", "furnace"],
        'instruction_paraphrases': [
            "Put the lantern within an oven enclosure.",
            "Set the light inside a kiln perimeter.",
            "Arrange the torch within the limits of the furnace wall.",
            "Position the lamp inside the furnace outline.",
            "Locate the torch within the kiln boundary."
        ],
        'check_lambda': lambda gd: is_item_in_closed_contour(gd, "torch", "furnace")
    },
    '24': {
        'instruction': "Place the plant inside a table boundary.",
        'items_name': ["plant", "table"],
        'instruction_paraphrases': [
            "Put the flower within a desk enclosure.",
            "Set the bush inside a counter perimeter.",
            "Arrange the plant within the limits of the table wall.",
            "Position the shrub inside the table outline.",
            "Locate the plant within the desk boundary."
        ],
        'check_lambda': lambda gd: is_item_in_closed_contour(gd, "plant", "table")
    },
    '25': {
        'instruction': "Place the stone inside a torch boundary.",
        'items_name': ["stone", "torch"],
        'instruction_paraphrases': [
            "Put the boulder within a lantern enclosure.",
            "Set the rock inside a light perimeter.",
            "Arrange the stone within the limits of the torch wall.",
            "Position the pebble inside the torch outline.",
            "Locate the stone within the lantern boundary."
        ],
        'check_lambda': lambda gt: is_item_in_closed_contour(gt, "stone", "torch")
    },
    '26': {
        'instruction': "Place the furnace inside a plant boundary.",
        'items_name': ["furnace", "plant"],
        'instruction_paraphrases': [
            "Put the kiln within a flower enclosure.",
            "Set the oven inside a shrub perimeter.",
            "Arrange the furnace within the limits of the plant wall.",
            "Position the stove inside the plant outline.",
            "Locate the furnace within the bush boundary."
        ],
        'check_lambda': lambda gd: is_item_in_closed_contour(gd, "furnace", "plant")
    }
}
