from craftext.checkers.base import was_item_collected_after_another

instructions = {
    '6': {
        'instruction': "Collect the pickaxe after you have collected the wood.",
        'items_name': ["wood", "pickaxe"],
        'instruction_paraphrases': [
            "Get the pickaxe after you have gathered the timber.",
            "Retrieve the pickaxe once you have collected the lumber.",
            "Pick up the pickaxe after you have obtained the wood.",
            "Collect the pickaxe following the collection of wooden logs.",
            "Acquire the pickaxe after securing the wooden material."
        ],
        'check_lambda': lambda gt: was_item_collected_after_another(gt, "wood", "pickaxe")
    },
    '7': {
        'instruction': "Gather the iron after you have obtained the coal.",
        'items_name': ["coal", "iron"],
        'instruction_paraphrases': [
            "Collect the iron once you have retrieved the coal.",
            "Get the iron after you have gathered the charcoal.",
            "Pick up the iron after you have secured the coal.",
            "Acquire the iron following the collection of carbon.",
            "Obtain the iron after you have gotten the coal."
        ],
        'check_lambda': lambda gt: was_item_collected_after_another(gt, "coal", "iron")
    },
    '8': {
        'instruction': "Obtain the sword after collecting the armor.",
        'items_name': ["armor", "sword"],
        'instruction_paraphrases': [
            "Collect the sword once you have gathered the armor.",
            "Get the sword after you have secured the protective gear.",
            "Retrieve the sword after you have obtained the armor.",
            "Pick up the sword following the collection of body armor.",
            "Acquire the sword after you have collected the defensive equipment."
        ],
        'check_lambda': lambda gt: was_item_collected_after_another(gt, "armor", "sword")
    },
    '9': {
        'instruction': "Secure the potion after gathering the iron.",
        'items_name': ["iron", "potion"],
        'instruction_paraphrases': [
            "Collect the potion once you have obtained the iron.",
            "Get the potion after you have gathered the metal.",
            "Retrieve the potion after you have secured the iron.",
            "Pick up the potion following the collection of iron ore.",
            "Acquire the potion after you have gotten the iron."
        ],
        'check_lambda': lambda gt: was_item_collected_after_another(gt, "iron", "potion")
    },
    '10': {
        'instruction': "Retrieve the coal after you have collected the wood.",
        'items_name': ["wood", "coal"],
        'instruction_paraphrases': [
            "Get the coal after you have gathered the timber.",
            "Collect the coal once you have obtained the wood.",
            "Pick up the coal after you have secured the wooden logs.",
            "Acquire the coal following the collection of lumber.",
            "Obtain the coal after you have gathered the wooden material."
        ],
        'check_lambda': lambda gt: was_item_collected_after_another(gt, "wood", "coal")
    }
}
