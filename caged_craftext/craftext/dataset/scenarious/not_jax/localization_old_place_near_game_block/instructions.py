from craftext.checkers.scenarious_map import is_object_near_target

instructions = {
    '27': {
        'instruction': "Place the crafting table near the tree.",
        'instruction_paraphrases': [
            "Put the crafting bench close to the oak.",
            "Position the crafting station adjacent to the woodland tree.",
            "Set up the crafting unit in proximity to the forest tree.",
            "Arrange the crafting equipment within a block of the timber.",
            "Deploy the crafting apparatus next to the sapling."
        ],
        'check_lambda': lambda game_data: is_object_near_target(game_data, "CRAFTING_TABLE", "TREE")[0]
    },
    '28': {
        'instruction': "Place the furnace north of the water.",
        'instruction_paraphrases': [
            "Put the furnace to the north of the stream.",
            "Position the furnace on the northern side of the water source.",
            "Set up the furnace to the north of the pond.",
            "Arrange the furnace within a block north of the river.",
            "Deploy the furnace to the northern edge of the lake."
        ],
        'check_lambda': lambda game_data: is_object_near_target(game_data, "FURNACE", "WATER")[1]
    },
    '29': {
        'instruction': "Put the chest south of the coal.",
        'instruction_paraphrases': [
            "Place the storage chest to the south of the charcoal.",
            "Position the container on the southern side of the coal deposit.",
            "Set the chest in proximity to the south of the carbon ore.",
            "Arrange the chest within a block south of the coal seam.",
            "Deploy the chest on the southern edge of the coal reserve."
        ],
        'check_lambda': lambda game_data: is_object_near_target(game_data, "CHEST", "COAL")[2]
    },
    '30': {
        'instruction': "Place the stone east of the iron.",
        'instruction_paraphrases': [
            "Put the stone block to the east of the iron ore.",
            "Position the stone on the eastern side of the ferrous metal.",
            "Set the stone in proximity to the east of the iron vein.",
            "Arrange the stone within a block east of the iron deposit.",
            "Deploy the stone on the eastern edge of the iron reserve."
        ],
        'check_lambda': lambda game_data: is_object_near_target(game_data, "STONE", "IRON")[3]
    },
    '31': {
        'instruction': "Place the fountain west of the diamond.",
        'instruction_paraphrases': [
            "Put the fountain to the west of the gem.",
            "Position the fountain on the western side of the diamond block.",
            "Set the fountain in proximity to the west of the precious stone.",
            "Arrange the fountain within a block west of the diamond jewel.",
            "Deploy the fountain to the western edge of the gemstone."
        ],
        'check_lambda': lambda game_data: is_object_near_target(game_data, "FOUNTAIN", "DIAMOND")[4]
    },
    '32': {
        'instruction': "Put the crafting table near the ruby.",
        'instruction_paraphrases': [
            "Place the crafting table close to the red gem.",
            "Position the crafting station adjacent to the ruby stone.",
            "Set up the crafting bench in proximity to the crimson jewel.",
            "Arrange the crafting unit within a block of the ruby gem.",
            "Deploy the crafting apparatus beside the ruby."
        ],
        'check_lambda': lambda game_data: is_object_near_target(game_data, "CRAFTING_TABLE", "RUBY")[0]
    },
    '33': {
        'instruction': "Place the enchantment table near the lava.",
        'instruction_paraphrases': [
            "Put the enchantment table close to the magma.",
            "Position the enchanting station adjacent to the molten rock.",
            "Set up the enchantment table in proximity to the lava flow.",
            "Arrange the enchanting unit within a block of the volcanic liquid.",
            "Deploy the enchanting apparatus beside the molten lava."
        ],
        'check_lambda': lambda game_data: is_object_near_target(game_data, "ENCHANTMENT_TABLE_FIRE", "LAVA")[0]
    },
    '34': {
        'instruction': "Put the furnace near the wall.",
        'instruction_paraphrases': [
            "Place the furnace close to the barrier.",
            "Position the furnace adjacent to the partition.",
            "Set the furnace in proximity to the dividing wall.",
            "Arrange the furnace within a block of the barricade.",
            "Deploy the furnace next to the stone wall."
        ],
        'check_lambda': lambda game_data: is_object_near_target(game_data, "FURNACE", "WALL")[0]
    },
    '35': {
        'instruction': "Place the chest near the path.",
        'instruction_paraphrases': [
            "Put the chest close to the trail.",
            "Position the chest adjacent to the walkway.",
            "Set the chest in proximity to the path road.",
            "Arrange the chest within a block of the footpath.",
            "Deploy the chest next to the pathway."
        ],
        'check_lambda': lambda game_data: is_object_near_target(game_data, "CHEST", "PATH")[0]
    },
    '36': {
        'instruction': "Place the stone near the grass.",
        'instruction_paraphrases': [
            "Put the stone close to the turf.",
            "Position the stone adjacent to the lawn.",
            "Set the stone in proximity to the grass patch.",
            "Arrange the stone within a block of the green grass.",
            "Deploy the stone next to the grassy field."
        ],
        'check_lambda': lambda game_data: is_object_near_target(game_data, "STONE", "GRASS")[0]
    }
}
