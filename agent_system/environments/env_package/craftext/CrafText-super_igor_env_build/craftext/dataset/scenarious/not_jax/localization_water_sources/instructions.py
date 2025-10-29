from craftext.checkers.scenarious_map import is_player_within_all_water_sources, is_player_within_south_water_sources, is_player_within_north_water_sources

instructions = {
    '37': {
        'instruction': "Move around the map and ensure that you have visited all the water sources.",
        'instruction_paraphrases': [
            "Explore the map and check that you've been to every water location.",
            "Traverse the game area and confirm that you've reached all water bodies.",
            "Wander across the map ensuring you have seen all the water spots.",
            "Roam the entire area and verify that you've come across all water features.",
            "Survey the map and ascertain that you've encountered every water source."
        ],
        'check_lambda': lambda game_data: is_player_within_all_water_sources(game_data)
    },
    '38': {
        'instruction': "Go to the northern part of the map and make sure you've visited all the water sources there.",
        'instruction_paraphrases': [
            "Head north on the map and check that you've been to all water bodies in that region.",
            "Navigate to the upper part of the map and confirm you’ve visited each water source in the north.",
            "Travel to the northern area and ensure you've explored all the water locations there.",
            "Move to the top of the map and verify that you've encountered all water features in the north.",
            "Proceed to the northern section and make certain you've covered every water source in that area."
        ],
        'check_lambda': lambda game_data: is_player_within_north_water_sources(game_data)
    },
    '39': {
        'instruction': "Go to the southern part of the map and ensure that you've visited all the water sources there.",
        'instruction_paraphrases': [
            "Move south on the map and verify that you've been to every water body in that area.",
            "Travel to the lower part of the map and confirm you've visited all water sources in the south.",
            "Head to the southern region and make sure you’ve explored each water spot there.",
            "Navigate to the bottom of the map and check that you've seen all the water features in the south.",
            "Proceed to the southern section of the map and ensure you've encountered all water sources."
        ],
        'check_lambda': lambda game_data: is_player_within_south_water_sources(game_data)
    },
    "40": {
        'instruction': "Move through all the water areas on the map.",
        'instruction_paraphrases': [
            "Walk over every water source on the map.",
            "Ensure you have visited each body of water.",
            "Make sure to traverse all water bodies across the map.",
            "Cross every aquatic zone to confirm your presence there.",
            "Verify that you've passed through each wetland region available on the map."
        ],
        'check_lambda': lambda game_data: is_player_within_all_water_sources(game_data)
    },
    "41": {
        'instruction': "Visit all water sources and make sure there's a crafting table nearby.",
        'instruction_paraphrases': [
            "Check all water bodies, ensuring a workbench is present close by.",
            "Go to each water source and verify the presence of a crafting station.",
            "Visit every aquatic zone and confirm there's a crafting bench adjacent to it.",
            "Explore all water regions, ensuring a crafting apparatus is nearby.",
            "Traverse each water area and ensure a crafting device is placed near each."
        ],
        'check_lambda': lambda game_data: is_player_within_all_water_sources(game_data, required_object="CRAFTING_TABLE")
    },
    "42": {
        'instruction': "Ensure you have walked across all water sources with a furnace nearby.",
        'instruction_paraphrases': [
            "Visit all water sources, ensuring a kiln is nearby.",
            "Check each water body and make sure there's an oven close by.",
            "Go over every water area and verify that a heater is placed nearby.",
            "Make sure to cross all aquatic zones with a smelter adjacent to them.",
            "Ensure each wetland has been visited and that a forge is located near it."
        ],
        'check_lambda': lambda game_data: is_player_within_all_water_sources(game_data, required_object="FURNACE")
    },
    "43": {
        'instruction': "Make sure all water sources have been visited, and an enchantment table is nearby.",
        'instruction_paraphrases': [
            "Visit every water source ensuring there's an enchantment altar close by.",
            "Ensure each water body has an enchantment pedestal next to it and is visited.",
            "Check that all water regions are traversed with a magic table present nearby.",
            "Go through each aquatic zone, confirming an arcane station is adjacent.",
            "Walk across every wetland cluster and make sure a spellbinding desk is near."
        ],
        'check_lambda': lambda game_data: is_player_within_all_water_sources(game_data, required_object="ENCHANTMENT_TABLE_FIRE")
    }, 
    "44": {
        'instruction': "Move through all the southern water areas on the map.",
        'instruction_paraphrases': [
            "Walk over every water source in the southern part of the map.",
            "Ensure you have visited each body of water in the south.",
            "Make sure to traverse all water bodies located in the southern region.",
            "Cross every southern aquatic zone to confirm your presence there.",
            "Verify that you've passed through each wetland region in the southern area of the map."
        ],
        'check_lambda': lambda game_data: is_player_within_south_water_sources(game_data)
    },
    "45": {
        'instruction': "Visit all southern water sources and make sure there's a crafting table nearby.",
        'instruction_paraphrases': [
            "Check all southern water bodies, ensuring a workbench is present close by.",
            "Go to each southern water source and verify the presence of a crafting station.",
            "Visit every southern aquatic zone and confirm there's a crafting bench adjacent to it.",
            "Explore all southern water regions, ensuring a crafting apparatus is nearby.",
            "Traverse each southern water area and ensure a crafting device is placed near each."
        ],
        'check_lambda': lambda game_data: is_player_within_south_water_sources(game_data, required_object="CRAFTING_TABLE")
    },
    "46": {
        'instruction': "Ensure you have walked across all southern water sources with a furnace nearby.",
        'instruction_paraphrases': [
            "Visit all southern water sources, ensuring a kiln is nearby.",
            "Check each southern water body and make sure there's an oven close by.",
            "Go over every southern water area and verify that a heater is placed nearby.",
            "Make sure to cross all southern aquatic zones with a smelter adjacent to them.",
            "Ensure each southern wetland has been visited and that a forge is located near it."
        ],
        'check_lambda': lambda game_data: is_player_within_south_water_sources(game_data, required_object="FURNACE")
    },
    "47": {
        'instruction': "Make sure all southern water sources have been visited, and an enchantment table is nearby.",
        'instruction_paraphrases': [
            "Visit every southern water source ensuring there's an enchantment altar close by.",
            "Ensure each southern water body has an enchantment pedestal next to it and is visited.",
            "Check that all southern water regions are traversed with a magic table present nearby.",
            "Go through each southern aquatic zone, confirming an arcane station is adjacent.",
            "Walk across every southern wetland cluster and make sure a spellbinding desk is near."
        ],
        'check_lambda': lambda game_data: is_player_within_south_water_sources(game_data, required_object="ENCHANTMENT_TABLE_FIRE")
    }, 
    "48": {
        'instruction': "Move through all the northern water areas on the map.",
        'instruction_paraphrases': [
            "Walk over every water source in the northern part of the map.",
            "Ensure you have visited each body of water in the north.",
            "Make sure to traverse all water bodies located in the northern region.",
            "Cross every northern aquatic zone to confirm your presence there.",
            "Verify that you've passed through each wetland region in the northern area of the map."
        ],
        'check_lambda': lambda game_data: is_player_within_north_water_sources(game_data)
    },
    "49": {
        'instruction': "Visit all northern water sources and make sure there's a crafting table nearby.",
        'instruction_paraphrases': [
            "Check all northern water bodies, ensuring a workbench is present close by.",
            "Go to each northern water source and verify the presence of a crafting station.",
            "Visit every northern aquatic zone and confirm there's a crafting bench adjacent to it.",
            "Explore all northern water regions, ensuring a crafting apparatus is nearby.",
            "Traverse each northern water area and ensure a crafting device is placed near each."
        ],
        'check_lambda': lambda game_data: is_player_within_north_water_sources(game_data, required_object="CRAFTING_TABLE")
    },
    "50": {
        'instruction': "Ensure you have walked across all northern water sources with a furnace nearby.",
        'instruction_paraphrases': [
            "Visit all northern water sources, ensuring a kiln is nearby.",
            "Check each northern water body and make sure there's an oven close by.",
            "Go over every northern water area and verify that a heater is placed nearby.",
            "Make sure to cross all northern aquatic zones with a smelter adjacent to them.",
            "Ensure each northern wetland has been visited and that a forge is located near it."
        ],
        'check_lambda': lambda game_data: is_player_within_north_water_sources(game_data, required_object="FURNACE")
    },
    "51": {
        'instruction': "Make sure all northern water sources have been visited, and an enchantment table is nearby.",
        'instruction_paraphrases': [
            "Visit every northern water source ensuring there's an enchantment altar close by.",
            "Ensure each northern water body has an enchantment pedestal next to it and is visited.",
            "Check that all northern water regions are traversed with a magic table present nearby.",
            "Go through each northern aquatic zone, confirming an arcane station is adjacent.",
            "Walk across every northern wetland cluster and make sure a spellbinding desk is near."
        ],
        'check_lambda': lambda game_data: is_player_within_north_water_sources(game_data, required_object="ENCHANTMENT_TABLE_FIRE")
    }


}
