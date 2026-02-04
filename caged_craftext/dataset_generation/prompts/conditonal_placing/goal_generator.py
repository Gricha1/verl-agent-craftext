import random

def generate_example_goals(num_goals: int, difficulty="EASY") -> tuple[list[str], list[str]]:
    """
    Generate a list of unique text-based goals for conditional placing with difficulty levels.
    
    Args:
        num_goals (int): Number of goals to generate.
        difficulty (str): Difficulty level ("EASY" or "MEDIUM").
    
    Returns:
        tuple[list[str], list[str]]: List of unique conditional placing goals and corresponding synonyms guidance.
    """
    inventory_items = {
        'EASY': ['WOOD', 'STONE', 'COAL', 'IRON', 'DIAMOND', 'SAPLING'],
        'MEDIUM': ["WOOD", "STONE", "COAL", "IRON", "DIAMOND", "SAPLING", "RUBY", "SAPPHIRE"]
    }
    placeable_objects = {
        'EASY': ['STONE', 'CRAFTING_TABLE', 'PLANT', 'FURNACE'],
        'MEDIUM': ['STONE', 'CRAFTING_TABLE', 'PLANT', 'FURNACE', 'ENCHANTMENT_TABLE_FIRE', 'ENCHANTMENT_TABLE_ICE', 'TORCH']
    }
    counts_to_collect = list(range(1, 6))  # Collect between 1 and 5 items
    counts_to_stand = list(range(1, 6))   # Place between 1 and 5 items

    # Generate weights for MEDIUM difficulty
    def generate_weights(items):
        """
        Generate weights for items with slower growth using square root or logarithm.
        """
        length = len(items)
        return [((i + 1) ** 0.5) for i in range(length)]  # Square root growth

    inventory_weights = generate_weights(inventory_items[difficulty]) if difficulty == "MEDIUM" else None
    placeable_weights = generate_weights(placeable_objects[difficulty]) if difficulty == "MEDIUM" else None

    combinations = set()
    make_synonims_to = []
    while len(combinations) < num_goals:
        # Weighted random selection for MEDIUM difficulty
        if difficulty == "MEDIUM":
            inventory_item = random.choices(inventory_items[difficulty], weights=inventory_weights, k=1)[0]
            object_to_place = random.choices(placeable_objects[difficulty], weights=placeable_weights, k=1)[0]
        else:
            inventory_item = random.choice(inventory_items[difficulty])
            object_to_place = random.choice(placeable_objects[difficulty])
        
        count_collect = random.choice(counts_to_collect)
        count_stand = random.choice(counts_to_stand)
        
        goal = (f"[{difficulty}] COLLECT {count_collect} {inventory_item} AND PLACE {count_stand} {object_to_place}")
        if goal not in combinations:
            synonyms_to = (
                f"WHEN GENERATING PARAPHRASES, USE VARIED SYNONYMS OR ALTERNATIVE EXPRESSIONS FOR THE TERMS '{inventory_item}' AND "
                f"'{object_to_place}'. ALSO, DIVERSIFY HOW YOU EXPRESS THE 'COUNT TO COLLECT' - '{count_collect}', AND THE 'COUNT TO STAND' - '{count_stand}', "
                f"BY USING NUMBERS (E.G., '7'), WORDS (E.G., 'SEVEN'), OR DESCRIPTIVE PHRASES (E.G., 'A TOTAL OF SEVEN ITEMS')."
            )
            make_synonims_to.append(synonyms_to)
        
        combinations.add(goal)

    return list(combinations), make_synonims_to
