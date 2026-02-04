import random
def generate_example_goals(num_goals: int, difficulty="EASY") -> list[str]:
    """
    Generate a list of unique text-based goals describing block placement patterns.
    
    Args:
        num_goals (int): Number of goals to generate.
    
    Returns:
        list[str]: List of unique goal descriptions.
    """
    block_types = {
        'EASY': ['STONE', 'CRAFTING_TABLE', 'PLANT', 'FURNACE'],
        'MEDIUM': ['STONE', 'CRAFTING_TABLE', 'PLANT', 'FURNACE', 'ENCHANTMENT_TABLE_FIRE', 'ENCHANTMENT_TABLE_ICE', 'TORCH']
    }
    
    if difficulty=='EASY':
        sizes = list(range(2, 5))
    else:
        sizes = list(range(3, 8))
    
    combinations = set()
    make_synonims_to = []
    
    while len(combinations) < num_goals:
       # difficulty = random.choice(list(block_types.keys()))
        block = random.choice(block_types[difficulty])
        size = random.choice(sizes)
        
        goal = f"SQUERE OF {block} WITH SIDE SIZE {size}"
        if goal not in combinations:
            synonyms_to=f"WHEN GENERATING PARAPHRASES, USE VARIED SYNONYMS OR ALTERNATIVE EXPRESSIONS FOR THE TERM '{block}', \
                        AND EXPLORE DIFFERENT WAYS TO EXPRESS THE SIZE '{size}', SUCH AS NUMBERS (E.G., '7'), WORDS (E.G., 'SEVEN'),\
                        OR PHRASES (E.G., 'A TOTAL OF SEVEN UNITS')."
            make_synonims_to.append(synonyms_to)
        combinations.add(goal)
    
    return list(combinations), make_synonims_to