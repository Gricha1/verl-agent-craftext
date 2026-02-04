import random
from typing import Tuple, List

def generate_example_goals(num_goals: int, difficulty="EASY") -> Tuple[List[str], List[str]]:
    """
    Generate a list of unique text-based goals describing block placement based on the time of day.
    
    Args:
        num_goals (int): Number of goals to generate.
        difficulty (str): Dataset difficulty level, affecting the time expressions.
    
    Returns:
        Tuple[List[str], List[str]]: List of unique goal descriptions and their paraphrasing guidelines.
    """
    
    block_types = {
        'EASY': ['STONE', 'CRAFTING_TABLE', 'PLANT', 'FURNACE'],
        'MEDIUM': ['STONE', 'CRAFTING_TABLE', 'PLANT', 'FURNACE', 'ENCHANTMENT_TABLE_FIRE', 'ENCHANTMENT_TABLE_ICE', 'TORCH']
    }
    
    # Времена суток в зависимости от сложности
    if difficulty.upper() == "MEDIUM":
        times_of_day = ['evening', 'morning']  # Варианты для Medium
    else:
        times_of_day = ['night', 'day']  # Варианты для Easy
    
    tasks = ['place_block']  # Основное действие – поставить блок
    
    combinations = set()
    make_synonyms_to = []
    blocks = block_types[difficulty.upper()]
    
    while len(combinations) < num_goals:
        time_of_day = random.choice(times_of_day)
        block = random.choice(blocks)
        task = random.choice(tasks)
        
        # Формируем цель
        goal = f"PLACE A {block} DURING THE {time_of_day.upper()}"
        
        if goal not in combinations:
            # Синонимы для времени суток и действия
            synonyms_to = f"WHEN GENERATING PARAPHRASES, USE VARIED SYNONYMS OR ALTERNATIVE EXPRESSIONS FOR THE TASK '{task.upper()}', " \
                          f"AND EXPLORE DIFFERENT WAYS TO EXPRESS THE TIME '{time_of_day.upper()}', SUCH AS 'IN THE EVENING', 'AT SUNRISE', " \
                          f"'DURING THE NIGHT', 'WHEN IT'S DAYTIME'. INCLUDE VARIANTS FOR THE ACTION, SUCH AS 'PLACE', 'SET', 'POSITION'."
            make_synonyms_to.append(synonyms_to)
        
        combinations.add(goal)
        print(f"len(combinations): {len(combinations)}, goal: {goal}")
    
    return list(combinations), make_synonyms_to
