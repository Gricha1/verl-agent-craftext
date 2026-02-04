import random
from typing import Tuple, List, Callable

def generate_example_goals(num_goals: int, difficulty="EASY", existing_task_generator: Callable[[], str] = None) -> Tuple[List[str], List[str]]:
    """
    Generate a list of unique text-based goals describing block placement and task execution based on the time of day.
    
    Args:
        num_goals (int): Number of goals to generate.
        difficulty (str): Dataset difficulty level, affecting the time expressions.
        existing_task_generator (Callable[[], str]): Function to fetch existing generated tasks.
    
    Returns:
        Tuple[List[str], List[str]]: List of unique goal descriptions and their paraphrasing guidelines.
    """
    # Времена суток в зависимости от сложности
    if difficulty.upper() == "MEDIUM":
        times_of_day = ['evening', 'morning']  # Варианты для Medium
    else:
        times_of_day = ['night', 'day']  # Варианты для Easy
    
    
    combinations = set()
    make_synonyms_to = []
    
    while len(combinations) < num_goals:
        time_of_day = random.choice(times_of_day)
        

        # Формируем цель с каскадным временем
        goal = f"DURING THE {time_of_day.upper()}, PERFORM THE TASK. TEXT OF THIS TASK IS: [TASK]."
        
        if goal not in combinations:
            # Синонимы для времени суток и действия
            synonyms_to = f"WHEN GENERATING PARAPHRASES, USE VARIED SYNONYMS OR ALTERNATIVE EXPRESSIONS FOR THE TASK [TASK], " \
                          f"AND EXPLORE DIFFERENT WAYS TO EXPRESS THE TIME '{time_of_day}', SUCH AS 'IN THE EVENING', 'AT SUNRISE', " \
                          f"'DURING THE NIGHT', 'WHEN IT'S DAYTIME'. INCLUDE VARIANTS FOR THE ACTION, SUCH AS 'PLACE', 'SET', 'POSITION'."
            make_synonyms_to.append(synonyms_to)
        
        combinations.add(goal)
    
    return list(combinations), make_synonyms_to
