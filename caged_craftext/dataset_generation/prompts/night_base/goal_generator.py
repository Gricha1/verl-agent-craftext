import random
from typing import Tuple, List

def generate_example_goals(num_goals: int, difficulty="EASY") -> Tuple[list[str], list[str]]:
    """
    Generate a list of unique text-based goals describing block placement patterns.
    
    Args:
        num_goals (int): Number of goals to generate.
    
    Returns:
        list[str]: List of unique goal descriptions.
    """
    # Времена суток
    times_of_day = ['night', 'day']  # Проверяем день и ночь
    tasks = ['build_line', 'build_square', 'conditional_placing', 'localization_placing']  # Список конкретных заданий

    combinations = set()
    make_synonims_to = []

    num_goals = 10  # Например, генерируем 10 целей

    while len(combinations) < num_goals:
        # Случайный выбор времени суток и задания
        time_of_day = random.choice(times_of_day)
        task = random.choice(tasks)

        # Формулируем цель в виде строки
        goal = f"PERFORM THE TASK '{task.upper()}' DURING THE {time_of_day.upper()}"
        
        if goal not in combinations:
            # Синонимы для времени суток и действия
            synonyms_to = f"WHEN GENERATING PARAPHRASES, USE VARIED SYNONYMS OR ALTERNATIVE EXPRESSIONS FOR THE TASK '{task}', " \
                        f"AND EXPLORE DIFFERENT WAYS TO EXPRESS THE TIME '{time_of_day}', SUCH AS 'AT NIGHT', 'IN THE DARK', " \
                        f"OR 'DURING THE DAY', 'WHEN THE SUN IS OUT'. INCLUDE VARIANTS FOR THE ACTION, SUCH AS 'PERFORM', 'EXECUTE', 'CARRY OUT'."
            make_synonims_to.append(synonyms_to)

        combinations.add(goal)


    return list(combinations), make_synonims_to