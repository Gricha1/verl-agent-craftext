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
    # Времена суток
    times_of_day = ['night', 'day']  # Проверяем день и ночь

    # Типы укрытий (shelters)
    shelters = ['house', 'cave', 'hut', 'tent', 'fortress', 'tower']  # Список возможных укрытий

    combinations = set()
    make_synonims_to = []

    num_goals = 10  # Например, генерируем 10 целей

    while len(combinations) < num_goals:
        # Случайный выбор времени суток и укрытия
        time_of_day = random.choice(times_of_day)
        shelter = random.choice(shelters)  # Выбираем случайное укрытие

        # Формулируем цель в виде строки
        goal = f"FIND THE SHELTER DURING THE {time_of_day.upper()}"
        
        if goal not in combinations:
            # Синонимы для укрытия и времени суток
            synonyms_to = f"WHEN GENERATING PARAPHRASES, USE VARIED SYNONYMS OR ALTERNATIVE EXPRESSIONS FOR THE TERM 'SHELTER', " \
                        f"INCLUDING {', '.join(shelters).upper()}. ALSO EXPLORE DIFFERENT WAYS TO EXPRESS THE TIME '{time_of_day}', " \
                        f"SUCH AS 'AT NIGHT', 'IN THE DARK', OR 'WHEN THE SUN SETS'. INCLUDE VARIANTS FOR THE ACTION, SUCH AS 'FIND', " \
                        f"'LOCATE', 'DISCOVER', 'SEARCH FOR'."
            make_synonims_to.append(synonyms_to)

        combinations.add(goal)

    return list(combinations), make_synonims_to