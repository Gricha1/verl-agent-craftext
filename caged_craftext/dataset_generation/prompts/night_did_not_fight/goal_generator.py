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
    times_of_day = ['night', 'day'] 

    mobs = ['cow', 'skeleton', 'zombie']


    combinations : set[str] = set()
    make_synonims_to = []


    while len(combinations) < num_goals:
        time_of_day = random.choice(times_of_day)
        mob = random.choice(mobs)

        goal = f"DO NOT FIGHT {mob.upper()} DURING THE {time_of_day.upper()}"
        
        if goal not in combinations:
            synonyms_to = f"WHEN GENERATING PARAPHRASES, USE VARIED SYNONYMS OR ALTERNATIVE EXPRESSIONS FOR THE ACTION 'FIGHT {mob.upper()}', " \
                        f"AND EXPLORE DIFFERENT WAYS TO EXPRESS THE TIME '{time_of_day}', SUCH AS 'AT NIGHT', 'IN THE DARK', " \
                        f"OR 'DURING THE NIGHT'. INCLUDE VARIANTS FOR THE ACTION, SUCH AS 'AVOID FIGHTING WITH {mob.upper()}', " \
                        f"'NOT ENGAGE WITH {mob.upper()}', 'REFRAIN FROM FIGHTING {mob.upper()}', 'STAY AWAY FROM {mob.upper()}'."
            make_synonims_to.append(synonyms_to)

        combinations.add(goal)


    return list(combinations), make_synonims_to