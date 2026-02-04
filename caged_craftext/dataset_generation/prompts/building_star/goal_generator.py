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
    block_types = {
        'EASY': ['STONE', 'CRAFTING_TABLE', 'PLANT', 'FURNACE'],
        'MEDIUM': ['STONE', 'CRAFTING_TABLE', 'PLANT', 'FURNACE', 'ENCHANTMENT_TABLE_FIRE', 'ENCHANTMENT_TABLE_ICE', 'TORCH']
    }
        
    sizes = [3, 5, 7]  # Размеры креста или звезды
    cross_types = ['direct', 'diagonal', 'combined']  # Типы крестов

    combinations: set[str] = set()
    make_synonims_to = []
    

    while len(combinations) < num_goals:
        # Случайно выбираем сложность и тип блока
        difficulty = random.choice(list(block_types.keys()))
        block = random.choice(block_types[difficulty])
        size = random.choice(sizes)
        
        if size > 3:
            cross_type = random.choice(cross_types)  # Выбираем тип креста
        else:
            cross_type = random.choice(cross_types[:-1])
            
        # Формулируем цель в виде строки
        goal = f"CROSS OF {str(block).upper()} WITH SIDE SIZE {str(size).upper()} AND {cross_type.upper()} SHAPE"
        
        if goal not in combinations:
            # Создаем инструкции с варьированием синонимов для блока, размера и формы
            synonyms_to = f"WHEN GENERATING PARAPHRASES, USE VARIED SYNONYMS OR ALTERNATIVE EXPRESSIONS FOR THE TERM '{str(block).upper()}', " \
                        f"AND EXPLORE DIFFERENT WAYS TO EXPRESS THE SIZE '{str(size).upper()}', SUCH AS NUMBERS (E.G., '7'), " \
                        f"WORDS (E.G., 'SEVEN'), OR PHRASES (E.G., 'A TOTAL OF SEVEN UNITS'). ALSO, INCLUDE VARIANTS OF THE " \
                        f"CROSS SHAPE: {cross_type.upper()} (DIRECT, DIAGONAL, OR COMBINED)."
            make_synonims_to.append(synonyms_to)

        combinations.add(goal)

    return list(combinations), make_synonims_to