import re
from typing import List, Tuple

# Ваш список действий из среды
ACTION_TO_TEXT: Tuple[str, ...] = (
    "NOOP",
    "LEFT",
    "RIGHT",
    "UP",
    "DOWN",
    "DO",
    "SLEEP",
    "PLACE_STONE",
    "PLACE_TABLE",
    "PLACE_FURNACE",
    "PLACE_PLANT",
    "MAKE_WOOD_PICKAXE",
    "MAKE_STONE_PICKAXE",
    "MAKE_IRON_PICKAXE",
    "MAKE_WOOD_SWORD",
    "MAKE_STONE_SWORD",
    "MAKE_IRON_SWORD",
)


# <--- НОВОЕ: Создаем словарь "натуральный текст -> ID действия"
# Это делается автоматически из вашего списка ACTION_TO_TEXT
# 'PLACE_STONE' -> 'place stone'
TEXT_TO_ACTION_ID = {text.lower().replace("_", " "): i for i, text in enumerate(ACTION_TO_TEXT)}


AVAILABLE_ACTIONS_STR = ", ".join([text.lower().replace("_", " ") for text in ACTION_TO_TEXT])


# <--- НОВОЕ: Обновляем шаблоны промптов
CRAFTEXT_TEMPLATE = f"""
Your goal is to complete the following task:
**TASK:** {{task_description}}

You have taken {{step_count}} actions so far. Here is the history of your recent actions:
{{action_history}}

You are now at step {{current_step}}.
This is what you currently see:
{{current_observation}}

First, think about what to do next. Then, choose one of the available actions and write it in the <action> tag.
Your available actions are: {AVAILABLE_ACTIONS_STR}
"""


CRAFTEXT_TEMPLATE_NO_HIS = f"""
Your goal is to complete the following task:
**TASK:** {{task_description}}

This is what you currently see:
{{current_observation}}

First, think about what to do next. Then, choose one of the available actions and write it in the <action> tag.
Your available actions are: {AVAILABLE_ACTIONS_STR}
"""

# Невалидное действие, которое среда точно не примет.
# Оно будет использоваться, если LLM сгенерирует что-то непонятное.
INVALID_ACTION_ID = -1


def craftext_projection(actions: List[str]):
    """
    ИСПРАВЛЕННАЯ ВЕРСИЯ: Парсит, валидирует и преобразует текстовый вывод LLM.
    Проверка на <think> сделана опциональной.
    """
    processed_actions = [INVALID_ACTION_ID] * len(actions)
    valids = [0] * len(actions) # 0 - невалидное, 1 - валидное

    for i, original_str in enumerate(actions):
        action_str_lower = original_str.lower()

        # 1. Извлекаем текстовое действие из тегов <action>
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = action_str_lower.find(start_tag)
        end_idx = action_str_lower.find(end_tag)

        # Если тегов <action> нет, действие точно невалидное
        if start_idx == -1 or end_idx == -1:
            continue

        extracted_action_text = action_str_lower[start_idx + len(start_tag) : end_idx].strip()

        # 2. Преобразуем текст в числовое действие, используя словарь
        action_id = TEXT_TO_ACTION_ID.get(extracted_action_text, INVALID_ACTION_ID)

        # 3. Финальная валидация
        is_known_action = action_id != INVALID_ACTION_ID
        has_no_chinese = not re.search(r"[\u4e00-\u9fff]", original_str)
        
        # <--- ГЛАВНОЕ ИЗМЕНЕНИЕ: Убрали обязательную проверку на has_think ---
        # Теперь для валидности достаточно, чтобы было найдено известное действие в тегах
        if is_known_action and has_no_chinese:
            valids[i] = 1
            processed_actions[i] = action_id
        # Если хотя бы одно условие не выполнено, действие остается INVALID_ACTION_ID, а valid - 0

    return processed_actions, valids
