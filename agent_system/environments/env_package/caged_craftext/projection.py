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

# Действия в формате для нового шаблона (заглавными буквами)
AVAILABLE_ACTIONS_STR_UPPERCASE = ", ".join([text for text in ACTION_TO_TEXT])


# <--- Функции для получения шаблонов с reasoning или без
def get_craftext_template(enable_reasoning: bool = True) -> str:
    """Возвращает шаблон промпта с reasoning или без."""
    if enable_reasoning:
        return f"""
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
    else:
        return f"""
Your goal is to complete the following task:
**TASK:** {{task_description}}

You have taken {{step_count}} actions so far. Here is the history of your recent actions:
{{action_history}}

You are now at step {{current_step}}.
This is what you currently see:
{{current_observation}}

Choose one of the available actions and write it in the <action> tag.
Your available actions are: {AVAILABLE_ACTIONS_STR}
"""


def get_craftext_template_no_his(enable_reasoning: bool = True) -> str:
    """Возвращает шаблон промпта без истории с reasoning или без."""
    if enable_reasoning:
        return f"""
Your goal is to complete the following task:
**TASK:** {{task_description}}

This is what you currently see:
{{current_observation}}

First, think about what to do next. Then, choose one of the available actions and write it in the <action> tag.
Your available actions are: {AVAILABLE_ACTIONS_STR}
"""
    else:
        return f"""
Your goal is to complete the following task:
**TASK:** {{task_description}}

This is what you currently see:
{{current_observation}}

Choose one of the available actions and write it in the <action> tag.
Your available actions are: {AVAILABLE_ACTIONS_STR}
"""


# Для обратной совместимости - используем reasoning по умолчанию
CRAFTEXT_TEMPLATE = get_craftext_template(enable_reasoning=True)
CRAFTEXT_TEMPLATE_NO_HIS = get_craftext_template_no_his(enable_reasoning=True)


CRAFTEXT_TEMPLATE_NO_HIS = f"""
Your goal is to complete the following task:
**TASK:** {{task_description}}

This is what you currently see:
{{current_observation}}

Choose one of the available actions and write it in the <action> tag.
Your available actions are: {AVAILABLE_ACTIONS_STR}
"""

def get_craftext_vl_template_no_his(enable_reasoning: bool = True) -> str:
    """Возвращает визуальный шаблон промпта без истории с reasoning или без."""
    if enable_reasoning:
        return f"""
Your goal is to complete the following task:
**TASK:** {{task_description}}

You currently see visual observation:

Picture 1: <image>

First, think about what to do next. Then, choose one of the available actions and write it in the <action> tag.
Your available actions are: {AVAILABLE_ACTIONS_STR}
"""
    else:
        return f"""
Your goal is to complete the following task:
**TASK:** {{task_description}}

You currently see visual observation:

Picture 1: <image>

Choose one of the available actions and write it in the <action> tag.
Your available actions are: {AVAILABLE_ACTIONS_STR}
"""

# Для обратной совместимости
CRAFTEXT_VL_TEMPLATE_NO_HIS = get_craftext_vl_template_no_his(enable_reasoning=True)

# Subtask-aware templates for subtask-based GiGPO
VALID_SUBTASKS_STR = "collect_wood, place_table, make_wood_pickaxe, make_wood_sword"

def get_craftext_subtask_template_no_his(enable_reasoning: bool = True) -> str:
    """Возвращает subtask шаблон промпта без истории с reasoning или без."""
    reasoning_text = "First, think about what subtask you're working on, then " if enable_reasoning else ""
    return f"""
Your goal is to complete the following task:
**TASK:** {{task_description}}

This is what you currently see:
{{current_observation}}

**IMPORTANT:** For each action, you must:
1. First, predict which subtask you are trying to complete at this moment
2. Then, choose and execute an action

**Valid subtasks:** {VALID_SUBTASKS_STR}

**Response format:**
<subtask>subtask_name</subtask>
<action>your_action_here</action>

{reasoning_text}Choose one of the available actions and write it in the <action> tag.
Your available actions are: {AVAILABLE_ACTIONS_STR}
"""

# Для обратной совместимости
CRAFTEXT_SUBTASK_TEMPLATE_NO_HIS = get_craftext_subtask_template_no_his(enable_reasoning=True)

def get_craftext_subtask_vl_template_no_his(enable_reasoning: bool = True) -> str:
    """Возвращает визуальный subtask шаблон промпта без истории с reasoning или без."""
    reasoning_text = "First, think about what subtask you're working on, then " if enable_reasoning else ""
    return f"""
Your goal is to complete the following task:
**TASK:** {{task_description}}

You currently see visual observation:

Picture 1: <image>

**IMPORTANT:** For each action, you must:
1. First, predict which subtask you are trying to complete at this moment
2. Then, choose and execute an action

**Valid subtasks:** {VALID_SUBTASKS_STR}

**Response format:**
<subtask>subtask_name</subtask>
<action>your_action_here</action>

{reasoning_text}Choose one of the available actions and write it in the <action> tag.
Your available actions are: {AVAILABLE_ACTIONS_STR}
"""

# Для обратной совместимости
CRAFTEXT_SUBTASK_VL_TEMPLATE_NO_HIS = get_craftext_subtask_vl_template_no_his(enable_reasoning=True)

# Расширенный шаблон с тегами <reasoning> и <answer>
def get_craftext_extended_template_no_his() -> str:
    """Возвращает расширенный шаблон промпта с тегами <reasoning> и <answer>."""
    # Используем точный список действий, указанный пользователем
    actions_list = "DO, DOWN, LEFT, MAKE_IRON_PICKAXE, MAKE_IRON_SWORD, MAKE_STONE_PICKAXE, MAKE_WOOD_PICKAXE, MAKE_WOOD_SWORD, NOOP, PLACE_FURNACE, PLACE_STONE, PLACE_TABLE, RIGHT, SKIP_SUBTASK, SLEEP, UP"
    return f"""
You are an expert agent analyzing a visual observation in a crafting environment. To answer the question correctly, you need to:
1. Carefully examine the observation
2. Reason step-by-step about what you observe
3. Provide your final answer

Important: Your reasoning process MUST be enclosed within <reasoning> </reasoning> tags. Your final answer MUST be enclosed within <answer> </answer> tags.

Observation:

Your goal is to complete the following task:
**TASK:** {{task_description}}

This is what you currently see:
{{current_observation}}

Your available actions are: {actions_list}
"""

# Для обратной совместимости
CRAFTEXT_EXTENDED_TEMPLATE_NO_HIS = get_craftext_extended_template_no_his()

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
