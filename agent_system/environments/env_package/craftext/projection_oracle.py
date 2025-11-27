"""
Projection функция для Craftext с поддержкой оракла.
Извлекает действия и вопросы из текстового вывода LLM.
"""
import re
from typing import List, Tuple, Dict, Optional
from .projection import ACTION_TO_TEXT, TEXT_TO_ACTION_ID, INVALID_ACTION_ID, AVAILABLE_ACTIONS_STR

# Шаблоны промптов с поддержкой вопросов к оракулу
CRAFTEXT_TEMPLATE_ORACLE = f"""
Your goal is to complete the following task:
**TASK:** {{task_description}}

You have taken {{step_count}} actions so far. Here is the history of your recent actions:
{{action_history}}

You are now at step {{current_step}}.
This is what you currently see:
{{current_observation}}

**ORACLE ASSISTANCE:** You can ask questions to an oracle assistant by using the <question> tag. 
The oracle can answer questions about the game mechanics and provide informatino about object locations in the game world.

**Response format:**
<question>your question here (optional)</question>
<action>your_action_here</action>

First, think about what to do next. If you need help, ask a question to the oracle. Then, choose one of the available actions.
Your available actions are: {AVAILABLE_ACTIONS_STR}
"""

CRAFTEXT_TEMPLATE_ORACLE_NO_HIS = f"""
Your goal is to complete the following task:
**TASK:** {{task_description}}

This is what you currently see:
{{current_observation}}

**ORACLE ASSISTANCE:** You can ask questions to an oracle assistant by using the <question> tag. 
The oracle can help you understand the game state, plan your actions, or answer questions about the game mechanics.

**Response format:**
<question>your question here (optional)</question>
<action>your_action_here</action>

First, think about what to do next. If you need help, ask a question to the oracle. Then, choose one of the available actions.
Your available actions are: {AVAILABLE_ACTIONS_STR}
"""

CRAFTEXT_VL_TEMPLATE_ORACLE_NO_HIS = f"""
Your goal is to complete the following task:
**TASK:** {{task_description}}

You currently see visual observation:

Picture 1: <image>

**ORACLE ASSISTANCE:** You can ask questions to an oracle assistant by using the <question> tag. 
The oracle can help you understand the game state, plan your actions, or answer questions about the game mechanics.

**Response format:**
<question>your question here (optional)</question>
<action>your_action_here</action>

First, think about what to do next. If you need help, ask a question to the oracle. Then, choose one of the available actions.
Your available actions are: {AVAILABLE_ACTIONS_STR}
"""


def craftext_projection_oracle(actions: List[str]) -> Tuple[List[int], List[int], List[Optional[str]]]:
    """
    Парсит текстовые действия и извлекает вопросы для оракла.
    
    Args:
        actions: Список текстовых действий от LLM
        
    Returns:
        Tuple of:
        - processed_actions: Список ID действий
        - valids: Список валидности действий (0 или 1)
        - questions: Список вопросов для оракла (None если вопроса нет)
    """
    processed_actions = [INVALID_ACTION_ID] * len(actions)
    valids = [0] * len(actions)
    questions = [None] * len(actions)

    for i, original_str in enumerate(actions):
        action_str_lower = original_str.lower()

        # 1. Извлекаем вопрос из тегов <question> (если есть)
        question_start = "<question>"
        question_end = "</question>"
        q_start_idx = action_str_lower.find(question_start)
        q_end_idx = action_str_lower.find(question_end)
        
        if q_start_idx != -1 and q_end_idx != -1:
            extracted_question = original_str[q_start_idx + len(question_start) : q_end_idx].strip()
            if extracted_question:
                questions[i] = extracted_question

        # 2. Извлекаем действие из тегов <action>
        action_start = "<action>"
        action_end = "</action>"
        a_start_idx = action_str_lower.find(action_start)
        a_end_idx = action_str_lower.find(action_end)

        # Если тегов <action> нет, действие невалидное
        if a_start_idx == -1 or a_end_idx == -1:
            continue

        extracted_action_text = action_str_lower[a_start_idx + len(action_start) : a_end_idx].strip()

        # 3. Преобразуем текст в числовое действие
        action_id = TEXT_TO_ACTION_ID.get(extracted_action_text, INVALID_ACTION_ID)

        # 4. Валидация
        is_known_action = action_id != INVALID_ACTION_ID
        has_no_chinese = not re.search(r"[\u4e00-\u9fff]", original_str)
        
        if is_known_action and has_no_chinese:
            valids[i] = 1
            processed_actions[i] = action_id

    return processed_actions, valids, questions

