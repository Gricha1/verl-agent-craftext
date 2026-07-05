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
Your available actions are: {AVAILABLE_ACTIONS_STR_UPPERCASE}
Write EXACTLY one of the action names above (UPPERCASE, underscores preserved) inside the <action> tag.
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
Your available actions are: {AVAILABLE_ACTIONS_STR_UPPERCASE}
Write EXACTLY one of the action names above (UPPERCASE, underscores preserved) inside the <action> tag.
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
Your available actions are: {AVAILABLE_ACTIONS_STR_UPPERCASE}
Write EXACTLY one of the action names above (UPPERCASE, underscores preserved) inside the <action> tag.
"""
    else:
        return f"""
Your goal is to complete the following task:
**TASK:** {{task_description}}

This is what you currently see:
{{current_observation}}

Choose one of the available actions and write it in the <action> tag.
Your available actions are: {AVAILABLE_ACTIONS_STR_UPPERCASE}
Write EXACTLY one of the action names above (UPPERCASE, underscores preserved) inside the <action> tag.
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
Your available actions are: {AVAILABLE_ACTIONS_STR_UPPERCASE}
Write EXACTLY one of the action names above (UPPERCASE, underscores preserved) inside the <action> tag.
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
Your available actions are: {AVAILABLE_ACTIONS_STR_UPPERCASE}
Write EXACTLY one of the action names above (UPPERCASE, underscores preserved) inside the <action> tag.
"""
    else:
        return f"""
Your goal is to complete the following task:
**TASK:** {{task_description}}

You currently see visual observation:

Picture 1: <image>

Choose one of the available actions and write it in the <action> tag.
Your available actions are: {AVAILABLE_ACTIONS_STR_UPPERCASE}
Write EXACTLY one of the action names above (UPPERCASE, underscores preserved) inside the <action> tag.
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
Your available actions are: {AVAILABLE_ACTIONS_STR_UPPERCASE}
Write EXACTLY one of the action names above (UPPERCASE, underscores preserved) inside the <action> tag.
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
Your available actions are: {AVAILABLE_ACTIONS_STR_UPPERCASE}
Write EXACTLY one of the action names above (UPPERCASE, underscores preserved) inside the <action> tag.
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


def get_single_token_action_template_no_his() -> str:
    """Prompt for one-token action output; legend lists token=ACTION for all 17 actions."""
    from .action_tokens import action_token_legend

    return f"""
Your goal is to complete the following task:
**TASK:** {{task_description}}

This is what you currently see:
{{current_observation}}

Reply with exactly ONE token — your chosen action (no tags, no explanation):
{action_token_legend()}
"""


def get_single_token_return_template_no_his() -> str:
    """Critic prompt: one token = discretized remaining return on a linear bin grid."""
    return """
Your goal is to complete the following task:
**TASK:** {task_description}

This is what you currently see:
{current_observation}

Reply with exactly ONE token — your estimate of total remaining reward from this state (no explanation):
{return_bin_legend}
"""


def get_single_token_action_vl_template_no_his() -> str:
    """VL actor prompt: one action token; observation is the game frame (<image>)."""
    from .action_tokens import action_token_legend

    return f"""
Your goal is to complete the following task:
**TASK:** {{task_description}}

You currently see visual observation:

Picture 1: <image>

Reply with exactly ONE token — your chosen action (no tags, no explanation):
{action_token_legend()}
"""


def get_single_token_return_vl_template_no_his() -> str:
    """VL value prompt: one return-bin token; observation is the game frame (<image>)."""
    return """
Your goal is to complete the following task:
**TASK:** {task_description}

You currently see visual observation:

Picture 1: <image>

Reply with exactly ONE token — your estimate of total remaining reward from this state (no explanation):
{return_bin_legend}
"""


def get_per_action_return_template_no_his() -> str:
    """Critic Q-at-first-step: one return token if the agent takes a specific action next."""
    return """
Your goal is to complete the following task:
**TASK:** {task_description}

This is what you currently see:
{current_observation}

If you take this action next: {action_name} (token {action_token})

Estimate the total remaining reward you can collect from this state until the task ends, assuming you take this action now and then play optimally.

Reply with exactly ONE token — your return estimate (no explanation):
{return_bin_legend}
"""


def format_per_action_return_prompt(
    *,
    task_description: str,
    current_observation: str,
    action_name: str,
    action_token: str,
    return_bin_legend: str,
    constraint: str = "",
) -> str:
    prompt = get_per_action_return_template_no_his().format(
        task_description=task_description or "No task",
        current_observation=current_observation or "(empty observation)",
        action_name=action_name,
        action_token=action_token,
        return_bin_legend=return_bin_legend,
    )
    if constraint:
        prompt += f"\n\n**CONSTRAINT:** {constraint}"
    return prompt


def get_all_actions_return_template_no_his() -> str:
    """One critic prompt: predict return token for every action in one reply."""
    return """
Your goal is to complete the following task:
**TASK:** {task_description}

This is what you currently see:
{current_observation}

Estimate the total remaining reward until the task ends for each possible next action (if you take that action now, then play optimally).

Actions in order (token=name):
{action_token_legend}

Reply with exactly {num_actions} consecutive tokens — one return estimate per action, in the same order as above (no spaces, no explanation):
{return_bin_legend}
"""


def format_all_actions_return_prompt(
    *,
    task_description: str,
    current_observation: str,
    return_bin_legend: str,
    num_actions: int = 17,
    constraint: str = "",
) -> str:
    from .action_tokens import action_token_legend

    prompt = get_all_actions_return_template_no_his().format(
        task_description=task_description or "No task",
        current_observation=current_observation or "(empty observation)",
        action_token_legend=action_token_legend(),
        num_actions=int(num_actions),
        return_bin_legend=return_bin_legend,
    )
    if constraint:
        prompt += f"\n\n**CONSTRAINT:** {constraint}"
    return prompt


# Невалидное действие, которое среда точно не примет.
# Оно будет использоваться, если LLM сгенерирует что-то непонятное.
INVALID_ACTION_ID = -1


def craftext_projection(actions: List[str]):
    """
    Parse LLM output into env action ids.
    Supports single-token labels (1..9, a..h), <action> tags, and bare action names.
    """
    from .action_tokens import parse_single_token_action

    processed_actions = [INVALID_ACTION_ID] * len(actions)
    valids = [0] * len(actions)

    for i, original_str in enumerate(actions):
        single_id = parse_single_token_action(original_str)
        if single_id != INVALID_ACTION_ID and not re.search(r"[\u4e00-\u9fff]", original_str):
            valids[i] = 1
            processed_actions[i] = single_id
            continue

        action_str_lower = original_str.lower()

        # 1. Извлекаем текстовое действие из тегов.
        # Поддерживаем несколько вариантов, т.к. разные chat templates/модели иногда оборачивают ответ в <response>.
        tag_pairs = [
            ("<action>", "</action>"),       # default_template
            ("<answer>", "</answer>"),       # extended_template
            ("<response>", "</response>"),   # некоторые chat-шаблоны
        ]
        extracted_action_text_raw = None
        for start_tag, end_tag in tag_pairs:
            start_idx = action_str_lower.find(start_tag)
            end_idx = action_str_lower.find(end_tag)
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                extracted_action_text_raw = original_str[start_idx + len(start_tag) : end_idx].strip()
                break

        # Если тегов нет, пробуем принять "голое" действие (например: UP / PLACE_STONE / place stone)
        if extracted_action_text_raw is None:
            extracted_action_text_raw = original_str.strip()
            if not extracted_action_text_raw:
                continue

        extracted_action_text_lower = extracted_action_text_raw.lower()

        # Normalize to ACTION_TO_TEXT style: UPPERCASE with underscores (PLACE_STONE)
        action_norm = extracted_action_text_raw.strip()
        action_norm = action_norm.replace("-", "_")
        action_norm = re.sub(r"\s+", "_", action_norm)
        action_norm = re.sub(r"_+", "_", action_norm)
        action_norm = action_norm.upper()

        # 2. Преобразуем текст в числовое действие, используя словарь
        if action_norm in ACTION_TO_TEXT:
            action_id = int(ACTION_TO_TEXT.index(action_norm))
        else:
            # Backward compatible: accept "place stone" / "place_stone" / mixed case
            compat = extracted_action_text_lower.replace("_", " ").strip()
            action_id = TEXT_TO_ACTION_ID.get(compat, INVALID_ACTION_ID)

        # 3. Финальная валидация
        is_known_action = action_id != INVALID_ACTION_ID
        has_no_chinese = not re.search(r"[\u4e00-\u9fff]", original_str)
        
        # Для валидности достаточно, чтобы было найдено известное действие в тегах
        if is_known_action and has_no_chinese:
            valids[i] = 1
            processed_actions[i] = action_id
        # Если хотя бы одно условие не выполнено, действие остается INVALID_ACTION_ID, а valid - 0

    return processed_actions, valids
