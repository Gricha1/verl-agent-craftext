"""
Модуль для сбора датасета с наблюдениями, вопросами и правильными ответами.
"""
import json
import base64
import io
from tqdm import tqdm
from PIL import Image
import numpy as np
import ray
from craftax.craftax_classic.renderer import render_craftax_pixels as render_classic
from craftax.craftax.constants import BLOCK_PIXEL_SIZE_HUMAN

from craftext_answer_functions import (
    answer_existence_question,
    answer_distance_question,
    answer_coordinates_question,
    answer_direction_question,
)


# Шаблоны вопросов
QUESTION_TEMPLATES = {
    "existence": "Question: Is there any {block_type} visible in the current observation?\n\nAnswer with ONLY 'Yes' or 'No'.\nAnswer:",
    "distance": "Question: Calculate the Manhattan distance (|x| + |y|) from the agent (0, 0) to the NEAREST {block_type}.\nIf no {block_type} is visible, answer '-1'.\n\nAnswer with ONLY the number (integer).\nAnswer:",
    "coordinates": "Question: Identify the coordinates (x, y) of the NEAREST {block_type}.\nIf multiple exist, choose the one with the smallest distance to (0,0).\nIf none exist, answer 'None'.\n\nAnswer in strict format: x, y\nAnswer:",
    "direction": "Target: nearest {block_type}\nAgent position: (0, 0)\n\nQuestion: In which direction is the nearest {block_type} located relative to the agent?\nOptions: [left, right, up, down, up-left, up-right, down-left, down-right, none]\n\nChoose 'none' if the object is not visible.\nAnswer with ONLY one option from the list.\nAnswer:"
}

# Функции для получения правильных ответов
ANSWER_FUNCTIONS = {
    "existence": answer_existence_question,
    "distance": answer_distance_question,
    "coordinates": answer_coordinates_question,
    "direction": answer_direction_question,
}


def collect_dataset(
    envs,
    num_samples=100,
    block_types=None,
    question_types=None,
    save_path="craftext_perception_dataset.jsonl"
):
    """
    Собирает датасет из наблюдений, вопросов и правильных ответов.
    
    Args:
        envs: среда Craftext
        num_samples: количество сэмплов для сбора
        block_types: типы блоков для вопросов (по умолчанию: ["tree", "stone", "water", "sand", "iron", "coal", "zombie"])
        question_types: типы вопросов (по умолчанию: ["existence", "distance", "coordinates", "direction"])
        save_path: путь для сохранения датасета
    
    Returns:
        Список записей датасета
    """
    if block_types is None:
        block_types = ["tree", "stone", "water", "sand", "iron", "coal", "zombie"]
    if question_types is None:
        question_types = ["existence", "distance", "coordinates", "direction"]
    
    dataset = []
    
    print(f"Сбор датасета: {num_samples} сэмплов...")
    
    for sample_idx in tqdm(range(num_samples)):
        # Сбрасываем среду
        obs, info = envs.reset({})
        state = ray.get(envs.envs._workers[0].get_state.remote())
        craftext_state = state.env_state
        
        # Получаем текстовое наблюдение
        text_observation = obs['anchor'][0]
        
        # Получаем визуальное наблюдение
        pixels = render_classic(craftext_state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN)
        pixels_for_plot = np.array(pixels).astype(np.uint8)
        
        # Конвертируем в PIL Image для сохранения
        image = Image.fromarray(pixels_for_plot)
        
        # Сохраняем изображение в байты
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        image_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
        
        # Генерируем вопросы для каждого типа блока и каждого типа вопроса
        for block_type in block_types:
            for question_type in question_types:
                # Формируем вопрос
                question = QUESTION_TEMPLATES[question_type].format(block_type=block_type)
                
                # Получаем правильный ответ
                answer_func = ANSWER_FUNCTIONS[question_type]
                correct_answer = answer_func(craftext_state, block_type)
                
                # Форматируем правильный ответ для сравнения
                if question_type == "existence":
                    expected_answer = str(correct_answer)
                elif question_type == "distance":
                    expected_answer = str(correct_answer)
                elif question_type == "coordinates":
                    # Для координат - список строк, но в вопросе ожидается один ответ
                    if isinstance(correct_answer, list):
                        expected_answer = correct_answer[0] if len(correct_answer) > 0 else "None"
                    else:
                        expected_answer = str(correct_answer)
                elif question_type == "direction":
                    # Для направлений - список, но в вопросе ожидается один ответ
                    if isinstance(correct_answer, list):
                        expected_answer = correct_answer[0] if len(correct_answer) > 0 else "none"
                    else:
                        expected_answer = str(correct_answer)
                
                # Создаем запись датасета
                dataset_entry = {
                    "sample_id": sample_idx,
                    "block_type": block_type,
                    "question_type": question_type,
                    "text_observation": text_observation,
                    "image_base64": image_base64,
                    "question": question,
                    "correct_answer": correct_answer,  # Сохраняем полный ответ (может быть списком)
                    "expected_answer": expected_answer,  # Ожидаемый ответ для сравнения
                }
                
                dataset.append(dataset_entry)
    
    # Сохраняем датасет
    print(f"\nСохранение датасета в {save_path}...")
    with open(save_path, 'w') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Датасет сохранен: {len(dataset)} записей")
    return dataset

