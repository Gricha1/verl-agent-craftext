"""
Модуль для оценки моделей на датасете восприятия Craftext.
"""
import json
import base64
import io
import re
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def load_dataset(dataset_path):
    """
    Загружает датасет из JSONL файла.
    
    Args:
        dataset_path: путь к JSONL файлу с датасетом
    
    Returns:
        Список записей датасета
    """
    dataset = []
    with open(dataset_path, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


def normalize_answer(answer, question_type):
    """
    Нормализует ответ модели для сравнения с правильным ответом.
    
    Args:
        answer: ответ модели (строка или другой тип)
        question_type: тип вопроса ("existence", "distance", "coordinates", "direction")
    
    Returns:
        Нормализованный ответ
    """
    if isinstance(answer, str):
        answer = answer.strip().lower()
    
    if question_type == "existence":
        # Нормализуем Yes/No
        if "yes" in answer:
            return "yes"
        elif "no" in answer:
            return "no"
        return answer
    
    elif question_type == "distance":
        # Извлекаем число
        numbers = re.findall(r'-?\d+', answer)
        if numbers:
            return int(numbers[0])
        return answer
    
    elif question_type == "coordinates":
        # Извлекаем координаты
        # Ищем паттерн типа "x, y" или "(x, y)"
        coords = re.findall(r'\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?', answer)
        if coords:
            return f"{coords[0][0]}, {coords[0][1]}"
        return answer.lower()
    
    elif question_type == "direction":
        # Нормализуем направление
        answer_lower = answer.lower() if isinstance(answer, str) else str(answer).lower()
        valid_directions = ["left", "right", "up", "down", "up-left", "up-right", "down-left", "down-right", "none"]
        for direction in valid_directions:
            if direction in answer_lower:
                return direction
        return answer_lower
    
    return answer


def _check_answer_correctness(normalized_answer, correct_answer, expected_answer, question_type):
    """
    Проверяет, является ли ответ правильным.
    
    Args:
        normalized_answer: нормализованный ответ модели
        correct_answer: правильный ответ (может быть списком)
        expected_answer: ожидаемый ответ для сравнения
        question_type: тип вопроса
    
    Returns:
        True если ответ правильный, False иначе
    """
    is_correct = False
    
    if question_type == "existence":
        expected_normalized = str(expected_answer).lower()
        is_correct = normalized_answer == expected_normalized
    
    elif question_type == "distance":
        expected_normalized = int(expected_answer) if isinstance(expected_answer, (int, str)) else expected_answer
        try:
            model_distance = int(normalized_answer) if isinstance(normalized_answer, (int, str)) else normalized_answer
            is_correct = model_distance == expected_normalized
        except:
            is_correct = False
    
    elif question_type == "coordinates":
        # Для координат проверяем, есть ли правильный ответ в списке
        if isinstance(correct_answer, list):
            # Нормализуем все правильные ответы для сравнения
            normalized_correct_answers = []
            for correct_coord in correct_answer:
                normalized_correct = normalize_answer(str(correct_coord), question_type)
                normalized_correct_answers.append(normalized_correct)
            is_correct = normalized_answer in normalized_correct_answers
        else:
            # Если не список, сравниваем с одним ожидаемым ответом
            expected_normalized = normalize_answer(str(expected_answer), question_type)
            is_correct = normalized_answer == expected_normalized
    
    elif question_type == "direction":
        # Для направлений проверяем, есть ли правильный ответ в списке
        if isinstance(correct_answer, list):
            # Нормализуем все правильные ответы для сравнения
            normalized_correct_answers = []
            for correct_dir in correct_answer:
                normalized_correct = normalize_answer(str(correct_dir), question_type)
                normalized_correct_answers.append(normalized_correct)
            is_correct = normalized_answer in normalized_correct_answers
        else:
            # Если не список, сравниваем с одним ожидаемым ответом
            expected_normalized = normalize_answer(str(expected_answer), question_type)
            is_correct = normalized_answer == expected_normalized
    
    return is_correct


def _display_verbose_info(entry, entry_idx, total, model_answer, normalized_answer, 
                          correct_answer, expected_answer, is_correct):
    """
    Отображает информацию в verbose режиме.
    
    Args:
        entry: запись датасета
        entry_idx: индекс записи
        total: общее количество записей
        model_answer: ответ модели
        normalized_answer: нормализованный ответ
        correct_answer: правильный ответ
        expected_answer: ожидаемый ответ
        is_correct: является ли ответ правильным
    """
    # Декодируем изображение
    image_base64 = entry['image_base64']
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes))
    image_array = np.array(image)
    
    # Создаем фигуру для отображения
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    axes.imshow(image_array)
    axes.axis('off')
    axes.set_title(
        f"Sample {entry_idx + 1}/{total} - {entry['block_type']} - {entry['question_type']}", 
        fontsize=12, 
        fontweight='bold'
    )
    
    # Формируем текст для отображения
    info_text = f"""
Question:
{entry['question']}

Model Answer: {model_answer}
Normalized: {normalized_answer}

Correct Answer: {correct_answer}
Expected: {expected_answer}

Status: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}
"""
    
    # Добавляем текст под изображением
    fig.text(
        0.5, 0.02, info_text, 
        ha='center', va='bottom', 
        fontsize=10, family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    plt.show()
    
    # Также выводим в консоль для удобства
    print(f"\n{'='*80}")
    print(f"Entry {entry_idx + 1}/{total}")
    print(f"Block Type: {entry['block_type']} | Question Type: {entry['question_type']}")
    print(f"Question: {entry['question']}")
    print(f"Model Answer: {model_answer}")
    print(f"Normalized Answer: {normalized_answer}")
    print(f"Correct Answer: {correct_answer}")
    print(f"Expected Answer: {expected_answer}")
    print(f"Status: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
    print(f"{'='*80}\n")


def evaluate_model_on_dataset(
    model,
    dataset,
    use_text_observation=True,
    verbose=False,
    batch_size=32,
    verbose_max_samples=10
):
    """
    Оценивает модель на датасете.
    
    Args:
        model: SimpleAgent (поддерживает как LLM, так и VLM модели)
        dataset: датасет для оценки
        use_text_observation: использовать ли текстовое наблюдение в промпте
        verbose: если True, отображает наблюдение (изображение), вопрос, ответ модели и правильный ответ
        batch_size: размер батча для обработки (только для LLM, VLM обрабатывается по одному)
        verbose_max_samples: максимальное количество примеров для отображения в verbose режиме (по умолчанию 10)
    
    Returns:
        Словарь с метриками:
        - overall_accuracy: общая точность
        - correct: количество правильных ответов
        - total: общее количество ответов
        - metrics_by_type: метрики по типам вопросов
        - results: детальные результаты для каждого примера
    """
    results = []
    correct = 0
    total = 0
    
    # Определяем, является ли модель VLM
    is_vlm = getattr(model, 'is_vlm', False)
    
    print(f"Оценка {'VLM' if is_vlm else 'LLM'} модели на {len(dataset)} примерах...")
    if not is_vlm:
        print(f"Используется батчинг с размером батча: {batch_size}")
    if verbose:
        print(f"Verbose режим: будет отображено максимум {verbose_max_samples} примеров")
    
    # Для LLM моделей используем батчинг
    if not is_vlm:
        # Собираем все промпты
        prompts = []
        entries_info = []  # Сохраняем информацию о каждом entry для последующей обработки
        
        for entry in dataset:
            question = entry['question']
            text_obs = entry['text_observation']
            
            # Формируем промпт
            if use_text_observation:
                prompt = f"{text_obs}\n\n{question}"
            else:
                prompt = question
            
            prompts.append(prompt)
            entries_info.append(entry)
        
        # Обрабатываем батчами
        model_answers = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="Обработка батчей"):
            batch_prompts = prompts[i:i+batch_size]
            try:
                # Формируем промпты для chat template
                batch_chats = [[{"role": "user", "content": p}] for p in batch_prompts]
                batch_formatted = [
                    model.tokenizer.apply_chat_template(
                        chat,
                        add_generation_prompt=True,
                        tokenize=False
                    ) for chat in batch_chats
                ]
                
                # Генерируем батч
                if model.use_lora and model.lora_path:
                    try:
                        from vllm.lora.request import LoRARequest
                        lora_request = LoRARequest(
                            lora_name=model.lora_name,
                            lora_int_id=1,
                            lora_path=model.lora_path,
                        )
                        batch_outputs = model.llm.generate(
                            batch_formatted,
                            model.sampling_params,
                            lora_request=lora_request
                        )
                    except Exception as e:
                        batch_outputs = model.llm.generate(batch_formatted, model.sampling_params)
                else:
                    batch_outputs = model.llm.generate(batch_formatted, model.sampling_params)
                
                # Извлекаем ответы
                batch_answers = [output.outputs[0].text.strip() for output in batch_outputs]
                model_answers.extend(batch_answers)
            except Exception as e:
                print(f"Ошибка при генерации батча {i//batch_size}: {e}")
                # В случае ошибки добавляем пустые ответы
                model_answers.extend([""] * len(batch_prompts))
        
        # Теперь обрабатываем результаты
        for entry_idx, (entry, model_answer) in enumerate(zip(entries_info, model_answers)):
            question_type = entry['question_type']
            correct_answer = entry['correct_answer']
            expected_answer = entry['expected_answer']
            
            # Нормализуем ответ
            normalized_answer = normalize_answer(model_answer, question_type)
            
            # Проверяем правильность
            is_correct = _check_answer_correctness(
                normalized_answer, correct_answer, expected_answer, question_type
            )
            
            if is_correct:
                correct += 1
            total += 1
            
            results.append({
                "sample_id": entry['sample_id'],
                "block_type": entry['block_type'],
                "question_type": question_type,
                "expected_answer": expected_answer,
                "model_answer": model_answer,
                "normalized_answer": normalized_answer,
                "is_correct": is_correct
            })
            
            # Отображение в verbose режиме (только первые verbose_max_samples примеров)
            if verbose and entry_idx < verbose_max_samples:
                _display_verbose_info(
                    entry, entry_idx, len(dataset), model_answer, normalized_answer,
                    correct_answer, expected_answer, is_correct
                )
    
    else:
        # Для VLM моделей обрабатываем по одному (не поддерживают батчинг в текущей реализации)
        for entry_idx, entry in enumerate(tqdm(dataset)):
            question = entry['question']
            text_obs = entry['text_observation']
            question_type = entry['question_type']
            correct_answer = entry['correct_answer']
            expected_answer = entry['expected_answer']
            
            # Формируем промпт
            if use_text_observation:
                prompt = f"{text_obs}\n\n{question}"
            else:
                prompt = question
            
            # Получаем ответ модели
            try:
                # Для VLM нужно декодировать изображение и передать его
                image_base64 = entry['image_base64']
                image_bytes = base64.b64decode(image_base64)
                image = Image.open(io.BytesIO(image_bytes))
                model_answer = model.predict(prompt, image=image)
            except Exception as e:
                print(f"Ошибка при генерации: {e}")
                model_answer = ""
            
            # Нормализуем ответ
            normalized_answer = normalize_answer(model_answer, question_type)
            
            # Проверяем правильность
            is_correct = _check_answer_correctness(
                normalized_answer, correct_answer, expected_answer, question_type
            )
            
            if is_correct:
                correct += 1
            total += 1
            
            results.append({
                "sample_id": entry['sample_id'],
                "block_type": entry['block_type'],
                "question_type": question_type,
                "expected_answer": expected_answer,
                "model_answer": model_answer,
                "normalized_answer": normalized_answer,
                "is_correct": is_correct
            })
            
            # Отображение в verbose режиме (только первые verbose_max_samples примеров)
            if verbose and entry_idx < verbose_max_samples:
                _display_verbose_info(
                    entry, entry_idx, len(dataset), model_answer, normalized_answer,
                    correct_answer, expected_answer, is_correct
                )
    
    if verbose and len(dataset) > verbose_max_samples:
        print(f"\n{'='*80}")
        print(f"Отображено только первые {verbose_max_samples} из {len(dataset)} примеров.")
        print(f"Для отображения всех примеров установите verbose_max_samples={len(dataset)}")
        print(f"{'='*80}\n")
    
    accuracy = correct / total if total > 0 else 0.0
    
    # Подсчитываем метрики по типам вопросов
    metrics_by_type = {}
    for question_type in ["existence", "distance", "coordinates", "direction"]:
        type_results = [r for r in results if r['question_type'] == question_type]
        if len(type_results) > 0:
            type_correct = sum(1 for r in type_results if r['is_correct'])
            metrics_by_type[question_type] = {
                "accuracy": type_correct / len(type_results),
                "total": len(type_results)
            }
    
    return {
        "overall_accuracy": accuracy,
        "correct": correct,
        "total": total,
        "metrics_by_type": metrics_by_type,
        "results": results
    }

