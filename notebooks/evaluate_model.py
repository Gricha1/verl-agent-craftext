#!/usr/bin/env python3
"""
Скрипт для оценки модели на датасете восприятия Craftext.
Сохраняет результаты в JSON файл для последующего анализа.
"""
import argparse
import json
import os
from pathlib import Path
from datetime import datetime

from simple_agent import SimpleAgent
from model_evaluation import load_dataset, evaluate_model_on_dataset


def save_results(results, model_name, output_dir="evaluation_results"):
    """
    Сохраняет результаты оценки в JSON файл.
    
    Args:
        results: словарь с результатами оценки
        model_name: имя модели (для имени файла)
        output_dir: директория для сохранения результатов
    """
    # Создаем директорию если её нет
    os.makedirs(output_dir, exist_ok=True)
    
    # Формируем имя файла
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Сохраняем результаты
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nРезультаты сохранены в: {filepath}")
    return filepath


def evaluate_and_save(
    model_name,
    base_model_path,
    dataset_path,
    checkpoint_path="",
    lora_rank=0,
    lora_alpha=64,
    target_modules="all-linear",
    max_response_length=128,
    do_sample=False,
    temperature=0.0,
    use_text_observation=True,
    batch_size=32,
    verbose=False,
    verbose_max_samples=10,
    output_dir="evaluation_results"
):
    """
    Оценивает модель и сохраняет результаты.
    
    Args:
        model_name: имя модели (для сохранения результатов)
        base_model_path: путь к базовой модели
        dataset_path: путь к датасету
        checkpoint_path: путь к чекпоинту (опционально)
        lora_rank: rank для LoRA (0 если не используется)
        lora_alpha: alpha для LoRA
        target_modules: модули для LoRA
        max_response_length: максимальная длина ответа
        do_sample: использовать ли сэмплирование
        temperature: температура для сэмплирования
        use_text_observation: использовать ли текстовое наблюдение
        batch_size: размер батча для LLM
        verbose: отображать ли детали
        verbose_max_samples: количество примеров для отображения
        output_dir: директория для сохранения результатов
    """
    print(f"=" * 80)
    print(f"Оценка модели: {model_name}")
    print(f"Базовая модель: {base_model_path}")
    print(f"Датасет: {dataset_path}")
    print(f"=" * 80)
    
    # Загружаем датасет
    print(f"\nЗагрузка датасета из {dataset_path}...")
    dataset = load_dataset(dataset_path)
    print(f"Загружено {len(dataset)} примеров")
    
    # Инициализируем модель
    print(f"\nИнициализация модели {base_model_path}...")
    agent = SimpleAgent(
        checkpoint_path=checkpoint_path,
        base_model_path=base_model_path,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        max_response_length=max_response_length,
        do_sample=do_sample,
        temperature=temperature,
    )
    
    # Оцениваем модель
    print(f"\nНачало оценки...")
    results = evaluate_model_on_dataset(
        agent,
        dataset,
        use_text_observation=use_text_observation,
        verbose=verbose,
        batch_size=batch_size,
        verbose_max_samples=verbose_max_samples
    )
    
    # Добавляем метаданные
    results['metadata'] = {
        'model_name': model_name,
        'base_model_path': base_model_path,
        'checkpoint_path': checkpoint_path,
        'lora_rank': lora_rank,
        'dataset_path': dataset_path,
        'dataset_size': len(dataset),
        'use_text_observation': use_text_observation,
        'batch_size': batch_size,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Сохраняем результаты
    filepath = save_results(results, model_name, output_dir)
    
    # Выводим краткую сводку
    print(f"\n{'=' * 80}")
    print(f"РЕЗУЛЬТАТЫ ОЦЕНКИ: {model_name}")
    print(f"{'=' * 80}")
    print(f"Общая точность: {results['overall_accuracy']:.2%}")
    print(f"Правильных ответов: {results['correct']} / {results['total']}")
    print(f"\nТочность по типам вопросов:")
    for qtype, metrics in results['metrics_by_type'].items():
        print(f"  {qtype:15s}: {metrics['accuracy']:.2%} ({metrics['total']} примеров)")
    print(f"{'=' * 80}\n")
    
    return results, filepath


def main():
    parser = argparse.ArgumentParser(description="Оценка модели на датасете восприятия Craftext")
    
    # Обязательные параметры
    parser.add_argument("--model-name", type=str, required=True,
                        help="Имя модели (для сохранения результатов)")
    parser.add_argument("--base-model-path", type=str, required=True,
                        help="Путь к базовой модели (HuggingFace model ID или локальный путь)")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Путь к датасету (JSONL файл)")
    
    # Параметры модели
    parser.add_argument("--checkpoint-path", type=str, default="",
                        help="Путь к чекпоинту (опционально)")
    parser.add_argument("--lora-rank", type=int, default=0,
                        help="Rank для LoRA (0 если не используется)")
    parser.add_argument("--lora-alpha", type=int, default=64,
                        help="Alpha для LoRA")
    parser.add_argument("--target-modules", type=str, default="all-linear",
                        help="Модули для LoRA")
    parser.add_argument("--max-response-length", type=int, default=128,
                        help="Максимальная длина ответа")
    parser.add_argument("--do-sample", action="store_true",
                        help="Использовать сэмплирование")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Температура для сэмплирования")
    
    # Параметры оценки
    parser.add_argument("--use-text-observation", action="store_true", default=True,
                        help="Использовать текстовое наблюдение в промпте")
    parser.add_argument("--no-text-observation", dest="use_text_observation", action="store_false",
                        help="Не использовать текстовое наблюдение (только для VLM)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Размер батча для LLM")
    parser.add_argument("--verbose", action="store_true",
                        help="Отображать детали оценки")
    parser.add_argument("--verbose-max-samples", type=int, default=10,
                        help="Количество примеров для отображения в verbose режиме")
    
    # Параметры сохранения
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                        help="Директория для сохранения результатов")
    
    args = parser.parse_args()
    
    # Запускаем оценку
    evaluate_and_save(
        model_name=args.model_name,
        base_model_path=args.base_model_path,
        dataset_path=args.dataset_path,
        checkpoint_path=args.checkpoint_path,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        max_response_length=args.max_response_length,
        do_sample=args.do_sample,
        temperature=args.temperature,
        use_text_observation=args.use_text_observation,
        batch_size=args.batch_size,
        verbose=args.verbose,
        verbose_max_samples=args.verbose_max_samples,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

