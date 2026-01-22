# Валидация модели Caged Craftext с визуализацией

Этот скрипт позволяет запустить валидацию обученной модели на среде Caged Craftext с визуализацией эпизодов в виде GIF-анимаций.

## Где сохраняются чекпоинты

Чекпоинты сохраняются в:
```
./checkpoints/verl_agent_caged_craftext/{RUN_NAME}/global_step_{N}/
```

Где:
- `verl_agent_caged_craftext` - это `trainer.project_name` из скрипта обучения
- `{RUN_NAME}` - это `trainer.experiment_name` (например, `run_ppo_qwen2.5_1.5b_caged_craftext_budgetary_water_20250120-120000`)
- `global_step_{N}` - номер шага обучения (например, `global_step_100`)

## Быстрый старт (РЕКОМЕНДУЕТСЯ)

**Автоматическая валидация с последним чекпоинтом:**

```bash
# Находит последний запуск и последний чекпоинт автоматически
bash evaluation/validate_latest_caged_craftext.sh
```

См. подробные примеры в [EXAMPLES_VALIDATION.md](EXAMPLES_VALIDATION.md)

## Использование

### Прямой запуск Python скрипта

```bash
python -m evaluation.evaluation_caged_craftext \
    --run_name "run_ppo_qwen2.5_1.5b_caged_craftext_budgetary_water_20250120-120000" \
    --global_step 100 \
    --craftext_settings "achievements_safe_budget_drink" \
    --num_episodes 10 \
    --output_dir "./runs"
```

### Использование bash скрипта

```bash
./evaluation/run_evaluation_caged_craftext.sh \
    "run_ppo_qwen2.5_1.5b_caged_craftext_budgetary_water_20250120-120000" \
    100 \
    "achievements_safe_budget_drink" \
    10 \
    "./runs"
```

## Параметры

- `--run_name` (обязательный): Имя запуска обучения (например, `run_ppo_qwen2.5_1.5b_caged_craftext_budgetary_water_20250120-120000`)
- `--global_step` (обязательный): Номер шага обучения для загрузки чекпоинта (например, `100`)
- `--craftext_settings` (опциональный, по умолчанию `achievements_safe_budget_drink`): Настройки датасета Caged Craftext
- `--num_episodes` (опциональный, по умолчанию `10`): Количество эпизодов для валидации
- `--output_dir` (опциональный, по умолчанию `./runs`): Директория для сохранения GIF-файлов
- `--checkpoint_path` (опциональный): Полный путь к чекпоинту (если не указан, используется стандартный путь)

## Структура чекпоинта

Скрипт автоматически ищет чекпоинт в следующих местах (в порядке приоритета):

1. Мердженная HF-модель: `{checkpoint_path}/actor/actor_hf_merged` или `{checkpoint_path}/actor_hf_merged`
2. LoRA адаптер: `{checkpoint_path}/actor/lora_adapter`
3. Базовая модель (fallback): `Qwen/Qwen2.5-1.5B-Instruct`

## Выходные файлы

Для каждого эпизода создается GIF-файл с именем:
```
{run_name}_episode_{N}_sr_{success_rate:.2f}_cost_{total_cost:.2f}.gif
```

Где:
- `N` - номер эпизода (1, 2, 3, ...)
- `success_rate` - успешность выполнения инструкции
- `total_cost` - общая стоимость эпизода (для Caged Craftext)

## Что визуализируется

Каждый кадр GIF содержит:
- Игровое поле Craftax
- Панель с промптом и выводом модели
- Информационный баннер с:
  - Номером шага
  - Наградой (reward)
  - Стоимостью (cost) - для Caged Craftext
  - Прогрессом выполнения инструкции
  - Успешностью (success rate)
  - Инструкцией
  - Ограничением (constraint) - для Caged Craftext
  - Действием агента

## Примеры использования

### Валидация модели после 100 шагов обучения

```bash
python -m evaluation.evaluation_caged_craftext \
    --run_name "run_ppo_qwen2.5_1.5b_caged_craftext_budgetary_water_20250120-120000" \
    --global_step 100 \
    --craftext_settings "achievements_safe_budget_drink" \
    --num_episodes 5 \
    --output_dir "./validation_results"
```

### Валидация с указанием полного пути к чекпоинту

```bash
python -m evaluation.evaluation_caged_craftext \
    --run_name "my_run" \
    --global_step 200 \
    --checkpoint_path "/path/to/checkpoint/global_step_200" \
    --craftext_settings "achievements_safe_caged" \
    --num_episodes 10
```

## Требования

- Обученная модель должна быть сохранена в формате, совместимом с `SimpleAgent`
- Среда должна быть настроена для работы с Caged Craftext
- Необходимы библиотеки: `jax`, `ray`, `transformers`, `vllm`, `PIL`

## Примечания

- Скрипт автоматически определяет, использовать ли мердженную модель или LoRA адаптер
- Визуализация включает информацию о cost и constraint, специфичную для Caged Craftext
- Каждый эпизод ограничен 50 шагами (можно изменить в коде)
