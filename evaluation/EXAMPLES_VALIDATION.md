# Примеры запуска валидации Caged Craftext

## Где сохраняются чекпоинты

Чекпоинты сохраняются в следующей структуре:
```
checkpoints/
└── verl_agent_caged_craftext/          # trainer.project_name
    └── {RUN_NAME}/                      # trainer.experiment_name (например, run_ppo_qwen2.5_1.5b_caged_craftext_budgetary_water_20250120-120000)
        ├── latest_checkpointed_iteration.txt  # Трекер последнего чекпоинта
        ├── global_step_0/
        │   ├── actor/
        │   │   ├── lora_adapter/        # LoRA адаптер (если используется)
        │   │   └── ...
        │   └── critic/
        ├── global_step_100/
        ├── global_step_200/
        └── ...
```

**Базовый путь**: `./checkpoints/verl_agent_caged_craftext/{RUN_NAME}/global_step_{N}`

## Примеры запуска

### 1. Автоматическая валидация с последним чекпоинтом (РЕКОМЕНДУЕТСЯ)

Скрипт автоматически найдет последний запуск и последний чекпоинт:

```bash
# Использовать последний запуск и последний чекпоинт
bash evaluation/validate_latest_caged_craftext.sh

# Или указать конкретный RUN_NAME
bash evaluation/validate_latest_caged_craftext.sh run_ppo_qwen2.5_1.5b_caged_craftext_budgetary_water_20250120-120000

# С дополнительными параметрами
bash evaluation/validate_latest_caged_craftext.sh \
    --num_episodes 5 \
    --max_steps 50 \
    --output_dir ./validation_results \
    --verbose
```

### 2. Валидация с указанием конкретного чекпоинта

```bash
bash evaluation/run_evaluation_caged_craftext.sh \
    --checkpoint_path ./checkpoints/verl_agent_caged_craftext/run_ppo_qwen2.5_1.5b_caged_craftext_budgetary_water_20250120-120000/global_step_100 \
    --craftext_settings achievements_safe_budget_drink \
    --num_episodes 10 \
    --output_dir ./runs
```

### 3. Валидация по run_name и global_step

```bash
bash evaluation/run_evaluation_caged_craftext.sh \
    --project_name verl_agent_caged_craftext \
    --run_name run_ppo_qwen2.5_1.5b_caged_craftext_budgetary_water_20250120-120000 \
    --global_step 100 \
    --craftext_settings achievements_safe_budget_drink \
    --num_episodes 5 \
    --do_sample \
    --temperature 0.4
```

### 4. Python скрипт для автоматического поиска последнего чекпоинта

```python
#!/usr/bin/env python3
"""Пример скрипта для автоматической валидации с последним чекпоинтом"""

import os
import sys
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.evaluation_caged_craftext import evaluation_caged_craftext, parse_args
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path

def find_latest_checkpoint(project_name="verl_agent_caged_craftext", run_name=None):
    """Находит последний чекпоинт для указанного или последнего запуска"""
    base_dir = Path("./checkpoints") / project_name
    
    if run_name is None:
        # Ищем последний запуск по дате модификации
        runs = sorted(base_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not runs:
            raise ValueError(f"Не найдено запусков в {base_dir}")
        run_name = runs[0].name
        print(f"Использован последний запуск: {run_name}")
    
    run_dir = base_dir / run_name
    if not run_dir.exists():
        raise ValueError(f"Запуск {run_name} не найден в {base_dir}")
    
    # Ищем последний чекпоинт через трекер
    latest_ckpt = find_latest_ckpt_path(str(run_dir))
    if latest_ckpt is None:
        # Fallback: ищем последний global_step_* по дате
        ckpt_dirs = sorted(run_dir.glob("global_step_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not ckpt_dirs:
            raise ValueError(f"Не найдено чекпоинтов в {run_dir}")
        latest_ckpt = str(ckpt_dirs[0])
    
    return latest_ckpt, run_name

if __name__ == "__main__":
    # Парсим аргументы
    args = parse_args()
    
    # Если checkpoint_path не указан, ищем последний
    if args.checkpoint_path is None:
        checkpoint_path, run_name = find_latest_checkpoint(
            project_name=args.project_name,
            run_name=args.run_name
        )
        args.checkpoint_path = checkpoint_path
        if args.run_name is None:
            args.run_name = run_name
        print(f"Найден последний чекпоинт: {checkpoint_path}")
    
    # Запускаем валидацию
    evaluation_caged_craftext(args)
```

## Параметры валидации

### Основные параметры

- `--checkpoint_path` - Полный путь к чекпоинту (если указан, остальные игнорируются)
- `--project_name` - Имя проекта (по умолчанию: `verl_agent_caged_craftext`)
- `--run_name` - Имя запуска (например, `run_ppo_qwen2.5_1.5b_caged_craftext_budgetary_water_20250120-120000`)
- `--global_step` - Номер шага обучения (например, `100`)
- `--craftext_settings` - Настройки датасета (по умолчанию: `achievements_safe_budget_drink`)

### Параметры валидации

- `--num_episodes` - Количество эпизодов для валидации (по умолчанию: `10`)
- `--max_steps` - Максимум шагов в эпизоде (по умолчанию: `50`)
- `--output_dir` - Директория для сохранения GIF (по умолчанию: `./runs`)
- `--seed` - Seed для воспроизводимости (по умолчанию: `42`)

### Параметры модели

- `--do_sample` - Использовать сэмплирование (как в обучении)
- `--temperature` - Температура для сэмплирования (по умолчанию: `0.4`)
- `--cuda_visible_devices` - Какие GPU использовать (например, `"0"` или `"0,1"`)

### Отладка

- `--verbose` - Подробный вывод

## Выходные файлы

GIF-файлы сохраняются в `--output_dir` с именами вида:
```
{run_name}_ep001_sr0.85_cost2.50.gif
{run_name}_ep002_sr1.00_cost0.00.gif
...
```

Где:
- `epXXX` - номер эпизода
- `srX.XX` - success rate (успешность выполнения инструкции)
- `costX.XX` - общая стоимость эпизода (для Caged Craftext)

## Типичные сценарии

### Проверка последнего обучения

```bash
# Просто запустить с последним чекпоинтом
bash evaluation/validate_latest_caged_craftext.sh
```

### Сравнение разных чекпоинтов

```bash
# Чекпоинт на шаге 100
bash evaluation/run_evaluation_caged_craftext.sh \
    --project_name verl_agent_caged_craftext \
    --run_name YOUR_RUN_NAME \
    --global_step 100 \
    --output_dir ./results_step100

# Чекпоинт на шаге 200
bash evaluation/run_evaluation_caged_craftext.sh \
    --project_name verl_agent_caged_craftext \
    --run_name YOUR_RUN_NAME \
    --global_step 200 \
    --output_dir ./results_step200
```

### Быстрая проверка (1 эпизод)

```bash
bash evaluation/validate_latest_caged_craftext.sh \
    --num_episodes 1 \
    --max_steps 30
```
