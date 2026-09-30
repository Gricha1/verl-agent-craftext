# Гайд по запуску обучения на aicenter2 и aicenteritl

## Обзор инфраструктуры

### aicenter2 (Docker)
- **Тип**: Docker контейнеры
- **GPU**: 4x NVIDIA A100 80GB
- **Образ**: `verlai/verl:vllm011.latest`
- **Особенности**: 
  - Нет нативного venv, всё работает в Docker
  - Пакеты (Gym/JAX/Craftax) устанавливаются при каждом запуске
  - Нельзя обновлять Torch/Transformers/vLLM/xFormers (уже в образе)
  - GPU внутри контейнера всегда маппятся на 0

### aicenteritl (Native venv)
- **Тип**: Нативный Python venv
- **GPU**: Доступны через SSH
- **Особенности**:
  - Прямой доступ к файловой системе
  - Быстрее старт (не нужно поднимать Docker)
  - Можно обновлять пакеты

## Запуск на aicenter2

### 1. Подготовка скрипта запуска

Создайте скрипт в `_exp_scripts/` на основе шаблона:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/storage/gorbov_gv/safe_rl_nlp}
IMAGE=${IMAGE:-verlai/verl:vllm011.latest}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}  # Хост GPU
RAY_TEMP_DIR=${RAY_TEMP_DIR:-/dev/shm/ray16}
HF_HOME=${HF_HOME:-/storage/gorbov_gv/models/hf}
MODEL_PATH=${MODEL_PATH:-/storage/gorbov_gv/models/hf/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}
DATA_DIR=${DATA_DIR:-/storage/gorbov_gv/data/verl-agent}

cd "$REPO"
mkdir -p "$RAY_TEMP_DIR" "$REPO/logdir" "$REPO/training_checkpoints"

# Preflight проверка
echo "[preflight] selected GPUs: $CUDA_VISIBLE_DEVICES"
for gpu in ${CUDA_VISIBLE_DEVICES//,/ }; do
  nvidia-smi -i "$gpu" --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
done

# Подтверждение запуска
if [[ ${CONFIRM_LAUNCH:-0} != 1 ]]; then
  echo "Запустите с CONFIRM_LAUNCH=1 для реального запуска"
  exit 0
fi

# Docker запуск
docker run --rm --cap-add=SYS_PTRACE --gpus "\"device=$CUDA_VISIBLE_DEVICES\"" \
  --ipc=host --network host --shm-size=32g \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e JAX_PLATFORMS=cpu \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -e RAY_TEMP_DIR="$RAY_TEMP_DIR" \
  -e HF_DATASETS_CACHE=/dev/shm/hf_datasets \
  -e RAY_FORK_CONTEXT=spawn \
  -e VLLM_WORKER_MULTIPROCESSING_START_METHOD=spawn \
  -e HF_DATASETS_NUM_PROC=1 \
  -e HF_HOME="$HF_HOME" \
  -e HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e COMET_API_KEY -e COMET_WORKSPACE -e COMET_PROJECT_NAME \
  -e COMET_DISABLED -e VALIDATION_ONLY -e MODEL_PATH="$MODEL_PATH" \
  -v "$REPO:/workspace" \
  -v "$HF_HOME:$HF_HOME:ro" \
  -v "$DATA_DIR:/root/data/verl-agent:ro" \
  -v "$RAY_TEMP_DIR:$RAY_TEMP_DIR" \
  -w /workspace --entrypoint bash "$IMAGE" -lc '
    # Установка зависимостей (каждый раз!)
    pip install --no-cache-dir \
      gym==0.26.2 jax==0.4.30 jaxlib==0.4.30 flax==0.10.4 chex==0.1.90 \
      optax==0.2.5 orbax-checkpoint==0.6.4 distrax==0.1.5 tensorstore==0.1.78 \
      ml_dtypes==0.5.3 gymnax==0.0.8 imageio pillow
    
    # Hydra overrides
    extra_args=(actor_rollout_ref.model.path="$MODEL_PATH" critic.model.path="$MODEL_PATH")
    
    # Запуск обучения
    HYPER_YAML=examples/ppo_trainer/config/ppo_debug_square_16x16_dual_reasoning_grounding.yaml \
      bash examples/ppo_trainer/ppo_debug_square_16x16_reasoning.sh "${extra_args[@]}"
  '
```

### 2. Конфигурация для разных режимов

#### No-Reasoning (single token actions)
**Файлы**:
- Config: `examples/ppo_trainer/config/ppo_debug_square_16x16_dual_no_reasoning_grounding.yaml`
- Script: `examples/ppo_trainer/ppo_debug_square_16x16.sh`
- Launch: `_exp_scripts/a2_prepare_ds16_dual_no_reasoning.sh`

**Ключевые параметры**:
```yaml
env.enable_reasoning=False
data.max_response_length=1  # Только 1 токен на действие
actor_rollout_ref.actor.single_token_actions=True
trainer.n_gpus_per_node=2  # 2 GPU для no-reasoning
```

**Позиционные аргументы** `run_caged_craftext_lora_job.sh`:
```bash
$1 = ENGINE              # caged_craftext
$2 = LOG_PROB_ACTION_ONLY # false
$3 = NO_REASONING        # true
$4 = train_data_size     # 64
$5 = max_response_length # 1
$6 = AUTO_RESET          # false
$7 = USE_ACTION_HEAD     # false
$8 = total_epochs        # 8000
$9 = USE_ACTOR_LORA      # true
$10 = PROMPT_TEMPLATE_TYPE # single_token_action
$11 = CRITIC_WARMUP      # 0
$12 = OBSERVATION_TYPE   # gt
```

#### With Reasoning (multi-token responses)
**Файлы**:
- Config: `examples/ppo_trainer/config/ppo_debug_square_16x16_dual_reasoning_grounding.yaml`
- Script: `examples/ppo_trainer/ppo_debug_square_16x16_reasoning.sh`
- Launch: `_exp_scripts/a2_prepare_ds16_dual_reasoning.sh`

**Ключевые параметры**:
```yaml
env.enable_reasoning=True
data.max_response_length=128  # До 128 токенов на ответ
actor_rollout_ref.actor.single_token_actions=False
trainer.n_gpus_per_node=1  # 1 GPU для reasoning
```

**Позиционные аргументы**:
```bash
$1 = ENGINE              # caged_craftext
$2 = LOG_PROB_ACTION_ONLY # true  # ВАЖНО для reasoning!
$3 = NO_REASONING        # false  # reasoning включён
$4 = train_data_size     # 64
$5 = max_response_length # 128
$6 = AUTO_RESET          # false
$7 = USE_ACTION_HEAD     # false
$8 = total_epochs        # 8000
$9 = USE_ACTOR_LORA      # true
$10 = PROMPT_TEMPLATE_TYPE # default_template  # НЕ single_token_action!
$11 = CRITIC_WARMUP      # 0
$12 = OBSERVATION_TYPE   # gt
```

### 3. Важные переменные окружения

#### VLLM_ATTENTION_BACKEND
**Проблема**: XFORMERS (по умолчанию) несовместим с длинными последовательностями в reasoning режиме.

**Решение**: Использовать TORCH_SDPA для reasoning:
```bash
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
```

В `run_caged_craftext_lora_job.sh` строка 98 должна быть:
```bash
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
```

Это позволяет переопределять backend через переменную окружения.

#### JAX_PLATFORMS=cpu
Craftax environment работает на CPU (не занимает GPU ресурсы):
```bash
-e JAX_PLATFORMS=cpu
-e XLA_PYTHON_CLIENT_PREALLOCATE=false
```

### 4. Запуск

```bash
# 1. Проверить конфигурацию (без запуска)
bash _exp_scripts/a2_prepare_ds16_dual_reasoning.sh

# 2. Реальный запуск
CONFIRM_LAUNCH=1 bash _exp_scripts/a2_prepare_ds16_dual_reasoning.sh

# 3. Запуск в background
nohup bash -c 'CONFIRM_LAUNCH=1 bash _exp_scripts/a2_prepare_ds16_dual_reasoning.sh' > launch.log 2>&1 &
```

### 5. Мониторинг

#### Проверка контейнеров
```bash
# Список активных контейнеров
docker ps --format '{{.Names}} {{.Status}}'

# GPU статус
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

#### Логи обучения
```bash
# Основной лог (путь из скрипта)
tail -f /storage/gorbov_gv/safe_rl_nlp/logdir/ds16_dual_reasoning_a2_20260917_211159.log

# Docker логи
docker logs great_lewin --tail 100

# Docker логи без Comet ошибок
docker logs great_lewin 2>&1 | grep -v 'ReadTimeoutError\|Retrying' | tail -50

# Ray worker логи
docker exec great_lewin tail -100 /dev/shm/ray16/session_latest/logs/worker-*.out
```

#### Ключевые события в логах
```
# Валидация завершена
[validation] per-action returns at t=0: [...]

# PPO начался
[PPO] Training loop entered.
[PPO phase 11:35:49 | global_step=1] ▶ ROLLOUT

# Rollout прогресс
[rollout train] env step 30 (active envs ≈ 64/64)

# PPO шаг завершён
[PPO phase 11:07:22 | global_step=2] ■ LOGGER done — this PPO step finished

# LoRA синхронизация
vLLM synced LoRA '123456' (id=123456), loaded_params: 392
```

### 6. Чекпоинты

**Настройка**: `trainer.save_freq=5` (сохранение каждые 5 шагов)

**Директории**:
```bash
# No-reasoning
training_checkpoints/verl_agent_caged_craftext_ds16_dual_no_reasoning_grounding

# Reasoning
training_checkpoints/verl_agent_caged_craftext_ds16_dual_reasoning_grounding
```

**Проверка**:
```bash
ls -lh /storage/gorbov_gv/safe_rl_nlp/training_checkpoints/
```

### 7. Comet ML эксперименты

**Просмотр**:
```bash
# Найти experiment key
cat /dev/shm/ray16/comet_experiment_key.txt

# URL эксперимента
https://www.comet.com/gregory-gorbov/verl-agent-caged-craftext/<experiment_key>
```

**Отключение Comet** (для тестов):
```bash
-e COMET_DISABLED=1
```

## Запуск на aicenteritl

### 1. Активация venv
```bash
ssh aicenteritl
source /path/to/venv/bin/activate
```

### 2. Конфигурация
Используются те же YAML конфиги и скрипты, но без Docker обёртки.

**Основное отличие**: Нет необходимости в:
- `docker run` команде
- Установке пакетов при каждом запуске
- Маппинге GPU (`-e CUDA_VISIBLE_DEVICES=0`)
- `--gpus` флаге

### 3. Пример запуска
```bash
cd /path/to/safe_rl_nlp

# Прямой запуск скрипта
HYPER_YAML=examples/ppo_trainer/config/ppo_debug_square_16x16_dual_reasoning_grounding.yaml \
  bash examples/ppo_trainer/ppo_debug_square_16x16_reasoning.sh

# С Hydra overrides
HYPER_YAML=examples/ppo_trainer/config/ppo_debug_square_16x16_dual_reasoning_grounding.yaml \
  bash examples/ppo_trainer/ppo_debug_square_16x16_reasoning.sh \
  actor_rollout_ref.model.path=/path/to/model \
  critic.model.path=/path/to/model
```

### 4. Переменные окружения
```bash
export CUDA_VISIBLE_DEVICES=0,1  # Прямой доступ к GPU
export HF_HOME=/path/to/models
export RAY_TEMP_DIR=/tmp/ray
```

## Типичные проблемы и решения

### 1. Paged KV cache block size must be divisible by 256
**Причина**: XFORMERS backend несовместим с длинными последовательностями

**Решение**:
```bash
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
```

### 2. Comet ReadTimeout errors
**Причина**: Сетевые проблемы из Docker контейнера

**Решение**: Не фатально, можно игнорировать. Или отключить Comet:
```bash
-e COMET_DISABLED=1
```

### 3. CUDA out of memory
**Причина**: Слишком много env или длинных последовательностей

**Решение**:
- Уменьшить `job.num_optimistic_envs` (например, 64 → 32)
- Уменьшить `data.max_response_length` (128 → 64)
- Увеличить `trainer.n_gpus_per_node` (1 → 2)

### 4. Validation takes too long
**Причина**: Reasoning генерирует до 128 токенов на траекторию

**Решение**: Нормально для первого шага. Последующие шаги быстрее благодаря vLLM warmup.

### 5. LoRA sync messages spam
**Причина**: vLLM синхронизирует LoRA веса для каждого env worker

**Решение**: Нормальное поведение, можно фильтровать:
```bash
docker logs great_lewin 2>&1 | grep -v 'vLLM synced LoRA'
```

## Производительность

### No-Reasoning (16x16, 2 GPU)
- PPO шаг: ~12 минут
- GPU memory: ~18 GB на GPU
- GPU utilization: 70-80%

### Reasoning (16x16, 1 GPU)
- PPO шаг: ~30-90 минут (первый шаг дольше)
- GPU memory: ~54 GB
- GPU utilization: 80-100%
- Validation: ~20-30 минут

## Полезные команды

```bash
# Проверить свободные GPU
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

# Остановить контейнер
docker stop great_lewin

# Посмотреть размер чекпоинтов
du -sh training_checkpoints/*

# Удалить старые чекпоинты
rm -rf training_checkpoints/verl_agent_caged_craftext_ds16_dual_reasoning_grounding/global_step_*

# Найти все логи
find /storage/gorbov_gv/safe_rl_nlp/logdir/ -name "*.log" -mtime -1

# Проверить Comet эксперименты
ls -lh /dev/shm/ray16/comet_experiment_key.txt
```

## Структура файлов

```
safe_rl_nlp/
├── _exp_scripts/
│   ├── a2_prepare_ds16_dual_no_reasoning.sh
│   └── a2_prepare_ds16_dual_reasoning.sh
├── examples/ppo_trainer/
│   ├── config/
│   │   ├── ppo_debug_square_16x16_dual_no_reasoning_grounding.yaml
│   │   └── ppo_debug_square_16x16_dual_reasoning_grounding.yaml
│   ├── ppo_debug_square_16x16.sh
│   ├── ppo_debug_square_16x16_reasoning.sh
│   └── run_caged_craftext_lora_job.sh
├── logdir/
│   └── ds16_dual_reasoning_a2_20260917_211159.log
├── training_checkpoints/
│   ├── verl_agent_caged_craftext_ds16_dual_no_reasoning_grounding/
│   └── verl_agent_caged_craftext_ds16_dual_reasoning_grounding/
└── verl/
    └── workers/sharding_manager/fsdp_vllm.py
```

## Контакты и ресурсы

- Comet ML: https://www.comet.com/gregory-gorbov/verl-agent-caged-craftext
- vLLM docs: https://docs.vllm.ai/
- Verl docs: внутренняя документация проекта
