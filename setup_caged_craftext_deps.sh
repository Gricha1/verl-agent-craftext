#!/bin/bash
# Скрипт для установки зависимостей для caged_craftext внутри Docker контейнера
# Использование: bash setup_caged_craftext_deps.sh

# Не используем set -e, чтобы скрипт продолжал работу при предупреждениях

echo "=== Установка зависимостей для caged_craftext ==="

# Проверяем, что мы в правильной директории
if [ ! -f "setup.py" ]; then
    echo "Ошибка: запустите скрипт из корня проекта (/usr/home/workspace)"
    exit 1
fi

# Проверяем conda окружение (Docker build uses conda run; interactive shell may need activate)
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Активируем conda окружение verl-agent-311..."
    if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
        source /opt/conda/etc/profile.d/conda.sh
    fi
    conda activate verl-agent-311 2>/dev/null || true
fi

echo "Текущее conda окружение: $CONDA_DEFAULT_ENV"

# Устанавливаем переменную окружения для craftax
export CRAFTAX_RELOAD_TEXTURES=True
echo "CRAFTAX_RELOAD_TEXTURES=$CRAFTAX_RELOAD_TEXTURES"

# Обновляем pip
echo "=== Обновление pip ==="
pip install -U pip setuptools wheel

# Базовые установки
echo "=== Установка базовых пакетов ==="
pip install psutil packaging
pip install datasets tensorboard comet_ml
pip install "ray[default]" msgspec cachetools openai gym gymnasium

# Установка craftext wrapper (без зависимостей, чтобы избежать конфликтов)
echo "=== Установка craftext wrapper (--no-deps) ==="
pip install --no-deps -e agent_system/environments/env_package/craftext/CrafText-super_igor_env_build || \
pip install --no-deps -e agent_system/environments/env_package/craftext/craftext || \
echo "Предупреждение: не удалось установить craftext wrapper"

# Удаляем старые версии JAX/Craftax (если есть)
echo "=== Очистка старых версий JAX/Craftax ==="
pip uninstall -y craftax flax jax jaxlib jax-cuda12-pjrt jax-cuda12-plugin chex optax orbax-checkpoint distrax tensorstore ml_dtypes || true

# Установка JAX экосистемы (для CUDA 11.8)
echo "=== Установка JAX экосистемы (CUDA 11.8) ==="
# Для CUDA 11.8 используем jaxlib без CUDA (CPU версия) или пробуем установить с CUDA 11
# Если CUDA версия не нужна, можно использовать CPU версию
pip install --no-cache-dir "jax==0.4.30" "jaxlib==0.4.30" "flax==0.10.4" "chex==0.1.90" "optax==0.2.5" "orbax-checkpoint==0.6.4" "distrax==0.1.5" "tensorstore==0.1.78" "ml_dtypes==0.5.3" "numpy==1.26.2" || \
echo "Предупреждение: некоторые пакеты JAX могут быть не установлены"

# Установка craftax (после JAX, без зависимостей)
echo "=== Установка craftax (--no-deps) ==="
pip install --no-cache-dir --no-deps craftax==1.4.3 || \
echo "Предупреждение: не удалось установить craftax"

# Установка зависимостей craftax (исключая jax/jaxlib, которые уже установлены)
echo "=== Установка зависимостей craftax ==="
pip install imageio matplotlib seaborn pygame gymnax treescope black pre-commit || \
echo "Предупреждение: некоторые зависимости craftax могут быть не установлены"

# Обновление pandas, scikit-learn, scipy
echo "=== Обновление pandas, scikit-learn, scipy ==="
pip install -U pandas scikit-learn scipy

# Установка caged_craftext (без зависимостей, чтобы избежать конфликтов)
echo "=== Установка caged_craftext (--no-deps) ==="
if [ -d "caged_craftext" ]; then
    cd caged_craftext
    pip install --no-deps -e . || echo "Предупреждение: не удалось установить caged_craftext"
    cd ..
else
    echo "Предупреждение: директория caged_craftext не найдена"
fi

# Установка остальных зависимостей из requirements.txt (исключая flash-attn, craftax, jax)
echo "=== Установка зависимостей из requirements.txt (исключая конфликтующие) ==="
if [ -f "requirements.txt" ]; then
    grep -v "^flash-attn" requirements.txt > /tmp/requirements_no_flash.txt || cp requirements.txt /tmp/requirements_no_flash.txt
    grep -v "^craftax" /tmp/requirements_no_flash.txt > /tmp/requirements_no_conflicts.txt || cp /tmp/requirements_no_flash.txt /tmp/requirements_no_conflicts.txt
    grep -v "^jax" /tmp/requirements_no_conflicts.txt > /tmp/requirements_final.txt || cp /tmp/requirements_no_conflicts.txt /tmp/requirements_final.txt
    pip install --no-deps -r /tmp/requirements_final.txt || echo "Предупреждение: некоторые зависимости из requirements.txt могут быть не установлены"
    rm -f /tmp/requirements_no_flash.txt /tmp/requirements_no_conflicts.txt /tmp/requirements_final.txt
else
    echo "Предупреждение: requirements.txt не найден"
fi

# Установка flash-attn в последнюю очередь (требуется для verl.workers.critic).
# ВАЖНО: только prebuilt wheel — сборка из исходников на H200 занимает часы.
echo "=== Установка flash-attn (prebuilt wheel, без компиляции) ==="
if python -c "import flash_attn" >/dev/null 2>&1; then
  echo "✓ flash-attn уже установлен, пропускаем"
else
  _torch_mm=$(python -c "import torch; v=torch.__version__.split('+')[0].split('.'); print(v[0]+'.'+v[1])" 2>/dev/null || echo "2.6")
  _abi=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')" 2>/dev/null || echo "FALSE")
  _tag="cu12torch${_torch_mm}cxx11abi${_abi}"
  _ver="2.7.4.post1"
  # torch>=2.8 → свежий flash-attn 2.8.3
  case "${_torch_mm}" in
    2.8|2.9) _ver="2.8.3" ;;
  esac
  WHL="flash_attn-${_ver}+${_tag}-cp311-cp311-linux_x86_64.whl"
  LOCAL_WHL="$(pwd)/.wheels/${WHL}"
  URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${_ver}/${WHL}"
  if [ -f "$LOCAL_WHL" ]; then
    echo "Installing flash-attn from local ${LOCAL_WHL}"
    pip install --no-cache-dir --no-deps "$LOCAL_WHL" && echo "✓ flash-attn из local wheel OK" \
      || echo "ОШИБКА: не удалось поставить local wheel ${LOCAL_WHL}"
  else
    echo "Downloading ${WHL} ..."
    if curl -fL --retry 5 --retry-delay 2 "$URL" -o "/tmp/${WHL}" && pip install --no-cache-dir --no-deps "/tmp/${WHL}"; then
      echo "✓ flash-attn из wheel OK"
    else
      echo "ОШИБКА: не удалось скачать/поставить wheel ${WHL}"
      echo "  Скачай на хост: mkdir -p .wheels && wget -O .wheels/${WHL} '${URL}'"
      echo "  (не компилируем из исходников — слишком долго)"
    fi
  fi
fi

# Проверки
echo "=== Проверка установки ==="
python -c "import numpy, pandas; print('✓ numpy', numpy.__version__, 'pandas', pandas.__version__)" || echo "✗ numpy/pandas не установлены"
python -c "import ray, tensordict; print('✓ ray', ray.__version__, 'tensordict', tensordict.__version__)" || echo "✗ ray/tensordict не установлены"
python -c "import vllm; import vllm.lora.models; print('✓ vllm OK', vllm.__version__)" || echo "✗ vllm не установлен"
python -c "import gymnasium; print('✓ gymnasium', gymnasium.__version__)" || echo "✗ gymnasium не установлен"
python -c "import jax; print('✓ jax', jax.__version__)" || echo "✗ jax не установлен"
python -c "import os; os.environ['CRAFTAX_RELOAD_TEXTURES']='True'; import craftax; print('✓ craftax OK')" || echo "✗ craftax не установлен"
python -c "import flash_attn; print('✓ flash-attn OK')" || echo "✗ flash-attn не установлен"

echo ""
echo "=== Установка завершена ==="
echo ""
echo "Для постоянного использования CRAFTAX_RELOAD_TEXTURES добавьте в ~/.bashrc:"
echo "  export CRAFTAX_RELOAD_TEXTURES=True"
echo ""
echo "Теперь можно запускать обучение:"
echo "  bash examples/ppo_trainer/run_caged_craftext_lora_job.sh vllm false false 32 512"
