#!/bin/bash
# set -x

source /home/jovyan/nsorokin/miniconda3/etc/profile.d/conda.sh
conda activate /home/jovyan/nsorokin/verl-agent-craftext/verl-agent-conda-venv-311/

python3 evaluation/evaluation_llm.py \
    --run_name "run_gigpo_qwen2.5_1.5b_from_sft_achievements_collect_wood_20251205-003530" \
    --episode_number 100