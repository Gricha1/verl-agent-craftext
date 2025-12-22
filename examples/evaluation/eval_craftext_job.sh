#!/bin/bash
# set -x

source /home/jovyan/nsorokin/miniconda3/etc/profile.d/conda.sh
conda activate /home/jovyan/nsorokin/verl-agent-craftext/verl-agent-conda-venv-311/

pip install -e . --no-deps

python3 evaluation/evaluation_llm.py