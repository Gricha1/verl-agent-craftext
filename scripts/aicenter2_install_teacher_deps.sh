#!/usr/bin/env bash
set -euo pipefail
venv=/home/gorbov_gv/craftext_cnn_ppo_expert_v1/venv
log=/home/gorbov_gv/craftext_teacher_deps_install.log
# DistilBERT is frozen and is evaluated only while the task table is built;
# JAX owns the A100s.  The CPU wheel avoids downloading an unrelated CUDA
# runtime stack into this JAX environment.
nohup bash -c '"$0" install --index-url https://download.pytorch.org/whl/cpu torch && "$0" install tensorboard' "$venv/bin/pip" >"$log" 2>&1 < /dev/null &
echo $! > /home/gorbov_gv/craftext_teacher_deps_install.pid
echo "DEPENDENCY_INSTALL_PID=$(cat /home/gorbov_gv/craftext_teacher_deps_install.pid) LOG=$log"
