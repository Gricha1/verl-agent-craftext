#!/usr/bin/env bash
set -euo pipefail
venv=/home/gorbov_gv/craftext_cnn_ppo_expert_v1/venv
log=/home/gorbov_gv/craftext_teacher_deps_install.log
nohup "$venv/bin/pip" install torch tensorboard >"$log" 2>&1 < /dev/null &
echo $! > /home/gorbov_gv/craftext_teacher_deps_install.pid
echo "DEPENDENCY_INSTALL_PID=$(cat /home/gorbov_gv/craftext_teacher_deps_install.pid) LOG=$log"
