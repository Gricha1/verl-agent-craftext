#!/usr/bin/env bash
set -euo pipefail

docker run --rm \
  --user 0:0 \
  --entrypoint /bin/bash \
  --volume /etc/passwd:/etc/passwd:ro \
  --volume /etc/group:/etc/group:ro \
  modern-verl-craftext:v091-smoke-craftext \
  -lc 'export HOME=/tmp USER=root LOGNAME=root USERNAME=root XDG_CACHE_HOME=/tmp/cache XDG_CONFIG_HOME=/tmp/config MPLCONFIGDIR=/tmp/mpl TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor; id; grep ":1007:" /etc/passwd; env | grep -E "^(HOME|USER|LOGNAME|USERNAME|XDG|MPL|TORCH)"; python3 -c "import os,getpass; print(os.getuid(), os.environ.get(\"USER\"), getpass.getuser())"'
