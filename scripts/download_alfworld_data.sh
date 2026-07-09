#!/bin/bash
# Download AlfWorld data when `alfworld-download` fails (GitHub timeout).
#
# Usage:
#   bash scripts/download_alfworld_data.sh
#   ALFWORLD_DATA=/path/to/cache bash scripts/download_alfworld_data.sh
#   FORCE=1 bash scripts/download_alfworld_data.sh   # re-download everything

set -euo pipefail

ALFWORLD_DATA="${ALFWORLD_DATA:-${HOME}/.cache/alfworld}"
FORCE="${FORCE:-0}"
DOWNLOAD_TIMEOUT="${DOWNLOAD_TIMEOUT:-300}"
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-10}"

JSON_URL="https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip"
PDDL_URL="https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_pddl.zip"
TW_PDDL_URL="https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip"
MRCNN_URL="https://github.com/alfworld/alfworld/releases/download/0.2.2/mrcnn_alfred_objects_sep13_004.pth"

mkdir -p "$ALFWORLD_DATA/detectors" "$ALFWORLD_DATA/logic"

echo "[INFO] ALFWORLD_DATA=$ALFWORLD_DATA"

download_file() {
  local url="$1"
  local dst="$2"
  local allow_skip="${3:-1}"
  python - "$url" "$dst" "$FORCE" "$allow_skip" "$DOWNLOAD_TIMEOUT" "$DOWNLOAD_RETRIES" <<'PY'
import os
import sys
import time

import requests

url, dst, force, allow_skip, timeout_s, retries = sys.argv[1:7]
force = force == "1"
allow_skip = allow_skip == "1"
timeout_s = int(timeout_s)
retries = int(retries)

if allow_skip and os.path.isfile(dst) and os.path.getsize(dst) > 0 and not force:
    print(f"[SKIP] already exists: {dst}")
    raise SystemExit(0)

os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
print(f"[DOWNLOAD] {url}")
print(f"           -> {dst}")

resume = 0
if os.path.isfile(dst):
    resume = os.path.getsize(dst)

headers = {}
if resume:
    headers["Range"] = f"bytes={resume}-"

last_err = None
for attempt in range(1, retries + 1):
    try:
        with requests.get(url, stream=True, headers=headers, timeout=timeout_s) as r:
            r.raise_for_status()
            mode = "ab" if resume and r.status_code == 206 else "wb"
            if mode == "wb":
                resume = 0
            total = resume + int(r.headers.get("Content-Length") or 0)
            done = resume
            with open(dst, mode) as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = 100.0 * done / total
                        print(f"\r           {done // (1024*1024)}MB / {total // (1024*1024)}MB ({pct:.0f}%)", end="", flush=True)
            print()
        if os.path.getsize(dst) == 0:
            raise RuntimeError("downloaded file is empty")
        print(f"[OK] saved {dst} ({os.path.getsize(dst) // (1024*1024)}MB)")
        raise SystemExit(0)
    except Exception as e:
        last_err = e
        resume = os.path.getsize(dst) if os.path.isfile(dst) else 0
        headers = {"Range": f"bytes={resume}-"} if resume else {}
        wait = min(30, 2 * attempt)
        print(f"[WARN] attempt {attempt}/{retries} failed: {e}; retry in {wait}s")
        time.sleep(wait)

raise SystemExit(f"[ERROR] download failed after {retries} tries: {last_err}")
PY
}

unzip_if_needed() {
  local zip_path="$1"
  local dst_dir="$2"
  local marker="$3"
  if [ -e "$marker" ] && [ "$FORCE" != "1" ]; then
    echo "[SKIP] already extracted: $marker"
    return 0
  fi
  echo "[UNZIP] $zip_path -> $dst_dir"
  unzip -o -q "$zip_path" -d "$dst_dir"
}

# --- json train (required) ---
if [ ! -d "$ALFWORLD_DATA/json_2.1.1/train" ] || [ "$FORCE" = "1" ]; then
  tmp_json="$(mktemp -u /tmp/alfworld_json.XXXXXX.zip)"
  download_file "$JSON_URL" "$tmp_json" 0
  unzip_if_needed "$tmp_json" "$ALFWORLD_DATA" "$ALFWORLD_DATA/json_2.1.1/train"
  rm -f "$tmp_json"
else
  echo "[SKIP] json train data present"
fi

# --- pddl (required) ---
if ! find "$ALFWORLD_DATA/json_2.1.1" -name '*.pddl' -print -quit 2>/dev/null | grep -q . || [ "$FORCE" = "1" ]; then
  tmp_pddl="$(mktemp -u /tmp/alfworld_pddl.XXXXXX.zip)"
  download_file "$PDDL_URL" "$tmp_pddl" 0
  unzip_if_needed "$tmp_pddl" "$ALFWORLD_DATA" "$ALFWORLD_DATA/json_2.1.1/valid_seen"
  rm -f "$tmp_pddl"
else
  echo "[SKIP] pddl data present"
fi

# --- tw-pddl (required for AlfredTWEnv) ---
if ! find "$ALFWORLD_DATA/json_2.1.1" -name 'game.tw-pddl' -print -quit 2>/dev/null | grep -q . || [ "$FORCE" = "1" ]; then
  tmp_tw="$(mktemp -u /tmp/alfworld_tw_pddl.XXXXXX.zip)"
  download_file "$TW_PDDL_URL" "$tmp_tw" 0
  echo "[UNZIP] $tmp_tw -> $ALFWORLD_DATA"
  unzip -o -q "$tmp_tw" -d "$ALFWORLD_DATA"
  rm -f "$tmp_tw"
else
  echo "[SKIP] tw-pddl data present"
fi

# --- detector weights ---
if [ ! -s "$ALFWORLD_DATA/detectors/mrcnn.pth" ] || [ "$FORCE" = "1" ]; then
  rm -f "$ALFWORLD_DATA/detectors/mrcnn.pth"
  download_file "$MRCNN_URL" "$ALFWORLD_DATA/detectors/mrcnn.pth" 1
else
  echo "[SKIP] mrcnn weights present"
fi

# --- logic files from pip package ---
echo "[INFO] copying logic files (alfred.pddl, alfred.twl2) ..."
python - <<'PY'
from alfworld.info import ALFRED_PDDL_PATH, ALFRED_TWL2_PATH, ALFWORLD_DATA
import os
import shutil

logic_dir = os.path.join(ALFWORLD_DATA, "logic")
os.makedirs(logic_dir, exist_ok=True)
shutil.copy(ALFRED_PDDL_PATH, os.path.join(logic_dir, "alfred.pddl"))
shutil.copy(ALFRED_TWL2_PATH, os.path.join(logic_dir, "alfred.twl2"))
print(f"[OK] logic -> {logic_dir}")
PY

# --- verify ---
errors=0
check_path() {
  if [ ! -e "$1" ]; then
    echo "[ERROR] missing: $1"
    errors=$((errors + 1))
  else
    echo "[OK] $1"
  fi
}

check_path "$ALFWORLD_DATA/json_2.1.1/train"
if ! find "$ALFWORLD_DATA/json_2.1.1/train" -name 'game.tw-pddl' -print -quit 2>/dev/null | grep -q .; then
  echo "[ERROR] missing: game.tw-pddl under json_2.1.1/train (tw-pddl zip not extracted?)"
  errors=$((errors + 1))
else
  echo "[OK] game.tw-pddl files in train/"
fi
check_path "$ALFWORLD_DATA/logic/alfred.pddl"
check_path "$ALFWORLD_DATA/logic/alfred.twl2"
check_path "$ALFWORLD_DATA/detectors/mrcnn.pth"

if [ "$errors" -gt 0 ]; then
  echo "[FAIL] AlfWorld data incomplete ($errors missing). Retry: FORCE=1 bash scripts/download_alfworld_data.sh"
  exit 1
fi

echo "[DONE] AlfWorld data ready in $ALFWORLD_DATA"
echo "       Run: bash examples/ppo_trainer/ppo_alfworld.sh"
