#!/bin/bash
# Base image only (conda + torch). caged_craftext deps install on first `docker/start.sh` via volume.
set -e
cd "$(dirname "$0")/.."
docker build -f docker/Dockerfile.A100 -t safe_llm_img .
