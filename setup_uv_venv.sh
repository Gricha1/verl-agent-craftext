uv venv -p 3.12

source .venv/bin/activate

uv pip install -e . "vllm==0.8.5"
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install psutil numpy
uv pip install flash-attn==2.7.4.post1 --no-build-isolation

uv pip install -e agent_system/environments/env_package/craftext/CrafText-super_igor_env_build
uv pip install datasets
uv pip install tensorboard