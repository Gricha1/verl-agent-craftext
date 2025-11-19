conda create -n verl-agent python==3.12 -y

conda activate verl-agent

pip install -e . "vllm==0.8.5"

pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install psutil numpy
pip install flash-attn==2.7.4.post1 --no-build-isolation

pip install -e agent_system/environments/env_package/craftext/CrafText-super_igor_env_build
pip install datasets
pip install tensorboard