conda create -n verl-agent-311 python==3.11 -y

conda activate verl-agent-311

pip install -e . "vllm==0.8.5"

pip install flash-attn==2.7.4.post1 --no-build-isolation

pip install -e agent_system/environments/env_package/craftext/CrafText-super_igor_env_build
pip install datasets
pip install tensorboard