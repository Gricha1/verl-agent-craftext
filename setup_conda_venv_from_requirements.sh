conda create -n verl-agent python==3.12 -y

conda activate verl-agent

pip install -r requirements_full_old.txt
pip install -e .

pip install -e agent_system/environments/env_package/craftext/CrafText-super_igor_env_build
pip install datasets