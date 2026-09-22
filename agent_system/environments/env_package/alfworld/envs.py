# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torchvision.transforms as T
import ray

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def get_obs_image(env):
    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i]*= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

def compute_reward(info, multi_modal=False):
    if multi_modal:
        raw = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        raw = 10.0 * float(info['won'])
    # Normalize to [0, 1] so actor-value return bins (e.g. 0..1) match MC targets.
    return raw / 10.0


def _info_at_index(info, idx):
    out = {}
    for key, value in info.items():
        if isinstance(value, (list, tuple)) and len(value) > idx:
            out[key] = value[idx]
        else:
            out[key] = value
    return out


class AlfworldWorker:
    """
    Single Ray actor with one TextWorld gym env (batch_size=N parallel slots).
    Avoids N separate Ray processes each registering the full AlfWorld game list.
    """

    def __init__(self, alf_config_path, seed, is_train, eval_dataset, batch_size):
        # Guard against JAX-threaded parent/import side-effects before TextWorld env creation.
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        self.batch_size = int(batch_size)
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        train_eval = 'train' if is_train else eval_dataset
        base_env = get_environment(env_type)(config, train_eval=train_eval)
        self.multi_modal = env_type == 'AlfredThorEnv'
        self.env = base_env.init_env(batch_size=self.batch_size)
        self.env.seed(seed)

    def step(self, actions):
        obs, scores, dones, infos = self.env.step(actions)
        infos['observation_text'] = obs
        return obs, scores, dones, infos

    def reset(self):
        obs, infos = self.env.reset()
        infos['observation_text'] = obs
        return obs, infos

    def getobs(self):
        image = get_obs_image(self.env)
        return image.cpu()

class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.num_processes = env_num * group_n
        self.group_n = group_n

        env_worker = ray.remote(**resources_per_worker)(AlfworldWorker)
        self.worker = env_worker.remote(
            alf_config_path,
            seed,
            is_train,
            eval_dataset,
            self.num_processes,
        )

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

        obs, scores, dones, infos = ray.get(self.worker.step.remote(actions))

        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        for i in range(self.num_processes):
            info = _info_at_index(infos, i)
            text_obs_list.append(obs[i])
            dones_list.append(dones[i])
            info_list.append(info)
            self.prev_admissible_commands[i] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def reset(self):
        obs, infos = ray.get(self.worker.reset.remote())

        text_obs_list = []
        image_obs_list = []
        info_list = []

        for i in range(self.num_processes):
            info = _info_at_index(infos, i)
            text_obs_list.append(obs[i])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list

    def getobs(self):
        images = ray.get(self.worker.getobs.remote())
        if isinstance(images, torch.Tensor) and images.shape[0] == self.num_processes:
            return [images[i] for i in range(self.num_processes)]
        return images

    @property
    def get_admissible_commands(self):
        return self.prev_admissible_commands

    def close(self):
        ray.kill(self.worker)

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)
