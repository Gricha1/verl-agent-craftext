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
import re
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import OmegaConf

from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.environments.prompts import *
from agent_system.memory import SearchMemory, SimpleMemory


def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


class Gsm8kEnvironmentManager(EnvironmentManagerBase):
    """Single-turn GSM8K: question from parquet env_kwargs, reward from rule-based scorer."""

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        if kwargs is None:
            raise ValueError("GSM8K requires env_kwargs in each parquet row / training batch")
        if isinstance(kwargs, np.ndarray):
            kwargs = kwargs.tolist()
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = list(obs)
        observations = {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": list(obs),
        }
        value_text = self.build_value_text_obs(infos, init=True)
        if value_text is not None:
            observations["value_text"] = value_text
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": list(next_obs),
        }
        value_text = self.build_value_text_obs(infos)
        if value_text is not None:
            next_observations["value_text"] = value_text
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)
        return next_observations, rewards, dones, infos

    def build_text_obs(self, questions: List[str], init: bool = False) -> List[str]:
        del init
        return [str(q) for q in questions]

    def build_value_text_obs(self, infos: List[Dict], init: bool = False) -> Optional[List[str]]:
        del init
        value_template_type = getattr(self.config.env, "value_prompt_template_type", None)
        if not value_template_type:
            return None
        if value_template_type != "single_token_return":
            raise ValueError(f"Unsupported value_prompt_template_type: {value_template_type!r}")

        from agent_system.environments.env_package.caged_craftext.projection import (
            get_single_token_return_template_no_his,
            extract_reasoning_text,
        )
        from agent_system.environments.env_package.caged_craftext.return_tokens import (
            return_bin_spec_from_env,
            return_token_legend_for_spec,
        )

        template_no_his = get_single_token_return_template_no_his()
        legend = return_token_legend_for_spec(return_bin_spec_from_env(self.config.env))
        prompts = []
        for info in infos:
            task = info.get("instruction") or ""
            observation = info.get("text_render") or "(awaiting solution)"
            prompts.append(
                template_no_his.format(
                    task_description=task,
                    current_observation=observation,
                    return_bin_legend=legend,
                )
            )
        return prompts

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item["active_masks"]:
                info = total_infos[batch_idx][i]
                won_value = float(info.get("won", 0.0))
                success["success_rate"].append(won_value)
                data_source = info.get("data_source")
                if data_source:
                    success[f"{data_source}_success_rate"].append(won_value)
                return


class SearchEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for SearchEnv.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SearchMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs

        self.memory.reset(batch_size=len(obs))

        observations = {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": obs.copy()
        }
        
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({
            "search": actions,
            "information": next_obs,
        })

        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": next_obs.copy()
        }
        
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False
    ) -> List[str]:
        postprocess_text_obs: List[str] = []

        if not init and self.config.env.history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search"
            )

        for i in range(len(text_obs)):
            if init or self.config.env.history_length <= 0:
                obs_i = SEARCH_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i]
                )
            else:
                obs_i = SEARCH_TEMPLATE.format(
                    task_description=self.tasks[i],
                    memory_context=memory_ctx[i],
                    step_count=len(self.memory[i]),
                )
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs


    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won_value)
                return  # Exit after finding the first active mask
            

class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        admissible_commands = self.envs.get_admissible_commands
        full_text_obs = self.build_text_obs(text_obs, admissible_commands, init=True)
        observations = {
            'text': full_text_obs,
            'image': image_obs,
            'anchor': text_obs,
            'admissible_commands': admissible_commands,
        }
        value_text = self.build_value_text_obs(text_obs, init=True)
        if value_text is not None:
            observations['value_text'] = value_text
        return observations, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        admissible_commands = self.envs.get_admissible_commands
        full_text_obs = self.build_text_obs(text_obs, admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            info['executed_action'] = actions[i]

        next_observations = {
            'text': full_text_obs,
            'image': image_obs,
            'anchor': text_obs,
            'admissible_commands': admissible_commands,
        }
        value_text = self.build_value_text_obs(text_obs)
        if value_text is not None:
            next_observations['value_text'] = value_text
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_value_text_obs(self, text_obs: List[str], init: bool = False) -> Optional[List[str]]:
        del init
        value_template_type = getattr(self.config.env, "value_prompt_template_type", None)
        if not value_template_type:
            return None
        if value_template_type != "single_token_return":
            raise ValueError(f"Unsupported value_prompt_template_type: {value_template_type!r}")

        from agent_system.environments.env_package.caged_craftext.projection import (
            get_single_token_return_template_no_his,
        )
        from agent_system.environments.env_package.caged_craftext.return_tokens import (
            return_bin_spec_from_env,
            return_token_legend_for_spec,
        )

        template_no_his = get_single_token_return_template_no_his()
        legend = return_token_legend_for_spec(return_bin_spec_from_env(self.config.env))
        prompts = []
        for i, obs in enumerate(text_obs):
            task = self.tasks[i] if i < len(self.tasks) else "No task"
            prompts.append(
                template_no_his.format(
                    task_description=task,
                    current_observation=self._observation_for_prompt(obs, task),
                    return_bin_legend=legend,
                )
            )
        return prompts
    
    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")

    def _observation_for_prompt(self, raw_obs: str, task: str) -> str:
        """Drop task line and TextWorld banner — task is already in the prompt header."""
        obs = raw_obs or ""
        task_marker = "Your task is to: "
        if task_marker in obs:
            obs = obs.replace(f"{task_marker}{task}", "", 1)
            if task_marker in obs:
                obs = obs.partition(task_marker)[0]
        obs = obs.replace("-= Welcome to TextWorld, ALFRED! =-", "").strip()
        return obs.strip()
        

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        prompt_template_type = getattr(self.config.env, "prompt_template_type", "default_template")
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")

        for i in range(len(text_obs)):
            if prompt_template_type == "single_token_action":
                from agent_system.environments.prompts.alfworld_single_token import (
                    ALFWORLD_SINGLE_TOKEN_ACTION_TEMPLATE,
                )
                from agent_system.environments.env_package.alfworld.action_tokens import (
                    build_admissible_legend,
                )

                pool = [s for s in admissible_actions[i] if s != "help"]
                task = self.tasks[i] if i < len(self.tasks) else "No task"
                obs = ALFWORLD_SINGLE_TOKEN_ACTION_TEMPLATE.format(
                    task_description=task,
                    current_observation=self._observation_for_prompt(text_obs[i], task),
                    action_legend=build_admissible_legend(pool),
                )
            else:
                reformatted_admissible_actions = "\n ".join(
                    f"'{s}'" for s in admissible_actions[i] if s != "help"
                )
                if init or self.config.env.history_length <= 0:
                    obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                        current_observation=text_obs[i],
                        admissible_actions=reformatted_admissible_actions,
                    )
                else:
                    obs = ALFWORLD_TEMPLATE.format(
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        history_length=valid_lens[i],
                        action_history=memory_contexts[i],
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                        admissible_actions=reformatted_admissible_actions,
                    )

            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]
        
        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break


class SokobanEnvironmentManager(EnvironmentManagerBase):
    ACTION_LOOKUP = {
        0: "Still",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }
    def __init__(self, envs, projection_f, config):
        self.is_multi_modal = envs.mode == 'rgb_array'
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs):
        obs, infos = self.envs.reset()
        if self.is_multi_modal:
            obs = np.array(obs, obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            observations = {
                'text': self.build_text_obs(infos, init=True), 
                'image': obs,   
                'anchor': obs
            }
        else:
            self.pre_text_obs = obs
            observations = {
                'text': self.build_text_obs(infos, obs, init=True),
                'image': None,
                'anchor': obs
            }
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        next_obs, rewards, dones, infos = self.envs.step(actions)

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        self.memory.store({'text_obs': self.pre_text_obs, 'action': [self.ACTION_LOOKUP[act] for act in actions]})
        if self.is_multi_modal:
            next_obs = np.array(next_obs, next_obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            next_observations = {
                'text': self.build_text_obs(infos),  
                'image': next_obs,
                'anchor': next_obs 
            }
        else:
            self.pre_text_obs = next_obs
            next_observations = {
                'text': self.build_text_obs(infos, next_obs),  
                'image': None, 
                'anchor': next_obs 
            }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self, infos, text_obs: List[str]=None, init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(infos)):
            if init or self.config.env.history_length <= 0:
                obs = SOKOBAN_VISUAL_TEMPLATE if self.is_multi_modal \
                 else SOKOBAN_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                )
            else:
                if self.is_multi_modal:
                    obs = SOKOBAN_VISUAL_TEMPLATE
                else:
                    obs = SOKOBAN_TEMPLATE.format(
                        step_count=len(self.memory[i]),
                        history_length=valid_lens[i],
                        action_history=memory_contexts[i],
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs


class GymCardEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(infos), 'image': obs, 'anchor': obs.copy()}
        
        return observations, infos

    def step(self, text_actions: List[str]):
        next_observations, rewards, dones, infos = super().step(text_actions)
        
        # add text observation to next_observations
        next_observations['text'] = self.build_text_obs(infos)
        next_observations['anchor'] = next_observations['image'].copy()

        return next_observations, rewards, dones, infos


    def build_text_obs(self, infos: Tuple[Dict]=None) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            if 'ezpoints' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_EZPOINTS_TEMPLATE.format(text_formula=text_formula)
            elif 'points24' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_POINTS24_TEMPLATE.format(text_formula=text_formula)
            elif 'numberline' in self.config.env.env_name.lower():
                obs = GYM_CARDS_NUMBERLINE_TEMPLATE
            elif "blackjack" in self.config.env.env_name.lower():
                obs = GYM_CARDS_BLACKJACK_TEMPLATE
            else:
                raise ValueError(f"Unsupported environment: {self.config.env.env_name}")
            postprocess_text_obs.append(obs)
        return postprocess_text_obs


class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.tasks = self.extract_task(obs)
        obs = self.format_obs(obs)
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(obs, infos, init=True), 
                        'image': None, 
                        'anchor': obs.copy()
                        }
        self.pre_text_obs = obs
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_obs = self.format_obs(next_obs)

        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = next_obs

        next_observations = {
            'text': self.build_text_obs(next_obs, infos),
            'image': None,
            'anchor': next_obs.copy()
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1]=='Instruction:'
            tasks.append(parts[2])
        return tasks
    
    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            except:
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs
    
    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions
            
    def build_text_obs(self, text_obs: List[str], infos: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            
            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

            if init or self.config.env.history_length <= 0:
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            else:
                obs = WEBSHOP_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
                if len(obs) > 13000:
                    print(f"Warning len(obs)={len(obs)} is too long")
                    obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info['task_score'])
                success['success_rate'].append(won_value)
                success['webshop_task_score (not success_rate)'].append(score_value)
                return

class AppWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, infos = self.envs.reset()
        
        self.supervisors = [info['supervisor'] for info in infos]
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = text_obs.copy()
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, init=True)
        return {'text': full_text_obs, 'image': None, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)

        self.memory.store({'text_obs': text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': None, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    

    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if init and self.supervisors is not None:
            for i in range(len(text_obs)):
                obs = APPWORLD_TEMPLATE_NO_HIS.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                    )
                postprocess_text_obs.append(obs)
        else:
            for i in range(len(text_obs)):
                # Get last `history_length` steps
                recent_history = self.memory[i][-self.config.env.history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.memory[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\nCode {step_number}: \n{action}\n\nResult {step_number}: \n{env_obs}\n"
                
                if len(action_history) > 10000:
                    action_history = "... " + action_history[-10000:]

                obs = APPWORLD_TEMPLATE.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
                postprocess_text_obs.append(obs)
        return postprocess_text_obs

from agent_system.environments.env_package.craftext.projection import CRAFTEXT_TEMPLATE, CRAFTEXT_TEMPLATE_NO_HIS, CRAFTEXT_VL_TEMPLATE_NO_HIS
from agent_system.environments.env_package.craftext.projection_oracle import (
    CRAFTEXT_TEMPLATE_ORACLE, 
    CRAFTEXT_TEMPLATE_ORACLE_NO_HIS, 
    CRAFTEXT_VL_TEMPLATE_ORACLE_NO_HIS
)
# Импортируем шаблоны для caged_craftext (используем те же, что и для обычного craftext)
from agent_system.environments.env_package.caged_craftext.projection import (
    CRAFTEXT_TEMPLATE, 
    CRAFTEXT_TEMPLATE_NO_HIS,
    get_craftext_template,
    get_craftext_template_no_his,
    get_craftext_extended_template_no_his,
    get_single_token_action_template_no_his,
    get_single_token_action_vl_template_no_his,
    get_single_token_return_template_no_his,
    get_single_token_return_vl_template_no_his,
    format_executed_actions_history,
    CRAFTEXT_EXTENDED_TEMPLATE_NO_HIS,
    ACTION_TO_TEXT as CAGED_ACTION_TO_TEXT,
)


class CraftextEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.tasks = [info.get('instruction', 'No instruction found') for info in infos]
        text_renders = [info.get('text_render', 'The world is empty.') for info in infos]
                
        if self.config.env.env_name == "craftext/CraftextEnv":
            observations = {
                'text': self.build_text_obs(text_renders, infos, init=True), 
                # 'image': obs,
                'anchor': text_renders.copy()
            }

        if self.config.env.env_name == "craftext/CraftextVLEnv":
            observations = {
                'text': self.build_text_obs(text_renders, infos, init=True), 
                'image': obs,
                'anchor': text_renders.copy()
            }

        self.pre_text_obs = text_renders
        self.memory.reset(batch_size=len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        action_ids, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(action_ids)
        next_text_renders = [info.get('text_render', 'The world is empty.') for info in infos]

        self.memory.store({'text_obs': self.pre_text_obs, 'action': text_actions})
        self.pre_text_obs = next_text_renders

        if self.config.env.env_name == "craftext/CraftextEnv":
            next_observations = {
                'text': self.build_text_obs(next_text_renders, infos),
                # 'image': next_obs,
                'anchor': next_text_renders.copy()
            }

        if self.config.env.env_name == "craftext/CraftextVLEnv":
            next_observations = {
                'text': self.build_text_obs(next_text_renders, infos),
                'image': next_obs,
                'anchor': next_text_renders.copy()
            }
        
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            # Also expose projected discrete action id for downstream logging/analysis.
            try:
                info['action_id'] = int(to_numpy(action_ids[i]))
            except Exception:
                info['action_id'] = -1
            # Raw model output (may include tags)
            info['action_text'] = text_actions[i] if i < len(text_actions) else ""
            # Parsed discrete action name (for debugging/video overlays)
            try:
                aid = int(info.get("action_id", -1))
                info["action_name"] = CAGED_ACTION_TO_TEXT[aid] if 0 <= aid < len(CAGED_ACTION_TO_TEXT) else ""
            except Exception:
                info["action_name"] = ""

        return next_observations, to_numpy(rewards), to_numpy(dones), infos

    def build_text_obs(self, text_renders: List[str], infos: List[Dict], init: bool = False) -> List[str]:
        final_prompts = []
        
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                self.config.env.history_length,
                obs_key="text_obs",
                action_key="action"
            )
        
        for i in range(len(text_renders)):
            # <--- ИЗМЕНЕНО: Шаблоны теперь самодостаточны, просто форматируем их
            if self.config.env.env_name == "craftext/CraftextVLEnv":
                prompt = CRAFTEXT_VL_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_renders[i]
                )
            else:
                if init or self.config.env.history_length <= 0:
                    prompt = CRAFTEXT_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i],
                        current_observation=text_renders[i]
                    )
                else:
                    prompt = CRAFTEXT_TEMPLATE.format(
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        action_history=memory_contexts[i],
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_renders[i]
                    )

            final_prompts.append(prompt)

        return final_prompts


class CagedCraftextEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager для Caged Craftext - безопасной версии Craftext с CMDP поддержкой.
    Аналогичен CraftextEnvironmentManager, но работает с caged_craftext окружениями.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def set_record_video(self, record: bool = True, env_idx: int = 0, env_idxs: list[int] | None = None):
        """Включить/выключить запись кадров для одного env (для записи видео траектории в Comet ML)."""
        if hasattr(self.envs, 'set_record_video_worker_idxs'):
            if env_idxs is not None:
                idxs = list(env_idxs) if record else None
            else:
                idxs = [env_idx] if record else None
            self.envs.set_record_video_worker_idxs(idxs)

    def set_validation_scenario_idx(self, idx: int | None):
        """Pin debug_square task (0=stone, 1=wood, 2=water) for the next env reset."""
        if hasattr(self.envs, "set_reset_instruction_override"):
            self.envs.set_reset_instruction_override(idx)

    def set_validation_scenario_pins(self, pins: dict[int, int] | None):
        """One-shot: worker slot -> scenario (e.g. {0:0, 1:1, 2:2}) on next reset."""
        if hasattr(self.envs, "set_validation_scenario_pins"):
            self.envs.set_validation_scenario_pins(pins)

    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.tasks = [info.get('instruction', 'No instruction found') for info in infos]
        text_renders = [info.get('text_render', 'The world is empty.') for info in infos]
        
        # Извлекаем constraint информацию, если есть
        constraints = [info.get('constraint', '') for info in infos]
                
        if self.config.env.env_name == "caged_craftext/CagedCraftextEnv":
            observations = {
                'text': self.build_text_obs(text_renders, infos, init=True), 
                'anchor': text_renders.copy()
            }
            value_text = self.build_value_text_obs(text_renders, infos, init=True)
            if value_text is not None:
                observations['value_text'] = value_text
            # Добавляем constraint в observations, если есть
            if any(constraints):
                observations['constraint'] = constraints

        if self.config.env.env_name == "caged_craftext/CagedCraftextVLEnv":
            observations = {
                'text': self.build_text_obs(text_renders, infos, init=True), 
                'image': obs,
                'anchor': text_renders.copy()
            }
            value_text = self.build_value_text_obs(text_renders, infos, init=True)
            if value_text is not None:
                observations['value_text'] = value_text

        self.pre_text_obs = text_renders
        self.memory.reset(batch_size=len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        action_ids, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(action_ids)
        next_text_renders = [info.get('text_render', 'The world is empty.') for info in infos]

        # Optimistic auto-reset can swap instruction_idx mid-rollout; keep prompts in sync.
        if not hasattr(self, "tasks") or len(self.tasks) != len(infos):
            self.tasks = ["No instruction found"] * len(infos)
        for i, info in enumerate(infos):
            instr = info.get("instruction")
            if instr:
                self.tasks[i] = instr

        # Store discrete action tokens for episode memory (cleaner than raw LLM text).
        try:
            from agent_system.environments.env_package.caged_craftext.action_tokens import (
                action_token_label,
            )
            stored_actions = []
            for i, raw in enumerate(text_actions):
                try:
                    aid = int(to_numpy(action_ids[i]))
                    # Invalid actions must not enter the history as full raw LLM text:
                    # with the reasoning template a raw response can be up to
                    # max_response_length (320) tokens and ~10 such entries overflow
                    # the prompt (observed crash: sequence_length=3228 > 3072).
                    stored_actions.append(action_token_label(aid) if aid >= 0 else action_token_label(0))
                except Exception:
                    stored_actions.append(action_token_label(0))
        except Exception:
            stored_actions = list(text_actions)
        # Extract reasoning text from LLM output (for reasoning mode)
        reasoning_texts = []
        try:
            from agent_system.environments.env_package.caged_craftext.projection import extract_reasoning_text
            for raw in text_actions:
                reasoning_texts.append(extract_reasoning_text(str(raw)))
        except Exception:
            reasoning_texts = [""] * len(text_actions)
        
        self.memory.store({'text_obs': self.pre_text_obs, 'action': stored_actions, 'reasoning': reasoning_texts})
        self.pre_text_obs = next_text_renders

        # Извлекаем constraint информацию, если есть
        constraints = [info.get('constraint', '') for info in infos]
        
        if self.config.env.env_name == "caged_craftext/CagedCraftextEnv":
            next_observations = {
                'text': self.build_text_obs(next_text_renders, infos),
                'anchor': next_text_renders.copy()
            }
            value_text = self.build_value_text_obs(next_text_renders, infos)
            if value_text is not None:
                next_observations['value_text'] = value_text
            # Добавляем constraint в observations, если есть
            if any(constraints):
                next_observations['constraint'] = constraints

        if self.config.env.env_name == "caged_craftext/CagedCraftextVLEnv":
            next_observations = {
                'text': self.build_text_obs(next_text_renders, infos),
                'image': next_obs,
                'anchor': next_text_renders.copy()
            }
            value_text = self.build_value_text_obs(next_text_renders, infos)
            if value_text is not None:
                next_observations['value_text'] = value_text
        
        for i, info in enumerate(infos):
            # Keep parity with CraftextEnvironmentManager: expose validity and discrete action ids.
            try:
                info['is_action_valid'] = to_numpy(valids[i])
            except Exception:
                info['is_action_valid'] = False
            try:
                info['action_id'] = int(to_numpy(action_ids[i]))
            except Exception:
                info['action_id'] = -1
            info['action_text'] = text_actions[i] if i < len(text_actions) else ""
            try:
                aid = int(info.get("action_id", -1))
                info["action_name"] = CAGED_ACTION_TO_TEXT[aid] if 0 <= aid < len(CAGED_ACTION_TO_TEXT) else ""
            except Exception:
                info["action_name"] = ""

        return next_observations, rewards, dones, infos

    def _executed_actions_history(self, env_idx: int, init: bool = False) -> str:
        """Format actions already taken in this episode for actor/critic prompts."""
        history_length = int(getattr(self.config.env, "history_length", 0) or 0)
        if init or history_length <= 0 or self.memory is None or self.memory._data is None:
            return "(none)"
        try:
            records = self.memory[env_idx]
        except Exception:
            return "(none)"
        if not records:
            return "(none)"
        recent = records[-history_length:]
        actions = [rec.get("action", "") for rec in recent]
        return format_executed_actions_history(actions)

    def _reasoning_history(self, env_idx: int, init: bool = False) -> str:
        """Format last N reasoning texts from this episode (oldest → newest)."""
        if init or self.memory is None or self.memory._data is None:
            return "(none)"
        try:
            records = self.memory[env_idx]
        except Exception:
            return "(none)"
        if not records:
            return "(none)"
        # N = env.reasoning_history_length (default 5, reference value)
        n_reasoning = int(getattr(self.config.env, "reasoning_history_length", 5))
        recent = records[-n_reasoning:]
        reasonings = [rec.get("reasoning", "") for rec in recent]
        # Filter out empty strings
        reasonings = [r for r in reasonings if r]
        if not reasonings:
            return "(none)"
        # Format as numbered list
        parts = []
        for i, r in enumerate(reasonings, 1):
            parts.append(f"{i}. {r}")
        return chr(10).join(parts)

    def build_text_obs(self, text_renders: List[str], infos: List[Dict], init: bool = False) -> List[str]:
        """
        Строит текстовые наблюдения из рендеров и инфо.
        Для Caged Craftext также может включать информацию о constraint.
        """
        # Prefer live instruction from infos (survives optimistic auto-reset).
        # Получаем тип шаблона из config (по умолчанию default_template)
        prompt_template_type = getattr(self.config.env, 'prompt_template_type', 'default_template')
        is_vl_env = "vlenv" in str(getattr(self.config.env, "env_name", "")).lower()
        
        # Получаем параметр enable_reasoning из config (по умолчанию True для обратной совместимости)
        enable_reasoning = getattr(self.config.env, 'enable_reasoning', True)
        history_length = int(getattr(self.config.env, "history_length", 0) or 0)
        
        # Выбираем шаблоны в зависимости от prompt_template_type
        if prompt_template_type == 'extended_template':
            # Используем extended шаблон (только без истории, так как history_length=0)
            template_no_his = get_craftext_extended_template_no_his()
            # Для extended шаблона не используем шаблон с историей, так как он не определен
            template = template_no_his  # Fallback
        elif prompt_template_type == 'single_token_action_reasoning':
            if is_vl_env:
                from agent_system.environments.env_package.caged_craftext.projection import get_single_token_action_reasoning_vl_template_no_his
                template_no_his = get_single_token_action_reasoning_vl_template_no_his()
            else:
                from agent_system.environments.env_package.caged_craftext.projection import get_single_token_action_reasoning_template_no_his
                template_no_his = get_single_token_action_reasoning_template_no_his()
            template = template_no_his
        elif prompt_template_type == 'single_token_action':
            if is_vl_env:
                template_no_his = get_single_token_action_vl_template_no_his()
            else:
                template_no_his = get_single_token_action_template_no_his()
            template = template_no_his
        else:
            # Используем обычные шаблоны с учетом enable_reasoning
            template = get_craftext_template(enable_reasoning=enable_reasoning)
            template_no_his = get_craftext_template_no_his(enable_reasoning=enable_reasoning)
        
        final_prompts = []
        
        for i, (text_render, info) in enumerate(zip(text_renders, infos)):
            task = info.get("instruction") or (
                self.tasks[i] if hasattr(self, "tasks") and i < len(self.tasks) else "No instruction found"
            )
            
            # Получаем constraint, если есть
            constraint = info.get('constraint', '')
            action_history_str = self._executed_actions_history(i, init=init)
            
            # Строим промпт
            if prompt_template_type == 'single_token_action':
                if is_vl_env:
                    prompt = template_no_his.format(
                        task_description=task,
                        action_history=action_history_str,
                    )
                else:
                    prompt = template_no_his.format(
                        task_description=task,
                        current_observation=text_render,
                        action_history=action_history_str,
                    )
            elif prompt_template_type == 'single_token_action_reasoning':
                reasoning_history_str = self._reasoning_history(i, init=init)
                if is_vl_env:
                    prompt = template_no_his.format(
                        task_description=task,
                        action_history=action_history_str,
                        reasoning_history=reasoning_history_str,
                    )
                else:
                    prompt = template_no_his.format(
                        task_description=task,
                        current_observation=text_render,
                        action_history=action_history_str,
                        reasoning_history=reasoning_history_str,
                    )
            elif prompt_template_type == 'extended_template' or history_length <= 0:
                if prompt_template_type == 'extended_template':
                    prompt = template_no_his.format(
                        task_description=task,
                        current_observation=text_render,
                    )
                elif action_history_str and action_history_str != "(none)":
                    # Обычный шаблон с историей (если есть)
                    try:
                        records = self.memory[i] if self.memory is not None else []
                    except Exception:
                        records = []
                    prompt = template.format(
                        task_description=task,
                        step_count=len(records),
                        action_history=action_history_str,
                        current_step=len(records),
                        current_observation=text_render
                    )
                else:
                    # Обычный шаблон без истории
                    prompt = template_no_his.format(
                        task_description=task,
                        current_observation=text_render
                    )
            else:
                # Обычный шаблон с историей
                try:
                    records = self.memory[i] if self.memory is not None else []
                except Exception:
                    records = []
                prompt = template.format(
                    task_description=task,
                    step_count=len(records),
                    action_history=action_history_str if action_history_str else "No actions taken yet.",
                    current_step=len(records),
                    current_observation=text_render
                )
            
            # Добавляем constraint информацию в промпт, если есть
            if constraint:
                prompt += f"\n\n**CONSTRAINT:** {constraint}"

            final_prompts.append(prompt)

        return final_prompts

    def build_value_text_obs(self, text_renders: List[str], infos: List[Dict], init: bool = False) -> Optional[List[str]]:
        """Build critic/value prompts (separate from actor prompt). Returns None if disabled."""
        value_template_type = getattr(self.config.env, 'value_prompt_template_type', None)
        if not value_template_type:
            return None
        is_vl_env = "vlenv" in str(getattr(self.config.env, "env_name", "")).lower()
        if value_template_type == 'single_token_return_vl':
            if not is_vl_env:
                raise ValueError("single_token_return_vl requires caged_craftext/CagedCraftextVLEnv")
            template_no_his = get_single_token_return_vl_template_no_his()
        elif value_template_type == 'single_token_return':
            template_no_his = get_single_token_return_template_no_his()
        else:
            raise ValueError(f"Unsupported value_prompt_template_type: {value_template_type!r}")

        from agent_system.environments.env_package.caged_craftext.return_tokens import (
            return_bin_spec_from_env,
            return_token_legend_for_spec,
        )

        legend = return_token_legend_for_spec(return_bin_spec_from_env(self.config.env))
        final_prompts = []
        for i, (text_render, info) in enumerate(zip(text_renders, infos)):
            task = info.get("instruction") or (
                self.tasks[i] if hasattr(self, "tasks") and i < len(self.tasks) else "No instruction found"
            )
            constraint = info.get('constraint', '')
            action_history_str = self._executed_actions_history(i, init=init)
            if value_template_type == 'single_token_return_vl':
                prompt = template_no_his.format(
                    task_description=task,
                    return_bin_legend=legend,
                    action_history=action_history_str,
                )
            else:
                prompt = template_no_his.format(
                    task_description=task,
                    current_observation=text_render,
                    return_bin_legend=legend,
                    action_history=action_history_str,
                )
            if constraint:
                prompt += f"\n\n**CONSTRAINT:** {constraint}"
            final_prompts.append(prompt)
        return final_prompts


class CraftextOracleEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager для Craftext с поддержкой оракла.
    Обрабатывает вопросы к оракулу и включает ответы в наблюдения.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.tasks = [info.get('instruction', 'No instruction found') for info in infos]
        text_renders = [info.get('text_render', 'The world is empty.') for info in infos]
                
        if self.config.env.env_name == "craftext/CraftextOracleEnv":
            observations = {
                'text': self.build_text_obs(text_renders, infos, init=True), 
                'anchor': text_renders.copy()
            }
        elif self.config.env.env_name == "craftext/CraftextOracleVLEnv":
            observations = {
                'text': self.build_text_obs(text_renders, infos, init=True), 
                'image': obs,
                'anchor': text_renders.copy()
            }
        else:
            # Fallback для неизвестных имен сред
            observations = {
                'text': self.build_text_obs(text_renders, infos, init=True), 
                'anchor': text_renders.copy()
            }

        self.pre_text_obs = text_renders
        self.memory.reset(batch_size=len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        # Используем oracle projection, которая возвращает actions, valids, questions
        action_ids, valids, questions = self.projection_f(text_actions)
        
        # Передаем вопросы в среду
        next_obs, rewards, dones, infos = self.envs.step(action_ids, questions)
        next_text_renders = [info.get('text_render', 'The world is empty.') for info in infos]

        self.memory.store({'text_obs': self.pre_text_obs, 'action': text_actions})
        self.pre_text_obs = next_text_renders

        if self.config.env.env_name == "craftext/CraftextOracleEnv":
            next_observations = {
                'text': self.build_text_obs(next_text_renders, infos),
                'anchor': next_text_renders.copy()
            }
        elif self.config.env.env_name == "craftext/CraftextOracleVLEnv":
            next_observations = {
                'text': self.build_text_obs(next_text_renders, infos),
                'image': next_obs,
                'anchor': next_text_renders.copy()
            }
        else:
            # Fallback для неизвестных имен сред
            next_observations = {
                'text': self.build_text_obs(next_text_renders, infos),
                'anchor': next_text_renders.copy()
            }
        
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            # Добавляем информацию о вопросе (если был)
            if questions[i] is not None:
                info['question'] = questions[i]

        return next_observations, to_numpy(rewards), to_numpy(dones), infos

    def build_text_obs(self, text_renders: List[str], infos: List[Dict], init: bool = False) -> List[str]:
        final_prompts = []
        
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                self.config.env.history_length,
                obs_key="text_obs",
                action_key="action"
            )
        
        for i in range(len(text_renders)):
            # Получаем ответ оракла (если есть)
            oracle_answer = infos[i].get('oracle_answer', None)
            
            # Формируем наблюдение с учетом ответа оракла
            observation_text = text_renders[i]
            if oracle_answer:
                observation_text += f"\n\n**Oracle's answer to your question:** {oracle_answer}"
            
            # Используем oracle шаблоны
            if self.config.env.env_name == "craftext/CraftextOracleVLEnv":
                prompt = CRAFTEXT_VL_TEMPLATE_ORACLE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=observation_text
                )
            else:
                if init or self.config.env.history_length <= 0:
                    prompt = CRAFTEXT_TEMPLATE_ORACLE_NO_HIS.format(
                        task_description=self.tasks[i],
                        current_observation=observation_text
                    )
                else:
                    prompt = CRAFTEXT_TEMPLATE_ORACLE.format(
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        action_history=memory_contexts[i],
                        current_step=len(self.memory[i]) + 1,
                        current_observation=observation_text
                    )

            final_prompts.append(prompt)

        return final_prompts

def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)

    if "gsm8k" in config.env.env_name.lower():
        from agent_system.environments.env_package.gsm8k import build_gsm8k_envs, gsm8k_projection

        score_method = str(getattr(config.env, "gsm8k_score_method", "strict"))
        format_score = float(getattr(config.env, "gsm8k_format_score", 0.0))
        correct_score = float(getattr(config.env, "gsm8k_correct_score", 1.0))
        _envs = build_gsm8k_envs(
            env_num=config.data.train_batch_size,
            group_n=group_n,
            score_method=score_method,
            format_score=format_score,
            correct_score=correct_score,
        )
        _val_envs = build_gsm8k_envs(
            env_num=config.data.val_batch_size,
            group_n=1,
            score_method=score_method,
            format_score=format_score,
            correct_score=correct_score,
        )
        projection_f = partial(gsm8k_projection)
        envs = Gsm8kEnvironmentManager(_envs, projection_f, config)
        val_envs = Gsm8kEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "search" in config.env.env_name.lower():
        from agent_system.environments.env_package.search import build_search_envs, search_projection
        _envs = build_search_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_search_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_config=config.env)

        projection_f = partial(search_projection)
        envs = SearchEnvironmentManager(_envs, projection_f, config)
        val_envs = SearchEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "gym_cards" in config.env.env_name.lower():
        from agent_system.environments.env_package.gym_cards import build_gymcards_envs, gym_projection
        _envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, resources_per_worker=resources_per_worker)
        _val_envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, resources_per_worker=resources_per_worker)
        
        projection_f = partial(gym_projection, env_name=config.env.env_name)
        envs = GymCardEnvironmentManager(_envs, projection_f, config)
        val_envs = GymCardEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import alfworld_projection, build_alfworld_envs
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        env_kwargs = {
            'eval_dataset': config.env.alfworld.eval_dataset, # 'eval_in_distribution' or 'eval_out_of_distribution'
        }
        _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        prompt_template_type = getattr(config.env, "prompt_template_type", "default_template")
        if prompt_template_type == "single_token_action":
            from agent_system.environments.env_package.alfworld.projection import (
                alfworld_single_token_projection,
            )

            projection_f = partial(alfworld_single_token_projection)
        else:
            projection_f = partial(alfworld_projection)

        # Check if we should use subtask-based GiGPO
        use_subtask_manager = config.env.get('use_subtask_gigpo', False)
        
        if use_subtask_manager:
            # Import here to avoid circular import
            from agent_system.environments.env_subtask_manager import AlfWorldSubtaskEnvironmentManager
            # Use subtask-based environment manager
            envs = AlfWorldSubtaskEnvironmentManager(_envs, projection_f, config)
            val_envs = AlfWorldSubtaskEnvironmentManager(_val_envs, projection_f, config)
        else:
            # Use standard environment manager
            envs = AlfWorldEnvironmentManager(_envs, projection_f, config)
            val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "sokoban" in config.env.env_name.lower():
        from agent_system.environments.env_package.sokoban import build_sokoban_envs, sokoban_projection
        env_kwargs = {
            'dim_room': config.env.sokoban.dim_room,
            'num_boxes': config.env.sokoban.num_boxes,
            'max_steps': config.env.max_steps,
            'search_depth': config.env.sokoban.search_depth
        }
        _envs = build_sokoban_envs(config.env.seed, config.data.train_batch_size, group_n, mode=config.env.sokoban.mode, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_sokoban_envs(config.env.seed + 1000, config.data.val_batch_size, 1, mode=config.env.sokoban.mode, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(sokoban_projection)
        envs = SokobanEnvironmentManager(_envs, projection_f, config)
        val_envs = SokobanEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        if config.env.webshop.use_small:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        else:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)

        projection_f = partial(webshop_projection)
        envs = WebshopEnvironmentManager(_envs, projection_f, config)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config)
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return envs, val_envs
    elif "appworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.appworld import appworld_projection, build_appworld_envs
        _envs = build_appworld_envs(dataset_name='train', seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, start_server_id=0, resources_per_worker=resources_per_worker)
        _val_envs = build_appworld_envs(dataset_name='test_normal', seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, start_server_id=config.data.train_batch_size*group_n, resources_per_worker=resources_per_worker)
        
        projection_f = partial(appworld_projection)
        envs = AppWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AppWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "oracle" in config.env.env_name.lower() and "craftext" in config.env.env_name.lower():
        # Модификация Craftext с поддержкой оракла
        # Проверяем наличие обоих слов "craftext" и "oracle" в имени среды
        print(f"[make_envs] Detected Oracle environment: {config.env.env_name}")
        from agent_system.environments.env_package.craftext.envs_oracle import build_craftext_envs_oracle
        from agent_system.environments.env_package.craftext.projection_oracle import craftext_projection_oracle

        # Параметры для среды Craftext
        env_kwargs = {
            'config_name': config.env.craftext_settings,
            'encode_form': 'embedding'
        }
        
        # Параметры оракла
        oracle_model_name = config.env.get('oracle_model_name', 'Qwen/Qwen2.5-1.5B-Instruct')
        
        # Создаем train и val среды с ораклом
        _envs = build_craftext_envs_oracle(
            seed=config.env.seed, 
            env_num=config.data.train_batch_size, 
            group_n=group_n, 
            is_train=True, 
            env_kwargs=env_kwargs, 
            resources_per_worker=resources_per_worker,
            oracle_model_name=oracle_model_name
        )
        _val_envs = build_craftext_envs_oracle(
            seed=config.env.seed + 1000, 
            env_num=config.data.val_batch_size, 
            group_n=1, 
            is_train=False, 
            env_kwargs=env_kwargs, 
            resources_per_worker=resources_per_worker,
            oracle_model_name=oracle_model_name
        )
        
        # Создаем функцию проекции с поддержкой вопросов
        projection_f = partial(craftext_projection_oracle)
        
        # Используем менеджер с поддержкой оракла
        envs = CraftextOracleEnvironmentManager(_envs, projection_f, config)
        val_envs = CraftextOracleEnvironmentManager(_val_envs, projection_f, config)
        
        return envs, val_envs
    elif "caged_craftext" in config.env.env_name.lower() or (config.env.env_name.lower().startswith("caged") and "craftext" in config.env.env_name.lower()):
        # 1. Импортируем все необходимое для Caged Craftext
        print(f"[make_envs] Detected Caged Craftext environment: {config.env.env_name}")
        from agent_system.environments.jax_device_config import (
            apply_jax_gpu_resources,
            configure_craftext_jax_backend,
            read_jax_gpu_settings,
        )
        use_jax_gpu, jax_gpu_fraction = read_jax_gpu_settings(config)
        configure_craftext_jax_backend(use_jax_gpu, gpu_mem_fraction=jax_gpu_fraction if use_jax_gpu else None)
        use_optimistic_parallel = bool(getattr(config.env, "use_optimistic_parallel", False))
        # MultiProcess Ray env workers: CPU only (N × jax_gpu_fraction GPU breaks 2-GPU nodes).
        if use_optimistic_parallel:
            resources_per_worker = apply_jax_gpu_resources(
                resources_per_worker, use_jax_gpu, jax_gpu_fraction
            )
        else:
            resources_per_worker = dict(resources_per_worker)
            resources_per_worker.pop("num_gpus", None)
            print(
                "[make_envs] use_optimistic_parallel=False: Ray env workers use CPU only "
                f"(num_cpus={resources_per_worker.get('num_cpus', '?')}); "
                "GPUs stay for vLLM/FSDP."
            )
        from agent_system.environments.env_package.caged_craftext.envs import (
            build_caged_craftext_envs,
            build_caged_craftext_envs_optimistic,
        )
        from agent_system.environments.env_package.caged_craftext.projection import craftext_projection

        use_pixel_obs = "vlenv" in str(config.env.env_name).lower()

        # 2. Указываем параметры для среды Caged Craftext
        env_kwargs = {
            'config_name': config.env.craftext_settings,  # Например, 'achievements_safe_caged'
            'encode_form': 'embedding',
            'observation_type': config.env.observation_type,
            # Avoid returning pixel observations for pure text/ascii envs to reduce RAM usage.
            'use_pixel_obs': use_pixel_obs,
            # Optimistic vec env: optional Ray actors (one per env) for parallel text_render only.
            'use_ray_text_render_workers': bool(getattr(config.env, "use_ray_text_render_workers", False)),
            # Per Ray text-render actor; too small → tasks queue / serial IPC. Used only for train env.
            'text_render_ray_num_cpus': float(getattr(config.env, "text_render_ray_num_cpus", 0.25)),
        }
        if str(config.env.craftext_settings) == "debug_square_8x8" or "debug_square" in str(
            config.env.craftext_settings
        ):
            env_kwargs['use_debug_square_map'] = True
            env_kwargs['config_name'] = str(config.env.craftext_settings)
        if getattr(config.env, "fixed_scenario_idx", None) is not None:
            env_kwargs["fixed_scenario_idx"] = int(config.env.fixed_scenario_idx)
        
        # 3. Создаем train и val среды
        optimistic_reset_ratio = getattr(config.env, "optimistic_reset_ratio", None)
        _debug_square = "debug_square" in str(config.env.craftext_settings)
        # debug_square val: 1 batched slot (stone/wood/water via scenario override), not 16 Ray workers.
        _val_env_num = 1 if _debug_square else int(config.data.val_batch_size)
        if use_optimistic_parallel:
            _envs = build_caged_craftext_envs_optimistic(
                seed=config.env.seed,
                env_num=config.data.train_batch_size,
                group_n=group_n,
                is_train=True,
                env_kwargs=env_kwargs,
                resources_per_worker=resources_per_worker,
                reset_ratio=optimistic_reset_ratio,
            )
            if _debug_square:
                # Same as validate_debug_square_actor_value_last_ckpt.sh: 1 Ray env worker, not optimistic JAX reset.
                print(
                    "[make_envs] debug_square: val env Ray MultiProcess env_num=1 "
                    "(stone/wood/water via set_validation_scenario_idx)"
                )
                _val_envs = build_caged_craftext_envs(
                    seed=config.env.seed + 1000,
                    env_num=1,
                    group_n=1,
                    is_train=False,
                    env_kwargs=env_kwargs,
                    resources_per_worker=resources_per_worker,
                )
            else:
                _val_envs = build_caged_craftext_envs_optimistic(
                    seed=config.env.seed + 1000,
                    env_num=int(config.data.val_batch_size),
                    group_n=1,
                    is_train=False,
                    env_kwargs=env_kwargs,
                    resources_per_worker=resources_per_worker,
                    reset_ratio=optimistic_reset_ratio,
                )
        else:
            _envs = build_caged_craftext_envs(
                seed=config.env.seed,
                env_num=config.data.train_batch_size,
                group_n=group_n,
                is_train=True,
                env_kwargs=env_kwargs,
                resources_per_worker=resources_per_worker,
            )
            _val_envs = build_caged_craftext_envs(
                seed=config.env.seed + 1000,
                env_num=_val_env_num,
                group_n=1,
                is_train=False,
                env_kwargs=env_kwargs,
                resources_per_worker=resources_per_worker,
            )
        
        # 4. Создаем функцию проекции (используем ту же, что и для обычного craftext)
        projection_f = partial(craftext_projection)
        
        # 5. Используем CagedCraftextEnvironmentManager
        envs = CagedCraftextEnvironmentManager(_envs, projection_f, config)
        val_envs = CagedCraftextEnvironmentManager(_val_envs, projection_f, config)
        
        return envs, val_envs
    elif "craftext" in config.env.env_name.lower():
        # 1. Импортируем все необходимое для Craftext
        print(f"[make_envs] Detected regular Craftext environment: {config.env.env_name}")
        from agent_system.environments.env_package.craftext.envs import build_craftext_envs
        from agent_system.environments.env_package.craftext.projection import craftext_projection

        # 2. Указываем параметры для среды Craftext (если нужны)
        env_kwargs = {
            'config_name': config.env.craftext_settings, # Пример
            'encode_form': 'embedding',   # Пример
            'observation_type': config.env.observation_type,
        }
        
        # 3. Создаем train и val среды
        _envs = build_craftext_envs(
            seed=config.env.seed, 
            env_num=config.data.train_batch_size, 
            group_n=group_n, 
            is_train=True, 
            env_kwargs=env_kwargs, 
            resources_per_worker=resources_per_worker
        )
        _val_envs = build_craftext_envs(
            seed=config.env.seed + 1000, 
            env_num=config.data.val_batch_size, 
            group_n=1, 
            is_train=False, 
            env_kwargs=env_kwargs, 
            resources_per_worker=resources_per_worker
        )
        
        # 4. Создаем функцию проекции
        projection_f = partial(craftext_projection)
        
        # 5. Выбираем менеджер: используем CraftextSubtaskEnvironmentManager если указано в конфиге
        # Import here to avoid circular import
        from agent_system.environments.env_subtask_manager import CraftextSubtaskEnvironmentManager
        
        use_subtask_manager = config.env.get('use_subtask_gigpo', False)
        
        if use_subtask_manager:
            # Используем менеджер с поддержкой subtask-based GiGPO
            envs = CraftextSubtaskEnvironmentManager(_envs, projection_f, config)
            val_envs = CraftextSubtaskEnvironmentManager(_val_envs, projection_f, config)
        else:
            # Используем стандартный менеджер
            envs = CraftextEnvironmentManager(_envs, projection_f, config)
            val_envs = CraftextEnvironmentManager(_val_envs, projection_f, config)

        return envs, val_envs
        
    else:
        print("Environment not supported")
        exit(1)