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

"""
Subtask-based Environment Managers.

These classes extend environment managers to use predicted subtasks as anchor states
for GiGPO grouping. The LLM agent should predict a subtask in its response, which will
be extracted and used as the anchor state instead of the raw observation.
"""

from typing import List, Dict, Any
import re
import numpy as np
from agent_system.environments.env_manager import (
    CraftextEnvironmentManager,
    AlfWorldEnvironmentManager,
    parse_gamefile,
    set_gamefile
)
from agent_system.environments.base import to_numpy
from agent_system.environments.env_package.craftext.projection import (
    CRAFTEXT_SUBTASK_TEMPLATE_NO_HIS,
    CRAFTEXT_SUBTASK_VL_TEMPLATE_NO_HIS
)
from agent_system.environments.prompts.alfworld import (
    ALFWORLD_SUBTASK_TEMPLATE_NO_HIS,
    ALFWORLD_SUBTASK_TEMPLATE,
    VALID_SUBTASKS_STR
)


class CraftextSubtaskEnvironmentManager(CraftextEnvironmentManager):
    """
    Craftext environment manager that uses subtask predictions as anchor states.
    
    The agent's response should include a subtask prediction in a specific format,
    e.g., "<subtask>collect_wood</subtask>" or "Subtask: collect_wood".
    This subtask will be extracted and used as the anchor state for GiGPO grouping.
    """
    
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
        # You can define valid subtasks here if needed
        self.valid_subtasks = [
            "collect_wood",
            "place_table",
            "make_wood_pickaxe",
            "make_wood_sword",
        ]
    
    def extract_subtask_from_action(self, text_action: str) -> str:
        """
        Extract subtask from the agent's text action/response.
        
        The agent should include the subtask in one of these formats:
        1. XML-style: <subtask>subtask_name</subtask>
        2. Prefix-style: Subtask: subtask_name
        3. JSON-style: {"subtask": "subtask_name"}
        
        Args:
            text_action: The full text response from the agent
            
        Returns:
            subtask: The extracted subtask label, or "unknown" if not found
        """
        # Method 1: XML-style tags
        xml_match = re.search(r'<subtask>(.*?)</subtask>', text_action, re.IGNORECASE)
        if xml_match:
            subtask = xml_match.group(1).strip().lower()
            if subtask in self.valid_subtasks:
                return subtask
        
        # Method 2: Prefix-style "Subtask: ..."
        prefix_match = re.search(r'Subtask:\s*(\w+)', text_action, re.IGNORECASE)
        if prefix_match:
            subtask = prefix_match.group(1).strip().lower()
            if subtask in self.valid_subtasks:
                return subtask
        
        # Method 3: JSON-style
        json_match = re.search(r'["\']subtask["\']\s*:\s*["\'](\w+)["\']', text_action, re.IGNORECASE)
        if json_match:
            subtask = json_match.group(1).strip().lower()
            if subtask in self.valid_subtasks:
                return subtask
        
        # Method 4: Simple keyword matching (fallback)
        # You can customize this based on your needs
        action_lower = text_action.lower()
        if any(word in action_lower for word in ["wood", "tree", "chop"]):
            return "collect_wood"
        elif any(word in action_lower for word in ["stone", "rock"]):
            return "collect_stone"
        elif any(word in action_lower for word in ["iron", "ore"]):
            return "collect_iron"
        elif any(word in action_lower for word in ["craft", "make", "tool"]):
            return "craft_tool"
        elif any(word in action_lower for word in ["build", "construct"]):
            return "build_structure"
        elif any(word in action_lower for word in ["place", "put"]):
            return "place_block"
        elif any(word in action_lower for word in ["mine", "dig"]):
            return "mine_resource"
        elif any(word in action_lower for word in ["move", "go", "explore"]):
            return "explore"
        
        # Default: return "unknown" if no subtask found
        return "unknown"
    
    def step(self, text_actions: List[str]):
        """
        Override step() to extract subtasks from actions and use them as anchor states.
        
        The key change: instead of using next_text_renders as anchor,
        we extract subtasks from text_actions and use those as anchors.
        """
        action_ids, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(action_ids)
        next_text_renders = [info.get('text_render', 'The world is empty.') for info in infos]

        self.memory.store({'text_obs': self.pre_text_obs, 'action': text_actions})
        self.pre_text_obs = next_text_renders

        # Extract subtasks from agent's responses
        subtasks = [self.extract_subtask_from_action(action) for action in text_actions]

        if self.config.env.env_name == "craftext/CraftextEnv":
            next_observations = {
                'text': self.build_text_obs(next_text_renders, infos),
                # 'image': next_obs,
                'anchor': subtasks  # Use subtasks as anchor instead of next_text_renders
            }

        if self.config.env.env_name == "craftext/CraftextVLEnv":
            next_observations = {
                'text': self.build_text_obs(next_text_renders, infos),
                'image': next_obs,
                'anchor': subtasks  # Use subtasks as anchor instead of next_text_renders
            }
        
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        return next_observations, to_numpy(rewards), to_numpy(dones), infos
    
    def reset(self, kwargs) -> Dict[str, Any]:
        """
        Override reset() to initialize with "unknown" subtasks.
        """
        observations, infos = super().reset(kwargs)
        # Initialize anchors with "unknown" subtask since we don't have agent actions yet
        batch_size = len(infos) if isinstance(infos, list) else 1
        observations['anchor'] = ["unknown"] * batch_size
        return observations, infos
    
    def build_text_obs(self, text_renders: List[str], infos: List[Dict], init: bool = False) -> List[str]:
        """
        Override build_text_obs() to use subtask-aware prompt templates.
        
        This method uses the subtask templates that instruct the agent to predict
        subtasks in its responses.
        """
        final_prompts = []
        
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                self.config.env.history_length,
                obs_key="text_obs",
                action_key="action"
            )
        
        for i in range(len(text_renders)):
            # Use subtask-aware templates
            if self.config.env.env_name == "craftext/CraftextVLEnv":
                prompt = CRAFTEXT_SUBTASK_VL_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_renders[i]
                )
            else:
                prompt = CRAFTEXT_SUBTASK_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_renders[i]
                )

            final_prompts.append(prompt)

        return final_prompts


class AlfWorldSubtaskEnvironmentManager(AlfWorldEnvironmentManager):
    """
    AlfWorld environment manager that uses subtask predictions as anchor states.
    
    The agent's response should include a subtask prediction in a specific format,
    e.g., "<subtask>find_object</subtask>" or "Subtask: find_object".
    This subtask will be extracted and used as the anchor state for GiGPO grouping.
    
    Valid subtasks:
    - find_object: Finding/locating an object in the environment
    - grab_object: Picking up/grasping an object
    - find_tool: Finding a tool or receptacle needed for the task
    - operate_tool: Using/operating a tool (e.g., microwave, fridge, etc.)
    - find_destination: Finding where to place an object
    - place_object: Placing an object at the destination
    """
    
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
        # Define valid subtasks for AlfWorld
        self.valid_subtasks = [
            "find_object",
            "grab_object",
            "find_tool",
            "operate_tool",
            "find_destination",
            "place_object",
        ]
    
    def extract_subtask_from_action(self, text_action: str) -> str:
        """
        Extract subtask from the agent's text action/response.
        
        The agent should include the subtask in one of these formats:
        1. XML-style: <subtask>subtask_name</subtask>
        2. Prefix-style: Subtask: subtask_name
        3. JSON-style: {"subtask": "subtask_name"}
        
        Args:
            text_action: The full text response from the agent
            
        Returns:
            subtask: The extracted subtask label, or "unknown" if not found
        """
        # Method 1: XML-style tags
        xml_match = re.search(r'<subtask>(.*?)</subtask>', text_action, re.IGNORECASE)
        if xml_match:
            subtask = xml_match.group(1).strip().lower()
            # Normalize subtask name (handle spaces, underscores, etc.)
            subtask = subtask.replace(" ", "_").replace("-", "_")
            if subtask in self.valid_subtasks:
                return subtask
        
        # Method 2: Prefix-style "Subtask: ..."
        prefix_match = re.search(r'Subtask:\s*([\w\s]+)', text_action, re.IGNORECASE)
        if prefix_match:
            subtask = prefix_match.group(1).strip().lower().replace(" ", "_").replace("-", "_")
            if subtask in self.valid_subtasks:
                return subtask
        
        # Method 3: JSON-style
        json_match = re.search(r'["\']subtask["\']\s*:\s*["\']([\w\s]+)["\']', text_action, re.IGNORECASE)
        if json_match:
            subtask = json_match.group(1).strip().lower().replace(" ", "_").replace("-", "_")
            if subtask in self.valid_subtasks:
                return subtask
        
        # Method 4: Simple keyword matching (fallback)
        action_lower = text_action.lower()
        if any(word in action_lower for word in ["find", "search", "locate", "look for"]) and \
           any(word in action_lower for word in ["object", "item", "thing"]):
            return "find_object"
        elif any(word in action_lower for word in ["grab", "pick", "take", "get", "pickup"]):
            return "grab_object"
        elif any(word in action_lower for word in ["find", "search", "locate"]) and \
             any(word in action_lower for word in ["tool", "receptacle", "microwave", "fridge", "cabinet", "drawer"]):
            return "find_tool"
        elif any(word in action_lower for word in ["operate", "use", "open", "turn on", "put in", "place in"]) and \
             any(word in action_lower for word in ["microwave", "fridge", "cabinet", "drawer", "tool"]):
            return "operate_tool"
        elif any(word in action_lower for word in ["find", "search", "locate"]) and \
             any(word in action_lower for word in ["destination", "place", "location", "where", "put"]):
            return "find_destination"
        elif any(word in action_lower for word in ["place", "put", "drop", "set"]):
            return "place_object"
        
        # Default: return "unknown" if no subtask found
        return "unknown"
    
    def step(self, text_actions: List[str]):
        """
        Override step() to extract subtasks from actions and use them as anchor states.
        
        The key change: instead of using text_obs as anchor,
        we extract subtasks from text_actions and use those as anchors.
        """
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # Extract subtasks from agent's responses
        subtasks = [self.extract_subtask_from_action(action) for action in text_actions]

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': subtasks}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def reset(self, kwargs):
        """
        Override reset() to initialize with "unknown" subtasks.
        """
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        # Initialize anchors with "unknown" subtask since we don't have agent actions yet
        batch_size = len(infos) if isinstance(infos, list) else 1
        return {'text': full_text_obs, 'image': image_obs, 'anchor': ["unknown"] * batch_size}, infos
    
    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False) -> List[str]:
        """
        Override build_text_obs() to use subtask-aware prompt templates.
        
        This method uses the subtask templates that instruct the agent to predict
        subtasks in its responses.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            # Use subtask-aware templates
            if init or self.config.env.history_length <= 0:
                obs = ALFWORLD_SUBTASK_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions,
                    valid_subtasks=VALID_SUBTASKS_STR
                )
            else:
                obs = ALFWORLD_SUBTASK_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions,
                    valid_subtasks=VALID_SUBTASKS_STR
                )

            postprocess_text_obs.append(obs)
        return postprocess_text_obs

