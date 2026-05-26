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
import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from verl.models.transformers.qwen2_vl import get_rope_index
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict, Optional
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
        latent_state: Optional[List[int]] = None,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        # `raw_prompt` is only present when data.return_raw_chat=True.
        # For many training runs we don't need it; avoid hard dependency.
        raw_prompt_list = gen_batch.non_tensor_batch.get('raw_prompt', None)
        raw_prompt = raw_prompt_list[item] if raw_prompt_list is not None else None
        data_source_list = gen_batch.non_tensor_batch.get('data_source', None)
        data_source = data_source_list[item] if data_source_list is not None else "unknown"
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Build chat structure
        # obs_content = raw_prompt[0]['content']
        # if '<image>' in obs_content: 
        #     obs_content = obs_content.replace('<image>', '')

        # Build chat structure
        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!")
        
        # Add latent state zt to prompt if available
        if latent_state is not None:
            latent_text = self.tokenizer.decode(latent_state, skip_special_tokens=False)
            obs_content += f"\n\nЛатентное состояние среды: {latent_text}"

        
        chat = np.array([{
            "content": obs_content,
            "role": "user",
        }])
        
        # Apply chat template (some tokenizers may not have chat_template)
        if getattr(self.tokenizer, "chat_template", None):
            prompt_with_chat_template = self.tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False
            )
        else:
            prompt_with_chat_template = "\n".join(
                [f"{m.get('role','user')}: {m.get('content','')}" for m in chat]
            )
        
        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            # If dataset did not provide `raw_prompt`, fall back to the prompt we just built.
            raw_prompt = raw_prompt if raw_prompt is not None else prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=self.config.data.max_prompt_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation=self.config.data.truncation,)
        
        

        if is_multi_modal:

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=image_grid_thw,
                    attention_mask=attention_mask[0],
                )
              ]  # (1, 3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict,
        latent_states: Optional[List[List[int]]] = None,
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # Process each sample in parallel
        for item in range(batch_size):
            # Extract per-sample observations
            latent_state = latent_states[item] if latent_states is not None else None
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
                latent_state=latent_state,
            )
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            episode_costs: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            tool_callings (np.ndarray): Number of tool callings for each environment
        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        effective_batch = []
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    # episode_costs (для Caged Craftext)
                    data['episode_costs'] = episode_costs[bs]
                    # tool_callings
                    data['tool_callings'] = tool_callings[bs]
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value

                    effective_batch.append(data)
            
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            world_model_trainer=None,
            record_video_env_idx: Optional[int] = None,
            is_train: bool = True,
            ):
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
            record_video_env_idx (int, optional): If set, collect render_frame from this env's infos for video
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            episode_costs (np.ndarray): Total costs for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            tool_callings (np.ndarray): Tool call counts
            validation_video_frames (list or None): Collected frames for record_video_env_idx, or None
        """

        batch_size = len(gen_batch.batch)
        validation_video_frames = [] if record_video_env_idx is not None else None
        validation_video_prompts = [] if record_video_env_idx is not None else None
        validation_video_actions = [] if record_video_env_idx is not None else None
        validation_video_action_ids = [] if record_video_env_idx is not None else None

        # Initial observations from the environment
        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))

        if is_train:
            print(
                f"[rollout train] env reset done, parallel slots={batch_size}, max_steps={int(self.config.env.max_steps)}, "
                f"auto_reset={bool(self.config.env.get('auto_reset', False))} — starting LLM↔env loop (no trainer metrics until this finishes)",
                flush=True,
            )

        if validation_video_frames is not None and record_video_env_idx is not None and record_video_env_idx < len(infos):
            frame = infos[record_video_env_idx].get('render_frame')
            if frame is not None:
                validation_video_frames.append(frame)
                # First frame: initial obs, no action yet
                prompt_text = obs.get('text', [None])[record_video_env_idx] if isinstance(obs.get('text'), list) else None
                # Ensure constraint is visible in validation text panel.
                try:
                    if isinstance(obs, dict) and "constraint" in obs and isinstance(obs["constraint"], list):
                        c = obs["constraint"][record_video_env_idx] if record_video_env_idx < len(obs["constraint"]) else ""
                        if c and (prompt_text is not None) and ("**CONSTRAINT:**" not in prompt_text):
                            prompt_text = (prompt_text or "") + f"\n\n**CONSTRAINT:** {c}"
                except Exception:
                    pass
                validation_video_prompts.append(prompt_text or "")
                validation_video_actions.append("")
                validation_video_action_ids.append(-1)

        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"
        
        if self.config.env.rollout.n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        episode_costs = np.zeros(batch_size, dtype=np.float32)  # Для Caged Craftext - накопление cost
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        # Списки завершённых эпизодов для метрик (среднее по эпизодам, как в caged_craftext baselines)
        completed_episode_returns = []
        completed_episode_lengths = []
        completed_episode_costs = []
        # Tracks env slots already pushed to completed_episode_* (avoids double-count at rollout end).
        episode_metrics_recorded = np.zeros(batch_size, dtype=bool)
        
        # Auto reset: проверяем, включен ли auto reset
        auto_reset_enabled = self.config.env.get('auto_reset', False)
        
        # Для auto reset: отслеживаем количество шагов для каждой среды
        if auto_reset_enabled:
            steps_per_env = np.zeros(batch_size, dtype=np.int32)  # Количество собранных шагов для каждой среды
            max_steps_per_env = int(self.config.env.max_steps)  # Максимальное количество шагов на среду
            # Для auto reset: храним награды и длины эпизодов отдельно для каждого эпизода
            current_episode_rewards = np.zeros(batch_size, dtype=np.float32)
            current_episode_lengths = np.zeros(batch_size, dtype=np.float32)
            current_episode_costs = np.zeros(batch_size, dtype=np.float32)

        # -------------------------
        # Saute-style safety shaping (optional)
        # -------------------------
        saute_cfg = None
        saute_enabled = False
        try:
            saute_cfg = self.config.algorithm.get("saute", None)
            saute_enabled = bool(saute_cfg and saute_cfg.get("enabled", False))
        except Exception:
            saute_cfg = None
            saute_enabled = False

        if saute_enabled:
            saute_gamma = float(saute_cfg.get("gamma", 1.0))
            safety_budget = float(saute_cfg.get("safety_budget", 1.0))
            unsafe_reward = float(saute_cfg.get("unsafe_reward", -10.0))
            violation_threshold = float(saute_cfg.get("violation_threshold", 0.0))
            append_safety_info_to_obs = bool(saute_cfg.get("append_safety_info_to_obs", True))

            # OmniSafe-style per-step budget normalization:
            # B_step = B * (1 - gamma^T)/(1-gamma) / T
            T = int(self.config.env.max_steps)
            if abs(saute_gamma - 1.0) < 1e-8:
                safety_budget_per_step = max(safety_budget, 1e-8)
            else:
                safety_budget_per_step = safety_budget * (1 - (saute_gamma ** T)) / (1 - saute_gamma) / max(T, 1)
                safety_budget_per_step = max(float(safety_budget_per_step), 1e-8)

            safety_obs = np.ones(batch_size, dtype=np.float32)
            violation_counts = np.zeros(batch_size, dtype=np.int32)
        else:
            saute_gamma = 1.0
            safety_budget_per_step = 1.0
            unsafe_reward = -10.0
            violation_threshold = 0.0
            append_safety_info_to_obs = False
            safety_obs = None
            violation_counts = None

        # Trajectory collection loop
        # Если auto reset включен, используем while цикл, иначе обычный for цикл
        _step = 0
        _rollout_iter = 0
        _logged_first_gen = False
        while True:
            # Определяем активные среды
            if auto_reset_enabled:
                # В режиме auto reset активны среды, которые еще не собрали достаточно шагов
                active_masks = steps_per_env < max_steps_per_env
                # Проверяем условие выхода для while цикла
                if np.all(steps_per_env >= max_steps_per_env):
                    break
            else:
                # В обычном режиме активны среды, которые еще не завершены
                active_masks = np.logical_not(is_done)
                # Проверяем условие выхода для for цикла
                if _step >= self.config.env.max_steps:
                    break
                _step += 1
            
            # Get latent state zt through encoder if world model is enabled and use_latent_in_policy is True
            latent_states = None
            use_latent_in_policy = self.config.trainer.get("world_model", {}).get("use_latent_in_policy", False)
            if use_latent_in_policy and world_model_trainer is not None:
                # Extract observation texts
                obs_texts = obs.get('text', None)
                if obs_texts is not None:
                    # Encode observations to get latent states zt
                    latent_states = world_model_trainer.encode_observation(
                        observations=obs_texts,
                        with_grad=False  # No gradients during rollout
                    )

            if is_train and not _logged_first_gen:
                print(
                    "[rollout train] first iteration: generate_sequences on actor (vLLM warmup can take minutes) …",
                    flush=True,
                )
                _logged_first_gen = True

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs, latent_states=latent_states)
            
            # Store latent states for later use (decode to text for storage)
            if latent_states is not None:
                batch.non_tensor_batch['latent_states'] = np.array([self.tokenizer.decode(zt, skip_special_tokens=False) for zt in latent_states], dtype=object)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info

            # pad to be divisible by dp_size
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            # # unpad
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid

            batch = batch.union(batch_output)
            
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            
            next_obs, rewards, dones, infos = envs.step(text_actions)

            _rollout_iter += 1
            if is_train and (_rollout_iter == 1 or _rollout_iter % 10 == 0):
                n_act = int(np.sum(active_masks)) if hasattr(active_masks, "sum") else batch_size
                print(
                    f"[rollout train] env step {_rollout_iter} (active envs ≈ {n_act}/{batch_size})",
                    flush=True,
                )

            # Конвертируем в numpy массивы, если они пришли как списки
            if isinstance(rewards, list):
                rewards = np.array(rewards, dtype=np.float32)
            if isinstance(dones, list):
                dones = np.array(dones, dtype=bool)
            
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            
            # Извлекаем cost из info для Caged Craftext (если есть)
            # cost - стоимость шага, episode_cost - накопленная стоимость эпизода
            if 'cost' in infos[0]:
                costs = np.array([info.get('cost', 0.0) for info in infos], dtype=np.float32)
                # Накапливаем cost для активных эпизодов
                episode_costs[active_masks] += costs[active_masks]
                batch.non_tensor_batch['cost'] = torch_to_numpy(costs, is_object=True)

                # Saute: update safety state + shape reward (optional)
                if saute_enabled:
                    # count violations as (cost > threshold)
                    violations = (costs > violation_threshold).astype(np.int32)
                    violation_counts[active_masks] += violations[active_masks]

                    # Update safety state: s <- (s - cost/B_step) / gamma
                    safety_obs[active_masks] -= (costs[active_masks] / safety_budget_per_step)
                    if abs(saute_gamma - 1.0) >= 1e-8:
                        safety_obs[active_masks] /= saute_gamma

                    # Reward gating: unsafe -> big penalty
                    original_rewards = rewards.copy()
                    safe_mask = (safety_obs > 0.0)
                    rewards = rewards.copy()
                    unsafe_active = np.logical_and(active_masks, np.logical_not(safe_mask))
                    rewards[unsafe_active] = unsafe_reward

                    batch.non_tensor_batch['original_rewards'] = torch_to_numpy(original_rewards, is_object=True)
                    batch.non_tensor_batch['saute/safety_state'] = torch_to_numpy(safety_obs, is_object=True)
                    batch.non_tensor_batch['saute/violation_count'] = torch_to_numpy(violation_counts.astype(np.float32), is_object=True)
            
            # Извлекаем episode_cost из info (если есть, используется как итоговая стоимость эпизода)
            # При завершении эпизода (done=True) обновляем episode_costs значением из info и записываем в списки завершённых эпизодов
            episode_costs_from_info = np.array([info.get('episode_cost', 0.0) for info in infos], dtype=np.float32) if infos and 'episode_cost' in infos[0] else None
            if episode_costs_from_info is not None:
                episode_costs[dones] = episode_costs_from_info[dones]
                batch.non_tensor_batch['episode_cost'] = episode_costs_from_info

            # Count this env step before logging completed episodes (fixes length off-by-one / min=0).
            if auto_reset_enabled:
                current_episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
                current_episode_lengths[active_masks] += 1
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            # Записываем завершённые эпизоды в списки для метрик (среднее по эпизодам)
            for i in range(batch_size):
                if dones[i]:
                    if auto_reset_enabled:
                        completed_episode_returns.append(float(current_episode_rewards[i]))
                        completed_episode_lengths.append(float(current_episode_lengths[i]))
                        cost_val = float(episode_costs_from_info[i]) if episode_costs_from_info is not None else float(current_episode_costs[i])
                        completed_episode_costs.append(cost_val)
                    else:
                        ep_len = float(episode_lengths[i])
                        completed_episode_returns.append(float(episode_rewards[i]))
                        completed_episode_lengths.append(ep_len)
                        completed_episode_costs.append(float(episode_costs[i]))
                        # Debug: flag implausibly short episodes (debug_square needs ~5+ steps).
                        if os.environ.get("CRAFTEXT_DEBUG_SHORT_EPISODES", "0") == "1" and ep_len <= 3:
                            instr = infos[i].get("instruction", "?") if i < len(infos) else "?"
                            aid = infos[i].get("action_id", -1) if i < len(infos) else -1
                            print(
                                f"[CRAFTEXT_DEBUG_SHORT_EPISODES] env={i} length={ep_len} "
                                f"reward={float(episode_rewards[i]):.3f} won={infos[i].get('won')} "
                                f"action_id={aid} instruction={str(instr)[:80]}",
                                flush=True,
                            )
                    episode_metrics_recorded[i] = True

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            
            # Store next observations only when needed (e.g., world model training).
            wm_cfg = self.config.trainer.get("world_model", {}) if hasattr(self.config, "trainer") else {}
            wm_enabled = bool(wm_cfg.get("enable", False))
            if wm_enabled:
                # Convert next_obs to a format that can be stored
                if isinstance(next_obs, dict):
                    if 'text' in next_obs and next_obs['text'] is not None:
                        batch.non_tensor_batch['next_obs_text'] = np.array(next_obs['text'], dtype=object)
                    if 'image' in next_obs and next_obs['image'] is not None:
                        batch.non_tensor_batch['next_obs_image'] = torch_to_numpy(next_obs['image'], is_object=True)
                else:
                    batch.non_tensor_batch['next_obs'] = torch_to_numpy(next_obs, is_object=True)
            
            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # Сбор кадров для записи видео (один env по record_video_env_idx) + промпт и действие для подписи
            if validation_video_frames is not None and record_video_env_idx is not None and record_video_env_idx < len(infos):
                frame = infos[record_video_env_idx].get('render_frame')
                if frame is not None:
                    validation_video_frames.append(frame)
                    prompt_text = obs.get('text', [None])[record_video_env_idx] if isinstance(obs.get('text'), list) else None
                    # Ensure constraint is visible in validation text panel.
                    try:
                        if isinstance(obs, dict) and "constraint" in obs and isinstance(obs["constraint"], list):
                            c = obs["constraint"][record_video_env_idx] if record_video_env_idx < len(obs["constraint"]) else ""
                            if c and (prompt_text is not None) and ("**CONSTRAINT:**" not in prompt_text):
                                prompt_text = (prompt_text or "") + f"\n\n**CONSTRAINT:** {c}"
                    except Exception:
                        pass
                    raw_action_text = text_actions[record_video_env_idx] if record_video_env_idx < len(text_actions) else ""
                    parsed_action_name = infos[record_video_env_idx].get("action_name", "") if record_video_env_idx < len(infos) else ""
                    if parsed_action_name:
                        action_text = f"{parsed_action_name} | raw: {raw_action_text}"
                    else:
                        action_text = raw_action_text
                    validation_video_prompts.append(prompt_text or "")
                    validation_video_actions.append(action_text or "")
                    try:
                        validation_video_action_ids.append(int(infos[record_video_env_idx].get("action_id", -1)))
                    except Exception:
                        validation_video_action_ids.append(-1)

            # Обновляем счетчики шагов для активных сред (если auto reset включен)
            if auto_reset_enabled:
                steps_per_env[active_masks] += 1
            
            # Update done states (для текущего эпизода)
            is_done = np.logical_or(is_done, dones)
            
            # Auto reset: перезапускаем среды, которые завершились, но еще не собрали достаточно шагов
            if auto_reset_enabled:
                need_reset = np.logical_and(dones, steps_per_env < max_steps_per_env)
                if np.any(need_reset):
                    # Сбрасываем счетчики эпизодов для перезапускаемых сред
                    current_episode_rewards[need_reset] = 0.0
                    current_episode_lengths[need_reset] = 0
                    current_episode_costs[need_reset] = 0.0
                    is_done[need_reset] = False

                    # Сбрасываем safety state для Saute, если используется
                    if saute_enabled:
                        safety_obs[need_reset] = 1.0
                        violation_counts[need_reset] = 0

                    use_optimistic_parallel = bool(
                        getattr(self.config.env, "use_optimistic_parallel", False)
                    )
                    if not use_optimistic_parallel:
                        # Multi-process env: explicit reset (restarts all workers).
                        reset_obs, reset_infos = envs.reset(kwargs=None)
                        active_envs = steps_per_env < max_steps_per_env
                        if isinstance(next_obs, dict) and isinstance(reset_obs, dict):
                            for i in range(batch_size):
                                if active_envs[i]:
                                    if 'text' in reset_obs and reset_obs['text'] is not None:
                                        next_obs['text'][i] = reset_obs['text'][i]
                                    if 'image' in reset_obs and reset_obs['image'] is not None:
                                        next_obs['image'][i] = reset_obs['image'][i]
                                    if 'anchor' in reset_obs and reset_obs['anchor'] is not None:
                                        next_obs['anchor'][i] = reset_obs['anchor'][i]
                        elif not isinstance(next_obs, dict):
                            for i in range(batch_size):
                                if active_envs[i]:
                                    next_obs[i] = reset_obs[i]
                    # Optimistic batched env: OptimisticResetVecEnvWrapper already reset done slots
                    # inside step(); next_obs from envs.step() is already the new episode.
                
            # Update observations for next step
            if saute_enabled and append_safety_info_to_obs:
                # Augment next observation text with safety info for the agent.
                # This matches the Saute idea of state augmentation, but in text form.
                if isinstance(next_obs, dict) and next_obs.get('text', None) is not None:
                    for i in range(batch_size):
                        # only meaningful for envs that are still active before the transition
                        if active_masks[i]:
                            next_obs['text'][i] = (
                                next_obs['text'][i]
                                + f"\n\n[SAFETY] Violations so far: {int(violation_counts[i])} | Safety state: {float(safety_obs[i]):.4f}"
                            )
            obs = next_obs

        if is_train:
            print(
                f"[rollout train] finished after {_rollout_iter} env steps — running reward/PPO update next",
                flush=True,
            )
        
        # Без auto_reset: эпизоды, не завершившиеся по done, — один раз в конце rollout (max_steps)
        if not auto_reset_enabled:
            for i in range(batch_size):
                if episode_metrics_recorded[i]:
                    continue
                completed_episode_returns.append(float(episode_rewards[i]))
                completed_episode_lengths.append(float(episode_lengths[i]))
                completed_episode_costs.append(float(episode_costs[i]))
        
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )

        return total_batch_list, episode_rewards, episode_lengths, episode_costs, success, traj_uid, tool_callings, validation_video_frames, validation_video_prompts, validation_video_actions, validation_video_action_ids, completed_episode_returns, completed_episode_lengths, completed_episode_costs
    
    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            world_model_trainer=None,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_episode_costs = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        total_completed_returns = []
        total_completed_lengths = []
        total_completed_costs = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, episode_costs, success, traj_uid, tool_callings, _vframes, _vprompts, _vactions, _vaction_ids, completed_returns, completed_lengths, completed_costs = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            total_completed_returns.extend(completed_returns)
            total_completed_lengths.extend(completed_lengths)
            total_completed_costs.extend(completed_costs)
            batch_list, episode_rewards, episode_lengths, episode_costs, success, traj_uid, tool_callings = filter_group_data(batch_list=batch_list, 
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                episode_costs=episode_costs,
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                tool_callings=tool_callings, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_episode_costs.append(episode_costs)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_episode_costs = np.concatenate(total_episode_costs, axis=0) if len(total_episode_costs) > 0 else np.zeros(len(total_episode_rewards), dtype=np.float32)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_episode_costs, total_success, total_traj_uid, total_tool_callings, total_completed_returns, total_completed_lengths, total_completed_costs

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            world_model_trainer=None,
            record_video_env_idx: Optional[int] = None,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).
            record_video_env_idx (int, optional): If set, collect render frames for this env for validation video.

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
            
        # Initial observations from the environment
        if self.config.algorithm.filter_groups.enable and is_train:
            # Dynamic Sampling (for DAPO and Dynamic GiGPO)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_episode_costs, total_success, total_traj_uid, totoal_tool_callings, completed_episode_returns, completed_episode_lengths, completed_episode_costs = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                world_model_trainer=world_model_trainer,
            )
            validation_video_frames = None
            validation_video_prompts = None
            validation_video_actions = None
        else:
            # Vanilla Sampling   
            total_batch_list, total_episode_rewards, total_episode_lengths, total_episode_costs, total_success, total_traj_uid, totoal_tool_callings, validation_video_frames, validation_video_prompts, validation_video_actions, validation_video_action_ids, completed_episode_returns, completed_episode_lengths, completed_episode_costs = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                world_model_trainer=world_model_trainer,
                record_video_env_idx=record_video_env_idx,
                is_train=is_train,
            )
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(totoal_tool_callings)
        # Для обратной совместимости, если episode_costs нет (для обычного craftext)
        if len(total_episode_costs) == 0:
            total_episode_costs = np.zeros(len(total_episode_rewards), dtype=np.float32)
        

        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            episode_costs=total_episode_costs,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=totoal_tool_callings,
        )
        if validation_video_frames is not None:
            gen_batch_output.meta_info['validation_video_frames'] = validation_video_frames
        if validation_video_prompts is not None:
            gen_batch_output.meta_info['validation_video_prompts'] = validation_video_prompts
        if validation_video_actions is not None:
            gen_batch_output.meta_info['validation_video_actions'] = validation_video_actions
        if validation_video_action_ids is not None:
            gen_batch_output.meta_info['validation_video_action_ids'] = validation_video_action_ids
        # Метрики по завершённым эпизодам (среднее по эпизодам, как в caged_craftext baselines)
        if completed_episode_returns is not None and len(completed_episode_returns) > 0:
            gen_batch_output.meta_info['completed_episode_returns'] = np.array(completed_episode_returns, dtype=np.float32)
        if completed_episode_lengths is not None and len(completed_episode_lengths) > 0:
            gen_batch_output.meta_info['completed_episode_lengths'] = np.array(completed_episode_lengths, dtype=np.float32)
        if completed_episode_costs is not None and len(completed_episode_costs) > 0:
            gen_batch_output.meta_info['completed_episode_costs'] = np.array(completed_episode_costs, dtype=np.float32)

        return gen_batch_output