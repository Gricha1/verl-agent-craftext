# ========= Standard Library Imports =========
import argparse
import json
import os
import random
import sys

# Set environment variables (it's best to set these before importing some third-party libraries)
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

# ========= Third-Party Libraries =========
import imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import dataclasses
import optax
import torch
import yaml

import pandas as pd
from tqdm import tqdm

# Additional third-party imports for machine learning
from flax.training.train_state import TrainState
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)

# ========= Local Modules =========
from craftax.craftax_env import make_craftax_env_from_name

# Imports from the baselines package
from baselines.experiments.super_igor.expert import PlansExpert
from baselines.experiments.super_igor.super_dataset import SuperDataset
from baselines.experiments.super_igor.view import add_text_to_image, CraftaxRenderer
from baselines.experiments.super_igor.persubtask_update import (
    initialize_reward_tracking,
    update_reward_sum_matrix_simple,
    get_update_mask,
    update_reward_sum_matrix_vectorized
)

from baselines.experiments.super_igor.craftext_wrappers.deterministic_inference import DetermOptimisticResetVecEnvWrapper
from baselines.experiments.super_igor.craftext_wrappers.old_encode_code import  QwenModelWrapper
from baselines.experiments.super_igor.craftext_wrappers.encoder import make_encoder_with_planning
from baselines.experiments.super_igor.craftext_wrappers.env_wrapper import SIPlanning, CustomInstructionWrapper
from baselines.experiments.super_igor.craftext_wrappers.scenarius_loader import create_scenarios_with_super_dataset

from craftext.instruction.wrappers.craftext_wrapper import InstructionWrapper
from craftext.encoders.craftext_base_model_encoder import EncodeForm

from baselines.models.actor_critic_with_text import create_actor_critic


# ========= Seed Configuration =========
# seed_value = 42  # Choose any fixed seed value

# random.seed(seed_value)           # Fix seed for the random module
# np.random.seed(seed_value)        # Fix seed for NumPy
# torch.manual_seed(seed_value)     # Fix seed for PyTorch
# torch.cuda.manual_seed_all(seed_value)  # Fix seed for all GPUs in PyTorch

achievement_dict = {
    0: "COLLECT_WOOD",
    1: "PLACE_TABLE",
    2: "EAT_COW",
    3: "COLLECT_SAPLING",
    4: "COLLECT_DRINK",
    5: "MAKE_WOOD_PICKAXE",
    6: "MAKE_WOOD_SWORD",
    7: "PLACE_PLANT",
    8: "DEFEAT_ZOMBIE",
    9: "COLLECT_STONE",
    10: "PLACE_STONE",
    11: "EAT_PLANT",
    12: "DEFEAT_SKELETON",
    13: "MAKE_STONE_PICKAXE",
    14: "MAKE_STONE_SWORD",
    15: "WAKE_UP",
    16: "PLACE_FURNACE",
    17: "COLLECT_COAL",
    18: "COLLECT_IRON",
    19: "COLLECT_DIAMOND",
    20: "MAKE_IRON_PICKAXE",
    21: "MAKE_IRON_SWORD",
}
class ResultManager:
    def __init__(self, experiment_name, craftext_settings):
        self.experiment_name = experiment_name
        self.craftext_settings = craftext_settings
        self.instructions = None
        self.functions = None
        self.success_rates = None
        self.std = None

    def update_results(self, instructions, functions, success_rates, std):
        self.instructions = instructions
        self.functions = functions
        self.success_rates = success_rates
        self.std = std

    def save_to_csv(self, output_path):
        if self.instructions is None or self.functions is None or self.success_rates is None:
            raise ValueError("Results are not initialized.")
        
        dataset = pd.DataFrame({
            'instructions': self.instructions,
            'functions': self.functions,
            'sr': self.success_rates,
            'std': self.std
        })
        dataset.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

class StepEvaluator:
    def __init__(self, scenario_data, achievement_dict):
        self.scenario_data = scenario_data
        self.achievement_dict = achievement_dict
        self.steps_made = {}

    def evaluate(self, action, env_state, step_now_implementing, indices_of_instruction_run):
        steps_str = []
        for i, j in zip(indices_of_instruction_run, step_now_implementing):
            full_steps = self.scenario_data.instructions_list[i].split("\n")
            steps_str.append(full_steps[j] if j < len(full_steps) else " - ")

        # reward[i].item()
        for i, a in enumerate(action):
            achievements_on_step = np.array(env_state.env_state.env_state.achievements[i].astype(int))
            current_step = step_now_implementing[i].item()
            instruction = self.scenario_data.instructions_list[indices_of_instruction_run[i]]

            if instruction not in self.steps_made:
                self.steps_made[instruction] = {}

            if current_step not in self.steps_made[instruction]:
                self.steps_made[instruction][current_step] = [
                    steps_str[i],
                    {'run_count': 0},
                    {self.achievement_dict[k]: 0 for k in range(len(achievements_on_step))}
                ]

            self.steps_made[instruction][current_step][1]['run_count'] += 1
            for k, val in enumerate(achievements_on_step):
                self.steps_made[instruction][current_step][2][self.achievement_dict[k]] += float(val)

    def save(self, path='steps_made.json'):
        with open(path, 'w') as file:
            json.dump(self.steps_made, file, indent=4)

def extract_planer_config(config):

    planner_config = {'model_config': \
                {
                'original_model_path': config['original_model_path'.upper()],
                'peft_weights_path': config['peft_weights_path'.upper()],
                },
              'generation_config':\
                {
                'num_paraphrases': config['num_paraphrases'.upper()],
                'beam_groups': config['beam_groups'.upper()],
                'beams_count': config['beams_count'.upper()],
                'max_new_tokens': config['max_new_tokens'.upper()],
                'prompt_template': config['prompt_template'.upper()],
                },
                'super_dataset':None,
                'augment': False #config['augment'.upper()]
             }
    return planner_config

class Experiment:
    def __init__(self, args):
        self.seeds_to_use = 30
        self.args = args
        self.calculate_per_step_score = args.per_step_scoring
        self.config = self._load_config()
        self.checkpoint_manager = self._initialize_checkpoint_manager()
        self.env, self.network = self._initialize_environment_and_network()
        self.use_expert = False
        # Load pairs instriction - plans
        if self.use_expert:
            self.plan_expert = PlansExpert()
            
                        
        self.super_dataset = self.env.scenario_handler.super_dataset 
        self.train_state = self._initialize_train_state()
        
        self.result_manager = ResultManager(args.experiment_name, args.craftext_settings)

    def _load_config(self):
        config_path = os.path.join(self.args.path, "config.yaml")
        
        with open(config_path) as f:
            raw_config = yaml.load(f, Loader=yaml.Loader)

        config = {key: value["value"] if isinstance(value, dict) and "value" in value else value
                  for key, value in raw_config.items()}
        
        planner_config = extract_planer_config(config)
        config['PLANER_CONFIG'] = planner_config
        config["NUM_ENVS"] = self.args.num_envs
        config["RATIO"] = self.args.ratio
        config['INFERENCE'] = self.args.inference
        config['LLM_PATH'] = self.args.llm_path
        config['DATASET_PATH'] = self.args.dataset_path
        config['SAVE_DATASET_PATH'] = self.args.save_dataset_path
        config['NUM_RETURN_SEQUENCES'] = self.args.num_return_sequences
        config['PLAN_WITH_LLM'] = self.args.plan_with_llm
        config['AUGMENT'] = bool(self.args.augment)
        config['CUSTOM_COMMAND'] = self.args.custom_command
        print("Planning?: ",  config['PLAN_WITH_LLM'])
       # exit()
        return config

    def _initialize_checkpoint_manager(self):
        orbax_checkpointer = PyTreeCheckpointer()
        options = CheckpointManagerOptions(max_to_keep=1, create=True)
        checkpoint_path = os.path.abspath(os.path.join(self.args.path, "checkpoint_restart_1"))
        return CheckpointManager(checkpoint_path, orbax_checkpointer, options)

    def _initialize_environment_and_network(self):
        is_classic = "-Text" not in self.config["ENV_NAME"]
        env_name = self.config["ENV_NAME"].replace("-Text", "")
        self.config["ENV_NAME"] = env_name

        env = make_craftax_env_from_name(env_name, False)
        actions_count = 18 if "Classic" in env_name else 43
       # network_class = ActorCriticConvWithFiLMonehot
       # network = create_actor_critic(actions_count, self.config["LAYER_SIZE"])
        network = create_actor_critic(ac_type=self.config["AC_TYPE"],
                                     vision_type=self.config["AC_VISION_TYPE"],
                                     text_encoder_type=self.config["AC_TEXT_ENCODER_TYPE"],
                                     text_mlp_sizes=self.config['AC_TEXT_MLP_SIZES'],
                                     nonlinearity=self.config['AC_NONLINEARITY'],
                                     vision_mlp_sizes=self.config['AC_VISION_MLP_SIZES'],
                                     layer_width=self.config["LAYER_SIZE"],
                                     action_dim=actions_count)

        #EncodeModel = QwenModelWrapper(self.config["LLM_PATH"], num_return_sequences=self.config['NUM_RETURN_SEQUENCES'])
        

        if not self.config['PLAN_WITH_LLM']:
            print("Use previos plans")
          #  exit()
            super_dataset = SuperDataset.load_from_json(self.config["DATASET_PATH"])
            self.config['PLANER_CONFIG']['super_dataset'] = super_dataset
            self.config['PLANER_CONFIG']['augment'] = self.config['AUGMENT']
            self.config['PLANER_CONFIG']['generation_config']['num_paraphrases'] = self.config['NUM_RETURN_SEQUENCES']
            print(self.config['PLANER_CONFIG'])
          #  exit()
            EncodeModel = make_encoder_with_planning(planer_type = "sd",
                                             planer_config=self.config['PLANER_CONFIG'],
                                             embedding_source=self.config['EMBEDDING_SOURCE'],
                                             step_by_step=self.config['STEP_BY_STEP'],
                                             full_sampled=False)

            
            # EncodeModel = SuperEncoder(super_dataset, form_to_use=EncodeForm.EMBEDDING,
            #                            num_return_sequences=self.config['NUM_RETURN_SEQUENCES'], n_splits=1,  augment=self.config['AUGMENT'], split_into_steps=True, make_one_hot=True)
            ScenariosClass = create_scenarios_with_super_dataset(self.config["DATASET_PATH"], load_preinited=True, update_sd=False)
        elif self.config['CUSTOM_COMMAND']:
            #TODO: Need to Fix
            EncodeModel = QwenModelWrapper(self.config["LLM_PATH"], num_return_sequences=1, augment=self.config['AUGMENT'], split_into_steps=True, do_plan=False)
            ScenariosClass = create_scenarios_with_super_dataset(self.config["DATASET_PATH"], load_preinited=True, update_sd=True)
        else:
            #EncodeModel = QwenModelWrapper(self.config["LLM_PATH"], num_return_sequences=self.config['NUM_RETURN_SEQUENCES'], augment=self.config['AUGMENT'], split_into_steps=True, make_one_hot=True)
            self.config['PLANER_CONFIG']['generation_config']['num_paraphrases'] = self.config['NUM_RETURN_SEQUENCES']
            # print(self.config['PLANER_CONFIG']['generation_config']['num_paraphrases'])
            # exit()
            EncodeModel = make_encoder_with_planning(planer_type = "llm",
                                             planer_config=self.config['PLANER_CONFIG'],
                                             embedding_source=self.config['EMBEDDING_SOURCE'],
                                             step_by_step=self.config['STEP_BY_STEP'],
                                             full_sampled=False)
            
            ScenariosClass = create_scenarios_with_super_dataset(self.config["DATASET_PATH"], load_preinited=False)
        
        env = InstructionWrapper(env, self.args.craftext_settings, scenario_handler_class=ScenariosClass,
                                  encode_model_class=EncodeModel,
                                  encode_form=EncodeForm.EMBEDDING)
        
        env = SIPlanning(env)
        checkers = env.scenario_handler.scenario_data.arguments
        plans = env.scenario_handler.scenario_data.instructions_list
        instructions = env.scenario_handler.scenario_data.original_instructions
        arguments = env.scenario_arguments
        for checker, args, plan, instruction in zip(checkers, arguments.achievements.achievement_mask, plans, instructions):
            print("+ + + "*10)
            print(instruction)
            print(args)
            print(plan)
            print(checker)

        if self.config['CUSTOM_COMMAND']:
           env = CustomInstructionWrapper(env, instruction=self.config['CUSTOM_COMMAND'].replace("\\n", "\n") )
        
        #EncodeModel = QwenModelWrapper(self.config["LLM_PATH"], num_return_sequences=self.config['NUM_RETURN_SEQUENCES'])
       # env = InstructionWrapper(env, self.args.craftext_settings,  encode_model_class=EncodeModel,
        #                          encode_form=EncodeForm.EMBED_CLS_FOR_SPLITS)
        
        print("N_INSTRUCTIOnS: ", env.n_instructions)
       # exit()
        if self.config['INFERENCE']:
            print("!-!")
            env = DetermOptimisticResetVecEnvWrapper(env, self.config["NUM_ENVS"], 
                                            min(self.config["RATIO"], self.config["NUM_ENVS"]),
                                            n_instructions=env.n_instructions, n_seeds=self.seeds_to_use)
        return env, network

    def _initialize_train_state(self):
        return self.checkpoint_manager.restore(int(self.config["TOTAL_TIMESTEPS"]))

    def view(self):
        rng = jax.random.PRNGKey(38)
        obs, env_state = self.env.reset(rng, self.env.default_params)
        step_fn = jax.jit(self.env.step)
        done = False
        renderer = CraftaxRenderer(self.env, self.env.default_params, pixel_render_size=1)
        steps = 0
        step_fn = jax.jit(self.env.step)
        params = self.train_state['runner_state'][0]["params"]
        observations = []
        agent_rng, _rng_ = jax.random.split(rng)
        instructions = []
        while not done:
            obs = jnp.expand_dims(obs, axis=0)
            if self.config['CUSTOM_COMMAND']:
                print(env_state.step_idx)
                o_instruction = self.config['CUSTOM_COMMAND'].replace("\\n", "\n") 
                steps_todo =  o_instruction.split("\n") 
                print(o_instruction, steps_todo)
                if env_state.step_idx < len(steps_todo):
                    instruction =steps_todo[env_state.step_idx]
                    
                    print("DO: ",instruction)
                instruction_emb = env_state.instruction.reshape(1, -1)
            else:
                instruction = self.env.scenario_handler.scenario_data.instructions_list[env_state.idx.item()]
                instruction_emb = env_state.instruction.reshape(1, -1)
                
            instructions.append(instruction)
            print(np.array(self.env.castom_initial_instruction).shape)
            print(np.array(instruction_emb).shape)
            pi, value = self.network.apply(params, obs, instruction_emb)
            agent_rng, _rng_ = jax.random.split(agent_rng)
            action = pi.sample(seed=agent_rng)[0]
            
           # action = pi.sample(seed=rng)[0]
            
            action = jax.device_put(action, device=jax.devices('gpu')[0])

            if action is not None:
               # rng, _rng = jax.random.split(rng)
                obs, env_state, reward, done, info = step_fn(
                    rng, env_state, action, self.env.default_params
                )
                steps += 1

            image = renderer.render_to_image(env_state.env_state)
            observations.append(image)
        gif_name = "_".join(instruction.split()[:5])
       # gif_name = inst.replace(" ", "_")
        ix = random.randint(0,200)
        folder_name = "animation"

        # Проверка и создание папки
        os.makedirs(folder_name, exist_ok=True)
        with imageio.get_writer(f'animation/{ix}_{gif_name}.gif', mode='I', duration=0.1) as writer:
            for i, image in enumerate(observations):
                instr = instructions[i]
                text = f"Step {i}, Instruction {instr}"
                image_with_text = add_text_to_image(image, text)
                writer.append_data(image_with_text.astype(np.uint8))
        print(f'Save with name animation/{ix}_{gif_name}.gif')

    def _update_rng_logs(self, rng_dict, indices, rng_values):
        for i, instruction in enumerate(indices):
            inst = int(instruction)
            if inst not in rng_dict:
                rng_dict[inst] = []
            rng_dict[inst].append(tuple(rng_values[i].tolist()))
    
    def _postprocess_and_save_rng_logs(self, rng_dict, filename):
        for instruction in rng_dict:
            rng_dict[instruction] = tuple(set(rng_dict[instruction]))
        with open(filename, "w", encoding="utf-8") as json_file:
            json.dump(rng_dict, json_file, ensure_ascii=False, indent=4)
            
    def run(self):
        rng = jax.random.PRNGKey(42)
        log_rngs = False    # Флаг отладки RNG
        render_obs = False  # Флаг отрисовки наблюдений
        N_INSTRUCTIONS = self.env.n_instructions
        
        if render_obs:
            self._init_rendering()

       # Variables for agent perfomance
        obs, env_state = self.env.reset(rng, self.env.default_params)
        step_fn = jax.jit(self.env.step)
        params = self.train_state['runner_state'][0]["params"]

        # Variables for SR measurement
        SR = []
        total_success_rate = np.zeros(N_INSTRUCTIONS+1)
        done_count = np.zeros(N_INSTRUCTIONS+1)
        prev_indx = env_state.env_state.idx
    
        # Additianal Variables for logginf
        instr_rngs = dict()
        instr_rngs_alt = dict()
        agent_rng, _rng_ = jax.random.split(rng)
        
        # Variables for track experiment progress
        remain_steps = 40000
        progress_bar = tqdm(total= N_INSTRUCTIONS * self.seeds_to_use, desc="Steps")
        steps = 0
        
        # Per step evaluator
        step_evaluator = StepEvaluator(
            scenario_data=self.env.scenario_handler.scenario_data,
            achievement_dict=achievement_dict  
        )

        num_goals = len(self.env.scenario_handler.scenario_data_jax.embeddings_list)
        max_subgoals = 30
        achievments_dim = 22
        reward_sum, episode_count = initialize_reward_tracking(num_goals, max_subgoals)
        per_achivment_sum = jnp.zeros((num_goals, max_subgoals, achievments_dim))

        prev_goals = env_state.env_state.idx
        prev_subgoals = env_state.env_state.step_idx
        old_achievments_done = env_state.env_state.env_state.achievements.astype(jnp.float32)
        
        done_in_step = dict()
        while steps < remain_steps:
            pi, _ = self.network.apply(params, obs, env_state.env_state.instruction)
            agent_rng, _rng_ = jax.random.split(agent_rng)
            action = pi.sample(seed=agent_rng)
            
            if action is not None:
                obs, env_state, reward, done, info = step_fn(rng, env_state, action, self.env.default_params)
                
                if self.calculate_per_step_score:
                    # --- calculate per-sub-goal reward
                    binary_reward = (reward > 0).astype(jnp.float32)
                    achievments_done = env_state.env_state.env_state.achievements.astype(jnp.float32)
                    curr_subgoals = env_state.env_state.step_idx
                    curr_goals =  env_state.env_state.idx
                    goals_changed = (curr_goals==prev_goals).astype(jnp.float32)
                    mask = get_update_mask(prev_subgoals, curr_subgoals, prev_goals, num_goals, max_subgoals)
                    reward_sum = update_reward_sum_matrix_simple(reward_sum, binary_reward, prev_goals, curr_subgoals)
                    per_achivment_sum = update_reward_sum_matrix_vectorized(reward_sum_matrix=per_achivment_sum,
                                            rewards=achievments_done - (old_achievments_done*goals_changed[:, None]),
                                            curr_goals=prev_goals,
                                            curr_subgoals=curr_subgoals)
                    prev_goals = curr_goals
                    prev_subgoals = curr_subgoals
                    episode_count = episode_count + mask.astype(jnp.int32)
                    old_achievments_done = achievments_done
                    # --- 
                
                steps += 1
                progress_bar.n = np.sum(done_count)
                
                # If it is hard to finish episodes
                if np.sum(done_count) not in done_in_step:
                    done_in_step = {np.sum(done_count):0}
                else:
                    done_in_step[np.sum(done_count)] += 1
                if done_in_step[np.sum(done_count)] > 20000:
                    break
                
                progress_bar.refresh()

                if render_obs:
                    self._render_observations(obs, action, env_state)

                prev_indx = self._update_success_metrics(prev_indx, env_state, done, info,
                                                        done_count, total_success_rate)
                print("-------------------")
                
                print("sum_sr:", total_success_rate)
                print("action:", action)
                print("step_idx:", env_state.env_state.step_idx)
                print("instruction_idx:", env_state.env_state.idx)

                if (np.sum(done_count)+5) >= self.env.n_instructions * self.seeds_to_use:
                    break

                if log_rngs:
                    self._update_rng_logs_if_needed(env_state, instr_rngs, instr_rngs_alt)

        progress_bar.close()
        success_rates = total_success_rate / done_count
        SR.append(success_rates)

        if log_rngs:
            self._postprocess_and_save_rng_logs(instr_rngs, "seeds.json")
            self._postprocess_and_save_rng_logs(instr_rngs_alt, "seeds_alt.json")

        mean_sr = np.mean(np.array(SR), axis=0)
        scores = []

        if self.use_expert:
            scores = self._expertize_plans(self.env.scenario_handler.scenario_data.instructions_list)
            total_score =  mean_sr[:len(self.env.scenario_handler.scenario_data.instructions_list)] * np.array(scores) 
        else:
            total_score =  mean_sr[:len(self.env.scenario_handler.scenario_data.instructions_list)] 
        per_subtaks_score = reward_sum/episode_count
        self._update_super_dataset(total_success_rate,done_count, total_score, reward_sum, episode_count,per_achivment_sum )
    
        
    def _init_rendering(self):
        import os
        self.save_dir = self.config.get("OBS_SAVE_PATH", "./observations_v2")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.num_envs = self.config["NUM_ENVS"]
        self.env_obs_count = {env_idx: 0 for env_idx in range(self.num_envs)}


    def _render_observations(self, obs, action, env_state):
        obs_np = np.array(obs)
        for env_idx in range(self.num_envs):
            env_folder = os.path.join(self.save_dir, f"env_{env_idx}")
            if not os.path.exists(env_folder):
                os.makedirs(env_folder)
            image_array = obs_np[env_idx]
            filename = os.path.join(env_folder, f"{env_idx}-{self.env_obs_count[env_idx]}.png")
            text = f"Act {action[env_idx]} - IDX{env_state.env_state.idx[env_idx]}"
            image_with_text = add_text_to_image(image_array * 255, text)
            plt.imsave(filename, image_with_text)
            self.env_obs_count[env_idx] += 1


    def _update_success_metrics(self, prev_indx, env_state, done, info, done_count, total_success_rate):
        instruction_done_float = info['SR']
        indices = np.where(instruction_done_float > 0)
        done_indices = np.where(done > 0)

        inst_done = []
        for inst in prev_indx[done_indices]:
            inst = int(inst)
            if done_count[inst] < self.seeds_to_use:
                done_count[inst] += 1
            if done_count[inst] == self.seeds_to_use:
                inst_done.append(inst)

        for inst in prev_indx[indices]:
            if inst not in inst_done:
                total_success_rate[int(inst)] += 1

        return env_state.env_state.idx


    def _update_rng_logs_if_needed(self, env_state, instr_rngs, instr_rngs_alt):
        indicec_used = np.array(env_state.env_state.idx)
        rngs_used = np.array(env_state.env_state.rng)
        self._update_rng_logs(instr_rngs, indicec_used, rngs_used)

        indicec_used_alt = np.array(env_state.v1)
        rngs_used_alt = np.array(env_state.v2)
        self._update_rng_logs(instr_rngs_alt, indicec_used_alt, rngs_used_alt)


    def _expertize_plans(self, plans):
        from tqdm import tqdm
        scores = []
        print("Make plans expertize...")
        for i in tqdm(range(0, len(plans), 5)):
            plans_batch = plans[i:i+5]
            expertize = self.plan_expert.check_plan_correctness(plans_batch)
            for yes, no in expertize:
                scores.append(1.0 if yes > no else 0.0)
        print("Finish")
        return scores


    def _update_super_dataset(self, sr_sum, counts, total_score, reward_sum, episode_count,per_achivment_sum):
        self.super_dataset.super_print()
        self.super_dataset.clear_scores()
        self.super_dataset.rebuild_mapping()
        self.super_dataset.batch_update_instruction(self.env.scenario_handler.scenario_data.original_instructions,
                                                    self.env.scenario_handler.scenario_data.instructions_list,
                                                    total_score)
        
        instructions = self.env.scenario_handler.scenario_data.original_instructions
        plans = self.env.scenario_handler.scenario_data.instructions_list
        srs = total_score
        
        print("- - + - - +"*5)
        for sum_sr, count_run, intsruction, plan, sr in zip(sr_sum, counts, instructions, plans, srs):
            print(intsruction)
            print(sum_sr)
            print(count_run)
            print(plan)
            print(sr)
        print("- - + - - +"*5)
        self.super_dataset.save_to_json(self.config["DATASET_PATH"])
        directory = os.path.dirname(self.config["DATASET_PATH"])
        
        if self.calculate_per_step_score:
        #  per_subtask_score = reward_sum / episode_count
            self.super_dataset.per_subtask_table(self.env.scenario_handler.scenario_data.original_instructions,
                                                self.env.scenario_handler.scenario_data.instructions_list, 
                                                total_score,
                                                episode_count,
                                                reward_sum,
                                                per_achivment_sum,
                                                directory)







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, type=str)
    parser.add_argument("--experiment_name", default=None, type=str)
    parser.add_argument("--plan_with_llm", default='False', type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--craftext_settings", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
    parser.add_argument("--ratio", type=int, default=1)
    parser.add_argument("--inference", type=int, default=False)
    parser.add_argument("--llm_path", type=str, default="Qwen/Qwen2.5-3B-Instruct-Advanced")
    parser.add_argument("--dataset_path", type=str, default="temp_dataset/super_dataset.json")
    parser.add_argument("--save_dataset_path", type=str, default="temp_dataset/super_dataset.json")
    parser.add_argument("--num_return_sequences", type=int, default=5)
    parser.add_argument("--augment", type=int, default=0)
    parser.add_argument("--per_step_scoring", type=int, default=0)
    parser.add_argument("--custom_command", type=str, default=None)
    
    parser.add_argument("--planer_type", type=str, default="llm")
    parser.add_argument("--embedding_source", type=int, default=1)
    parser.add_argument("--step_by_step", type=bool, default=True)
    parser.add_argument("--original_model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--peft_weights_path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--num_paraphrases", type=int, default=15)
    parser.add_argument("--beam_groups", type=int, default=15)
    parser.add_argument("--beams_count", type=int, default=15)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--prompt_template", type=int, default=2)
  #  parser.add_argument("--augment", type=bool, default=False)
    
        #Observation Encoder config
    parser.add_argument("--ac_type", type=str, default="ac_model")
    parser.add_argument("--ac_vision_type", type=str, default="resnet_impala")
    parser.add_argument("--ac_text_encoder_type", type=str, default="mlp")
    parser.add_argument("--ac_text_mlp_sizes", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--ac_nonlinearity", type=str, default="relu")
    parser.add_argument("--ac_vision_mlp_sizes",type=int, nargs="+", default=[256,])

    
    args, rest_args = parser.parse_known_args(sys.argv[1:])
    
    
    
    args.plan_with_llm = False if args.plan_with_llm=="False" else True

  #  args.augment = False #False if args.augment==0 else True
    
    print(args.plan_with_llm)
    # exit()
    if args.path is None:
        args.path = f"./wandb/{args.experiment_name}/files/"
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    experiment = Experiment(args)
   # experiment.view()
    if args.debug:
        with jax.disable_jit():
            if not args.inference:
                experiment.view()
            else:
                experiment.run()
    else:
        if not args.inference:
                experiment.view()
        else:
                experiment.run()
