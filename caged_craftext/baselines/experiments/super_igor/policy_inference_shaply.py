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
# Imports from the baselines package
from baselines.experiments.deterministic_inference import DetermOptimisticResetVecEnvWrapper
from baselines.experiments.super_igor.encoder import QwenEncodeModel, QwenModelWrapper, SuperEncoder
from baselines.experiments.super_igor.expert import PlansExpert
from baselines.experiments.super_igor.scenarius_loader_v2 import (
    CrafTextScenariosWithSuperDataset,
    create_scenarios_with_super_dataset
)
from baselines.experiments.super_igor.super_dataset import SuperDataset
from baselines.experiments.super_igor.view import add_text_to_image, CraftaxRenderer
from baselines.models.actor_critic import ActorCriticConvWithBERT 

# Imports from other local packages
from craftax.craftax_env import make_craftax_env_from_name
from craftext.craftext_encoder import EncodeForm
from craftext.craftext_wrapper import InstructionWrapper

# ========= Seed Configuration =========
seed_value = 42  # Choose any fixed seed value

random.seed(seed_value)           # Fix seed for the random module
np.random.seed(seed_value)        # Fix seed for NumPy
torch.manual_seed(seed_value)     # Fix seed for PyTorch
torch.cuda.manual_seed_all(seed_value)  # Fix seed for all GPUs in PyTorch


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

    
class Experiment:
    def __init__(self, args):
        self.seeds_to_use = 30
        self.args = args
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
        config["NUM_ENVS"] = self.args.num_envs
        config["RATIO"] = self.args.ratio
        config['INFERENCE'] = self.args.inference
        config['LLM_PATH'] = self.args.llm_path
        config['DATASET_PATH'] = self.args.dataset_path
        config['SAVE_DATASET_PATH'] = self.args.save_dataset_path
        config['NUM_RETURN_SEQUENCES'] = self.args.num_return_sequences
        config['PLAN_WITH_LLM'] = self.args.plan_with_llm
        # print("Planning?: ",  config['PLAN_WITH_LLM'])
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
        actions_count = 17 if "Classic" in env_name else 43
        network_class = ActorCriticConvWithBERT
        network = network_class(actions_count, self.config["LAYER_SIZE"])

        #EncodeModel = QwenModelWrapper(self.config["LLM_PATH"], num_return_sequences=self.config['NUM_RETURN_SEQUENCES'])
        

        if not self.config['PLAN_WITH_LLM']:
            print("Use previos plans")
         #   exit()
            super_dataset = SuperDataset.load_from_json(self.config["DATASET_PATH"])
            EncodeModel = SuperEncoder(super_dataset, form_to_use=EncodeForm.EMBEDDING, num_return_sequences=1, n_splits=5, shap=True)
            ScenariosClass = create_scenarios_with_super_dataset(self.config["DATASET_PATH"], load_preinited=True, update_sd=True)
    
        env = InstructionWrapper(env, self.args.craftext_settings, scenario_handler_class=ScenariosClass,
                                  encode_model_class=EncodeModel,
                                  encode_form=EncodeForm.EMBED_CLS_FOR_SPLITS)
        

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
        rng = jax.random.PRNGKey(42)
        obs, env_state = self.env.reset(rng, self.env.default_params)
        step_fn = jax.jit(self.env.step)
        done = False
        renderer = CraftaxRenderer(self.env, self.env.default_params, pixel_render_size=1)
        steps = 0
        step_fn = jax.jit(self.env.step)
        params = self.train_state['runner_state'][0]["params"]
        observations = []
        while not done and steps < 5000:
            obs = jnp.expand_dims(obs, axis=0)
            instruction =self.env.scenario_handler.scenario_data.instructions_list[env_state.idx.item()]
            pi, value = self.network.apply(params, obs, env_state.instruction.reshape(1, -1))
            action = pi.sample(seed=rng)[0]
            
            action = jax.device_put(action, device=jax.devices('gpu')[0])

            if action is not None:
               # rng, _rng = jax.random.split(rng)
                obs, env_state, reward, done, info = step_fn(
                    rng, env_state, action, self.env.default_params
                )
                steps += 1

            image = renderer.render_to_image(env_state.env_state)
            observations.append(image)
        gif_name = "_".join(self.env.scenario_handler.scenario_data.instructions_list[env_state.idx.item()].split()[:5])
       # gif_name = inst.replace(" ", "_")
        ix = random.randint(0,200)
        folder_name = "animation"

        # Проверка и создание папки
        os.makedirs(folder_name, exist_ok=True)
        with imageio.get_writer(f'animation/{ix}_{gif_name}.gif', mode='I', duration=0.1) as writer:
            for i, image in enumerate(observations):
                text = f"Step {i}, Instruction {self.env.scenario_handler.scenario_data.instructions_list[env_state.idx.item()]}"
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
        remain_steps = 15000
        progress_bar = tqdm(total= N_INSTRUCTIONS * self.seeds_to_use, desc="Steps")
        steps = 0

        while steps < remain_steps:
            pi, _ = self.network.apply(params, obs, env_state.env_state.instruction)
            agent_rng, _rng_ = jax.random.split(agent_rng)
            action = pi.sample(seed=agent_rng)
            if action is not None:
                obs, env_state, reward, done, info = step_fn(rng, env_state, action, self.env.default_params)
                steps += 1
                progress_bar.n = np.sum(done_count)
                progress_bar.refresh()

                if render_obs:
                    self._render_observations(obs, action, env_state)

                prev_indx = self._update_success_metrics(prev_indx, env_state, done, info,
                                                        done_count, total_success_rate)

                if np.sum(done_count) >= self.env.n_instructions * self.seeds_to_use:
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
        self._update_super_dataset(total_score)
    
        
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


    def _update_super_dataset(self, total_score):
        self.super_dataset.super_print()
        self.super_dataset.clear_scores()
        self.super_dataset.rebuild_mapping()
        self.super_dataset.batch_update(self.env.scenario_handler.scenario_data.instructions_list, total_score)
        self.super_dataset.save_to_json(self.config["DATASET_PATH"])







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, type=str)
    parser.add_argument("--experiment_name", default=None, type=str)
    parser.add_argument("--plan_with_llm", default='False', type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--craftext_settings", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
    parser.add_argument("--ratio", type=int, default=1)
    parser.add_argument("--inference", type=bool, default=False)
    parser.add_argument("--llm_path", type=str, default="Qwen/Qwen2.5-3B-Instruct-Advanced")
    parser.add_argument("--dataset_path", type=str, default="temp_dataset/super_dataset.json")
    parser.add_argument("--save_dataset_path", type=str, default="temp_dataset/super_dataset.json")
    parser.add_argument("--num_return_sequences", type=int, default=5)

  
    args, rest_args = parser.parse_known_args(sys.argv[1:])
    args.plan_with_llm = False if args.plan_with_llm=="False" else True
    # print(args.plan_with_llm)
    # exit()
    if args.path is None:
        args.path = f"./wandb/{args.experiment_name}/files/"
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    experiment = Experiment(args)
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
