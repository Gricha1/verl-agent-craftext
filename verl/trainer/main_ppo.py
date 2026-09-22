# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os

# Must be set before craftax.constants loads TEXTURES pickle (JAX ShapedArray version mismatch otherwise).
os.environ.setdefault("CRAFTAX_RELOAD_TEXTURES", "True")

import hydra
import ray

from verl.utils.ray_utils import ray_local_fs_capacity_system_config, silence_ray_disk_usage_warnings
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    task_runner_runtime_env = None
    if not ray.is_initialized():
        silence_ray_disk_usage_warnings()
        # this is for local ray cluster
        ray_init_kwargs = {
            "runtime_env": {"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true"}},
            "num_cpus": config.ray_init.num_cpus,
        }
        # Добавляем num_gpus, если указано в конфиге
        # Если не указано, пытаемся определить из trainer.n_gpus_per_node
        if hasattr(config.ray_init, 'num_gpus') and config.ray_init.num_gpus is not None:
            ray_init_kwargs["num_gpus"] = config.ray_init.num_gpus
        elif hasattr(config.trainer, 'n_gpus_per_node') and config.trainer.n_gpus_per_node is not None:
            # Автоматически определяем количество GPU из конфигурации trainer
            total_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
            ray_init_kwargs["num_gpus"] = total_gpus
            print(f"[INFO] Ray init: автоматически установлено num_gpus={total_gpus} из trainer.n_gpus_per_node={config.trainer.n_gpus_per_node} * nnodes={config.trainer.nnodes}")
        # Raylet (C++) disk spam: see ray_config_def.h local_fs_capacity_threshold (default 0.95)
        _prev_sys = ray_init_kwargs.get("_system_config") or {}
        ray_init_kwargs["_system_config"] = {**_prev_sys, **ray_local_fs_capacity_system_config(config)}
        ray_init_kwargs["runtime_env"]["env_vars"] = dict(
            ray_init_kwargs["runtime_env"].get("env_vars", {})
        )
        ray_init_kwargs["runtime_env"]["env_vars"].setdefault(
            "CRAFTAX_RELOAD_TEXTURES", os.environ.get("CRAFTAX_RELOAD_TEXTURES", "True")
        )
        # Keep JAX on CPU for default Ray workers (vLLM/FSDP). GPU env only in TaskRunner.
        ray_init_kwargs["runtime_env"]["env_vars"].setdefault("JAX_PLATFORMS", "cpu")
        use_jax_gpu, jax_gpu_fraction = _read_craftext_jax_gpu_settings(config)
        task_runner_runtime_env = None
        if use_jax_gpu:
            from agent_system.environments.jax_device_config import (
                task_runner_cuda_visible_devices,
                task_runner_jax_runtime_env,
            )

            task_runner_runtime_env = {
                "env_vars": task_runner_jax_runtime_env(
                    ray_init_kwargs["runtime_env"]["env_vars"],
                    use_jax_gpu=True,
                    jax_gpu_fraction=jax_gpu_fraction,
                    config=config,
                )
            }
            print(
                f"[INFO] Craftext JAX on GPU in TaskRunner only "
                f"(CUDA_VISIBLE_DEVICES={task_runner_cuda_visible_devices(config)}, "
                f"no Ray GPU reservation — keeps {config.trainer.n_gpus_per_node} GPUs for vLLM/FSDP). "
                f"XLA_PYTHON_CLIENT_MEM_FRACTION={jax_gpu_fraction}"
            )
        ray.init(**ray_init_kwargs)

    # Do not set num_gpus on TaskRunner: Ray would subtract it from the pool and
    # fail when actor/critic need n_gpus_per_node=2 (e.g. 2 - 0.15 < 2).
    runner_opts = {"num_cpus": 1}
    if task_runner_runtime_env is not None:
        runner_opts["runtime_env"] = task_runner_runtime_env
    runner = TaskRunner.options(**runner_opts).remote()
    ray.get(runner.run.remote(config))


def _read_craftext_jax_gpu_settings(config) -> tuple[bool, float]:
    from agent_system.environments.jax_device_config import read_jax_gpu_settings

    return read_jax_gpu_settings(config)


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # print initial config
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        from agent_system.environments import make_envs

        env_name = str(getattr(config.env, "env_name", "") or "").lower()
        needs_jax = "craftext" in env_name
        if needs_jax:
            from agent_system.environments.jax_device_config import (
                configure_craftext_jax_backend_with_fallback,
                craftext_jax_device_summary,
                read_jax_gpu_settings,
            )

            use_jax_gpu, jax_gpu_fraction = read_jax_gpu_settings(config)
            use_jax_gpu = configure_craftext_jax_backend_with_fallback(
                use_jax_gpu,
                gpu_mem_fraction=jax_gpu_fraction if use_jax_gpu else None,
            )
        else:
            # AlfWorld/TextWorld: never import/configure JAX before env workers fork.
            use_jax_gpu = False
        envs, val_envs = make_envs(config)
        if needs_jax:
            print(f"[INFO] {craftext_jax_device_summary()}")

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)  # used for multimodal LLM, could be none

        # vllm early verify
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # define worker classes
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        actor_pool_id = "actor_pool"
        critic_pool_id = "critic_pool"
        half_gpus = int(config.trainer.n_gpus_per_node * 0.5)
        resource_pool_spec = {
            actor_pool_id: [half_gpus] * config.trainer.nnodes,
            critic_pool_id: [half_gpus] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: actor_pool_id,
            Role.Critic: critic_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = actor_pool_id

        # use reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = actor_pool_id

        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        if reward_manager_name == 'episode':
            from agent_system.reward_manager import EpisodeRewardManager
            reward_manager_cls = EpisodeRewardManager
        else:
            raise NotImplementedError

        use_episode_return = bool(
            config.reward_model.get("use_episode_return_as_token_reward", True)
        )
        use_remaining_return = bool(
            config.reward_model.get("use_remaining_return_as_token_reward", False)
        )
        remaining_return_gamma = float(
            config.reward_model.get(
                "remaining_return_gamma",
                config.algorithm.get("gamma", 1.0),
            )
        )
        reward_fn = reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=0,
            normalize_by_length=False,
            use_episode_return_as_token_reward=use_episode_return,
            use_remaining_return_as_token_reward=use_remaining_return,
            remaining_return_gamma=remaining_return_gamma,
        )

        # Note that we always use function-based RM for validation
        val_reward_fn = reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=1,
            normalize_by_length=False,
            use_episode_return_as_token_reward=use_episode_return,
            use_remaining_return_as_token_reward=use_remaining_return,
            remaining_return_gamma=remaining_return_gamma,
        )

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        assert config.actor_rollout_ref.rollout.n == 1, "In verl, actor_rollout_ref.rollout.n>1 is for GRPO. In verl+env, we keep n=1, and achieve GRPO by env.rollout.n"

        from agent_system.multi_turn_rollout import TrajectoryCollector
        traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
        )
        trainer.init_workers()
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """Create a dataset.

    Arguments:
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset")
    else:
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    # use sampler for better ckpt resume
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    return sampler


def set_memory_limits():
    """Ограничить использование GPU до 30 GB"""
    
    # 1. PyTorch общее ограничение
    max_memory_gb = 30
    import torch
    total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    fraction = max_memory_gb / total_memory_gb
    
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    print(f"✓ PyTorch memory limited to {max_memory_gb} GB ({fraction:.2%})")
    

if __name__ == "__main__":
    # set_memory_limits()
    main()
