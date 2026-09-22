#!/bin/bash
#SBATCH --job-name=verl_ppo_caged_craftext_16x16_dual_reasoning_2gpu
#SBATCH --output=verl_ppo_caged_craftext_16x16_dual_reasoning_2gpu_%j.out
#SBATCH --error=verl_ppo_caged_craftext_16x16_dual_reasoning_2gpu_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00

export COMET_API_KEY="${COMET_API_KEY:-}"
export RAY_LOCAL_FS_CAPACITY_THRESHOLD=0.999

# === Configuration for 16x16 environment with reasoning and DUAL PPO ===
# Classic DUAL PPO: separate actor and critic models
# Actor: CE loss (cross-entropy), NOT CMAE
# Reasoning: multi-token response with reasoning text + <action>X</action> tag
# Using 2 GPUs on aicenteritl
#
# PPO/reward/env hyperparameters are pinned to the reference run
# 90deefed8ddc458da4f27349a14bf4f4 (ds8_dual_gt_g099_h50, aicenter3, 2026-09-11).
# Allowed differences (reasoning/16x16): prompt_template_type=single_token_action_reasoning,
# enable_reasoning=True, reasoning_history_length=5, max_response_length=320,
# craftext_settings=debug_square_16x16, max_prompt_length=2048,
# single_token_actions=False.
# Forced technical/resource differences: use_remove_padding=False (no flash_attn on
# aicenteritl), train_batch_size=32 (RAM limit, 64 envs OOMs), save_freq=4 (user decision:
# grounding needs initial/early/middle/final checkpoints; reference had -1).

MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
SAVE_PATH=${SAVE_PATH:-"/home/gorbov_gv/safe_rl_nlp/checkpoints/verl/ppo_caged_craftext_16x16_dual_reasoning_2gpu/"}

# Environment parameters (reference: env.seed=0, env.max_steps=50, env.history_length=50)
ENV_SEED=${ENV_SEED:-0}
HISTORY_LENGTH=${HISTORY_LENGTH:-50}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-50}

# PPO parameters (reference: total_epochs=8000, ppo_epochs=1)
TOTAL_EPOCHS=${TOTAL_EPOCHS:-8000}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-1}

# Batch sizes (reference: train_batch_size=64, mini=64, micro=16)
# aicenteritl RAM limits the env count: 64 workers OOM (125GB), user-authorized 32.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-16}
PPO_MICRO_BATCH=${PPO_MICRO_BATCH:-16}
PPO_MINI_BATCH=${PPO_MINI_BATCH:-64}
CRITIC_MICRO_BATCH=${CRITIC_MICRO_BATCH:-16}
CRITIC_MINI_BATCH=${CRITIC_MINI_BATCH:-64}

# Learning rates (reference)
ACTOR_LR=${ACTOR_LR:-1e-6}
CRITIC_LR=${CRITIC_LR:-1e-5}

# Sequence lengths (reference: prompt 1024, response 1)
# 16x16 + reasoning need: prompt 2048 (user-approved), response 320 (user-confirmed).
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-320}

# LoRA parameters (reference: actor 64/64, critic 64/64)
ACTOR_LORA_RANK=${ACTOR_LORA_RANK:-64}
ACTOR_LORA_ALPHA=${ACTOR_LORA_ALPHA:-64}
CRITIC_LORA_RANK=${CRITIC_LORA_RANK:-64}
CRITIC_LORA_ALPHA=${CRITIC_LORA_ALPHA:-64}

# GPU memory utilization for vLLM (reference: 0.55)
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.55}

# Trainer cadence (reference: test_freq=20, val_before_train=True, save_freq=-1)
# save_freq=4 per user decision: grounding requires initial/early/middle/final checkpoints.
TEST_FREQ=${TEST_FREQ:-20}
SAVE_FREQ=${SAVE_FREQ:-4}

# Logging
PROJECT_NAME=${PROJECT_NAME:-"verl-agent-caged-craftext"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"ppo_caged_craftext_16x16_dual_reasoning_2gpu"}

# === Launch training ===
export VLLM_ATTENTION_BACKEND=TRITON_ATTN

cd ~/safe_rl_nlp

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    +algorithm.log_prob_action_only=false \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    algorithm.gae_by_trajectory=False \
    algorithm.use_actor_value_token=False \
    algorithm.use_kl_in_reward=False \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.return_raw_chat=False \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    reward_model.use_episode_return_as_token_reward=False \
    reward_model.use_remaining_return_as_token_reward=True \
    reward_model.remaining_return_gamma=0.99 \
    ++env.env_name=caged_craftext/CagedCraftextEnv \
    ++env.craftext_settings=debug_square_16x16 \
    ++env.observation_type=ascii \
    env.seed=$ENV_SEED \
    env.max_steps=$MAX_EPISODE_STEPS \
    env.history_length=$HISTORY_LENGTH \
    +env.auto_reset=False \
    ++env.use_jax_gpu=False \
    ++env.jax_gpu_fraction=0.15 \
    +env.use_optimistic_parallel=True \
    +env.optimistic_reset_ratio=8 \
    +env.use_ray_text_render_workers=False \
    +env.value_return_min=-5 \
    +env.value_return_max=6 \
    +env.value_return_bin_step=0.4 \
    env.resources_per_worker.num_cpus=0.03 \
    +env.enable_reasoning=True \
    +env.prompt_template_type=single_token_action_reasoning \
    +env.reasoning_history_length=5 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.lora_rank=$ACTOR_LORA_RANK \
    actor_rollout_ref.model.lora_alpha=$ACTOR_LORA_ALPHA \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.actor.entropy_over_valid_actions=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.actor.actor_value_token=False \
    actor_rollout_ref.actor.single_token_actions=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTIL \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=$N_SAMPLES_PER_PROMPT \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    critic.optim.lr=$CRITIC_LR \
    critic.model.path=$MODEL_PATH \
    critic.model.enable_gradient_checkpointing=True \
    critic.model.use_remove_padding=False \
    critic.model.lora_rank=$CRITIC_LORA_RANK \
    critic.model.lora_alpha=$CRITIC_LORA_ALPHA \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    critic.ppo_micro_batch_size_per_gpu=$CRITIC_MICRO_BATCH \
    critic.ppo_mini_batch_size=$CRITIC_MINI_BATCH \
    trainer.critic_warmup=0 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.val_before_train=True \
    trainer.test_freq=$TEST_FREQ \
    trainer.save_freq=$SAVE_FREQ \
    trainer.resume_mode=disable \
    trainer.env_val_video_freq=200000 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.logger=[console,comet] \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    ++ray_init.local_fs_capacity_threshold=0.999
