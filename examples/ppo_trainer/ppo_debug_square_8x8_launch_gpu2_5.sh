#!/usr/bin/env bash
# Launch 8x8 on GPU 2+5 with ISOLATED Ray cluster (FORCE_NEW_RAY_CLUSTER=1)
# Fixes NCCL deadlock from GPU 3 contention
set -euo pipefail

cd /home/gorbov_gv/safe_rl_nlp

export CUDA_VISIBLE_DEVICES=2,5
export N_GPUS=2
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export FORCE_NEW_RAY_CLUSTER=1
export RAY_TMPDIR=/tmp/ray_temp
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
export NCCL_TIMEOUT=600

echo ==========================================
echo "PPO 8x8 on GPU 2+5 (ISOLATED cluster)"
echo ==========================================
echo [INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES N_GPUS=$N_GPUS
echo [INFO] FORCE_NEW_RAY_CLUSTER=$FORCE_NEW_RAY_CLUSTER
echo [INFO] RAY_TMPDIR=$RAY_TMPDIR

bash examples/ppo_trainer/run_caged_craftext_lora_job.sh \
  vllm \
  false \
  true \
  64 \
  1 \
  false \
  false \
  8000 \
  true \
  single_token_action \
  0 \
  ascii \
  ++env.craftext_settings='debug_square_8x8' \
  +env.use_optimistic_parallel=True \
  +env.optimistic_reset_ratio=8 \
  +env.use_ray_text_render_workers=False \
  +env.value_prompt_template_type=single_token_return \
  +env.value_return_min=-5 \
  +env.value_return_max=6 \
  +env.value_return_bin_step=0.4 \
  ++env.use_jax_gpu=False \
  ++env.jax_gpu_fraction=0.15 \
  env.history_length=50 \
  reward_model.use_episode_return_as_token_reward=False \
  reward_model.use_remaining_return_as_token_reward=True \
  reward_model.remaining_return_gamma=0.99 \
  algorithm.gamma=1.0 \
  algorithm.lam=1.0 \
  algorithm.gae_by_trajectory=False \
  algorithm.use_actor_value_token=True \
  actor_rollout_ref.actor.actor_value_token=True \
  actor_rollout_ref.actor.actor_value_loss_coef=1.0 \
  actor_rollout_ref.actor.actor_value_separate_optimizer_steps=true \
  actor_rollout_ref.actor.actor_value_target_encoding=two_hot \
  actor_rollout_ref.actor.actor_value_loss_type=clipped_mae \
  actor_rollout_ref.actor.actor_value_clipped_mae_rho=0.2 \
  actor_rollout_ref.actor.actor_value_entropy_coef=0.1 \
  actor_rollout_ref.actor.actor_value_target_from_returns=True \
  actor_rollout_ref.actor.actor_value_return_min=-5 \
  actor_rollout_ref.actor.actor_value_return_max=6 \
  actor_rollout_ref.actor.actor_value_return_bin_step=0.4 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.actor.entropy_coeff=0.01 \
  actor_rollout_ref.actor.entropy_over_valid_actions=True \
  actor_rollout_ref.actor.entropy_action_single_token_fastpath=True \
  actor_rollout_ref.actor.entropy_action_batched_forward=False \
  actor_rollout_ref.actor.entropy_action_length_normalize=True \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  ray_init.local_fs_capacity_threshold=0.999 \
  trainer.critic_warmup=0 \
  trainer.resume_mode=disable \
  trainer.save_freq=5 \
  trainer.actor_lora_only_checkpoint=True \
  trainer.test_freq=20 \
  trainer.max_actor_ckpt_to_keep=2 \
  trainer.max_critic_ckpt_to_keep=1 \
  trainer.default_local_dir=training_checkpoints/verl_agent_caged_craftext_debug_square_8x8_shared_av_gt_g099_cmae \
  trainer.actor_value_online_reward_wm.enable=False \
  trainer.actor_value_online_plan_q_wm.enable=False \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1
SCRIPT
chmod +x /home/gorbov_gv/safe_rl_nlp/examples/ppo_trainer/ppo_debug_square_8x8_launch_gpu2_5.sh
echo Script created
