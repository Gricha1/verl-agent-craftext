# Modern veRL CrafText PPO smoke path

`run_ppo_8x8_smoke.sh` is the first correctness benchmark, not a throughput
benchmark: DP=1, RSSM disabled, 4 CrafText lanes, three PPO updates, and free
form `<think>…</think><action>…</action>` output with a 320-token total response
cap. One custom Ray rollout worker owns one JAX-vmapped CrafText instance for
the complete batch; it does not create an environment actor/process per lane.
Actor and critic remain separate FSDP roles, co-located only because this is a
one-GPU smoke test.

The loop publishes one PPO item per environment action.  Its response contains
only that action's generated reasoning and `<action>` tag (at most 320 tokens);
the next observation, canonical action history, and bounded reasoning history
are rebuilt on the prompt side for the next step.  The environment reward is
written to the last trainable token of that one response, replacing veRL's
generic terminal-reward convention in `CraftextAgentLoopOutput.as_dict()`.

Run only on a confirmed free GPU, for example:

```bash
MODEL_PATH=/models/Qwen2.5-1.5B-Instruct \
CRAFTEXT_LEGACY_ROOT=/work/verl-agent-craftext \
RUN_ROOT=/work/modern_verl_runs/ppo_8x8_smoke_001 \
GPU_ID=0 PYTHON_BIN=/path/to/modern-verl/.venv/bin/python \
bash craftext_integration/run_ppo_8x8_smoke.sh
```

The defaults are deliberately a three-update smoke.  A longer isolated PPO
run uses the same entrypoint with, for example, `NUM_ENVS=32`,
`MAX_ENV_STEPS=50`, and a chosen `TOTAL_EPOCHS`; this preserves the one-PPO-
item-per-CrafText-action contract.

## Controlled 8x8 benchmark

`run_ppo_8x8_benchmark_docker.sh` is the separate measurable benchmark, not a
replacement for a running smoke/correctness job. It creates a fresh container,
Ray directory and `RUN_ROOT`; it requires `COMET_API_KEY` and logs to Comet,
console and `RUN_ROOT/logs/metrics.jsonl`. It also writes:

- `dataset/validation_seeds.json`: immutable base seed and JAX-vector lane
  contract for 64 greedy validation episodes;
- `validation_trajectories/step_*.jsonl`: first ten raw validation episodes;
- `profiles/gpu_samples.csv`: host-level 1 Hz GPU utilisation/VRAM/power/PID
  samples from process start through exit;
- `profiles/torch/`: a separate two-PPO-step CUDA/CPU/memory torch trace.

The headline validation values are `val/success_rate` (CrafText's real success
condition), mean/median episode return and length, plus action parse/invalid/
`NOOP`/repeat rates.  `train/reward_mean` remains a per-action dense reward and
is deliberately not presented as success rate.  Do not run it while a
comparison run shares either GPU, CPU, filesystem or Ray resources.

For an actual GPU test use `run_ppo_8x8_smoke_docker.sh` with a unique
`CONTAINER_NAME`. It mounts code/model/CrafText read-only, and creates a
distinct Ray directory, dataset, checkpoints, timeline, and logs under
`RUN_ROOT`; it neither connects to nor clears any existing Ray cluster.
