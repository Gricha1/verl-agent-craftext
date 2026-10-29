# vlm_pipeline — PPO + BERT + CNN demonstrations -> visual SLM -> RL fine-tune

One checkout of this branch runs all three phases of the project pipeline, for any CrafText
task set (`achievements_wood`, `achievements_all`, ...):

| Phase | What | Code | Launcher (aicenter3) |
|---|---|---|---|
| 1 | Teacher **A** = PPO + DistilBERT + CNN trained on the task set | `caged_craftext/baselines/train_teacher_multitask.py` | `launch/phase1_teacher_ppo.sh` |
| 1→2 | Demonstrations: deterministic rollouts of A, with A's logits / action probabilities | `caged_craftext/baselines/collect_flax_teacher_trajectories.py` (refuses a teacher of another task set) | `launch/phase1_collect.sh` |
| 2 | **B_ce** = visual 5M SLM + CNN from ORIGINAL random weights: CE on A's action, or distillation of A's full action distribution | `vlm_pipeline/phase2_slm/slm_ce/` | `launch/phase2_ce.sh` (train -> materialise -> closed-loop gate) |
| 3 | **B_rl** = B_ce fine-tuned by RL in veRL (veRL Agents, PPO) | `vlm_pipeline/phase3_verl/` | `launch/phase3_rl.sh` |

Rules (see `safe_rl_nlp.md`): one task set = one teacher (demonstrations only from the teacher
trained on that task set); phase 2 never starts from a checkpoint; phase 3 starts from the phase-2
checkpoint that passed the closed-loop gate; on aicenter3 our jobs use only GPU2 and GPU3
(`launch/common.sh` refuses others).

## Layout

- `caged_craftext/` (repo root) — CrafText + Craftax environment, PPO teachers, collectors.
  `ppo_with_instruction.py` is the older restart-based PPO trainer (the r4 `achievements_all` teacher);
  its collector is `collect_ppo_achievements_all_trajectories.py`.
- `phase2_slm/slm_ce/`
  - `prepare_teacher_dataset.py` — teacher Parquet -> SLM rows (prompt protocol, episode split,
    successful episodes, copied frames, `teacher_probs`);
  - `crafter_slm.py` — BPE tokenizer fitted on the train split, 5M Qwen3-style decoder;
  - `crafter_vision_slm.py` — CNN soft-token prefix + decoder, `--target argmax|teacher_distribution`;
  - `closed_loop_eval.py` — greedy closed loop in the teacher env (`--mode teacher`), in the veRL
    adapter env (`--mode adapter`), or offline accuracy;
  - `checkpoint_accuracy.py`, `test_distill_loss.py` — diagnostics / loss check.
- `phase3_verl/` — the veRL tree that runs phase 3: upstream veRL v0.9.1 (Apache-2.0, `LICENSE`)
  plus our patches in `verl/` (critic loading, tracking, FSDP/engine configs, v1 trainer) and
  `craftext_integration/` (vector agent loop, CrafText env adapter, visual SLM model / vLLM plugin,
  checkpoint materialiser, launch scripts). The 147 MB `craftext_integration/wheels/` used to build
  the docker images is not versioned.

## Docker images (aicenter3)

`modern-verl-craftext:jax0430-ppo-r4-collector` (phase 1), `modern-verl-craftext:v091-smoke-craftext`
(phase-2 training), `modern-verl-craftext:v091-smoke-craftext-trl027` (materialise, closed loop, phase 3).

## Example (wood)

```bash
cd $BASE/repo/verl-agent-craftext            # this branch
GPU_ID=2 SETTINGS=achievements_wood RUN_NAME=ppo_teacher_wood_gpu2_20261101_r1 bash vlm_pipeline/launch/phase1_teacher_ppo.sh
GPU_ID=2 SETTINGS=achievements_wood TEACHER=$RUNS/ppo_teacher_wood_gpu2_20261101_r1/checkpoints/teacher_best.pt \
  SEEDS=20261500:20261550 TASKS=0,1,2,3 RUN_NAME=wood_demos_gpu2_20261102_r1 bash vlm_pipeline/launch/phase1_collect.sh
GPU_ID=2 SETTINGS=achievements_wood CORPUS=$RUNS/wood_demos_gpu2_20261102_r1/corpus TARGET=teacher_distribution \
  RUN_NAME=slm5m_wood_distill_gpu2_20261102_r1 bash vlm_pipeline/launch/phase2_ce.sh
GPU_ID=3 SETTINGS=achievements_wood MODEL_PATH=$RUNS/slm5m_wood_distill_gpu2_20261102_r1/materialized_epoch_48 \
  RUN_NAME=slm5m_wood_verl_rl_gpu3_20261103_r1 bash vlm_pipeline/launch/phase3_rl.sh
```

Every launcher writes `code_version.txt` (commit, branch, dirty files) into its run directory.
