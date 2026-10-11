# Offline reward-RSSM QA suite

`evaluate_offline_reward_rssm_qa.py` is a read-only, two-stage evaluator for a
saved `reward_continuation_heads=true` RSSM checkpoint on `debug_square_8x8`.
It does not construct an optimizer or modify a checkpoint.

It first records `checkpoint_identity.json` with the checkpoint's absolute
path, SHA-256, byte length, saved training step, resolved config, and source
commit. It refuses any checkpoint unless its resolved config declares exactly
five `h` soft tokens, five `z` soft tokens, and reward/continuation heads.

Run the mandatory one-episode sanity pass on the H200 first:

```bash
python scripts/evaluate_offline_reward_rssm_qa.py \
  --run-dir /data/homes/gorbov_gv/latent_wm/runs/<reward-rssm-run> \
  --checkpoint /data/homes/gorbov_gv/latent_wm/runs/<reward-rssm-run>/checkpoint_step_<N>.pt \
  --output-dir /data/homes/gorbov_gv/latent_wm/evaluation/<reward-rssm-run>_sanity \
  --device cuda:0
```

Inspect `sanity.json`, `checkpoint_identity.json`, one frame under `frames/`,
and `gifs/episode_*/` before the full run.  The full run is explicit:

```bash
python scripts/evaluate_offline_reward_rssm_qa.py \
  --run-dir /data/homes/gorbov_gv/latent_wm/runs/<reward-rssm-run> \
  --checkpoint /data/homes/gorbov_gv/latent_wm/runs/<reward-rssm-run>/checkpoint_step_<N>.pt \
  --output-dir /data/homes/gorbov_gv/latent_wm/evaluation/<reward-rssm-run>_full \
  --device cuda:0 --full --candidate-batch-size 64
```

The evaluator creates fixed trajectories before asking the models questions;
therefore Base, HZ, H-only, and Z-only are scored on identical states. HZ is
hard-asserted to receive exactly `soft(h)×5 + soft(z)×5`, and its text prompt
cannot contain the observation, action-history template, or reasoning-history
template. Real observations are used solely by the RSSM posterior.

Key artifacts:

- `per_state_questions.csv` and `.html`: every scored question and model answer;
- `question_summary.csv`, `group_macro.csv`, and `summary.md`;
- `paired_trajectories.json`: executed fixed actions and simulator ground truth;
- `actor_probe.json` and `actor_probe_summary.json`;
- `autonomous_base.json`, `autonomous_hz.json`, and `actor_summary.csv`;
- `free_generation.json`: qualitative `<think>/<answer>` traces, not used in
  accuracy;
- `frames/` and `gifs/`: auditable visualizations.
