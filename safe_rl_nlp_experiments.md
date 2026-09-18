# safe_rl_nlp — Experiment Tracker

Рабочая тетрадь экспериментов `safe_rl_nlp`. Здесь фиксируются:

1. Canonical pending experiments (что мы **хотим** запустить).
2. Matched configuration (что **общее** между 8×8 и 16×16).
3. Valid baselines (что считаем опорным).
4. Recent experiment history (что реально запускали, с Comet keys и статусами).
5. Invalid / diagnostic runs (что в мусор / не сравнимо).
6. Текущие гипотезы.
7. Checkpoint policy.
8. Next launch checklist (что проверить перед запуском).

Machine-readable источник истины по метаданным:
[`experiments/registry.yaml`](experiments/registry.yaml). Гипотезы:
[`experiments/hypotheses.yaml`](experiments/hypotheses.yaml).

---

## PLANNED EXPERIMENTS -- grounding and 16x16 priority

None of the entries in this section has been run. No Comet IDs are assigned.
Common invariants where architecturally applicable: same starting base checkpoint,
history 50, episode maximum 50, current environment reward, invalid-action penalty
enabled with coefficient `0.1`, discounted remaining return `G_t^0.99`, and the
current baseline KL/reference configuration. CMAE is excluded from this main
grounding matrix.

| Priority / planned name | Architecture | Reasoning | Status | Source of truth / required settings |
|---|---|---:|---|---|
| 16x16-A `ds16_dual_no_reasoning_grounding` | DUAL PPO | OFF | **PLANNED / NOT RUN** | Existing 16x16 DUAL launcher/config; 16x16, history 50, one coded action token, preserve checkpoint snapshots. |
| 16x16-B `ds16_dual_reasoning_grounding` | DUAL PPO | ON | **PLANNED / NOT RUN** | Matched to A except reasoning: response <=64, last five reasoning responses, final parseable `<action>code</action>`, target prompt limit 4096 after measuring clipping. |
| 8x8-G1 `ds8_dual_no_reasoning_grounding` | DUAL PPO | OFF | **PLANNED / NOT RUN** | Same 8x8 environment/action history/reward/checkpoint base as G2/G3. |
| 8x8-G2 `ds8_dual_reasoning_grounding` | DUAL PPO | ON | **PLANNED / NOT RUN** | Same as G1 except reasoning; response <=64, five reasoning history entries, `max_prompt_length=1536`. |
| 8x8-G3 `ds8_shared_no_reasoning_grounding` | SINGLE/SHARED actor-value | OFF | **PLANNED / NOT RUN** | Same base/env/reward where applicable; no CMAE factor in the main comparison. |
| 8x8-G4 `ds8_shared_reasoning_grounding` | SINGLE/SHARED actor-value | ON | **BLOCKED / NOT RUN** | Current state-level value-bin implementation has no token-wise V/GAE semantics over reasoning prefixes. |

For 16x16-A use the existing DUAL configuration rather than inventing a new one.
For 16x16-B preserve environment/action semantics and use the same coded action map:
`1=NOOP, 2=LEFT, 3=RIGHT, 4=UP, 5=DOWN, 6=DO, 7=SLEEP, 8=PLACE_STONE,
9=PLACE_TABLE, a=PLACE_FURNACE, b=PLACE_PLANT, c=MAKE_WOOD_PICKAXE,
d=MAKE_STONE_PICKAXE, e=MAKE_IRON_PICKAXE, f=MAKE_WOOD_SWORD,
g=MAKE_STONE_SWORD, h=MAKE_IRON_SWORD`. The reasoning parser must map a final
`<action>4</action>` compatibly without changing environmental action semantics.

Required B diagnostics: `valid_action_rate`, `format_error_rate`,
`invalid_action_rate`, `invalid_action_penalty`, response length mean/p95/truncation
rate, prompt length mean/p95/max/clip ratio, success rate, and reward. A format
error (unparseable action/tag) must remain distinct from a parsed but environment-
invalid action. Preserve initial, early, middle and final checkpoints for all
canonical runs.

## 1. Canonical experiments to run

Сейчас **два основных matched experiments** — 8×8 и 16×16 на единой
shared actor-value архитектуре с CMAE value loss. Reasoning выключен в обоих.

### 1.1 8×8 shared AV CMAE NO reasoning (draft id: `ds8_shared_av_gt_g099_cmae_h50_no_reasoning`)

| Поле | Значение |
|------|----------|
| Map | `debug_square_8x8` |
| Architecture | shared / single actor-value (one LLM = actor + value) |
| Value head | `actor_value_token=True` (через actor LoRA) |
| Critic | `separate_critic=False` (`critic.lora_rank=0`, frozen base) |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` + LoRA r=64 alpha=64 |
| Reasoning | **OFF** — `enable_reasoning=False` (или флаг НЕ передаётся) |
| History length | 50 |
| Max steps | 50 |
| Action | single token, `prompt_template_type=single_token_action` |
| `max_response_length` | 1 |
| Reward | per-step `r_t = d_t − d_{t+1} + success_bonus` |
| Token reward | remaining return `G_t^0.99` = `r_t + 0.99·r_{t+1} + 0.99²·r_{t+2} + ...` |
| `remaining_return_gamma` | 0.99 |
| `use_remaining_return_as_token_reward` | True |
| `gae_by_trajectory` | False |
| Reward-WM, Q-WM, plan-Q | OFF |
| Value target encoding | two-hot |
| Value bins 8×8 | `[-5, 6]`, step 0.4 |
| Value loss | two-hot **clipped MAE** (CMAE), `rho = 0.2` |
| `actor_value_target_from_returns` | True |
| `actor_value_loss_coef` | 1.0 |
| `actor_value_separate_optimizer_steps` | True |
| `actor_value_entropy_coef` | 0.1 |
| `value_prompt_template_type` | `single_token_return` |
| Total epochs | 8000 |
| `train_batch_size` | **TBD — match validated shared-AV baseline** |
| `max_prompt_length` | **MEASURE BEFORE LAUNCH** (см. §8 checklist) |
| GPUs | **TBD — какие свободны** |
| Checkpoint strategy | rolling actor-only, save_freq=5, keep=2 |
| Comet | ON |

Ближайший реально использованный шаблон: `_exp_scripts/a3_launch_ds8_cmae_13.sh`
(см. resolved overrides в `logdir/ppo_debug_square_8x8_shared_av_gt_g099_cmae_20260914_195710.log`,
Comet `1f29addc63504d96b78f7fc5fe3c5611` — INVALID_SETUP, см. §5).

### 1.2 16×16 shared AV CMAE NO reasoning (draft id: `ds16_shared_av_gt_g099_cmae_h50_no_reasoning`)

| Поле | Значение |
|------|----------|
| Map | `debug_square_16x16` |
| Architecture | shared / single actor-value (тот же) |
| Value head | `actor_value_token=True` (тот же) |
| Critic | `separate_critic=False` (тот же) |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` + LoRA r=64 alpha=64 |
| Reasoning | **OFF** — **отличается от последних 16×16 CMAE попыток (6e817, 83dc44), которые были с `enable_reasoning=True`** |
| History length | 50 |
| Max steps | 50 |
| Action | single token, `prompt_template_type=single_token_action` |
| `max_response_length` | 1 |
| Reward | per-step `r_t` (та же) |
| Token reward | remaining return `G_t^0.99` |
| `remaining_return_gamma` | 0.99 |
| `gae_by_trajectory` | False |
| Value target encoding | two-hot |
| Value bins 16×16 | `[-26, 16]`, step **1.5** |
| Value loss | two-hot **clipped MAE** (CMAE), `rho = 0.2` |
| `actor_value_target_from_returns` | True |
| `value_prompt_template_type` | `single_token_return` |
| Total epochs | 8000 |
| `train_batch_size` | **TBD — match 8×8** |
| `max_prompt_length` | **MEASURE BEFORE LAUNCH** |
| GPUs | **TBD — какие свободны** |
| Checkpoint strategy | rolling actor-only, save_freq=5, keep=2 |
| Comet | ON |

Ближайшие реально использованные шаблоны:
`_exp_scripts/a3_launch_ds16_cmae_*.sh` (см. resolved overrides в
`logdir/ppo_debug_square_16x16_shared_av_gt_g099_cmae_*.log`,
Comet `6e817a47...` / `83dc44c0...` — оба с `enable_reasoning=True`,
поэтому **не** валидный NO-reasoning baseline, см. §4/§5).

---

## 2. Canonical matched configuration (8×8 vs 16×16)

Столбцы — только то, что **должно совпадать** между двумя canonical runs.

| Поле | 8×8 | 16×16 |
|------|-----|-------|
| `env.enable_reasoning` | **False** | **False** |
| `env.craftext_settings` | `debug_square_8x8` | `debug_square_16x16` |
| `env.observation_type` | `ascii` | `ascii` |
| `env.history_length` | 50 | 50 |
| `env.max_steps` | 50 | 50 |
| `env.prompt_template_type` | `single_token_action` | `single_token_action` |
| `env.value_prompt_template_type` | `single_token_return` | `single_token_return` |
| `env.value_return_min` | -5 | **-26** |
| `env.value_return_max` | 6 | **16** |
| `env.value_return_bin_step` | 0.4 | **1.5** |
| `actor_value_return_min` | -5 | **-26** |
| `actor_value_return_max` | 6 | **16** |
| `actor_value_return_bin_step` | 0.4 | **1.5** |
| `algorithm.gamma` | 1.0 | 1.0 |
| `algorithm.lam` | 1.0 | 1.0 |
| `algorithm.gae_by_trajectory` | False | False |
| `algorithm.use_actor_value_token` | True | True |
| `actor.actor_value_token` | True | True |
| `actor.actor_value_loss_type` | `clipped_mae` | `clipped_mae` |
| `actor.actor_value_clipped_mae_rho` | 0.2 | 0.2 |
| `actor.actor_value_target_encoding` | `two_hot` | `two_hot` |
| `actor.actor_value_target_from_returns` | True | True |
| `actor.actor_value_loss_coef` | 1.0 | 1.0 |
| `actor.actor_value_entropy_coef` | 0.1 | 0.1 |
| `actor.actor_value_separate_optimizer_steps` | True | True |
| `actor.single_token_actions` | True | True |
| `actor.use_kl_loss` / `kl_loss_coef` | True / 0.01 | True / 0.01 |
| `actor.clip_ratio` | 0.2 | 0.2 |
| `actor.kl_loss_type` | `low_var_kl` | `low_var_kl` |
| `actor.use_invalid_action_penalty` | True / 0.1 | True / 0.1 |
| `actor.entropy_coeff` | 0.01 | 0.01 |
| `actor.entropy_over_valid_actions` | True | True |
| `critic.lora_rank` | 0 (frozen base) | 0 (frozen base) |
| `critic.optim.lr` | 1e-5 | 1e-5 |
| `model.path` | `Qwen/Qwen2.5-1.5B-Instruct` | (same) |
| `model.lora_rank` / `lora_alpha` | 64 / 64 | 64 / 64 |
| `rollout.gpu_memory_utilization` | 0.6 | 0.6 |
| `rollout.enforce_eager` | True | True |
| `rollout.val_kwargs.temperature` | 1.0 | 1.0 |
| `rollout.tensor_model_parallel_size` | 1 | 1 |
| `data.train_batch_size` | **TBD** (match historical shared-AV) | **TBD** |
| `data.max_prompt_length` | **MEASURE BEFORE LAUNCH** | **MEASURE BEFORE LAUNCH** |
| `data.max_response_length` | 1 | 1 |
| `data.filter_overlong_prompts` | True | True |
| `data.truncation` | `error` | `error` |
| `reward_model.use_remaining_return_as_token_reward` | True | True |
| `reward_model.remaining_return_gamma` | 0.99 | 0.99 |
| `reward_model.use_episode_return_as_token_reward` | False | False |
| `trainer.logger` | `[console, comet]` | `[console, comet]` |
| `trainer.total_epochs` | 8000 | 8000 |
| `trainer.critic_warmup` | 0 | 0 |
| `trainer.actor_lora_only_checkpoint` | True | True |
| `trainer.save_freq` / `max_actor_ckpt_to_keep` | 5 / 2 | 5 / 2 |
| `trainer.test_freq` | 20 | 20 |
| `trainer.val_before_train` | True | True |
| `trainer.actor_value_online_reward_wm.enable` | False | False |
| `trainer.actor_value_online_plan_q_wm.enable` | False | False |

**Что отличается и не должно меняться внутри серии:**
- `craftext_settings` (8×8 vs 16×16)
- `value_return_min/max/bin_step` и `actor_value_return_min/max/bin_step`
  (для 16×16 расширены из-за большего диапазона return-to-go)
- `gpu_memory_utilization` для rollout может отличаться на разных картах

**Что пока TBD** (см. §8 checklist):
- Точное значение `train_batch_size`
- Точное значение `max_prompt_length` (нужно замерить на каждой карте с
  NO reasoning конфигом; предыдущие измерения — с reasoning ON — нерелевантны)
- Какие именно GPU использовать (зависит от текущих владельцев на aicenter3)

---

## 3. Valid baselines

### 3.1 8×8 dual — valid baselines

Эти runs дают **baseline** для 8×8 dual архитектуры (отдельный critic,
без shared actor-value). Исторические, не запускались в этой/предыдущей
сессии — но они **эмпирически валидные** для сравнения.

| Comet key | id | env | reward | γ | hist | status | success | notes |
|-----------|-----|-----|--------|---|------|--------|---------|-------|
| `02e2744ddcdb4573ad263487a710e956` | ds8_dual_rt_h0 | 8×8 dual | `r_t` | 1.0 | 0 | successful | max 1.0, last ~0.99 | Репродукция FINAL dual baseline |
| `874bc480b5224efb9b92795d271ca81c` | ds8_dual_rt_h50 | 8×8 dual | `r_t` | 1.0 | 50 | successful | max 0.96, last ~0.96 | History=50 **не** ломает learning с `r_t` |
| `983925d38dda4ef18af90f87919fa37d` | ds8_dual_final | 8×8 dual | `r_t` | 1.0 | 0 | successful (historical) | — | Опорный dual |
| `680dfbf1125a4397bdc8ad3726110d25` | ds8_dual_valent | 8×8 dual | `r_t` | 1.0 | 0 | successful (historical) | — | valid-action entropy, близко к FINAL |
| `1a064d01c9ae4e8596a54d06d78f674d` | ds8_dual_gt_h50 | 8×8 dual | `G_t` (claimed) | 1.0 | 50 | failed | — | token reward остался `r_t` (баг кода) |
| `2a42e153bdb0448789a56140e70fdf50` | ds8_dual_gt_h50_b | 8×8 dual | `G_t` | 1.0 | 50 | failed | max 0.12, last ~0.09 | успех ~12% на step 79; vf_explained bad |
| `c3a5e727c5414fa3a4e5ff57657351b1` | ds8_dual_rt_trajgae | 8×8 dual | `r_t` + traj GAE | 1.0 | 50 | inconclusive | max 0.06 | policy success не стартует |

### 3.2 8×8 shared — valid baselines

| Comet key | id | env | arch | loss | γ | status | notes |
|-----------|-----|-----|------|------|---|--------|-------|
| `535900d8711a44a0b3a37ab55458b48f` | ds8_shared_av_gt_g099_a3 | 8×8 shared AV | shared | two_hot CE | 1.0 (rtg γ=0.99) | failed (disk_full) | step ~28, success_max 0.077, value_loss 1.633, acc 0.367, vf_explained -0.033, ep_reward -0.406 |
| `4be19de5e50949359891ce852700d565` | ds8_av (historical) | 8×8 actor_value | sep actor+critic | value bins | 1.0 | inconclusive | uses value bins, **не** dual |
| `96b48ca1b11340c3801d1c828b39e573` | (unverified) | 8×8 shared | shared | CE | — | failed (disk_full) | зафиксировано в заметках 535900d8 как параллельный запуск, **comet_key unverified** — использовать как baseline **нельзя** |

### 3.3 16×16 dual — valid baselines

| Comet key | id | env | reward | γ | hist | status | notes |
|-----------|-----|-----|--------|---|------|--------|-------|
| `b126b4532767420f9a0fd031b4d27739` | ds16_dual_rt_h50 (2 GPU) | 16×16 dual | `r_t` | 1.0 | 50 | oom | Early OOM на vLLM wake_up/kv_cache ~step 2-4 |
| `da03a539d25e4c46882ebed96a3cafc4` | ds16_dual_rt_h50 (restart) | 16×16 dual | `r_t` | 1.0 | 50 | aborted | Короткий restart |
| `da63a962175b4f1a82606c3fc0fe82d8` | ds16_dual_rt_h50 (long) | 16×16 dual | `r_t` | 1.0 | 50 | failed (long) | step ~351; success_max 0.125, last ~0.0; **доказывает что 16×16 проблема — не про G_t**, а про partial observability (H9) |

### 3.4 16×16 shared — valid baselines

| Comet key | id | env | arch | loss | γ | status | notes |
|-----------|-----|-----|------|------|---|--------|-------|
| `3f6f151820dc4ad69b700bff3ddae686` | ds16_av (trajGAE over G_t) | 16×16 actor_value | sep | bins | 1.0 | oom | double-count GAE; returns exploded |
| `1ac1a602403c44e183f13a9c340040b4` | ds16_av (gae=False) | 16×16 actor_value | sep | bins | 1.0 | failed | policy weak / success low |
| `14831ad097444a28ab3d2e3d2926aee6` | ds16_av (FINAL-ish) | 16×16 actor_value | sep | bins | 1.0 | failed | success 0; hist=0 |
| `4d803c6eef934d5a9f95f47b79b53eea` | ds16_shared_av_gt_g099_ce | 16×16 shared AV | shared | two_hot CE | 1.0 (rtg γ=0.99) | failed (oom) | bins [-26,16]/1.5; relaunched как aa0714bb |
| `aa0714bbaaca4cdbbd2536551f065fe7` | ds16_shared_av_gt_g099_ce (relaunch) | 16×16 shared AV | shared | two_hot CE | 1.0 (rtg γ=0.99) | stalled (disk_full) | global_step 28; success 0; value_loss 4.7→1.225, value_acc 0.007→0.634, vf_explained -0.175. Видна value-classifier learning, **но training horizon слишком короткий для итогов** |
| (5e4492cda67d…) | (pre-4d803c6e) | 16×16 shared AV | shared | CE | 1.0 | dead (NCCL hang) | superseded → 4d803c6e |
| (19afee57ab7b…) | (pre-4d803c6e) | 16×16 shared AV | shared | CE | 1.0 | dead (vllm_mem=0.75 OOM) | superseded → 4d803c6e |

---

## 4. Recent experiment history (chronological)

Это **новейшая** история — последние запуски. Внешние ссылки на Comet
приведены; полные resolved overrides — в соответствующих локальных логах.

### 4.1 2026-09-13 (aicenter3, 8×8 shared AV CE)

| time | id | comet | env | arch | loss | reasoning | status | notes |
|------|-----|-------|-----|------|------|-----------|--------|-------|
| 17:36 | ds8_shared_av_gt_g099_a3 | `535900d8…` | 8×8 | shared AV | two_hot CE | OFF (default) | failed (disk_full) | Реальный running ~28 PPO steps. value_loss 1.633, value_acc 0.367, vf_explained -0.033, ep_reward -0.406. Crash от `/tmp/ray` disk full (aicenter3 incident). Checkpoint stages `[40,110]` — **никаких checkpoint'ов не сохранилось** |

### 4.2 2026-09-13 (aicenter3, 16×16 shared AV CE relaunch)

| time | id | comet | env | arch | loss | reasoning | status | notes |
|------|-----|-------|-----|------|------|-----------|--------|-------|
| 17:36 | ds16_shared_av_gt_g099_ce (relaunch) | `aa0714bb…` | 16×16 | shared AV | two_hot CE | OFF | stalled (disk_full) | step 28 / 89600 env steps (~635 s/step). Twin 8×8 CE baseline в архитектуре, но map=16×16 и bins [-26,16]/1.5. **Видна value-classifier learning** (loss 4.7→1.225, acc 0.007→0.634), vf_explained -0.175, но training horizon слишком короткий. Завис в UPDATE phase из-за того же disk_full |

### 4.3 2026-09-14 (aicenter3, 16×16 shared AV CMAE attempts)

| time | comet | arch | loss | reasoning | PPO steps | status | notes |
|------|-------|------|------|-----------|-----------|--------|-------|
| 17:36 | `56dd190a05354ec6b78f1bbc4bad4349` | shared AV | CMAE rho=0.2 | ON (default) | 0 | **INVALID_SETUP** | AF_UNIX path >107 bytes (`/mnt/aicenter1-datasets/gorbov_gv/...`) |
| 17:39 | `d30ba0e096814b0b9106e6cbda8b27b2` | shared AV | CMAE rho=0.2 | ON | partial init | stopped (SIGTERM) | Не дошёл до первого PPO update; внешний kill |
| 17:42 | `351b9d90c040407b9b47fc50877c6d9a` | shared AV | CMAE rho=0.2 | ON | partial init | stopped (SIGTERM) | То же |
| 18:06 | `40b2cdb9861541d899389d441ee24234` | shared AV | CMAE rho=0.2 | ON | partial init | stopped | Не дошёл до training |
| 18:17 | `6e817a4776b940b49544c68618e2e44a` | shared AV | CMAE rho=0.2 | ON | **3** PPO updates | stopped (SIGTERM) | step 1 logged: success 0, value_loss 0.632, value_acc 0.007, value_entropy 2.307. **Diagnostic metrics (`p_target_mass`, `clipped_mae/unclipped_mae`) не появлялись** в логе — потерян `_actor_value_diag` |
| 18:55 | `83dc44c02e92498093f6ac324d7a7493` | shared AV | CMAE rho=0.2 | ON | **~25** PPO updates | **CRASHED** (ActorDiedError → SIGTERM) | На шаге 1 — diagnostics появились (`p_target_mass_le_0.2:0.905`, `clipped_mae:0.861`, `p_left/mean:0.017`, etc.). Шаг 26 упал в `compute_actor_token_values`: ActorDiedError → Worker exit type SYSTEM_ERROR → SIGTERM. Checkpoint stages `[40,110]` — **сохранённых checkpoint'ов нет** |

### 4.4 2026-09-14 (aicenter3, 8×8 shared AV CMAE + 8×8/16×16 retries)

| time | id | comet | env | arch | loss | reasoning | status | notes |
|------|-----|-------|-----|------|------|-----------|--------|-------|
| 19:58 | ds8_shared_av_gt_g099_cmae_13 | `1f29addc63504d96b78f7fc5fe3c5611` | 8×8 | shared AV | CMAE rho=0.2 | **OFF** | **INVALID_SETUP** | Тот же AF_UNIX path issue: `ray_temp_dir=/mnt/aicenter1-datasets/gorbov_gv/ray_ds8_cmae`. **Конфиг уже был с `enable_reasoning=False`** — это правильный NO-reasoning шаблон для будущих запусков |
| 23:09–23:55 | ds8_gpu3_4_launch_v9 (2 attempts) | `bb4f96e…` | 8×8 | shared AV | CE? + value bins | **ON** (`+env.enable_reasoning=True`) | CRASHED | Упал с `NotImplementedError: sequence_length=1587 > max_length=1024` на v9_231607. **v9 НЕ был NO-reasoning**, как предполагалось в более ранней документации |
| 23:24, 23:32 | ds8_gpu3_4_launch_v10 (2 attempts) | (тот же `bb4f96e…`) | 8×8 | shared AV | CE? + value bins | **ON** | CRASHED | Упал с `NotImplementedError: sequence_length=6514 > max_length=6144` на v10_233211. Главная гипотеза: раздувание prompt'а из-за `value_prompt_template_type=single_token_return` + `actor_value_token=True` |

### 4.5 8×8 CMAE recent log — fresh resolved overrides (1f29addc, 19:58)

Полный resolved override list (полезно как шаблон для будущих canonical 8×8 NO-reasoning CMAE запусков) — из лога
`logdir/ppo_debug_square_8x8_shared_av_gt_g099_cmae_20260914_195710.log`.

```
algorithm.adv_estimator=gae
data.train_files=/home/gorbov_gv/data/verl-agent/text/train.parquet
data.val_files=/home/gorbov_gv/data/verl-agent/text/test.parquet
data.train_batch_size=64
data.val_batch_size=16
data.max_prompt_length=1024
data.max_response_length=1                    # single token action
data.filter_overlong_prompts=True
data.truncation=error
data.return_raw_chat=False
actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct
actor_rollout_ref.model.lora_rank=64
actor_rollout_ref.model.lora_alpha=64
actor_rollout_ref.actor.optim.lr=1e-6
actor_rollout_ref.actor.ppo_mini_batch_size=128
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=64
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.01
actor_rollout_ref.actor.clip_ratio=0.2
actor_rollout_ref.actor.kl_loss_type=low_var_kl
actor_rollout_ref.actor.use_invalid_action_penalty=True
actor_rollout_ref.actor.invalid_action_penalty_coef=0.1
critic.optim.lr=1e-5
critic.model.use_remove_padding=True
critic.model.path=Qwen/Qwen2.5-1.5B-Instruct
critic.model.lora_rank=0                     # frozen base, no critic LoRA
critic.model.lora_alpha=16
critic.ppo_mini_batch_size=128
critic.ppo_micro_batch_size_per_gpu=64
algorithm.use_kl_in_reward=False
reward_model.use_episode_return_as_token_reward=False
+algorithm.log_prob_action_only=false
env.env_name=caged_craftext/CagedCraftextEnv
+env.enable_reasoning=False                  # ← KEY: OFF
+env.craftext_settings=achievements_safe_budget_energy_collect_wood
+env.observation_type=ascii
+env.prompt_template_type=single_token_action
actor_rollout_ref.actor.single_token_actions=True
actor_rollout_ref.actor.actor_value_token=False
+env.auto_reset=False
++env.use_jax_gpu=False
+actor_rollout_ref.model.use_action_head=False
+actor_rollout_ref.model.num_actions=17
env.seed=0
env.max_steps=50
env.history_length=50
env.resources_per_worker.num_cpus=0.03
trainer.critic_warmup=0
trainer.logger=[console,comet]
trainer.project_name=verl-agent-caged-craftext
trainer.experiment_name=ds8_shared_av_gt_g099_cmae_h50
trainer.n_gpus_per_node=2
trainer.nnodes=1
trainer.save_freq=100                         # overridden to 5 below
trainer.test_freq=0                           # overridden to 20 below
trainer.total_epochs=8000
trainer.val_before_train=True
# --- file-specific overrides ---
++env.craftext_settings=debug_square_8x8
+env.use_optimistic_parallel=True
+env.optimistic_reset_ratio=8
+env.use_ray_text_render_workers=False
+env.value_prompt_template_type=single_token_return
+env.value_return_min=-5
+env.value_return_max=6
+env.value_return_bin_step=0.4
++env.use_jax_gpu=False
++env.jax_gpu_fraction=0.15
env.history_length=50
reward_model.use_episode_return_as_token_reward=False
reward_model.use_remaining_return_as_token_reward=True
reward_model.remaining_return_gamma=0.99
algorithm.gamma=1.0
algorithm.lam=1.0
algorithm.gae_by_trajectory=False
algorithm.use_actor_value_token=True
actor_rollout_ref.actor.actor_value_token=True
actor_rollout_ref.actor.actor_value_loss_coef=1.0
actor_rollout_ref.actor.actor_value_separate_optimizer_steps=true
actor_rollout_ref.actor.actor_value_target_encoding=two_hot
actor_rollout_ref.actor.actor_value_loss_type=clipped_mae
actor_rollout_ref.actor.actor_value_clipped_mae_rho=0.2
actor_rollout_ref.actor.actor_value_entropy_coef=0.1
actor_rollout_ref.actor.actor_value_target_from_returns=True
actor_rollout_ref.actor.actor_value_return_min=-5
actor_rollout_ref.actor.actor_value_return_max=6
actor_rollout_ref.actor.actor_value_return_bin_step=0.4
actor_rollout_ref.rollout.gpu_memory_utilization=0.6
actor_rollout_ref.rollout.enforce_eager=True
actor_rollout_ref.rollout.val_kwargs.temperature=1.0
actor_rollout_ref.actor.entropy_coeff=0.01
actor_rollout_ref.actor.entropy_over_valid_actions=True
actor_rollout_ref.actor.entropy_action_single_token_fastpath=True
actor_rollout_ref.actor.entropy_action_batched_forward=False
actor_rollout_ref.actor.entropy_action_length_normalize=True
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16
actor_rollout_ref.actor.ppo_mini_batch_size=64
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8
ray_init.local_fs_capacity_threshold=0.999
trainer.critic_warmup=0
trainer.resume_mode=disable
trainer.save_freq=5
trainer.actor_lora_only_checkpoint=True
trainer.test_freq=20
trainer.max_actor_ckpt_to_keep=2
trainer.max_critic_ckpt_to_keep=1
trainer.default_local_dir=training_checkpoints/verl_agent_caged_craftext_debug_square_8x8_shared_av_gt_g099_cmae
trainer.actor_value_online_reward_wm.enable=False
trainer.actor_value_online_plan_q_wm.enable=False
trainer.n_gpus_per_node=2
trainer.nnodes=1
```

### 4.6 16×16 CMAE recent log — fresh resolved overrides (83dc44c0, 18:55)

То же самое для 16×16 (полезно как шаблон для будущих canonical 16×16 NO-reasoning
CMAE запусков — но **нужно убрать `+env.enable_reasoning=True`**).

Отличия от 8×8 (1f29addc):
- `craftext_settings=debug_square_16x16` вместо `debug_square_8x8`
- `value_return_min=-26, max=16, bin_step=1.5` вместо `-5/6/0.4`
- `actor_value_return_min=-26, max=16, bin_step=1.5`
- `+env.enable_reasoning=True` (НЕПРАВИЛЬНО для canonical — должно быть False)
- vLLM setup может отличаться (для 16×16 нужен реальный тест)
- ВСЁ остальное — то же

---

## 5. Invalid / diagnostic runs

Не использовать для итогов и не считать baseline:

| Comet key | id | env | status | причина |
|-----------|-----|-----|--------|---------|
| `56dd190a05354ec6b78f1bbc4bad4349` | 16×16 CMAE attempt 1 | 16×16 | INVALID_SETUP | `validate_socket_filename failed: AF_UNIX path length >107` — `ray_temp_dir=/mnt/aicenter1-datasets/...` слишком длинный для unix socket |
| `40b2cdb9861541d899389d441ee24234` | 16×16 CMAE attempt 4 | 16×16 | INVALID_SETUP | Тот же AF_UNIX issue; не дошёл до training |
| `1f29addc63504d96b78f7fc5fe3c5611` | ds8_shared_av_gt_g099_cmae_13 | 8×8 | INVALID_SETUP | **Тот же** AF_UNIX issue с `ray_temp_dir=/mnt/aicenter1-datasets/gorbov_gv/ray_ds8_cmae`. Конфиг правильный (`enable_reasoning=False`, CMAE rho=0.2), но run не запустился. **Конфиг пригоден как шаблон** (см. §4.5) |
| (5e4492cda67d…) | 16×16 shared AV CE pre-4d803c6e | 16×16 | dead (NCCL hang) | superseded |
| (19afee57ab7b…) | 16×16 shared AV CE pre-4d803c6e | 16×16 | dead (vllm_mem OOM) | superseded |
| `ds8_gpu3_4_launch_v9_230928` | local log only | 8×8 | INVALID_SETUP | Hydra override syntax error (`env.use_optimistic_parallel` без `+`) |
| `ds8_gpu2_5_launch_v9_.log` | local log only | 8×8 | INVALID_SETUP | `python: command not found` — venv issue |
| `535900d8711a44a0b3a37ab55458b48f` | ds8_shared_av_gt_g099_a3 | 8×8 | failed (disk_full) | **НЕ INVALID** — это валидный запуск, упавший из-за инфраструктуры (aicenter3 `/tmp/ray` 99% full). Но checkpoint'ы НЕ сохранились (stages [40,110], умер на step 28). См. §3.2 |

### 5.1 Runs со спорным статусом

- `6e817a4776b940b49544c68618e2e44a` (16×16 CMAE): формально прошёл 3 PPO update,
  но `p_target_mass` и `clipped_mae/unclipped_mae` не доходили до logger
  (потеря `_actor_value_diag`). **Не** использовать как итоговый CMAE baseline.
  Статус: STOPPED (diagnostics logging bug).
- `83dc44c02e92498093f6ac324d7a7493` (16×16 CMAE post-fix): прошёл ~25 PPO updates,
  diagnostics появились на шаге 1, но run **CRASHED** на шаге 26
  (`ActorDiedError` → `Worker exit type SYSTEM_ERROR` → `SIGTERM`
  в `compute_actor_token_values`). Checkpoint'ы НЕ сохранены (stages [40,110]).

---

## 6. Current hypotheses

### 6.1 H9 — 16×16 проблема существует независимо от G_t vs r_t

Статус: **OPEN / partially supported**.

Доказательства:
- `da63a962` (16×16 dual, `r_t`, history=50, ~351 PPO updates, no G_t): success_max 0.125, last ~0.0. Не learning.
- `2a42e153` (8×8 dual, `G_t`, history=50, ~79 PPO updates): success_max 0.12.
- `aa0714bb` (16×16 shared AV, CE): success 0 за 28 шагов.

`da63a962` — самый сильный аргумент, что **G_t не виноват** в 16×16 провале:
даже с `r_t` (история 50) на 16×16 нет learning.

**Гипотеза:** partial observability + GAE/credit assignment плохо работают
на больших картах с history=50. Возможный фикс — actor-value shared head
с более узким value target (CMAE).

### 6.2 H11/H12 — CMAE vs CE для shared actor-value

Статус: **OPEN / under experiment**.

CE loss:
```
L_CE = -Σ_j y_j · log p_j
```

CMAE loss (для two-hot):
```
L_MAE = 0.5 · Σ_j |p_j - y_j|
p_target_mass = p_left + p_right
mask = 1[p_target_mass ≤ 0.2]
L_CMAE = mask · L_MAE
```

Гипотеза:
```
partial observability
  → ambiguous value targets
  → low p_target
  → strong CE gradients
  → interference with shared actor
```

CMAE может уменьшить это давление (mask гасит вклад low-confidence bins).

Текущее состояние: **не доказано**. У `83dc44c0` 90.5% шагов имели
`p_target_mass ≤ 0.2` (значит маска CMAE активна почти всегда — это
говорит о том, что value target действительно low-confidence, что
согласуется с гипотезой H11). Но run не дожил до проверки эффекта.

### 6.3 H1, H2 — discounted G_t

См. [`experiments/hypotheses.yaml`](experiments/hypotheses.yaml). Не разбирается здесь.

---

## 7. Checkpoint policy

Для обоих canonical runs:

**Сохранять:**
- shared actor/model LoRA weights (`actor_lora_only_checkpoint=True`)

**Не сохранять:**
- optimizer state
- scheduler
- separate critic (`critic.lora_rank=0`)
- full trainer state

**Стратегия:** rolling
- `trainer.save_freq=5`
- `trainer.max_actor_ckpt_to_keep=2`
- `trainer.test_freq=20`

**Рекомендуемые точки сохранения (для новых runs):**
- step 15–20 (early)
- step ~middle (mid)
- step late

**Проблема из прошлого:**
- `aa0714bb` и `535900d8` планировали stages `[40, 110]`, run умер на
  step ~28 — **никаких checkpoint'ов не сохранено**. Это потерянная работа.
- Решение: rolling save_freq=5 + keep=2 гарантирует, что даже при crash
  на шаге 30 у нас есть минимум 1-2 checkpoint'а для inspection.

---

## 8. Next launch checklist

### 8.1 Before launch

- [ ] `nvidia-smi` — проверить владельцев каждой GPU, **не занимать чужие**
- [ ] Подтвердить `env.craftext_settings` (8×8 vs 16×16)
- [ ] Подтвердить `algorithm.use_actor_value_token=True`
- [ ] Подтвердить `actor.actor_value_token=True`
- [ ] Подтвердить `algorithm.gae_by_trajectory=False`
- [ ] Подтвердить `actor_value_loss_type=clipped_mae`, `clipped_mae_rho=0.2`
- [ ] Подтвердить `actor_value_target_encoding=two_hot`, `target_from_returns=True`
- [ ] Подтвердить `+env.value_prompt_template_type=single_token_return`
- [ ] Подтвердить `value_return_min/max/bin_step` соответствует карте:
  - 8×8: `[-5, 6] / 0.4`
  - 16×16: `[-26, 16] / 1.5`
- [ ] Подтвердить `critic.lora_rank=0` (отдельный critic заморожен)
- [ ] Подтвердить `env.enable_reasoning=False` (или флаг НЕ передаётся)
- [ ] Подтвердить `prompt_template_type=single_token_action`, `single_token_actions=True`
- [ ] Подтвердить `data.max_response_length=1`
- [ ] **`+env.use_optimistic_parallel=True`**
- [ ] **`ray_temp_dir` НЕ на `/mnt/aicenter1-datasets/`** — из-за AF_UNIX 107-byte limit
- [ ] **`data.max_prompt_length` MEASURED** на каноническом NO-reasoning конфиге (НЕ брать 2048/1024 «из v9» — v9 был с reasoning=ON и нерелевантен)
- [ ] `train_batch_size` — match validated shared-AV baseline (например `64` для 535900d8/1f29addc)
- [ ] `trainer.save_freq=5`, `max_actor_ckpt_to_keep=2`, `actor_lora_only_checkpoint=True`
- [ ] `trainer.test_freq=20`, `val_before_train=True`
- [ ] `trainer.logger=[console, comet]` (Comet ON — стандарт)
- [ ] `trainer.critic_warmup=0`
- [ ] `actor_value_online_reward_wm.enable=False`, `actor_value_online_plan_q_wm.enable=False`
- [ ] Записать `trainer.experiment_name`, `local_log_path`, `checkpoint_dir`
- [ ] Добавить запись в [`experiments/registry.yaml`](experiments/registry.yaml) со статусом `planned`
- [ ] Записать `git_commit` запускаемой версии кода

### 8.2 After first PPO update (sanity check)

- [ ] `success_rate` залогирован и не отрицательный
- [ ] `value_token_loss` залогирован и падает / стабилен
- [ ] `value_token_accuracy` залогирован
- [ ] `p_target_mass` залогирован (CMAE diagnostic)
- [ ] `clipped_mae` / `unclipped_mae` залогированы (CMAE diagnostic)
- [ ] `clipped_fraction` залогирован
- [ ] `value_grad_norm` залогирован и finite
- [ ] GPU memory stable (без OOM)
- [ ] `/tmp/ray` НЕ заполнен (защита от disk_full)
- [ ] Comet experiment URL доступен

---

## 9. Useful links

### Master docs
- [`safe_rl_nlp.md`](safe_rl_nlp.md) — основной документ проекта
- [`experiments/registry.yaml`](experiments/registry.yaml) — machine-readable source of truth
- [`experiments/hypotheses.yaml`](experiments/hypotheses.yaml) — гипотезы

### Local logs (this / previous session)

- 8×8 shared AV CE aicenter3: `logdir/ppo_debug_square_shared_av_gt_g099_a3_20260913_173619.log`
- 16×16 shared AV CE relaunch: `logdir/ppo_debug_square_16x16_shared_av_gt_g099_ce_20260913_173647.log`
- 16×16 CMAE attempts: `logdir/ppo_debug_square_16x16_shared_av_gt_g099_cmae_*.log`
- 8×8 CMAE attempt (INVALID_SETUP, но правильный NO-reasoning конфиг):
  `logdir/ppo_debug_square_8x8_shared_av_gt_g099_cmae_20260914_195710.log`
- Recent v9/v10 retries (reasoning=ON — **не** валидный NO-reasoning):
  `logdir/ds8_gpu3_4_launch_v9_*.log`, `logdir/ds8_gpu3_4_launch_v10_*.log`

### Comet experiments

- 8×8 dual `r_t` history=0 (FINAL repro): https://www.comet.com/gregory-gorbov/verl-agent-caged-craftext/02e2744ddcdb4573ad263487a710e956
- 8×8 dual `r_t` history=50: https://www.comet.com/gregory-gorbov/verl-agent-caged-craftext/874bc480b5224efb9b92795d271ca81c
- 8×8 dual `G_t` history=50 (repeat): https://www.comet.com/gregory-gorbov/verl-agent-caged-craftext/2a42e153bdb0448789a56140e70fdf50
- 8×8 shared AV CE (`535900d8`): https://www.comet.com/gregory-gorbov/verl-agent-caged-craftext/535900d8711a44a0b3a37ab55458b48f
- 16×16 dual `r_t` long (`da63a962`): https://www.comet.com/gregory-gorbov/verl-agent-caged-craftext/da63a962175b4f1a82606c3fc0fe82d8
- 16×16 shared AV CE relaunch (`aa0714bb`): https://www.comet.com/gregory-gorbov/verl-agent-caged-craftext/aa0714bbaaca4cdbbd2536551f065fe7
- 16×16 CMAE post-fix (`83dc44c0`): https://www.comet.com/gregory-gorbov/verl-agent-caged-craftext/83dc44c02e92498093f6ac324d7a7493
- 16×16 CMAE diagnostic-loss run (`6e817a47`): https://www.comet.com/gregory-gorbov/verl-agent-caged-craftext/6e817a4776b940b49544c68618e2e44a
- 8×8 CMAE INVALID_SETUP, но правильный NO-reasoning шаблон (`1f29addc`):
  https://www.comet.com/gregory-gorbov/verl-agent-caged-craftext/1f29addc63504d96b78f7fc5fe3c5611

### Hardware
- aicenter3: GPU 0,1,2,3,4,5 (80 GB каждый), GPU 6,7 (статус уточнить)

---

## 10. TBD — что нужно закрыть перед запуском

1. **`train_batch_size`** для canonical 8×8 и 16×16. Не выбирать 8 «потому что так было в v10». Совпадает с validated shared-AV baseline (`64` для 535900d8 и 1f29addc) — использовать это значение, если нет причины менять.
2. **`max_prompt_length`** — измерить на каноническом NO-reasoning конфиге на каждой карте. **Не** использовать 2048 «из v9» (v9 был с reasoning=ON).
3. **GPU allocation** — какие конкретно GPU на aicenter3 свободны прямо сейчас.
4. **`ray_temp_dir`** — корректное место, НЕ `/mnt/aicenter1-datasets/` (AF_UNIX limit).
5. **`trainer.experiment_name`** — уникальное имя для нового 8×8 и 16×16 CMAE runs.
6. **git_commit** — текущий dirty-tree vs release commit. Зафиксировать.
