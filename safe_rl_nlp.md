# safe_rl_nlp — Project Master Document

**Read this file first.** It is the single human-readable research context for AI agents

> **2026-09-16 research update.** Sections 17--19 below specify the planned
> checkpoint-based grounding protocol and the 16x16 reasoning study. All newly
> listed experiments are **PLANNED / NOT RUN**; SHARED + reasoning is **BLOCKED /
> NOT RUN**.

## 17. Grounding and 16x16 study plan (PLANNED / NOT RUN)

**Grounding question.** Does joint actor-value training improve the actor's
understanding of the environment relative to standard DUAL PPO? We will answer
this by direct LLM QA from frozen PPO checkpoints--not a linear probe, extra
classifier, or post-PPO training. The primary 8x8 matrix is DUAL/no-reasoning,
DUAL/reasoning, SHARED/no-reasoning, and SHARED/reasoning. CMAE is not a factor in
this comparison; it remains a separate 16x16 hypothesis.

### Grounding evaluator

`scripts/evaluate_debug_square_grounding.py` accepts an arbitrary saved full or
LoRA checkpoint, never creates an optimizer or updates weights, and writes
`per_question.csv` plus `metrics.json`. It builds deterministic held-out states
with the production `generate_debug_square_world`, uses the production
CagedCraftext ASCII renderer and actor no-history representation for reset states,
and performs constrained candidate scoring. Every answer is therefore one of the
predefined permitted options. It is usable for DUAL actor checkpoints now and is
structured to accept SHARED checkpoints later.

The production 8x8/16x16 maps have one each of **STONE, WOOD, WATER** at three
inner corners, shuffled every reset. `TREE` is a fixed solid border and `GRASS` is
floor; neither is an object-localization target. The nine question families are:

1. target direction (`RIGHT_UP`, `RIGHT_DOWN`, `LEFT_UP`, `LEFT_DOWN`);
2. target shortest-path distance (0--30; Manhattan is exact in the unobstructed
   interior);
3. UP consequence;
4. DOWN consequence;
5. LEFT consequence;
6. RIGHT consequence (each `CLOSER`, `FARTHER`, or `SAME`);
7. STONE direction;
8. WOOD direction;
9. WATER direction.

States with aligned row/column are excluded from quadrant questions. A blocked
tree-border move is `SAME`. Required exact metrics are
`target_direction_accuracy`, `target_distance_accuracy`,
`action_consequence_accuracy`, `object_direction_accuracy`, and their macro mean;
the CSV/JSON retain per-action and per-object results.

For a grounding curve preserve at least checkpoint_0/before-PPO, early, middle,
and final. Saving every five PPO updates is fine only if retention preserves those
selected checkpoints--a rolling “keep last 2” policy alone is insufficient.

### Reasoning semantics and 16x16 hypotheses

With DUAL reasoning, critic input is `[prompt | z1 ... zn | action]`; causal values
are `V(z1)=V(prompt)`, `V(z2)=V(prompt,z1)`, ..., and
`V(action)=V(prompt,z1,...,zn)`. Thus token-level PPO/GAE sees reasoning prefixes;
there is no separate text value prompt for the DUAL critic.

The current SHARED value-token method makes one return-bin prediction from an env
state prompt and has no token-wise value semantics for reasoning prefixes. Hence
**SHARED + reasoning is BLOCKED / NOT RUN** until that design is resolved. The
future target is action history 50 plus only the last five reasoning responses
(without duplicating their final action), max response length 64, 8x8
`max_prompt_length=1536`, and 16x16 target `max_prompt_length=4096`, all with
measured clipping diagnostics.

The 16x16 target is often initially invisible, unlike 8x8: it requires exploration
and memory. First test whether ordinary DUAL PPO learns 16x16 with and without
reasoning. Shared CE interference on ambiguous states and CMAE rho=0.2 are
secondary hypotheses; defer Q-learning until this baseline issue is understood.
working on this project. It explains the research goal, stable findings, the experiment
map, and the current state.

- **Machine-readable source of truth for run metadata:** `experiments/registry.yaml`
  (+ `experiments/hypotheses.yaml`, `experiments/meta.yaml`). This document does NOT
  replace the registry and must NOT become a raw log dump.
- **Separation of concerns:** stable knowledge, current state, open hypotheses, and
  next experiments are in separate sections below.
- Update this document only on *meaningful* events (new significant experiment,
  hypothesis confirmed/refuted, architecture or roadmap change, important failure
  mode) — never per training step. Record such updates in the Changelog.

Canonical repo paths on hosts (all at `47d9d7a`, branch `safe`):
- **aicenter3:** `/home/gorbov_gv/safe_rl_nlp`
- **aicenteritl:** `/home/gorbov_gv/repos/verl-agent-craftext`
- **aicenter1:** `/home/gorbov_gv/repos/verl-agent-craftext` (NFS server only)
- **aicenter2:** `/mnt/home/gorbov_gv/repos/verl-agent-craftext`

GitHub: `github.com/Gricha1/verl-agent-craftext`

---

## 1. Project goal

Test whether **one and the same LLM** can simultaneously act as:

1. a **policy actor**, and
2. a **task-oriented implicit world model**.

We deliberately do NOT do classical reconstruction-style world modelling
(`state/action -> next pixels` or `-> full next observation`). The approach is closer
to TD-MPC value learning:

- `state -> V_env`
- `state + action -> Q_env`
- `state + action sequence -> plan-Q`
- optionally success/done prediction
- reward prediction is secondary

Scientific claim to establish: the LLM **internalizes task/world structure** relevant
to decision making, rather than merely fitting the final reward.

## 2. Core hypothesis

Value/Q supervision in the *same* LLM (via token prediction on the LM head) grounds
the model in the environment and may improve sample efficiency — provided that
(a) credit assignment is correct for multi-turn embodied settings, and
(b) gradient interference between policy and value objectives is controlled.

### Research questions

- **RQ1.** Can a shared LLM actor+value train as well as ordinary PPO with a separate critic?
- **RQ2.** Does task-oriented value/Q supervision ground the LLM in the environment?
- **RQ3.** Does it improve sample efficiency?
- **RQ4.** How to do credit assignment in a multi-turn embodied environment where
  one LLM response ≠ a whole environment episode?
- **RQ5.** How does partial observability affect value-token supervision?
- **RQ6.** Does gradient interference arise between policy and value objectives?
- **RQ7.** Can MAE/CMAE make supervision more robust when value targets are
  partially unpredictable?
- **RQ8.** Does Q / multi-horizon Q help more than immediate reward prediction?

## 3. Proposed architecture

**Same LLM, same LM head, different prompts / token targets** — value and Q do NOT
require a separate neural head in the main proposed architecture:

| Function | Mapping |
|---|---|
| Policy | state/history -> action token(s) |
| Value | state/history -> value-bin token(s) |
| Q | state/history + candidate action -> Q-bin token(s) |
| Plan-Q | state/history + candidate action sequence -> plan-value token(s) |

Implementation: `verl/utils/actor_value_token.py` — dual-prompt actor-value PPO
(actor prompt -> action token; value prompt -> return-bin token), constrained
decoding over return-bin tokens, two-hot encoding, value loss (CE; MAE exists on
hosts — see §6). Bin grids and return tokens: `agent_system/environments/env_package/caged_craftext/return_tokens.py`.

### Two architecture variants

- **Dual PPO baseline** — actor LLM + separate critic LLM. Less gradient
  interference, but two models. This is the reference baseline, NOT the proposal.
- **Shared / single actor-value** — ONE LLM; actor and value are prompt-separated
  token tasks. This is the main research architecture.

## 4. Environments

### debug_square 8x8 (CrafText/Crafter-like navigation)

Agent starts near the center; on 8x8 the target is usually visible in the
observation window. Dense potential-shaped reward:

```
r_t = (d_t - d_{t+1}) + 1_success      # d = Manhattan distance
```

closer -> +1, farther -> -1, unchanged -> 0, success -> extra bonus.

### debug_square 16x16

Same task, but the target is often **outside the local observation window** —
the task becomes partially observable. Critical for value learning: identical
observable prompts can correspond to different hidden world states and different
future returns.

### sparse debug_square

Success-only reward. A control close to ALFWorld in credit-assignment structure
(`r = [0..0, 1]` → on success `G_t == R`).

### ALFWorld

Text environment, mostly sparse reward (0...0, 1 on success). At gamma=1,
`full R ≈ G_t` on a successful trajectory. Natural-language semantic prior,
admissible/constrained actions. Registry entry: `alfworld_import_pending`
(Comet project `gregory-gorbov/verl-agent-alfworld`, historical import pending).

### GSM8K

One env step ≈ one full generated math solution; sparse final correctness reward.
A reasoning-domain test of shared actor/value — but weaker evidence for
physical/world grounding. Credit assignment is simpler than in embodied CrafText.

## 5. Credit assignment (multi-turn problem)

In CrafText each env step produces one LLM response, often `response_len = 1`
action token. Vanilla VERL response-level GAE with `response_len=1` does not see
the whole env trajectory. Distinguish **response-level return** vs **env-level return**.

Variants used:

- **Immediate reward:** `r_t`
- **Undiscounted remaining return:** `G_t = r_t + r_{t+1} + ...`
- **Discounted remaining return:** `G_t^γ = r_t + γ r_{t+1} + γ² r_{t+2} + ...`
  (currently γ = 0.99; implemented via `compute_remaining_return_scalars` in
  `verl/utils/actor_value_token.py`, tested by `tests/test_remaining_returns_gamma.py`)
- **Trajectory GAE:** separate env-level GAE implementation.

**Never** combine an already-computed `G_t` with trajectory GAE — future reward
gets double-counted (this exact bug caused returns explosion in `ds16_av_3f6f1518`).

### Why undiscounted G_t fails on debug_square (dense)

With `r_t = d_t - d_{t+1} + bonus` and γ=1 the sum telescopes:

```
G_t = d_t - d_final + bonus
```

Local action credit is lost: a bad action can be fully compensated later
(`-1 + 1 = 0`). With γ=0.99 telescoping is broken (`-1 + 0.99·(+1) = -0.01`),
so bad actions leave a negative trace. This is hypothesis **H1** (strong evidence).

## 6. Value/Q formulation

Value targets are encoded as **bins** with **two-hot** encoding (main variant;
one-hot also supported). Loss: categorical cross-entropy. For a target between
adjacent bins:

```
y_i = α,  y_{i+1} = 1-α
L_CE = -α·log p_i - (1-α)·log p_{i+1}
```

Bins must cover the actual return range. Always log: target min/max, p01/p99,
edge-bin fraction, clipping fraction.

### CE vs MAE hypothesis (OPEN — not a proven fact)

Motivated by PaW-like robust world-model losses. For one-hot targets:
`CE: L = -log p_y` vs `MAE: L = 1 - p_y`. When `p_y << 1`, CE gradient is large
while MAE gradient is ~`p_y` — much smaller. Two-hot generalization:
`L_MAE = 0.5 · Σ_j |p_j - y_j|`.

Hypothesized chain on 16x16: partial observability → ambiguous return targets →
low target probability → strong CE gradients → shared actor/value gradient
interference. MAE may reduce the harm — but it does NOT recover missing
information and may ignore hard-but-useful examples.

**Matched experiment needed (same everything else):**
`16x16 shared AV + CE` vs `16x16 shared AV + MAE`.

MAE loss status: unit tests exist locally (`tests/test_categorical_mae_value_loss.py`
tests `categorical_mae_loss` and `value_loss_type="mae"`), but the local copy of
`verl/utils/actor_value_token.py` does NOT contain them — the MAE implementation
lives in the newer version on the hosts. A GSM8K MAE launch script exists:
`examples/ppo_trainer/ppo_gsm8k_actor_value_mae_lora64.sh`.

## 7. Key experimental results (verified against registry / metrics caches)

All CraftExt runs are in Comet project `gregory-gorbov/verl-agent-caged-craftext`.
Model: Qwen2.5-family (1.5B-Instruct for shared AV), LoRA r64 unless noted.

### 7.1 debug_square 8x8, dual

| Run | Key | Setup | Result |
|---|---|---|---|
| `ds8_dual_rt_h0_02e2744` | `02e2744ddcdb4573ad263487a710e956` | r_t, hist=0 | **success ~0.99–1.0** — current baseline/best |
| `ds8_dual_rt_h50_874bc480` | `874bc480b5224efb9b92795d271ca81c` | r_t, hist=50 | success ~0.96 — **history=50 does NOT break learning** |
| `ds8_dual_gt_h50_1a064d01` | `1a064d01c9ae4e8596a54d06d78f674d` | undisc. G_t, hist=50 | failed |
| `ds8_dual_gt_h50_2a42e153` | `2a42e153bdb0448789a56140e70fdf50` | undisc. G_t (repeat) | success ≤ ~0.12 by step 79; returns ∈ [-8,8]; vf_explained ≤ 0 — **undiscounted dense G_t ≪ r_t** |
| `ds8_dual_rt_trajgae_h50_c3a5e727` | `c3a5e727c5414fa3a4e5ff57657351b1` | r_t + env traj-GAE | inconclusive: better critic/VF metrics, success_max 0.06 — **better value fit ≠ automatically better policy** |

Historical references: `ds8_dual_final_983925d3` (`983925d3...`), `ds8_dual_valent_680dfbf1` (valid-action entropy), `ds8_av_4be19de5` (early actor-value line, inconclusive).

### 7.2 debug_square 16x16

| Run | Key | Setup | Result |
|---|---|---|---|
| `ds16_dual_rt_h50_b126b453` | `b126b453...` | dual r_t, 2 GPU | early OOM (vLLM kv_cache, step ~2–4) |
| `ds16_dual_rt_h50_da63a962` | `da63a962...` | dual r_t, 1 GPU batch32 | failed: to step ~351, success_max 0.125, no learning — working 8x8 recipe does NOT transfer to 16x16; partial observability/exploration suspected (H9) |
| `ds16_av_3f6f1518` | `3f6f1518...` | AV, trajGAE over G_t | **double-count bug** → returns exploded → OOM |
| `ds16_av_1ac1a602` | `1ac1a602...` | AV, gae=False | failed: targets OK after fix, policy weak |
| `ds16_av_14831ad0` | `14831ad0...` | AV, hist=0 | failed: success 0 |

### 7.3 Discounted G_t line (dual 8x8, γ=0.99)

Metrics cache `experiments/metrics_cache/gt_g099_compare.json` compares
r_t (`874bc480`) vs undiscounted G_t (`2a42e153`) vs discounted G_t γ=0.99
(`4e293f9890f14822a3ef5cf912b04671`, name "ds8_dual_gt_g099_h50"). At scrape time
the γ=0.99 run's metrics were all null — **no verified numbers locally**. Dead
related keys: `c7dcda4e...` (TOKEN_CLS), `979bc517...` (qwen_vl, timeout).
The γ=0.99 curve was reported (chat context) to look better than undiscounted G_t,
but the run may have been stopped early — **not a confirmed conclusion**.
Registry keys `90deefed`/`aa4f5c97` appear only in `_exp_scripts/inspect_dual_and_prep.sh`
(inspecting log `logdir/ppo_debug_square_gt_g099_20260911_070145.log`); they are
**not registered** in registry.yaml. `90deefed8ddc458da4f27349a14bf4f4` is
confirmed by the user as a real Comet experiment (the dual G_t γ=0.99 run) and
appears in the shared compare view together with `535900d8`, `874bc480`,
`02e2744d`.

### 7.4 Shared actor-value, discounted G_t γ=0.99

- **8x8, aicenter3** — `ds8_shared_av_gt_g099_h50_a3_535900d8`, Comet key
  **`535900d8711a44a0b3a37ab55458b48f`**, hist=50, LoRA r64, bins [-5,6]/0.4,
  two-hot CE, phys GPU 0+6 (NCCL P2P off). Reached global_step ~28 / ~85.8k env
  steps; success peaked **0.077** (0.016 at step 27), value_token_loss 1.633,
  value_token_accuracy 0.367. **Crashed 2026-09-13 ~23:53: OSError
  "No space left on device"** (aicenter3 root disk full). An earlier 8x8 shared
  attempt (2026-09-11) also died of disk-full at ~step 27; its reported key
  `96b48ca1...` remains unverified. A parallel aicenter2 docker attempt never
  trained (`ModuleNotFoundError: gym` in the container).
- **16x16 CE baseline, aicenter3** — first run `4d803c6e...` died early
  (2026-09-12; reported CUDA OOM from a foreign GPU process). **Relaunch
  `ds16_shared_av_gt_g099_ce_h50_aa0714bb`, Comet key `aa0714bbaaca4cdbbd2536551f065fe7`**:
  reached global_step 28 / 89600 env steps, success 0.0, value CE loss 4.7→1.225,
  value accuracy 0.007→0.634 — then **hung in the UPDATE phase after the same
  2026-09-13 disk-full incident** (`/tmp/ray` 99% full). Both runs too short to
  judge the method.

### 7.5 16x16 clipped-MAE (CMAE) line — CE-vs-MAE matched experiment

- **CMAE loss implemented** on top of the existing categorical MAE
  (`verl/utils/actor_value_token.py`): `p_target_mass = p_left + p_right`
  (two-hot), mask = `1[p_target_mass <= rho]`, `L_CMAE = mask · 0.5·Σ|p_j − y_j|`,
  **rho = 0.2** (PaW confidence threshold; their sensitivity best). Loss type
  `clipped_mae`; unit tests in `tests/test_categorical_mae_value_loss.py`
  (one-hot/two-hot MAE, active/zero-loss sides of rho, finite backward, CE path
  intact).
- **Logging bugfix (2026-09-14):** `verl/workers/actor/dp_actor.py` — the
  `_actor_value_diag` metrics dict (p_target_mass stats, clipped/unclipped MAE)
  was silently dropped when the entropy term created a new loss tensor; now
  propagated. Diag keys: `actor_value/p_target_mass/*`,
  `actor_value/p_target_mass_le_0.2`, `actor_value/clipped_high_conf_fraction`,
  `actor_value/unclipped_mae`, `actor_value/clipped_mae`.
- **GPU-mapping pitfall on aicenter3 (2026-09-14):** physical GPU2 is broken and
  **skipped by CUDA enumeration**, shifting all later CUDA indices by −1:
  `CUDA_VISIBLE_DEVICES=4,5` lands on **phys 5+6**. UUID-based CVD is NOT usable
  (verl does `int(gpu_id)` on Ray GPU ids). Fix in
  `_exp_scripts/a3_launch_ds16_cmae_45.sh`: `CVD=3,4` (= phys 4,5) + a torch
  PCI-bus preflight guard (expects 0x81/0xA1, aborts if enumeration changed).
- **Dead keys this line (never trained, do not cite):** `40b2cdb9...`
  (UUID-CVD crash), `6e817a47...` (stopped at step ~3 — diagnostic logging bug;
  `_actor_value_diag` was lost when the entropy term created a new loss tensor).
- **Most recent CMAE attempt:** `ds16_shared_av_gt_g099_cmae_h50`, Comet key
  **`83dc44c02e92498093f6ac324d7a7493`**, launched 2026-09-14 ~18:55 UTC on
  **phys GPU 4+5** (contexts verified by UUID), Ray temp on NFS
  (`/mnt/aicenter1-datasets/gorbov_gv/ray`, spilling threshold 0.999).
  Diagnostics DID appear at step 1 (`p_target_mass_le_0.2:0.905`,
  `clipped_mae:0.861`, `p_left/mean:0.017`, etc.), confirming the logging fix.
  **CRASHED** at step ~26 (`ActorDiedError` → `Worker exit type SYSTEM_ERROR`
  → `SIGTERM` in `compute_actor_token_values`). Checkpoint stages `[40,110]` —
  **no checkpoints saved**. Compare pair candidate: CE=`aa0714bb` vs
  CMAE=`83dc44c0` (both too short to judge).
- **8×8 CMAE first attempt:** `ds8_shared_av_gt_g099_cmae_h50`, Comet key
  **`1f29addc63504d96b78f7fc5fe3c5611`**, launched 2026-09-14 ~19:58 UTC.
  **INVALID_SETUP** — `validate_socket_filename failed: AF_UNIX path length
  >107 bytes`, caused by `ray_temp_dir=/mnt/aicenter1-datasets/gorbov_gv/ray_ds8_cmae`.
  Resolved overrides (`enable_reasoning=False`, CMAE rho=0.2, single-token-action)
  **are correct** — this log is the right template for a future 8×8 NO-reasoning
  CMAE launch, but `ray_temp_dir` must NOT be on `/mnt/aicenter1-datasets/`.

**Note (2026-09-14):** All 16×16 CMAE attempts above had `+env.enable_reasoning=True`
(default). For the canonical 16×16 NO-reasoning CMAE run, explicitly set
`+env.enable_reasoning=False`. See [`safe_rl_nlp_experiments.md`](safe_rl_nlp_experiments.md) §1.2.

### 7.6 GSM8K (shared actor/value in a reasoning domain)

GSM8K runs are NOT in registry.yaml; numbers below are from
`experiments/metrics_cache/gsm8k_sample_efficiency.json` and
`gsm8k_compare_value_diag.json`:

| Key | Setup | AUC | step→0.5 | Notes |
|---|---|---|---|---|
| `0a0271de2c93...` | dual PPO, one-hot, LoRA128 | 0.472 | 62 | baseline |
| `d5b5b7b9d70a...` | shared AV, two-hot, plan_q, separate opt | 0.544 | 37 | best matched pair vs `0a02` (still confounded) |
| `d048907ba952...` | shared, full FT, MC-Q WM | 0.665 | 17 | |
| `b7573e3eacd9...` | dual PPO, one-hot, LoRA64 | 0.322 | 87 | weak dual variant |
| `98b6c4aa71a4...` | shared, full FT, av+reward_wm | **0.700** | 17 | |

Cached verdict: **`causal_claim_shared_worse_than_dual = FALSE`** — every pair is
confounded (plan_q, two_hot, lora_rank, full-vs-LoRA, mc_q). Value-token accuracy
in shared runs grows (e.g. `d5b5`: 0.13 → ~0.68; loss 2.83 → 0.55). Note: for
two-hot, `exp(-CE) ≠ p_target`; the historical CE→probability proxy is invalid.
GSM8K MAE value-loss experiment: script exists (`ppo_gsm8k_actor_value_mae_lora64.sh`,
host aicenter3); **no verified results locally** — check Comet.

## 8. Current hypotheses

Registry of record: `experiments/hypotheses.yaml` (H1–H9). Summary:

| # | Hypothesis | Status | Evidence | Next test |
|---|---|---|---|---|
| H1 | Dense G_t (γ=1) telescoping destroys local action credit | **strong evidence (active)** | r_t runs succeed, G_t runs fail (7.1) | sparse control, γ=0.99/0.95, n-step |
| H2 | G_t scale/variance is hard for the critic | high / active | returns [-8,8], vf_explained ≤ 0 on G_t | γ=0.99, n-step |
| H3 | ALFWorld sparse R/G_t ≠ dense square G_t | high / active | semantics differ | sparse square control, ALFWorld import |
| H4 | MC G_t noisy due to stochastic future policy | medium / open | — | n-step 3/5 |
| H5 | Poor critic fit of G_t → noisy advantages | medium / open | G_t runs; counter: traj-GAE c3a5e727 | γ=0.99 |
| H6 | Implementation bug in G_t | low / largely ruled out | grouping/auto_reset checked | — |
| H7 | history=50 breaks policy | **EXCLUDED** | r_t+hist50 → 0.96 | — |
| H8 | Bin range explains dual G_t failures | **EXCLUDED** | dual critic uses no bins | — |
| H9 | 16x16 failure dominated by partial observability | medium / open | da63a962 | obs/curriculum fixes |
| H10 | Shared actor-value suffers gradient interference (RQ6) | open | — | matched dual vs shared |
| H11 | CE gradients on low-probability bins destabilize shared training | open | — | 16x16 CE vs MAE |
| H12 | MAE stabilizes shared AV under partial observability | experimental hypothesis | — | 16x16 CE vs MAE |
| H13 | Value-bin range/clipping can hurt | must always diagnose | — | bin diagnostics every run |
| H14 | Q/plan-Q gives better task-oriented WM supervision than reward | planned | — | Q, multi-horizon Q |

(H10–H14 are formulated here for the master doc; add them to
`experiments/hypotheses.yaml` when their first experiments are registered.)

## 9. Current experiment state (as of 2026-09-15, ~13:30 UTC)

- **Registry last updated:** 2026-09-14; local repo copy may lag the hosts.
- **NO RUNNING TRAINING** on aicenter3 today. Last activity: v9/v10 retries on
  GPU 3+4 (reasoning=ON) crashed 2026-09-14 23:55 with
  `NotImplementedError: sequence_length=6514 > max_length=6144`.
- **Immediate experimental goal** (canonical, documented in
  [`safe_rl_nlp_experiments.md`](safe_rl_nlp_experiments.md) §1):
  **matched 8×8 vs 16×16** under a single shared actor-value architecture
  with the **same** everything else: **NO reasoning**, `G_t^0.99`, two-hot,
  **CMAE rho=0.2**. 8×8 bins `[-5,6]/0.4`, 16×16 bins `[-26,16]/1.5`. Single
  token action, history 50, max_steps 50.
- **Stopped/cleaned by us (2026-09-14):** hung CE run `aa0714bb` (its GPU 4+5
  workers and stale Ray session removed; foreign processes untouched).
- **8x8 shared AV (`535900d8...`)** — still down (crashed 2026-09-13, disk
  full); relaunch pending.
- **Infrastructure lessons:**
  - 2026-09-13: one aicenter3 root-disk-full event killed/stalled BOTH
    shared-AV runs. Ray temp now on NFS (`/mnt/aicenter1-datasets/gorbov_gv/ray`).
  - 2026-09-14: aicenter3 phys GPU2 broken → CUDA indices shift; use CVD=3,4
    for phys 4,5 + PCI preflight guard (see 7.5).
  - 2026-09-14: foreign jobs appeared mid-work on GPU0 (korolkova dynalang)
    and GPU6 (ivanov_la vllm serve, root). Do not touch; re-check `nvidia-smi`
    before every launch.
  - **AF_UNIX 107-byte socket path limit**: `ray_temp_dir` MUST NOT live under
    `/mnt/aicenter1-datasets/` (path too long). Choose a shorter mount.
- **Planned, not launched** (registry): sparse control `ds8_dual_sparse_R_h50_seed1`
  (priority 1), dual γ=0.99/0.95, n-step 3/5, ALFWorld import.
- **Pre-launch checklist** lives in [`safe_rl_nlp_experiments.md`](safe_rl_nlp_experiments.md) §8.

## 10. Infrastructure

Hosts: **aicenteritl** (historical main host, coordination host in `meta.yaml`),
**aicenter2**, **aicenter3** (also seen: aicenter1, ml3/ml4, h200 in fleet cache).
Operational scripts: `_exp_scripts/` (prefixes `a2_`, `a3_`, `itl_` per host).

- **aicenter2:** VM planned for recreation. Persistent paths: `/storage`,
  `/mnt/aicenter1-datasets`. Artifacts must go to `/storage/gorbov_gv/...` —
  **never keep the only copy of checkpoints/logs on the ephemeral VM filesystem**.
  Launches there run in docker containers.
- **aicenter3:** known issues — root disk almost full, `/storage` mount issues,
  foreign GPU processes, one broken GPU (GPU2). Used for 16x16 shared AV CE and
  GSM8K MAE.
- **aicenteritl:** historical main host; experiment dashboard runs here (port 8770).
- **aicenter1:** NFS server only (`/mnt/aicenter1-datasets/`), no training launches.
- **Always `nvidia-smi` before any training launch.**

### 10.1 Concrete launch how-to (verified 2026-09-18)

Full reference: [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md). Summary below.

**Git repository:** All hosts run from clones of `github.com/Gricha1/verl-agent-craftext`,
branch `safe`, commit `47d9d7a` (2026-09-18). Exact paths:
- **aicenter3:** `/home/gorbov_gv/safe_rl_nlp` (reference tree, converted in place)
- **aicenteritl:** `/home/gorbov_gv/repos/verl-agent-craftext`
- **aicenter1:** `/home/gorbov_gv/repos/verl-agent-craftext` (NFS server only, no training)
- **aicenter2:** `/mnt/home/gorbov_gv/repos/verl-agent-craftext` (local disk, not s3fs)

**IMPORTANT:** A2 containers currently mount `/storage/gorbov_gv/safe_rl_nlp` (s3fs, stale
tree without git). To run from the current code, the bind mount must be changed to the
git clone path and the container restarted.

**aicenter2 = Docker only** (no native venv). Image `verlai/verl:vllm011.latest`.
Each launch installs only env-side packages (Gym/JAX/Craftax) at runtime and must
NOT upgrade Torch/Transformers/vLLM/xFormers. Host GPU is selected with
`--gpus '"device=X"'`; **inside the container it is always remapped to GPU 0**
(`-e CUDA_VISIBLE_DEVICES=0`).

```bash
# preflight (no run):
bash _exp_scripts/a2_prepare_ds16_dual_reasoning.sh
# actual launch (single host GPU 3 → container GPU 0):
CONFIRM_LAUNCH=1 bash _exp_scripts/a2_prepare_ds16_dual_reasoning.sh
```

Two matched 16x16 configs live in `examples/ppo_trainer/config/` and are driven by
`run_caged_craftext_lora_job.sh` via 12 positional args. The reasoning variant is
launched through `ppo_debug_square_16x16_reasoning.sh`.

| knob | no-reasoning | reasoning |
|---|---|---|
| `env.enable_reasoning` | False | True |
| `LOG_PROB_ACTION_ONLY` (arg 2) | false | **true** (log-prob only over `<action>` tokens) |
| `NO_REASONING` (arg 3) | true | **false** |
| `max_response_length` (arg 5) | 1 | **128** |
| `PROMPT_TEMPLATE_TYPE` (arg 10) | `single_token_action` | **`default_template`** (reasoning path is ignored under `single_token_action`) |
| `trainer.n_gpus_per_node` | 2 | 1 |
| `VLLM_ATTENTION_BACKEND` | XFORMERS (ok) | **TORCH_SDPA** (XFORMERS hits `Paged KV cache block size must be divisible by 256` at len 128) |

`VLLM_ATTENTION_BACKEND` is overridable in `run_caged_craftext_lora_job.sh`
(`export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}`), committed in 47d9d7a.

**aicenteritl / aicenter3 = native venv.** Same configs and `run_caged_craftext_lora_job.sh`,
but no Docker wrapper, no per-run pip install, and real host GPU indices via
`CUDA_VISIBLE_DEVICES` directly:

```bash
source <venv>/bin/activate
HYPER_YAML=examples/ppo_trainer/config/ppo_debug_square_16x16_dual_reasoning_grounding.yaml \
  bash examples/ppo_trainer/ppo_debug_square_16x16_reasoning.sh \
  actor_rollout_ref.model.path="$MODEL_PATH" critic.model.path="$MODEL_PATH"
```

**Checkpoints:** `trainer.save_freq=5` → every 5 PPO steps, into
`training_checkpoints/verl_agent_caged_craftext_ds16_dual_{no_,}reasoning_grounding/`
(under `/storage/gorbov_gv/...`, never the ephemeral VM FS). Comet ReadTimeout from
Docker is non-fatal (network egress); training continues — set `COMET_DISABLED=1` to silence.

### 10.2 Work only from the git repository (mandatory rule)

**Rule.** On every working host — `aicenteritl`, `aicenter1`, `aicenter2`, `aicenter3` —
the project is used **only from a clone of
`github.com/iameteron/verl-agent-craftext`**. No standalone non-git copy of the project
is ever a training target.

1. **No hand-copied code.** Sync between hosts happens by `git push` + `git pull`/`clone`.
   Transplanting individual files from one host to another is prohibited: it produces
   hybrid trees that match no commit and cannot be reproduced.
2. **Pre-launch verification is mandatory, before any training start** (in the exact
   directory the launch will run from — on aicenter2 that is the container's `/workspace`,
   i.e. the host path it mounts):
   ```bash
   git rev-parse --abbrev-ref HEAD     # branch
   git rev-parse HEAD                  # commit hash
   git status --porcelain              # must be empty for a training launch
   git log -1 --format='%H %ci %s'     # what this commit is
   ```
   If the tree is dirty, resolve the diff first (commit it or stash it with `-u`);
   never launch training from an unspecified dirty state.
3. **Order of operations for any change:** edit inside the repo → verify → commit →
   push → pull/clone on the other hosts. Other servers never receive the change by
   file copy.
4. **Nothing is overwritten blindly.** Before any sync or rollback on a host with
   possible local work: show `git status` and `git diff` first. `reset --hard`,
   `checkout -- <path>`, `clean -fd` and overwriting a modified file are forbidden
   until local changes have been inspected and either saved or explicitly approved
   for discard by the user. Useful work must never vanish silently.
5. **Every experiment is pinned to a commit.** Record `git_branch` + `git_commit` in
   `experiments/registry.yaml` and in the run log header, together with the resolved
   config (`outputs/<date>/<time>/.hydra/config.yaml` or the `HYPER_YAML` file plus
   its effective overrides).
6. **Code + config ↔ commit must be unambiguous.** Given a commit hash and the run's
   YAML, the experiment must be reconstructable exactly: `git archive <sha>` reproduces
   the tree. This is not true today — see the status note below.
7. **Stale non-git copies are quarantined, not used.** An old copy that is not a
   repository must not be launched. Rename it out of the default path
   (`<path>_pre_git_backup_<date>`) rather than deleting it, and launch from the clone.
8. **On any cross-host code divergence, compare git first** — `git log`,
   `git rev-parse HEAD`, `git diff <shaA> <shaB>`, `git log -S <symbol>` — and only
   then decide. Do not "fix" a divergence by copying the file that looks right; that
   is how the current hybrid trees were made.
9. **Never commit credentials.** No `COMET_API_KEY`, tokens or passwords in any
   committed file, including launch YAMLs and Datasphere descriptors — reference them
   as `${oc.env:COMET_API_KEY}` from an untracked `.env`.

**Historical state (2026-09-18).** The paths below describe the old migration only;
they are not an authorization to launch from those directories.

- **aicenter3:** `/home/gorbov_gv/safe_rl_nlp` — reference tree converted in place,
  `.git` copied in, `git checkout -- .`. Pre-conversion backup:
  `/home/gorbov_gv/backups/a3_pre_git_20260918.tgz`.
- **aicenteritl:** `/home/gorbov_gv/repos/verl-agent-craftext` — clean clone, plus
  local branch `itl-local-work-20260918` (`2cd125c`) preserving dashboard-only work.
- **aicenter1:** `/home/gorbov_gv/repos/verl-agent-craftext` — clean clone (NFS server
  only, no training launches).
- **aicenter2:** `/mnt/home/gorbov_gv/repos/verl-agent-craftext` — clean clone on local
  disk (not s3fs). The legacy `/storage/gorbov_gv/safe_rl_nlp` s3fs tree is stale and
  must not be used for training.
- **GitHub:** `github.com/Gricha1/verl-agent-craftext` (created 2026-09-18; previously
  the placeholder `diameteron/verl-agent-craftext` did not exist).
- **A2 Docker launch recipe** (wrappers, reasoning launcher, grounding configs,
  SKIP_DATA_PREP, overridable VLLM_ATTENTION_BACKEND, COMET_DISABLED) committed in
  `47d9d7a`. The action-history implementation (`format_executed_actions_history`,
  `_executed_actions_history`) is also committed (part of `f1d2fb6`).
- **A2 container migration pending:** live containers still mount
  `/storage/gorbov_gv/safe_rl_nlp` (s3fs, stale). To run from current code, recreate
  the bind mount to the git clone path and restart.

**Strict active-launch rule (2026-09-24).** The only permitted launch source on ITL
and A3 is a clean clone named
`/home/gorbov_gv/verl-agent-craftext-safe-<full-or-short-current-SHA>` on branch
`safe`, at exactly the same pushed commit on both hosts. `~/safe_rl_nlp`,
`~/safe_rl_nlp_safe_*`, and an older `verl-agent-craftext-safe-*` directory are
legacy source trees: never launch from them. The Python environments and checkpoints
are kept outside a source tree (`~/venvs/verl-itl` on ITL and
`~/quantization/async/.venv` on A3; `~/verl_checkpoints/<experiment>` on both).

Before launch, record in the launcher log the branch, full SHA, clean status, host,
experiment name, and resolved overrides; require both hosts to report the same SHA.
After a new clean clone has been verified, remove the explicitly listed stale source
trees rather than keeping several plausible launch targets. Do not delete an unknown
repository merely because its name resembles this project.

## 11. Checkpoint & evaluation strategy

For new shared actor-value experiments, save **lightweight snapshots** —
shared actor/model weights or LoRA adapter only (NO optimizer/scheduler/
trainer state, no separate critic) — at **early / mid / late** stages
(minimum 2 meaningful checkpoints, preferably 3). Record per checkpoint:
global_step, env_steps, Comet key, git commit, config, stage.
(Example: the 16x16 CE run uses `checkpoint_mode: actor_lora_only`, stages [40,110].)

### 11.1 Mandatory checkpoint mode for the active 16x16 DUAL reasoning line

The current DUAL PPO runs save **one actor LoRA adapter only**, at PPO step 4:

```bash
trainer.actor_lora_only_checkpoint=True
'trainer.actor_lora_stage_steps=[4]'
trainer.save_freq=-1
```

This path invokes `save_lora_adapter_only`; it must not write FSDP model shards,
optimizer, scheduler, critic, trainer state, or dataloader state. The checkpoint lives
under `~/verl_checkpoints/<experiment>/actor_lora_stage_4/` and its metadata must
include the experiment name and exact Git SHA. `max_actor_checkpoints_to_keep=1` is
kept as a guard, but the stage list contains exactly one entry. Never approximate this
with a generic `save_freq=4`: generic FSDP saving serializes actor/critic state and
previously exhausted host disk.

**Value-understanding evaluation (future research block)** on saved checkpoints:

- value vs distance to target;
- value before/after moving toward the target;
- value when the target is invisible;
- same observation-like states with different hidden target positions;
- calibration; sensitivity to action history;
- does value represent task progress;
- do representations improve *before* the policy improves.

Purpose: prove the LLM internalizes task/world structure, not just reward fitting.

## 12. Research roadmap

1. **Matched 8×8 vs 16×16 shared actor-value + CMAE rho=0.2 + NO reasoning**.
   Match everything except `craftext_settings` (8×8 vs 16×16) and value bins
   (`[-5,6]/0.4` vs `[-26,16]/1.5`). Detailed canonical config in
   [`safe_rl_nlp_experiments.md`](safe_rl_nlp_experiments.md) §1–§2.
2. **Stable 8x8 shared AV + discounted G_t γ=0.99** run (independent of #1,
   for direct comparison with `535900d8`).
3. **Stable 16x16 shared AV CE** baseline (independent of #1, for direct
   comparison with `aa0714bb`).
4. Checkpoint analysis: early/mid/late value understanding (§11).
5. **Sparse square control** (sparse R/G_t) vs ALFWorld-like outcome credit.
6. Return to auxiliary tasks: actor+V → actor+V+Q → actor+V+multi-horizon-Q → plan-Q
   (inference: sample candidate plans, score with plan-Q, execute first action, replan).
7. Harder sparse achievement tasks: collect → prerequisite chain → craft table →
   wood pickaxe → stone pickaxe (ladder in `meta.yaml`).

Reward prediction history: actor+value+reward was often worse than actor+value
(possible causes: gradient interference, redundant immediate-reward signal,
limited LoRA capacity). Reward prediction remains a **secondary ablation**; the
main line is Q_env and plan-Q.

## 13. Experiment registry & dashboard

- `experiments/registry.yaml` — machine-readable experiment history (source of truth).
- `experiments/hypotheses.yaml` — H1–H9 with status/evidence.
- `experiments/meta.yaml` — research goal, baseline (`ds8_dual_rt_h0_02e2744`),
  compare presets, project rules.
- `experiments/metrics_cache/` — cached metric scrapes (may be stale/partial;
  many Comet read timeouts).
- **Dashboard** (`experiment_dashboard/`, FastAPI, port 8770, aicenteritl):
  tabs Overview / Experiments / Compare / Hypotheses / Planned / Launch
  (dry-run unless `EXPERIMENT_DASHBOARD_ALLOW_LAUNCH=1`).
  API: `/api/overview`, `/api/experiments`, `/api/hypotheses`,
  `/api/compare/preset/rt_vs_gt_8x8`, `/api/gpu`, `/api/launch/*`.
- This master doc keeps only key runs, conclusions, IDs, and the research
  frontier — it deliberately does not mirror all registry entries.

## 14. Rules for agents

1. Before ANY training launch: `nvidia-smi`; warn about foreign GPU jobs.
2. Never kill foreign processes.
3. If the user says STOP — stop. No automatic restart.
4. Never launch training without explicit user confirmation.
5. Do not silently change batch size, memory utilization, GPU count, learning
   rate, or reward target.
6. Any meaningful config change = a new experiment.
7. Register every experiment in `experiments/registry.yaml` (+ snapshot).
8. Save Comet key + local log path + config + git commit for every run.
9. Do not commit without explicit user request.
10. Never infer architecture from an experiment name alone — check the resolved
    config (prefer log header + code over Comet parameter names).
11. Always distinguish: dual critic vs shared actor-value; r_t vs full R vs
    undiscounted G_t vs discounted G_t vs trajectory GAE.
12. Never combine G_t with trajectory GAE without checking for double-counting.
13. On ephemeral machines, put checkpoints/logs on `/storage`.
14. Keep `txt_plan.txt` out of any docs/commits — it contains plaintext
    credentials (should be moved out of the repo and rotated).

## 14a. Правило воспроизводимости экспериментов и изменения гиперпараметров (обязательное)

1. **Референс-сравнение перед экспериментом.** Перед каждым запуском обучения —
   обязательное сравнение с референсным Comet-запуском: его точный run ID,
   полный effective (resolved) конфиг, git commit и незакоммиченные рабочие
   изменения на момент запуска. Предположения и текущие дефолты за параметры
   референса не выдаются.
2. **Запрещены самостоятельные изменения** гиперпараметров, архитектуры, reward,
   PPO loss, размеров батчей и длительности обучения.
3. **Запрещена опора на непроверенные дефолты форка.** Каждое обучающе-значимое
   значение подтверждается по референсу (log header, overrides, код), а не по
   дефолту конфига форка.
4. **Обработка падений.** При падении запуска показывать лог/трейсбек и анализ
   последствий (что именно затронуто); не маскировать падение изменением методики.
5. **Самостоятельно исправлять можно только подтверждённые программные ошибки**
   (мёртвые ключи конфига, неверная передача параметра в код) — с тестами и
   diff-отчётом, не меняя согласованную методику обучения.
6. **Гиперпараметры как workaround — только с согласия пользователя.** Если для
   запуска нужно уменьшить batch, длину истории, max_prompt_length или изменить
   другие обучающие параметры — сначала согласовать.
7. **Перед запуском показывать:** ID референсного запуска, git commit, полный
   resolved config, таблицу различий, вынужденные технические отличия (с точной
   ошибкой и альтернативами) и ожидаемое влияние на результат.
8. **Запуск без сравнения с референсом и без разрешения запрещён.**
9. **После запуска** сохранить в реестре экспериментов (`experiments/registry.yaml`
   + snapshot) точный конфиг, git commit и Comet run ID.

## 15. Related ideas (brief)

- **Senna** — one VLM with auxiliary planning-oriented token tasks for grounding.
- **PaW** — policy + next-observation world-model co-training. Key difference:
  PaW predicts `o_{t+1}`; we predict task-relevant **V/Q/plan-Q** — internalize
  dynamics relevant to decision making, not reconstruct observations. PaW also
  motivated the robust MAE/CMAE investigation for noisy token prediction.

## 16. 16x16 reasoning prompt incident: token actions vs action names (2026-09-23)

### Symptom

Two nominally similar 16x16 reasoning/DUAL PPO runs had a large gap in both
`episode/valid_action_ratio` and success rate. The reference curve started its
first rollout near **0.66--0.68** valid actions; the current token-prompt run
started near **0.36**. This was present before the first optimizer update, so it
could not be caused by PPO, KL, entropy, critic learning, or reward shaping.

### Root cause

The reference B run was launched before commit `393e031`. Although its config
contained `reasoning_template=single_token_action_reasoning`, that field was
not consumed by the old environment manager. It therefore used the generic
action-**name** prompt (e.g. `UP`, `PLACE_STONE`). The later C4 run actually
used the new prompt with a `token=name` legend and expected numeric action
tokens. Thus the two runs did not have the same initial prompt.

The encoding also made history needlessly harder to read (`2=LEFT`,
`1=NOOP`) and invalid model output could previously expand into raw response
text, increasing prompt length and eventually harming action formatting.

### Evidence

- Zero-shot first-action A/B with the same base model: the reconstructed
  action-name B prompt produced **12/24 = 0.50** valid actions; the token
  reasoning prompt produced **1/24 = 0.0417**.
- After changing the reasoning actor prompt to action names, a 24-state
  zero-shot probe produced **21/24 = 0.875** valid actions.
- With `history_length=50`, action history rendered as names, and
  `reasoning_history_length=3`, a 50-step static-observation parser/prompt
  probe produced **143/150 = 0.9533** valid actions. This last number verifies
  prompt/history and parsing, not task success, because the observation was
  deliberately held fixed to avoid mixing in Craftax/JAX worker behaviour.

### Resolution and current convention

1. The reasoning actor prompt requests one action name in
   `<action>NAME</action>`; the allowed list contains names, not numeric IDs.
2. Executed action history is stored/rendered as names only, for example
   `LEFT, NOOP, DOWN`; an invalid action executes and is remembered as `NOOP`.
3. `history_length=50`; only the last three non-empty reasoning snippets are
   included. No reasoning section is inserted until there is actual reasoning
   from a prior turn.
4. The current 16x16 training run is
   `ppo_caged_craftext_16x16_action_names_h50_r3` on aicenteritl. Its actor
   uses this prompt. Its critic currently shares the actor text prompt because
   `env.value_prompt_template_type` is not set; it does **not** use the
   optional one-token return prompt.

## 17. Current 16x16 DUAL reasoning investigation (2026-09-24)

### What is established

- The actor now asks for an action **name** in `<action>NAME</action>`.  If it
  cannot be parsed, the environment executes `NOOP`; action history then
  contains `NOOP`, not `?` or a numeric action ID.
- `history_length=50` stores executed action names.  `reasoning_history_length=3`
  adds only the last three reasoning snippets to the next actor prompt.
- With `env.store_raw_reasoning_on_missing_action_tag=True`, an unparsed answer
  still executes as `NOOP`, but its bounded raw text (at most 200 characters)
  is retained as a reasoning snippet.  This is intentionally separate from
  action parsing: an invalid action does not erase its text memory.
- The action-name change solved the initial valid-action regression: a
  static-history zero-shot probe reached `143/150 = 0.9533` valid actions.
  In a separate 32-episode transformer zero-shot A/B, empty reasoning memory
  was `94.56%` valid and raw fallback was `95.74%`.  Thus raw fallback changes
  formatting only slightly and is **not** evidence for a large reward/success
  improvement by itself.

### Why the old curves are not a clean baseline

The Comet experiment `f39114bc...` was reused by later launches through a
shared Comet key file.  Its current parameter page therefore combines values
from different sessions and cannot be treated as the resolved config of the
early metric points.  The early strong runs also predate the reproducible DUAL
launcher and were from a NOGIT/dirty tree.  Their exact source is unavailable.

The short-history run `fc25dfa9...` did have a materially different observed
configuration: action history `2`, reasoning history `5`, `max_prompt=2560`,
KL `0.001`, entropy `0.001`.  Its early success spike was not stable: success
rose to about `0.77` near step 4704 and later collapsed.  It must therefore be
treated as a hypothesis source, not as a proven target curve.

### Active matched DUAL ablation

Both active runs are DUAL PPO (separate actor and critic), 16x16, 32
environments, action history 50, reasoning history 3, 3072 prompt tokens,
reasoning enabled and raw fallback enabled.  Both are on safe commit
`789414a`.

| Host | Experiment name | KL | Entropy |
| --- | --- | ---: | ---: |
| aicenteritl | `ds16_dual_reasoning_raw_kl001_itl_retry_20260924` | 0.001 | 0.01 |
| aicenter3 | `ds16_dual_reasoning_raw_kl001_ent001_a3_retry_20260924` | 0.001 | 0.001 |

This isolates the effect of reducing KL by 10x and then additionally reducing
entropy by 10x.  Do not compare either with shared actor-value/CMAE runs as a
policy-learning baseline.

### Launch problems found and fixed

1. A3 initially scheduled vLLM onto busy GPUs, so free memory was below its
   `gpu_memory_utilization=0.55` requirement.  The retry is pinned to GPU 0,1.
2. A3's installed Craftax has replaced module-level
   `render_craftax_pixels` with `make_craftax_pixel_renderer`.  This caused
   imports to fail before any rollout.  Commits `92b6f64`, `5c7ac9a`, and
   `789414a` add a lazy compatibility wrapper in the regular, oracle, and
   caged environments.  It affects only pixel/video rendering; the ASCII
   observation prompt and RL method are unchanged.
3. Each retry has its own `RAY_TEMP_DIR` and `COMET_EXPERIMENT_KEY_FILE`; never
   reuse the shared key file, otherwise a new run attaches to an old Comet
   experiment and pollutes its parameters and curves.

### Remaining hypotheses, in priority order

1. **Optimization constraint:** current KL/entropy `0.01` may have prevented
   the policy from moving enough; the active pair tests `0.001` directly.
2. **Prompt/history distribution:** old short-history settings (`h_action=2`,
   `h_reasoning=5`) may make early exploration and credit assignment easier
   than 50 action records, even though the latter parses well.  This needs a
   clean matched A/B after the current KL/entropy result.
3. **Old dirty-tree behaviour:** reward, sampling, prompt construction, or
   rollout semantics in the unversioned historical run could differ.  This
   cannot be inferred from the contaminated Comet parameter page.
4. **Raw reasoning memory:** it is now tested as a controlled feature, but the
   zero-shot A/B says it is unlikely to explain the large success-rate gap on
   its own.
5. **Parser/valid-action rate is not the main remaining explanation:** current
   zero-shot validity is about 95%, while the weak training curve still has low
   success.  Valid syntax and task-solving/credit assignment must be measured
   separately.

## 18. Changelog

### 17.1 16x16 shared actor/value CMAE with reasoning (2026-09-23)

The third 16x16 line is a shared actor/value model with reasoning enabled. Its
actor emits `<think>...</think><action>ACTION_NAME</action>`; it retains 50
executed action names and the last three non-empty reasoning snippets. The
value prompt is separate but is predicted by the same LLM/LoRA as the actor.

CMAE is `actor_value_loss_type=clipped_mae`: a two-hot return-bin target uses
categorical MAE only if its target-support mass is at most `rho=0.2`:
`L = 1[p_target_mass <= 0.2] * 0.5 * sum_j |p_j-y_j|`. Expected diagnostics:
`actor_value/value_loss_type=2`, `actor_value/p_target_mass_le_0.2`,
`actor_value/clipped_mae`, and `actor_value/unclipped_mae`. It is an ablation,
not a proven improvement; the CE and CMAE reasoning configs are otherwise
matched.

- **2026-09-21** — Added §14a «Правило воспроизводимости экспериментов и
  изменения гиперпараметров» (обязательное референс-сравнение перед запуском,
  запрет самостоятельных изменений методики, порядок фиксации запусков) после
  аудита конфигурации DUAL PPO + reasoning относительно референса `90deefed`.
- **2026-09-14** — Created the project master document. Consolidated debug_square
  r_t/G_t results, the discounted G_t line, shared-AV CE-vs-MAE hypothesis,
  GSM8K cached comparisons, infrastructure notes, and the checkpoint policy.
- **2026-09-14 (later)** — Verified live run status via SSH: registered 8x8
  shared AV `535900d8...` (crashed, disk full) and 16x16 CE relaunch
  `aa0714bb...` (hung after disk-full) in `experiments/registry.yaml`; marked
  `4d803c6e...` as failed; rewrote §7.4 and §9 with verified data. User-confirmed
  `90deefed...` as the dual G_t γ=0.99 Comet experiment.
