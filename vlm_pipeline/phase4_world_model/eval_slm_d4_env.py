#!/usr/bin/env python3
"""Closed-loop success on the wood tasks of an SLM over frozen Dreamer 4 latents (train_slm_reason_act_d4.py checkpoint),
optionally choosing among several sampled plans in Dreamer 4 imagination by the SLM's own value head.

Environment: the teacher env of the B_ce gate (closed_loop_eval.py --mode teacher): InstructionWrapper(Craftax-Classic-Pixels-v1,
achievements_wood), 4 tasks x seeds 20264000..20264031, at most 400 steps, success = state.instruction_done; the same key
schedule (reset with PRNGKey(seed), then key, k = split(key) before every step), all 128 episodes stepped in one vmapped batch.

At step t of an episode (frames x_0..x_t, executed actions a_0..a_{t-1}):
  z_t     = Dreamer 4 encoder(x_{t-3..t})  (uint8 63 x 63 frame = round(255 obs), the history window as in training)
  Z_t     = (z_{t-3}, z_{t-2}, z_{t-1}, z_t), the first latent repeated before the episode start
  text    = SLM(System + Z_t + Task) decoded greedy / sampled at --temperature / sampled from the --top-k tokens
  action  = <action> of the text (one-action checkpoints) or the first element of <plan> (--plan checkpoints);
            an unparsable answer plays Noop.

--imagine (plan checkpoints): N = --plans candidate plans per state (--greedy-candidate: the first is greedy; --keep-previous:
  one more candidate, the rest of the plan chosen at t-1). For every candidate p = (p_1..p_H):
  dream   d_1..d_H = Dreamer 4 dynamics, windowed rollout (window 4 = trained seq_len, context z_{t-2..t}, actions a_{t-2},
          a_{t-1}, p_1..p_H; K denoising steps), as dynamics_eval.windowed_rollout
  value   v_k = V(g, (S_k..S_{k+3})) on the sequence S = (z_{t-3..t}, d_1..d_H), k = 1..H, i.e. the SLM value of the window
          that ends at the k-th imagined frame
  score   max_k v_k (--score max) or v_H (--score last), over the steps the plan covers (shorter plans: Noop after its end,
          not scored)
  The candidate with the best score is chosen and its first --execute actions are played (default 1: replanning at every
  step); a lane plans again when its queue is empty.
  --score mean: mean_k v_k.
  --score reward (needs --heads, a Dreamer 4 phase-2 checkpoint of train_heads.py: finetuned dynamics + reward head, which
  then also produces the dream): with r_k the expected reward that arrives with d_k (reward head, MTP distance 0, task g)
  and p_k its terminal probability (the continue head if the heads have one, i.e. native reward; else the dataset's rule on
  the reward head's bins),
          score = sum_{k<=L} gamma^(k-1) A_k r_k + gamma^L A_{L+1} v_L,   A_k = prod_{j<k} (1 - p_j),  L = plan length (<= H)
  i.e. Dreamer's predicted rewards along the plan plus the SLM value of the last imagined state. Heads with a continue head
  (native env reward, Craftax / 50: the task and death have no reward of their own, both only end the episode) add
  sum_{k<=L} gamma^k A_k p_k v_k: an imagined end is worth the SLM value of its frame (G = gamma^(T-1-t) on success, so ~1
  one action before the task is done and ~0 before a death), otherwise finishing the task would cost the plan its tail.
SLM checkpoints trained with --dreamer-h read the Dreamer agent tokens h too (needs --heads, the same phase-2 model): h of the
real window ending at t when writing the plan, h of each imagined window for its value (cache_dreamer_h.py conventions).
Online Dyna loop (dyna_loop.sh): --critic <SLM .pt> takes every value from that frozen SLM instead of the acting one (GRPO
moves the policy's backbone, its value head is not trained); --save-traj <dir> writes the played episodes as a Dreamer 4
WoodEpisodeDataset with the native reward (frames 64 x 64, actions, rewards, terminals = done, wood_manifest.json, the
build_wood_d4_heads.py native conventions) plus z.pt, the per-frame latents the SLM saw, for GRPO starts.

--game crafter (Qwen-teacher Crafter data of Alexey, 2026-11-19): stock Crafter 1.8.3 (crafter.Env(size=(64, 64)), the
  63 x 63 view of its 64 x 64 frame as in build_qwen_crafter64.py), one world per seed of --seeds, the one task of the SLM,
  native reward (+1 per new achievement, +-0.1 per health point); an episode ends at death or --max-steps. Result: score =
  mean number of distinct achievements per episode (ZeroHero protocol: seeds 150000000..150000127, 2000 steps), the
  unlock rate of each achievement, and success_rate = score so that the loop logs read the same; an episode counts as
  "success" in the saved trajectories when no death ended it (the failed episodes of the GRPO start sampler).

  python eval_slm_d4_env.py --run <SLM run dir> --checkpoint step_030000.pt --out <json> [--plans 8 --decode sample --imagine
     --dyn <dreamer4 dyn>/checkpoints/latest.pt]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="train_slm_reason_act_d4.py run dir (with run/checkpoints)")
    p.add_argument("--checkpoint", default="latest.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--dyn", default="", help="Dreamer 4 dynamics checkpoint (for --imagine)")
    p.add_argument("--tok", default="", help="Dreamer 4 tokenizer checkpoint that encodes the env frames (default: the one the "
                   "SLM was trained with; the Dyna loop with tokenizer finetuning passes the current one)")
    p.add_argument("--teacher-root", default="/w/repo/caged_craftext")
    p.add_argument("--game", choices=("craftax", "crafter"), default="craftax",
                   help="not --env: the aicenter3 docker wrapper strips -e / --env from every docker run")
    p.add_argument("--settings", default="achievements_wood")
    p.add_argument("--seeds", default="20264000:20264032")
    p.add_argument("--tasks", default="", help="comma-separated task indices (default all)")
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--decode", choices=("greedy", "sample", "topk"), default="greedy")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--plans", type=int, default=1, help="candidate plans per state")
    p.add_argument("--greedy-candidate", action="store_true", help="the first candidate is decoded greedy")
    p.add_argument("--keep-previous", action="store_true", help="the rest of the previous chosen plan is one more candidate")
    p.add_argument("--imagine", action="store_true", help="choose the candidate by the value of its Dreamer 4 dream")
    p.add_argument("--horizon", type=int, default=0, help="imagined steps (0 = the plan length of the checkpoint)")
    p.add_argument("--score", choices=("max", "last", "mean", "reward"), default="max")
    p.add_argument("--value-ensemble", default="", help="ensemble.pt of train_value_heads_d4_slm.py (needs --heads, the same "
                   "phase-2 checkpoint): the value of an imagined window is the ridge-stacked multi-horizon heads")
    p.add_argument("--value-mix", type=lambda s: [float(x) for x in s.split(",")], default=None,
                   help="wS,wD,k,b (needs --heads): the value of an imagined window is wS v_SLM + wD clip(V_Dreamer, 0, 1)^k + b, "
                        "V_Dreamer the phase-2 value head on the same window (k = log gamma_SLM / log gamma_Dreamer puts it on "
                        "the SLM's discount); weights by least squares on held-out teacher frames (compare_values_d4_slm.py)")
    p.add_argument("--execute", type=int, default=1, help="actions of the chosen plan played before planning again")
    p.add_argument("--heads", default="", help="Dreamer 4 phase-2 checkpoint (train_heads.py) for --score reward")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--K", type=int, default=4, help="denoising steps per imagined frame")
    p.add_argument("--chunk", type=int, default=1024, help="rows per forward pass")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--critic", default="", help="SLM checkpoint (.pt) whose value head scores the imagined windows instead "
                   "of the acting SLM's (the Dyna loop keeps the value of the start SLM while GRPO moves the policy)")
    p.add_argument("--dump-dir", default="", help="dir: per episode (written when it ends) every step's real frame, all N plan "
                   "texts with their scores, the chosen one, and the decoded Dreamer dreams of the best and the worst plan")
    p.add_argument("--counterfactual", type=int, default=0, help="N > 0 (with --imagine --score reward): every N-th step each "
                   "candidate plan of every planning lane is also played in a copy of the real environment (Craftax state / deep-"
                   "copied Crafter env) and scored by the same formula with real rewards, real ends and the critic on the real "
                   "frames; <out>.cf.json holds dream vs real scores per group (does the dream rank the plans of one state like "
                   "reality, i.e. are GRPO's group advantages right?)")
    p.add_argument("--cf-dump", default="", help="dir: with --counterfactual, the decoded dream frames and the real frames of all "
                   "candidates of the first --cf-dump-max groups (npz) + plan texts and score parts (json)")
    p.add_argument("--cf-dump-max", type=int, default=64)
    p.add_argument("--cf-dump-per-check", type=int, default=3, help="groups dumped per counterfactual step (spread over time)")
    p.add_argument("--save-traj", default="",help="dir: the played episodes in the WoodEpisodeDataset format with the native "
                   "reward (build_wood_d4_heads.py native conventions) + per-frame latents z.pt + episodes.json")
    a = p.parse_args()
    t_start = time.time()
    torch.manual_seed(a.seed)
    device = torch.device("cuda")
    run = Path(a.run)
    state = torch.load(run / "run" / ("checkpoints" if a.checkpoint.startswith("step_") else "") / a.checkpoint,
                       map_location="cpu", weights_only=False)
    cfg = state["args"]
    sys.path.insert(0, f"{cfg['repo']}/vlm_pipeline/phase4_world_model")
    from train_slm_reason_act_d4 import EOS, NAMES, SYSTEM, SYSTEM_PLAN, SLMReasonAct, parse_action, parse_plan
    from dreamer4.models.tokenizer import FrozenTokenizer
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
    K = cfg.get("plan", 0)
    H = a.horizon or K
    if a.imagine and not K:
        raise ValueError("--imagine needs a --plan checkpoint")
    need_world = a.imagine or bool(a.heads)
    text = PreTrainedTokenizerFast(tokenizer_file=str(Path(cfg["text_tokenizer"]) / "tokenizer.json"))
    enc = lambda s: text.encode(s, add_special_tokens=False)
    lm = AutoModelForCausalLM.from_pretrained(cfg["init_checkpoint"], dtype=torch.float32)
    lm.resize_token_embeddings(len(text)); lm.tie_weights()
    n_h = int(state["model"]["h_slot"].shape[0]) if "h_slot" in state["model"] else 0
    d_h = int(state["model"]["h_adapter.0.weight"].shape[1]) if n_h else 0
    model = SLMReasonAct(lm, cfg["readout"], n_h=n_h, d_h=d_h).to(device)
    model.load_state_dict(state["model"]); model.eval()
    critic = model
    if a.critic:   # frozen value SLM (same architecture and text tokenizer as the acting one)
        cst = torch.load(a.critic, map_location="cpu", weights_only=False)
        clm = AutoModelForCausalLM.from_pretrained(cst["args"]["init_checkpoint"], dtype=torch.float32)
        clm.resize_token_embeddings(len(text)); clm.tie_weights()
        critic = SLMReasonAct(clm, cst["args"]["readout"], n_h=n_h, d_h=d_h).to(device)
        from hl_critic import attach_for   # an HL-Gauss critic checkpoint (train_slm_reason_act_d4.py --critic-from) reads its mean
        attach_for(critic, cst).load_state_dict(cst["model"]); critic.eval()
        assert cst["tasks"] == state["tasks"], "critic and policy SLMs have different task lists"
        print("value from the critic", a.critic, flush=True)
    if n_h and not a.heads:
        raise ValueError("this SLM reads Dreamer h tokens: pass --heads (and --dyn) of the phase-2 model it was trained on")
    tasks = state["tasks"]
    tok = FrozenTokenizer(a.tok or cfg["tok"], history=4, device=device)
    if need_world:
        from dreamer4.dynamics_eval import windowed_rollout
        from dreamer4.train.train_dynamics import load_dynamics_checkpoint
        dyn, dck = load_dynamics_checkpoint(a.dyn, device=device)
        dyn.eval()
        window, k_max, tau_ctx = dck["config"]["data"]["seq_len"], dck["config"]["model"]["k_max"], dck["config"]["objective"]["tau_ctx"]
        ctx = window - 1
        if a.heads:   # phase-2 world model: finetuned dynamics with agent tokens + reward head; it also produces the dream
            from dreamer4.models.agent_heads import build_agent_heads, flatten_agent_out, terminal_bins
            from dreamer4.models.dynamics import _log2_int
            from dreamer4.models.world_model import WorldModel
            hck = torch.load(a.heads, map_location="cpu", weights_only=False)
            world = WorldModel(a.dyn, device=device, n_agent=int(hck["n_agent"]))
            world.load_dynamics_weights(hck["dyn"])
            dyn = world.dyn.eval()
            heads = build_agent_heads(hck["agent_cfg"]).to(device).eval()
            heads.load_state_dict(hck["agent"])
            term_bin = terminal_bins(heads.reward, world.continues_from_reward, device)
            bin_values = heads.reward.twohot.bin_values.to(device).float()
            head_task = ({g: i for i, g in enumerate(world.dataset.manifest["task_names"])} if "task_names" in world.dataset.manifest
                         else {g: 0 for g in tasks})   # Qwen Crafter data: one task, no names
    if a.score == "reward" and not a.heads:
        raise ValueError("--score reward needs --heads")
    if a.value_mix and not (a.heads and a.imagine):
        raise ValueError("--value-mix needs --heads and --imagine")
    if a.value_ensemble:   # train_value_heads_d4_slm.py: heads on frozen SLM / Dreamer features + ridge weights
        if not (a.heads and a.imagine) or a.value_mix:
            raise ValueError("--value-ensemble needs --heads and --imagine, and excludes --value-mix")
        from train_value_heads_d4_slm import ValueHeads
        ens = torch.load(a.value_ensemble, map_location="cpu", weights_only=False)
        ens_slm = ValueHeads(ens["d_slm"], len(ens["gammas"])).to(device).eval(); ens_slm.load_state_dict(ens["slm_head"])
        ens_d4 = ValueHeads(ens["d_dreamer"], len(ens["gammas"])).to(device).eval(); ens_d4.load_state_dict(ens["dreamer_head"])
        ens_w = torch.tensor(ens["weights"], dtype=torch.float32, device=device)
        print("value ensemble", dict(zip(ens["names"], ens["weights"])), "bias", ens["bias"], flush=True)

    system = SYSTEM_PLAN.format(K) if K else SYSTEM
    pre = torch.tensor([enc("System:\n" + system + "\nUser:\n")], device=device)
    posts = [enc(f"\nTask: {t}\nAssistant:\n") for t in tasks]
    Lp = max(map(len, posts))
    post_ids = torch.zeros(len(tasks), Lp, dtype=torch.long, device=device)
    post_mask = torch.zeros_like(post_ids)
    for i, ids in enumerate(posts):
        post_ids[i, Lp - len(ids):] = torch.tensor(ids); post_mask[i, Lp - len(ids):] = 1
    max_tokens = cfg.get("gen_tokens") or (80 + 6 * K if K else 96)

    # ---------------- environment: all episodes in one vmapped batch ----------------
    crafter_env = a.game == "crafter"
    if crafter_env:   # stock Crafter, one python env per world seed, stepped in a loop
        import crafter
        instrs = list(tasks)
        first_seed, last_seed = map(int, a.seeds.split(":"))
        task_ids = [0]
        lanes = [(0, s) for s in range(first_seed, last_seed)]
        L = len(lanes)
        g_of_lane = torch.zeros(L, dtype=torch.long, device=device)
        envs = [crafter.Env(size=(64, 64), seed=s, length=10 ** 9) for _, s in lanes]
        obs = np.stack([e.reset() for e in envs])
        to_frame = lambda o: torch.from_numpy(np.ascontiguousarray(o[:, :63, :63])).to(device)   # [L, 63, 63, 3] uint8
        unlocked = [set() for _ in range(L)]
        died = np.zeros(L, bool)

        def crafter_step(acts, live_mask):
            r, done, vit = np.zeros(L, np.float32), np.zeros(L, bool), np.zeros((L, 4), np.float32)
            for lane in np.flatnonzero(live_mask):
                o, rr, dd, info = envs[lane].step(int(acts[lane]))
                obs[lane] = o; r[lane] = rr; done[lane] = dd
                inv = info["inventory"]
                vit[lane] = (inv["health"], inv["food"], inv["drink"], inv["energy"])
                unlocked[lane] |= {k for k, v in info["achievements"].items() if v > 0}
                died[lane] |= bool(dd) and inv["health"] <= 0
            return r, done, vit
    else:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        sys.path[:0] = [a.teacher_root, f"{a.teacher_root}/Craftax"]
        import jax
        import jax.numpy as jnp
        from craftax.craftax_env import make_craftax_env_from_name
        from craftext.environment.craftext_wrapper import InstructionWrapper
        base = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", False)
        params = base.default_params
        env = InstructionWrapper(base, a.settings)
        instrs = list(env.scenario_handler.scenario_data.instructions_list)
        assert set(instrs) <= set(tasks), (instrs, tasks)
        first_seed, last_seed = map(int, a.seeds.split(":"))
        task_ids = [int(x) for x in a.tasks.split(",")] if a.tasks else list(range(len(instrs)))
        lanes = [(ti, s) for ti in task_ids for s in range(first_seed, last_seed)]
        L = len(lanes)
        g_of_lane = torch.tensor([tasks.index(instrs[ti]) for ti, _ in lanes], device=device)
        keys = jnp.stack([jax.random.PRNGKey(s) for _, s in lanes])
        reset_v = jax.jit(jax.vmap(lambda k, i: env.reset(k, params, instruction_idx=i)))

        def step_one(key, st, act):
            key, k = jax.random.split(key)
            obs, st, r, done, _ = env.step(k, st, act, params)
            es = st.env_state   # Craftax state: why an episode ended (health <= 0; lava; food / drink at 0 drain health)
            lava = es.map[es.player_position[0], es.player_position[1]] == 14   # BlockType.LAVA in Craftax-Classic
            vit = jnp.stack([es.player_health, es.player_food, es.player_drink, es.player_energy]).astype(jnp.float32)
            return key, obs, st, r, done, st.instruction_done, vit, lava

        step_v = jax.jit(jax.vmap(step_one))
        obs, est = reset_v(keys, jnp.array([ti for ti, _ in lanes], jnp.int32))
        to_frame = lambda o: torch.from_numpy(np.round(np.asarray(o) * 255).astype(np.uint8)).to(device)   # [L, 63, 63, 3]

    # ---------------- SLM helpers ----------------
    answer_entropy = []   # per answer of the last sample_text call: mean token entropy (nats)

    @torch.no_grad()
    def sample_text(Z, gi, greedy_rows, Hd=None):
        """Decode one answer per row; rows flagged in greedy_rows are greedy, the others follow --decode."""
        out_rows = []
        answer_entropy.clear()
        for s in range(0, Z.shape[0], a.chunk):
            Zc, gc, gr = Z[s:s + a.chunk], gi[s:s + a.chunk], greedy_rows[s:s + a.chunk]
            ent_sum = torch.zeros(Zc.shape[0], device=device); ent_n = torch.zeros(Zc.shape[0], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                emb, mask, _ = model.prefix(Zc, pre, post_ids[gc], post_mask[gc], None if Hd is None else Hd[s:s + a.chunk])
                o = model.backbone(inputs_embeds=emb, attention_mask=mask, use_cache=True)
            past, h = o.past_key_values, o.last_hidden_state[:, -1]
            B = Zc.shape[0]
            done = torch.zeros(B, dtype=torch.bool, device=device)
            written = []
            for _ in range(max_tokens):
                logits = model.lm_head(h).float()
                greedy = logits.argmax(-1)
                lsm = (logits / a.temperature).log_softmax(-1)
                ent_sum += torch.where(done, torch.zeros_like(ent_sum), -(lsm.exp() * lsm).sum(-1)); ent_n += (~done).float()
                if a.decode == "greedy":
                    tk = greedy
                else:
                    lg = logits / a.temperature
                    if a.decode == "topk":
                        kth = lg.topk(a.top_k, -1).values[:, -1:]
                        lg = lg.masked_fill(lg < kth, float("-inf"))
                    tk = torch.multinomial(lg.softmax(-1), 1)[:, 0]
                    tk = torch.where(gr, greedy, tk)
                tk = torch.where(done, torch.zeros_like(tk), tk)
                written.append(tk)
                done |= tk == EOS
                if bool(done.all()):
                    break
                mask = torch.cat([mask, torch.ones(B, 1, dtype=mask.dtype, device=device)], 1)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = model.backbone(inputs_embeds=model.embed(tk[:, None]).float(), attention_mask=mask,
                                       past_key_values=past, use_cache=True)
                past, h = o.past_key_values, o.last_hidden_state[:, -1]
            for row in torch.stack(written, 1).tolist():
                row = row[:row.index(EOS)] if EOS in row else row
                out_rows.append(text.decode(row))
            answer_entropy.extend((ent_sum / ent_n.clamp_min(1)).tolist())
        return out_rows

    @torch.no_grad()
    def value_of(Z, gi, Hd=None, hidden=False):
        vs, hs = [], []
        for s in range(0, Z.shape[0], a.chunk):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                emb, mask, _ = critic.prefix(Z[s:s + a.chunk], pre, post_ids[gi[s:s + a.chunk]], post_mask[gi[s:s + a.chunk]],
                                             None if Hd is None else Hd[s:s + a.chunk])
                hh = critic.backbone(inputs_embeds=emb, attention_mask=mask).last_hidden_state
                vs.append(critic.value_head(hh[:, -1]).squeeze(-1).float())
            hs.append(hh[:, -1])
        return (torch.cat(vs), torch.cat(hs)) if hidden else torch.cat(vs)

    @torch.no_grad()
    def dream(zc, acts):
        """zc [B, ctx, 64, 16] real context, acts [B, ctx - 1 + H] action ids -> [B, H, 64, 16] imagined latents."""
        outs = []
        for s in range(0, zc.shape[0], a.chunk):
            zf = torch.cat([zc[s:s + a.chunk], zc[s:s + a.chunk, -1:].expand(-1, H, -1, -1)], 1)
            oh = F.one_hot(acts[s:s + a.chunk], 17).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                d, _, _ = windowed_rollout(dyn, z_full=zf, actions=oh, window=window, ctx=ctx, K=a.K, k_max=k_max, tau_ctx=tau_ctx)
            outs.append(d.float())
        return torch.cat(outs)

    @torch.no_grad()
    def agent_scan(S, into, fresh, gi, ks):
        """Phase-2 agent tokens on the windows S[:, k:k+4] for k in ks. S [C, n, 64, 16] latents, into [C, n - 1] the action that
        led into S[:, i + 1], fresh [n] bool (False: S[i] repeats S[i - 1] at the episode start -> start flag), gi [C] SLM task
        ids. Returns h [C, len(ks), n_agent, d], expected arriving reward r and terminal probability p [C, len(ks)]."""
        C = S.shape[0]
        task = torch.tensor([head_task[tasks[int(g)]] for g in gi.tolist()], device=device)
        sig = min(int(round((1.0 - tau_ctx) * k_max)), k_max)
        hs, rs, ps = [], [], []
        for k in ks:
            zw = S[:, k:k + 4]                                  # window S[k..k+3]
            av = torch.zeros(C, 4, 18, device=device)
            av[:, 0, 17] = 1.0                                  # window start flag
            for j in range(1, 4):
                if fresh[k + j]:
                    av[:, j, :17] = F.one_hot(into[:, k + j - 1], 17).float()
                else:
                    av[:, j, 17] = 1.0
            out_h, out_r, out_p = [], [], []
            for s0 in range(0, C, a.chunk):
                sl = slice(s0, s0 + a.chunk)
                B = zw[sl].shape[0]
                st = torch.full((B, 4), _log2_int(k_max), device=device, dtype=torch.long)
                sg = torch.full((B, 4), sig, device=device, dtype=torch.long)
                ag = heads.tokens((B, 4), task_ids=task[sl].reshape(B, 1).expand(B, 4), device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = dyn(av[sl], st, sg, (1 - tau_ctx) * zw[sl], agent_tokens=ag)
                hh = out[-1][:, -1].float()                         # (B, n_agent, d)
                lr_ = heads.reward.at(flatten_agent_out(hh), 0).float()   # symexp two-hot: decode with mean()
                out_h.append(hh); out_r.append(heads.reward.mean(lr_))
                out_p.append(torch.sigmoid(heads.cont(flatten_agent_out(hh)).squeeze(-1).float()) if hasattr(heads, "cont")
                             else (lr_.softmax(-1) * term_bin).sum(-1))   # continue head (native reward) or reward bins
            hs.append(torch.cat(out_h)); rs.append(torch.cat(out_r)); ps.append(torch.cat(out_p))
        return torch.stack(hs, 1), torch.stack(rs, 1), torch.stack(ps, 1)

    # ---------------- closed loop ----------------
    xs, zs, played = [], [], []       # per step: frames [L,63,63,3], latents [L,64,16], actions [L]
    rews, dones = [], []              # per step: env reward of the action, done after it [L]
    said = []                         # per step: the text of the plan chosen at that step ("" when the lane played its queue)
    active = np.ones(L, bool)
    success = np.zeros(L, bool)
    steps = np.zeros(L, int)
    ret = np.zeros(L)
    counts = [collections.Counter() for _ in range(L)]
    texts = [[] for _ in range(L)]
    unparsed = np.zeros(L, int)
    chosen_stats = collections.Counter()
    best_scores, now_values = [[] for _ in range(L)], [[] for _ in range(L)]
    previous = [None] * L               # rest of the chosen plan, for --keep-previous
    queue = [[] for _ in range(L)]      # actions of the chosen plan still to play (--execute)
    reward_terms = [[] for _ in range(L)]
    last_vit, min_health, in_lava = np.zeros((L, 4)), np.full(L, 9.0), np.zeros(L, bool)
    N = max(1, a.plans)
    timing = collections.defaultdict(float)
    dump = [dict(frame=[], dream_best=[], dream_worst=[], steps=[]) for _ in range(L)]
    cf_log = []
    cf_dumped = [0, 0]   # groups dumped in total, in this counterfactual step
    rng_dump = np.random.default_rng(a.seed + 99)

    @torch.no_grad()
    def counterfactual(t, act_lanes, flat, pa, valid, sc, rsum, tail, S=None, cands=None, rk=None, pk=None, vv=None, ents=None):
        cf_dumped[1] = 0
        """Every candidate of the planning lanes played in a copy of the real environment from the current state: real rewards
        r_k, the real end (wood: 1 on the task, 0 on a death; Crafter: 0 on a death) and otherwise the critic V on the real
        window at the plan's last step, in the dream's score formula
            real = sum_{k<=e} gamma^(k-1) r_k + (gamma^e end_value  if the episode ended at step e  else  gamma^L V(real x_{t+L}))"""
        C = len(flat); lanes_c = [int(act_lanes[j]) for j, _ in flat]
        Ls = valid.long().sum(1).tolist(); pa_np = pa.cpu().numpy()
        frames = [[xs[max(0, t - 3 + i)][l] for i in range(4)] for l in lanes_c]
        r_real = np.zeros((C, H), np.float32); end_k = np.full(C, -1); end_val = np.zeros(C, np.float32)
        if crafter_env:
            import copy
            for f, l in enumerate(lanes_c):
                e = copy.deepcopy(envs[l])
                for k in range(Ls[f]):
                    o, rr, dd, info = e.step(int(pa_np[f, k]))
                    r_real[f, k] = rr; frames[f].append(torch.from_numpy(np.ascontiguousarray(o[:63, :63])).to(device))
                    if dd:
                        end_k[f] = k; end_val[f] = 0.0
                        break
        else:
            rows = np.array(lanes_c)
            sub = jax.tree_util.tree_map(lambda x: x[rows], est); kk = keys[rows]
            live_ = np.ones(C, bool)
            for k in range(H):
                kk, ob, sub, r, done, ok, _, _ = step_v(kk, sub, jnp.asarray(pa_np[:, k].astype(np.int32)))
                r, done, ok, obn = np.asarray(r), np.asarray(done), np.asarray(ok), np.asarray(ob)
                for f in range(C):
                    if live_[f] and k < Ls[f]:
                        r_real[f, k] = r[f]
                        frames[f].append(torch.from_numpy(np.round(obn[f] * 255).astype(np.uint8)).to(device))
                        if done[f]:
                            live_[f] = False; end_k[f] = k; end_val[f] = 1.0 if ok[f] else 0.0
        n_real = [len(fr) - 4 for fr in frames]
        seq = torch.stack([torch.stack(fr + [fr[-1]] * (4 + H - len(fr))) for fr in frames])     # [C, 4 + H, 63, 63, 3]
        lat = torch.cat([tok.encode_frames(seq[s0:s0 + 128]).float() for s0 in range(0, C, 128)])   # [C, 4 + H, 64, 16]
        for f, l in enumerate(lanes_c):   # the real past frames keep the latents the planner saw (full history)
            lat[f, :4] = torch.stack([zs[max(0, t - 3 + i)][l] for i in range(4)])
        Zw = torch.stack([lat[f, n_real[f]:n_real[f] + 4] for f in range(C)])
        gi = g_of_lane[torch.tensor(lanes_c, device=device)]
        Vr = value_of(Zw, gi).cpu().numpy()
        disc = a.gamma ** np.arange(H)
        # V of the real window ending at the k-th real frame, k = 1..H (meaningless after the end; not used there)
        Vall = value_of(torch.stack([lat[f, k:k + 4] for f in range(C) for k in range(1, H + 1)]), gi.repeat_interleave(H)).view(C, H).cpu().numpy()
        KS = [k for k in (1, 4, 8, 16) if k <= H]
        rk_, pk_, vd_ = rk.cpu().numpy(), pk.cpu().numpy(), vv.masked_fill(~valid, 0).cpu().numpy()
        alive_d = np.cumprod(np.concatenate([np.ones((C, 1)), 1 - pk_], 1), 1)
        cont = hasattr(heads, "cont")

        def dream_at(f, k):   # the dream score of the first k steps of the plan (its own formula, truncated)
            k = min(k, Ls[f]); j = np.arange(k)
            v_ = (disc[j] * alive_d[f, j] * rk_[f, j]).sum() + a.gamma ** k * alive_d[f, k] * vd_[f, k - 1]
            if cont:
                v_ += (a.gamma * disc[j] * alive_d[f, j] * pk_[f, j] * vd_[f, j]).sum()
            return float(v_)

        def real_at(f, k):
            k = min(k, Ls[f]); e = end_k[f]
            if 0 <= e < k:
                return float((disc[:e + 1] * r_real[f, :e + 1]).sum() + a.gamma ** (e + 1) * end_val[f])
            return float((disc[:k] * r_real[f, :k]).sum() + a.gamma ** k * Vall[f, k - 1])
        out = []
        for j in sorted({j for j, _ in flat}):
            fs = [f for f, (jj, _) in enumerate(flat) if jj == j]
            rec = dict(t=t, lane=lanes_c[fs[0]], dream=[float(sc[f]) for f in fs], dream_r=[float(rsum[f]) for f in fs],
                       dream_tail=[float(tail[f]) for f in fs], real=[], real_r=[], real_tail=[], end=[], plan_len=[Ls[f] for f in fs],
                       first=[int(pa_np[f, 0]) for f in fs])
            rec["V_real_last"] = [float(Vr[f]) for f in fs]
            rec["ks"] = KS
            rec["dream_k"] = [[dream_at(f, k) for k in KS] for f in fs]
            rec["real_k"] = [[real_at(f, k) for k in KS] for f in fs]
            rec["dream_death_k"] = [[float((alive_d[f, :min(k, Ls[f])] * pk_[f, :min(k, Ls[f])] * (rk_[f, :min(k, Ls[f])] < 0)).sum())
                                     for k in KS] for f in fs]
            rec["real_death_k"] = [[float(0 <= end_k[f] < min(k, Ls[f]) and end_val[f] == 0) for k in KS] for f in fs]
            rec["dream_r_k"] = [[float((disc[:min(k, Ls[f])] * alive_d[f, :min(k, Ls[f])] * rk_[f, :min(k, Ls[f])]).sum()) for k in KS] for f in fs]
            rec["real_r_k"] = [[float((disc[:min(k, Ls[f], (end_k[f] + 1) if end_k[f] >= 0 else 99)] *
                                       r_real[f, :min(k, Ls[f], (end_k[f] + 1) if end_k[f] >= 0 else 99)]).sum()) for k in KS] for f in fs]
            rec["v_dream_k"] = [[float(vd_[f, min(k, Ls[f]) - 1]) for k in KS] for f in fs]
            rec["v_real_k"] = [[(float(Vall[f, min(k, Ls[f]) - 1]) if not (0 <= end_k[f] < min(k, Ls[f])) else None) for k in KS] for f in fs]
            rec["plan_actions"] = [[int(x) for x in pa_np[f, :Ls[f]]] for f in fs]
            if ents is not None:
                rec["entropy"] = [ents[f] for f in fs]
            for f in fs:
                e = end_k[f]; n = (e + 1) if e >= 0 else Ls[f]
                rr = float((disc[:n] * r_real[f, :n]).sum())
                tl = float(a.gamma ** (e + 1) * end_val[f]) if e >= 0 else float(a.gamma ** Ls[f] * Vr[f])
                rec["real"].append(rr + tl); rec["real_r"].append(rr); rec["real_tail"].append(tl)
                rec["end"].append("none" if e < 0 else ("success" if end_val[f] > 0 else "death"))
            if (a.cf_dump and S is not None and cf_dumped[0] < a.cf_dump_max and cf_dumped[1] < a.cf_dump_per_check
                    and np.std(rec["real"]) > 1e-6 and rng_dump.random() < 0.25):   # groups whose 8 plans really differ
                cf_dumped[0] += 1; cf_dumped[1] += 1
                dd = Path(a.cf_dump); dd.mkdir(parents=True, exist_ok=True)
                tag = f"t{t:04d}_lane{lanes_c[fs[0]]:03d}"
                img = tok.decode_latents(S[fs]).float()[:, 4:].permute(0, 1, 3, 4, 2).clamp(0, 1)        # [8, H, 63, 63, 3]
                dream_u8 = (img * 255).round().to(torch.uint8).cpu().numpy()
                real_u8 = np.zeros_like(dream_u8)
                for q, f in enumerate(fs):
                    for k, fr in enumerate(frames[f][4:4 + H]):
                        real_u8[q, k] = fr.cpu().numpy()
                start = torch.stack(frames[fs[0]][:4]).cpu().numpy()
                np.savez_compressed(dd / f"{tag}.npz", dream=dream_u8, real=real_u8, start=start,
                                    n_real=np.array([len(frames[f]) - 4 for f in fs]))
                j_ = flat[fs[0]][0]
                meta = dict(rec, texts=[cands[j_][c][2] for _, c in [flat[f] for f in fs]],
                            plans=[[NAMES[x] for x in pa_np[f, :Ls[f]]] for f in fs])
                (dd / f"{tag}.json").write_text(json.dumps(meta, ensure_ascii=False))
            out.append(rec)
        return out

    for t in range(a.max_steps):
        t0 = time.time()
        xs.append(to_frame(obs))
        win = torch.stack(xs[max(0, t - 3):t + 1], 1)
        with torch.no_grad():
            zs.append(tok.encode_frames(win)[:, -1].float())
        timing["encode"] += time.time() - t0
        live = np.flatnonzero(active)
        act_lanes = np.array([lane for lane in live if not queue[lane]], dtype=int)   # lanes that plan now
        idx = torch.tensor(act_lanes, device=device, dtype=torch.long)
        hist = [zs[max(0, t - 3 + j)][idx] for j in range(4)]
        Zt = torch.stack(hist, 1)                                              # [A, 4, 64, 16]
        A = len(act_lanes)
        # candidates
        rows_Z = Zt.repeat_interleave(N, 0)
        rows_g = g_of_lane[idx].repeat_interleave(N, 0)
        fresh_real = [False] + [t - 3 + i >= 1 for i in range(1, 4)]          # S[0..3] = z_{t-3..t}
        into_real = torch.tensor([[played[t - 3 + i][lane] if t - 3 + i >= 0 else 0 for i in range(3)] for lane in act_lanes],
                                 device=device, dtype=torch.long).reshape(A, 3)   # a_{t-3}, a_{t-2}, a_{t-1}
        Hnow = None
        if n_h and A:
            Hnow = agent_scan(Zt, into_real, fresh_real, g_of_lane[idx], [0])[0][:, 0]   # [A, n_agent, d]
        greedy_rows = torch.zeros(A * N, dtype=torch.bool, device=device)
        if a.greedy_candidate:
            greedy_rows[::N] = True
        t1 = time.time()
        answers = sample_text(rows_Z, rows_g, greedy_rows, None if Hnow is None else Hnow.repeat_interleave(N, 0)) if A else []
        timing["text"] += time.time() - t1
        cands = []                                                              # per active lane: list of (plan, source)
        cand_ent = [answer_entropy[j * N:(j + 1) * N] for j in range(A)] if A else []
        for j, lane in enumerate(act_lanes):
            cl = []
            for n in range(N):
                txt = answers[j * N + n]
                if K:
                    plan = [x for x in (parse_plan(txt) or []) if x is not None]
                else:
                    x = parse_action(txt)
                    plan = [x] if x is not None else []
                if not plan:
                    unparsed[lane] += 1
                    plan = [0]
                cl.append((plan, "greedy" if (a.greedy_candidate and n == 0) or a.decode == "greedy" else "sampled", txt))
            if a.keep_previous and previous[lane]:
                cl.append((previous[lane], "previous", ""))
            cands.append(cl)
            if len(texts[lane]) < 3:
                texts[lane].append(answers[j * N])
        choice = [0] * A
        if A and a.imagine and (N > 1 or a.keep_previous):
            t2 = time.time()
            flat = [(j, c) for j, cl in enumerate(cands) for c in range(len(cl))]
            lane_of = torch.tensor([j for j, _ in flat], device=device)
            ctx_lat = torch.stack([zs[max(0, t - ctx + 1 + i)][idx] for i in range(ctx)], 1)[lane_of]   # z_{t-2..t}
            before = [[played[t - ctx + 1 + i][lane] if t - ctx + 1 + i >= 0 else 0 for i in range(ctx - 1)] for lane in act_lanes]
            acts, valid = [], []
            for j, c in flat:
                plan = cands[j][c][0][:H]
                acts.append(before[j] + plan + [0] * (H - len(plan)))
                valid.append([k < len(plan) for k in range(H)])
            acts = torch.tensor(acts, device=device)
            valid = torch.tensor(valid, device=device)
            d = dream(ctx_lat, acts)                                            # [C, H, 64, 16]
            timing["dream"] += time.time() - t2
            t3 = time.time()
            S = torch.cat([Zt[lane_of], d], 1)                                  # [C, 4 + H, 64, 16]
            Wk = torch.stack([S[:, k:k + 4] for k in range(1, H + 1)], 1)       # [C, H, 4, 64, 16]
            if a.heads:   # Dreamer agent tokens of every imagined window: reward head, and the SLM's h input
                into = torch.cat([into_real[lane_of], acts[:, ctx - 1:]], 1)    # actions into S[1..3 + H]
                Himg, rk, pk = agent_scan(S, into, fresh_real + [True] * H, g_of_lane[idx][lane_of], list(range(1, H + 1)))
            v = value_of(Wk.flatten(0, 1), g_of_lane[idx][lane_of].repeat_interleave(H, 0),
                         Himg.flatten(0, 1) if n_h else None, hidden=bool(a.value_ensemble))
            if a.value_ensemble:   # stacked multi-horizon heads on the SLM hidden state and the Dreamer agent tokens
                v, hsS = v
                hf = flatten_agent_out(Himg.flatten(0, 1).float())
                vd0 = heads.value.mean(heads.value.at(hf, 0).float()).float()
                with torch.no_grad():
                    X = torch.cat([ens_slm(hsS.float()), ens_d4(hf), v[:, None], vd0.clamp(0, 1)[:, None] ** ens["kpow"]], 1)
                v = X @ ens_w + ens["bias"]
            v = v.view(-1, H)
            if a.value_mix:   # v = wS v_SLM + wD clip(V_Dreamer, 0, 1)^k + b on every imagined window
                wS, wD, kk, b0 = a.value_mix
                hf = flatten_agent_out(Himg.flatten(0, 1).float())
                vd = heads.value.mean(heads.value.at(hf, 0).float()).float().view(-1, H)
                v = wS * v + wD * vd.clamp(0, 1) ** kk + b0
            v = v.masked_fill(~valid, float("-inf"))
            last = valid.long().sum(1) - 1
            if a.score == "max":
                sc = v.max(1).values
            elif a.score == "mean":
                sc = v.masked_fill(~valid, 0).sum(1) / valid.sum(1)
            elif a.score == "last":
                sc = v.gather(1, last[:, None])[:, 0]
            else:
                alive = torch.cumprod(torch.cat([torch.ones_like(pk[:, :1]), 1 - pk], 1), 1)   # A_1..A_{H+1}
                disc = a.gamma ** torch.arange(H, device=device, dtype=torch.float32)
                rsum = (disc * alive[:, :H] * rk * valid).sum(1)
                vl = v.gather(1, last[:, None])[:, 0]
                tail = a.gamma ** (last + 1).float() * alive.gather(1, (last + 1)[:, None])[:, 0] * vl
                if hasattr(heads, "cont"):   # native reward: an imagined end is valued by the SLM value of its frame
                    tail = tail + (a.gamma * disc * alive[:, :H] * pk * v.masked_fill(~valid, 0) * valid).sum(1)
                sc = rsum + tail
                for j, (jj, c) in enumerate(flat):
                    if c == 0:
                        reward_terms[act_lanes[jj]].append((float(rsum[j]), float(tail[j])))
            v_now = value_of(Zt, g_of_lane[idx], Hnow)
            timing["value"] += time.time() - t3
            if a.counterfactual and a.score == "reward" and t % a.counterfactual == 0:
                t5 = time.time()
                ent_f = [cand_ent[j][c] if c < len(cand_ent[j]) else None for j, c in flat]
                cf_log.extend(counterfactual(t, act_lanes, flat, acts[:, ctx - 1:], valid, sc, rsum, tail, S, cands, rk, pk, v, ent_f))
                timing["counterfactual"] += time.time() - t5
            sc = sc.tolist()
            pos = 0
            for j, cl in enumerate(cands):
                part = sc[pos:pos + len(cl)]
                choice[j] = int(np.argmax(part))
                if a.dump_dir:   # the real frame, every candidate and the dreams of the best and the worst plan
                    lane = act_lanes[j]; fb, fw = pos + choice[j], pos + int(np.argmin(part))
                    with torch.no_grad():
                        img = tok.decode_latents(S[[fb, fw]]).float()[:, 4:].permute(0, 1, 3, 4, 2).clamp(0, 1)
                    img = (img * 255).round().to(torch.uint8).cpu().numpy()            # [2, H, 63, 63, 3]
                    dump[lane]["frame"].append(xs[t][lane].cpu().numpy()); dump[lane]["dream_best"].append(img[0])
                    dump[lane]["dream_worst"].append(img[1])
                    dump[lane]["steps"].append(dict(t=t, choice=choice[j], worst=int(np.argmin(part)), scores=part,
                                                    texts=[c[2] for c in cl], plans=[[NAMES[x] for x in c[0]] for c in cl]))
                best_scores[act_lanes[j]].append(float(part[choice[j]]))
                now_values[act_lanes[j]].append(float(v_now[j]))
                pos += len(cl)
        acts_now = np.zeros(L, np.int32)
        said.append([""] * L)
        for j, lane in enumerate(act_lanes):
            plan, src, said[-1][lane] = cands[j][choice[j]]
            chosen_stats[src] += 1
            queue[lane] = list(plan[:max(1, a.execute)])
            previous[lane] = plan[len(queue[lane]):] or None
        for lane in live:
            acts_now[lane] = queue[lane].pop(0)
            counts[lane][NAMES[acts_now[lane]]] += 1
        played.append(acts_now)
        t4 = time.time()
        if crafter_env:
            r, done, vit = crafter_step(acts_now, active)
            env_done = done & died                                       # a death; the timeout below has no done flag
            done = done | (steps + 1 >= a.max_steps)
            ok, lava = np.zeros(L, bool), np.zeros(L, bool)
        else:
            keys, obs, est, r, done, ok, vit, lava = step_v(keys, est, jnp.asarray(acts_now))
            r, done, ok, vit, lava = np.asarray(r), np.asarray(done), np.asarray(ok), np.asarray(vit), np.asarray(lava)
            env_done = done
        rews.append(r.astype(np.float32)); dones.append(env_done.copy())
        timing["env"] += time.time() - t4
        for lane in live:
            ret[lane] += float(r[lane]); steps[lane] = t + 1
            last_vit[lane] = vit[lane]; min_health[lane] = min(min_health[lane], float(vit[lane][0])); in_lava[lane] = bool(lava[lane])
            success[lane] |= bool(ok[lane])
            if done[lane]:
                active[lane] = False
                queue[lane] = []
                if a.dump_dir and dump[lane]["frame"]:   # written at once, so the first finished episodes can be looked at
                    dd = Path(a.dump_dir); dd.mkdir(parents=True, exist_ok=True)
                    ti, seed = lanes[lane]
                    tag = f"{'ok' if success[lane] else 'fail'}_{instrs[ti].split()[-1].strip('.').lower()}_{seed}"
                    np.savez_compressed(dd / f"{tag}.npz", frame=np.stack(dump[lane]["frame"]),
                                        dream_best=np.stack(dump[lane]["dream_best"]), dream_worst=np.stack(dump[lane]["dream_worst"]),
                                        actions=np.array([p[lane] for p in played]))
                    (dd / f"{tag}.json").write_text(json.dumps(dict(task=instrs[ti], seed=seed, success=bool(success[lane]),
                                                                    steps=t + 1, vitals=[float(x) for x in vit[lane]],
                                                                    lava=bool(lava[lane]) if not crafter_env else False,
                                                                    per_step=dump[lane]["steps"]), ensure_ascii=False))
                    dump[lane] = None
        if t % 20 == 0 or not active.any():
            print(json.dumps(dict(t=t, active=int(active.sum()), success=int(success.sum()) if not crafter_env else
                                  round(float(np.mean([len(u) for u in unlocked])), 3), lanes=L,
                                  sec=round(time.time() - t_start, 1), **{k: round(v, 1) for k, v in timing.items()})), flush=True)
        if not active.any():
            break
    episodes = []
    for lane, (ti, seed) in enumerate(lanes):
        h, f, dr, en = last_vit[lane]
        if crafter_env:   # "success" of a Crafter episode in the saved trajectories: no death ended it
            end = "death" if died[lane] else "timeout" if steps[lane] >= a.max_steps else "other"
            success[lane] = not died[lane]
        else:
            end = ("success" if success[lane] else "lava" if in_lava[lane] else
                   ("death_starve" if min(f, dr) <= 0 else "death_damage") if h <= 0 else "timeout" if steps[lane] >= a.max_steps else "other")
        episodes.append(dict(task=instrs[ti], seed=seed, steps=int(steps[lane]), success=bool(success[lane]), ret=float(ret[lane]),
                             achievements=sorted(unlocked[lane]) if crafter_env else None,
                             end=end, final_health=float(h), final_food=float(f), final_drink=float(dr), final_energy=float(en),
                             min_health=float(min_health[lane]),
                             top_actions=counts[lane].most_common(4), unparsed=int(unparsed[lane]), first_texts=texts[lane],
                             mean_best_score=float(np.mean(best_scores[lane])) if best_scores[lane] else None,
                             mean_value_now=float(np.mean(now_values[lane])) if now_values[lane] else None,
                             mean_reward_term=float(np.mean([x[0] for x in reward_terms[lane]])) if reward_terms[lane] else None,
                             mean_value_term=float(np.mean([x[1] for x in reward_terms[lane]])) if reward_terms[lane] else None))
    by_task = {instrs[ti]: float(np.mean([e["success"] for e in episodes if e["task"] == instrs[ti]])) for ti in task_ids}
    extra = {}
    if crafter_env:   # ZeroHero score: distinct achievements per episode; unlock rate of each achievement
        extra = dict(score=float(np.mean([len(u) for u in unlocked])), mean_return=float(ret.mean()),
                     achievement_rates={k: float(np.mean([k in u for u in unlocked])) for k in sorted(crafter.constants.achievements)})
        by_task = extra["achievement_rates"]
    ends = dict(collections.Counter(e["end"] for e in episodes))
    res = dict(run=run.name, checkpoint=a.checkpoint, step=state["step"], args=vars(a), plan=K, horizon=H, ends=ends, **extra,
               success_rate=extra["score"] if crafter_env else float(success.mean()), successes=int(success.sum()), episodes_n=L,
               by_task=by_task,
               chosen=dict(chosen_stats), unparsed_share=float(unparsed.sum() / max(1, steps.sum() * N)),
               mean_steps=float(steps.mean()), seconds=time.time() - t_start, timing=dict(timing), episodes=episodes)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    if cf_log:   # dream vs real ranking of the plans of one state
        def avg_rank(x):
            r = np.empty(len(x)); r[np.argsort(x, kind="mergesort")] = np.arange(len(x))
            for u in np.unique(x):
                r[x == u] = r[x == u].mean()
            return r
        rho, top1, regret, adv_corr, n_g = [], [], [], [], 0
        for rec in cf_log:
            d, r = np.array(rec["dream"]), np.array(rec["real"])
            if len(d) < 2 or r.std() < 1e-6 or d.std() < 1e-9:
                continue
            n_g += 1
            rho.append(float(np.corrcoef(avg_rank(d), avg_rank(r))[0, 1])); top1.append(float(r[d.argmax()] >= r.max() - 1e-6))
            regret.append(float(r.max() - r[d.argmax()]))
            adv_corr.append(float(np.corrcoef((d - d.mean()) / d.std(), (r - r.mean()) / r.std())[0, 1]))
        summ = dict(groups=len(cf_log), groups_with_real_spread=n_g, spearman_mean=float(np.nanmean(rho)) if rho else None,
                    top1_agree=float(np.mean(top1)) if top1 else None, regret_mean=float(np.mean(regret)) if regret else None,
                    adv_corr_mean=float(np.nanmean(adv_corr)) if adv_corr else None,
                    share_groups_real_tied=1 - n_g / max(1, len(cf_log)))
        Path(a.out + ".cf.json").write_text(json.dumps(dict(summary=summ, groups=cf_log)))
        print("COUNTERFACTUAL", json.dumps(summ), flush=True)
    if a.save_traj:   # build_wood_d4_heads.py native conventions: frame x_t, action a_t, reward of a_t on t -> t+1
        sd = Path(a.save_traj); sd.mkdir(parents=True, exist_ok=True)
        names = sorted(instrs)
        fr, zz, ac, rw, tm, starts, lengths, task_idx, bc = [], [], [], [], [], [], [], [], []
        first = 0
        for lane, e in enumerate(episodes):
            n = e["steps"]
            x = torch.stack([xs[t][lane] for t in range(n)]).cpu().numpy()
            fr.append(np.pad(x, ((0, 0), (0, 1), (0, 1), (0, 0))))          # 64 x 64 like wood2200 (black row / column)
            zz.append(torch.stack([zs[t][lane] for t in range(n)]).half().cpu())
            ac.append(np.array([played[t][lane] for t in range(n)], np.int64))
            r_ep = np.array([rews[t][lane] for t in range(n)], np.float32)
            rr = np.zeros(n, np.float32); tt = np.zeros(n, bool)
            if n >= 2:
                rr[:n - 1] = r_ep[:n - 1]; rr[n - 2] += r_ep[n - 1]
                tt[n - 2] = bool(dones[n - 1][lane])                        # ended with done (task or death), not a timeout
            rw.append(rr); tm.append(tt)
            starts.append(first); lengths.append(n); first += n
            task_idx.append(names.index(e["task"])); bc.append(int(e["success"]))
        np.save(sd / "frames.npy", np.concatenate(fr)); np.save(sd / "actions.npy", np.concatenate(ac))
        np.save(sd / "rewards.npy", np.concatenate(rw)); np.save(sd / "terminals.npy", np.concatenate(tm))
        torch.save(torch.cat(zz), sd / "z.pt")
        (sd / "wood_manifest.json").write_text(json.dumps(dict(
            starts=starts, lengths=lengths, episode_ids=[-(1 + i) for i in range(len(episodes))], n_actions=17, tasks=task_idx,
            bc=[0] * len(bc) if crafter_env else bc, task_names=names, terminal_rule="native",
            reward_rule=("native Crafter reward" if crafter_env else "native environment reward (Craftax / 50)") +
                        ", last action's reward added to the transition into x_{n-1}",
            source=dict(run=run.name, checkpoint=a.checkpoint, seeds=a.seeds, heads=a.heads, critic=a.critic))))
        (sd / "plans.json").write_text(json.dumps([said[t][lane] for lane, e in enumerate(episodes) for t in range(e["steps"])]))
        (sd / "episodes.json").write_text(json.dumps([dict(task=e["task"], seed=e["seed"], steps=e["steps"], success=e["success"],
                                                           end=e["end"]) for e in episodes]))
        print("saved", len(episodes), "episodes", first, "frames to", sd, flush=True)
    print(json.dumps({k: res[k] for k in ("run", "step", "success_rate", "successes", "episodes_n", "by_task", "ends", "chosen",
                                          "unparsed_share", "mean_steps", "seconds") + (("score", "mean_return") if crafter_env else ())}),
          flush=True)
    print("EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
