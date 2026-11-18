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
  The candidate with the best score is chosen and its first action played; replanning at every step.

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
    p.add_argument("--teacher-root", default="/w/repo/caged_craftext")
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
    p.add_argument("--score", choices=("max", "last"), default="max")
    p.add_argument("--K", type=int, default=4, help="denoising steps per imagined frame")
    p.add_argument("--chunk", type=int, default=1024, help="rows per forward pass")
    p.add_argument("--seed", type=int, default=0)
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
    text = PreTrainedTokenizerFast(tokenizer_file=str(Path(cfg["text_tokenizer"]) / "tokenizer.json"))
    enc = lambda s: text.encode(s, add_special_tokens=False)
    lm = AutoModelForCausalLM.from_pretrained(cfg["init_checkpoint"], dtype=torch.float32)
    lm.resize_token_embeddings(len(text)); lm.tie_weights()
    model = SLMReasonAct(lm, cfg["readout"]).to(device)
    model.load_state_dict(state["model"]); model.eval()
    tasks = state["tasks"]
    tok = FrozenTokenizer(cfg["tok"], history=4, device=device)
    if a.imagine:
        from dreamer4.dynamics_eval import windowed_rollout
        from dreamer4.train.train_dynamics import load_dynamics_checkpoint
        dyn, dck = load_dynamics_checkpoint(a.dyn, device=device)
        dyn.eval()
        window, k_max, tau_ctx = dck["config"]["data"]["seq_len"], dck["config"]["model"]["k_max"], dck["config"]["objective"]["tau_ctx"]
        ctx = window - 1

    system = SYSTEM_PLAN.format(K) if K else SYSTEM
    pre = torch.tensor([enc("System:\n" + system + "\nUser:\n")], device=device)
    posts = [enc(f"\nTask: {t}\nAssistant:\n") for t in tasks]
    Lp = max(map(len, posts))
    post_ids = torch.zeros(len(tasks), Lp, dtype=torch.long, device=device)
    post_mask = torch.zeros_like(post_ids)
    for i, ids in enumerate(posts):
        post_ids[i, Lp - len(ids):] = torch.tensor(ids); post_mask[i, Lp - len(ids):] = 1
    max_tokens = 80 + 6 * K if K else 96

    # ---------------- environment: all episodes in one vmapped batch ----------------
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
        return key, obs, st, r, done, st.instruction_done

    step_v = jax.jit(jax.vmap(step_one))
    obs, est = reset_v(keys, jnp.array([ti for ti, _ in lanes], jnp.int32))
    to_frame = lambda o: torch.from_numpy(np.round(np.asarray(o) * 255).astype(np.uint8)).to(device)   # [L, 63, 63, 3]

    # ---------------- SLM helpers ----------------
    @torch.no_grad()
    def sample_text(Z, gi, greedy_rows):
        """Decode one answer per row; rows flagged in greedy_rows are greedy, the others follow --decode."""
        out_rows = []
        for s in range(0, Z.shape[0], a.chunk):
            Zc, gc, gr = Z[s:s + a.chunk], gi[s:s + a.chunk], greedy_rows[s:s + a.chunk]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                emb, mask, _ = model.prefix(Zc, pre, post_ids[gc], post_mask[gc])
                o = model.backbone(inputs_embeds=emb, attention_mask=mask, use_cache=True)
            past, h = o.past_key_values, o.last_hidden_state[:, -1]
            B = Zc.shape[0]
            done = torch.zeros(B, dtype=torch.bool, device=device)
            written = []
            for _ in range(max_tokens):
                logits = model.lm_head(h).float()
                greedy = logits.argmax(-1)
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
        return out_rows

    @torch.no_grad()
    def value_of(Z, gi):
        vs = []
        for s in range(0, Z.shape[0], a.chunk):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                emb, mask, _ = model.prefix(Z[s:s + a.chunk], pre, post_ids[gi[s:s + a.chunk]], post_mask[gi[s:s + a.chunk]])
                hh = model.backbone(inputs_embeds=emb, attention_mask=mask).last_hidden_state
                vs.append(model.value_head(hh[:, -1]).squeeze(-1).float())
        return torch.cat(vs)

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

    # ---------------- closed loop ----------------
    xs, zs, played = [], [], []       # per step: frames [L,63,63,3], latents [L,64,16], actions [L]
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
    N = max(1, a.plans)
    timing = collections.defaultdict(float)
    for t in range(a.max_steps):
        t0 = time.time()
        xs.append(to_frame(obs))
        win = torch.stack(xs[max(0, t - 3):t + 1], 1)
        with torch.no_grad():
            zs.append(tok.encode_frames(win)[:, -1].float())
        timing["encode"] += time.time() - t0
        act_lanes = np.flatnonzero(active)
        idx = torch.tensor(act_lanes, device=device)
        hist = [zs[max(0, t - 3 + j)][idx] for j in range(4)]
        Zt = torch.stack(hist, 1)                                              # [A, 4, 64, 16]
        A = len(act_lanes)
        # candidates
        rows_Z = Zt.repeat_interleave(N, 0)
        rows_g = g_of_lane[idx].repeat_interleave(N, 0)
        greedy_rows = torch.zeros(A * N, dtype=torch.bool, device=device)
        if a.greedy_candidate:
            greedy_rows[::N] = True
        t1 = time.time()
        answers = sample_text(rows_Z, rows_g, greedy_rows)
        timing["text"] += time.time() - t1
        cands = []                                                              # per active lane: list of (plan, source)
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
        if a.imagine and (N > 1 or a.keep_previous):
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
            v = value_of(Wk.flatten(0, 1), g_of_lane[idx][lane_of].repeat_interleave(H, 0)).view(-1, H)
            v = v.masked_fill(~valid, float("-inf"))
            if a.score == "max":
                sc = v.max(1).values
            else:
                last = valid.long().sum(1) - 1
                sc = v.gather(1, last[:, None])[:, 0]
            v_now = value_of(Zt, g_of_lane[idx])
            timing["value"] += time.time() - t3
            sc = sc.tolist()
            pos = 0
            for j, cl in enumerate(cands):
                part = sc[pos:pos + len(cl)]
                choice[j] = int(np.argmax(part))
                best_scores[act_lanes[j]].append(float(part[choice[j]]))
                now_values[act_lanes[j]].append(float(v_now[j]))
                pos += len(cl)
        acts_now = np.zeros(L, np.int32)
        for j, lane in enumerate(act_lanes):
            plan, src, _ = cands[j][choice[j]]
            chosen_stats[src] += 1
            acts_now[lane] = plan[0]
            previous[lane] = plan[1:] if len(plan) > 1 else None
            counts[lane][NAMES[plan[0]]] += 1
        played.append(acts_now)
        t4 = time.time()
        keys, obs, est, r, done, ok = step_v(keys, est, jnp.asarray(acts_now))
        r, done, ok = np.asarray(r), np.asarray(done), np.asarray(ok)
        timing["env"] += time.time() - t4
        for lane in act_lanes:
            ret[lane] += float(r[lane]); steps[lane] = t + 1
            success[lane] |= bool(ok[lane])
            if done[lane]:
                active[lane] = False
        if t % 20 == 0 or not active.any():
            print(json.dumps(dict(t=t, active=int(active.sum()), success=int(success.sum()), lanes=L,
                                  sec=round(time.time() - t_start, 1), **{k: round(v, 1) for k, v in timing.items()})), flush=True)
        if not active.any():
            break
    episodes = []
    for lane, (ti, seed) in enumerate(lanes):
        episodes.append(dict(task=instrs[ti], seed=seed, steps=int(steps[lane]), success=bool(success[lane]), ret=float(ret[lane]),
                             top_actions=counts[lane].most_common(4), unparsed=int(unparsed[lane]), first_texts=texts[lane],
                             mean_best_score=float(np.mean(best_scores[lane])) if best_scores[lane] else None,
                             mean_value_now=float(np.mean(now_values[lane])) if now_values[lane] else None))
    by_task = {instrs[ti]: float(np.mean([e["success"] for e in episodes if e["task"] == instrs[ti]])) for ti in task_ids}
    res = dict(run=run.name, checkpoint=a.checkpoint, step=state["step"], args=vars(a), plan=K, horizon=H,
               success_rate=float(success.mean()), successes=int(success.sum()), episodes_n=L, by_task=by_task,
               chosen=dict(chosen_stats), unparsed_share=float(unparsed.sum() / max(1, steps.sum() * N)),
               mean_steps=float(steps.mean()), seconds=time.time() - t_start, timing=dict(timing), episodes=episodes)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ("run", "step", "success_rate", "successes", "episodes_n", "by_task", "chosen",
                                          "unparsed_share", "mean_steps", "seconds")}), flush=True)
    print("EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
