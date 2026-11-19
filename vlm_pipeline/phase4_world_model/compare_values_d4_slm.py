#!/usr/bin/env python3
"""Value of the Dreamer 4 phase-2 heads vs value of the SLM on the same held-out frames (wood2200_d4/val).

For every validation frame t of every episode (n frames, actions a_0..a_{n-1}):
  Dreamer  V_D(t) = value head (two-hot, decoded with mean()) on the agent tokens h_t of the real window z_{t-3..t}, actions
           leading into it, task of the episode (exactly how eval_slm_d4_env.py reads the phase-2 model on a real window).
           Its training target (train_heads add_returns on the build_wood_d4_heads.py labels): ret_t = sum_k gamma_D^k r_{t+k},
           with the reward-v1 label r_{n-2} = 1 on success -> ret_t = gamma_D^(n-2-t) on success (t <= n-2), else 0.
  SLM      V_S(t) = value_head of the SLM at "Assistant:\n" on Z_t = z_{t-3..t} and the task.
           Its training target (add_value_targets): G_t = gamma_S^(n-2-t) on success, else 0; frame n-1 has no target.
Both targets mark the same event (one action before the task is done) and differ only by the discount. Saved per frame:
V_D, V_S, both targets, the frames to the end, success / death of the episode, task, episode index.

  python compare_values_d4_slm.py --slm <step.pt> --dyn <phase-1 dyn> --heads <phase-2 latest.pt> --raw-data <raw>
     --val <wood2200_d4/val> --latent-cache <dir with z_val.pt> --out values.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def main():
    p = argparse.ArgumentParser()
    for k in ("--slm", "--dyn", "--heads", "--raw-data", "--val", "--latent-cache", "--out"):
        p.add_argument(k, required=True)
    p.add_argument("--repo", default="/w/repo")
    p.add_argument("--chunk", type=int, default=512)
    a = p.parse_args()
    sys.path.insert(0, f"{a.repo}/vlm_pipeline/phase4_world_model")
    from train_slm_reason_act_d4 import SYSTEM, SYSTEM_PLAN, SLMReasonAct
    from train_slm_recon_d4 import windows
    from train_slm_world_model import add_value_targets, load_episodes
    from dreamer4.models.agent_heads import build_agent_heads, flatten_agent_out
    from dreamer4.models.dynamics import _log2_int
    from dreamer4.models.world_model import WorldModel
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
    dev = torch.device("cuda")

    base = torch.load(a.slm, map_location="cpu", weights_only=False)
    cfg, tasks = base["args"], base["tasks"]
    K = cfg.get("plan") or 0
    text = PreTrainedTokenizerFast(tokenizer_file=str(Path(cfg["text_tokenizer"]) / "tokenizer.json"))
    enc = lambda s: text.encode(s, add_special_tokens=False)
    lm = AutoModelForCausalLM.from_pretrained(cfg["init_checkpoint"], dtype=torch.float32)
    lm.resize_token_embeddings(len(text)); lm.tie_weights()
    assert "h_slot" not in base["model"], "SLMs that read Dreamer h are not supported here"
    model = SLMReasonAct(lm, cfg["readout"]).to(dev)
    model.load_state_dict(base["model"]); model.eval()
    pre = torch.tensor([enc("System:\n" + (SYSTEM_PLAN.format(K) if K else SYSTEM) + "\nUser:\n")], device=dev)
    posts = [enc(f"\nTask: {t}\nAssistant:\n") for t in tasks]
    Lp = max(map(len, posts))
    post_ids = torch.zeros(len(tasks), Lp, dtype=torch.long, device=dev); post_mask = torch.zeros_like(post_ids)
    for i, ids in enumerate(posts):
        post_ids[i, Lp - len(ids):] = torch.tensor(ids); post_mask[i, Lp - len(ids):] = 1

    hck = torch.load(a.heads, map_location="cpu", weights_only=False)
    world = WorldModel(a.dyn, device=dev, n_agent=int(hck["n_agent"]))
    world.load_dynamics_weights(hck["dyn"])
    dyn = world.dyn.eval()
    heads = build_agent_heads(hck["agent_cfg"]).to(dev).eval()
    heads.load_state_dict(hck["agent"])
    head_task = {g: i for i, g in enumerate(world.dataset.manifest["task_names"])}
    k_max, tau = world.k_max, world.tau_ctx
    sig = min(int(round((1.0 - tau) * k_max)), k_max)
    gamma_d = float(hck["train_cfg"]["gamma"]); gamma_s = float(cfg.get("value_gamma", 0.99))

    episodes = load_episodes(Path(a.raw_data), None, None)
    add_value_targets(episodes, gamma_s)
    root = Path(a.val)
    man = json.loads(((root / "manifest.json") if (root / "manifest.json").exists() else (root / "wood_manifest.json")).read_text())
    z = torch.load(Path(a.latent_cache) / "z_val.pt").to(dev)
    cols = {k: [] for k in ("idx", "first", "t", "n", "g", "gd", "prev", "tS", "mS", "tD", "succ", "died", "ep")}
    first = 0
    for e, (key, n) in enumerate(zip(man["episode_ids"], man["lengths"])):
        ep = episodes[int(key)]
        assert len(ep) == n
        ok, died = bool(ep[-1]["success"]), (not ep[-1]["success"]) and bool(ep[-1]["done"]) and n < 1000
        moves = [int(it["action"]) for it in ep]
        g = tasks.index(ep[0]["instruction"])
        for t in range(n):
            cols["idx"].append(first + t); cols["first"].append(first); cols["t"].append(t); cols["n"].append(n)
            cols["g"].append(g); cols["gd"].append(head_task[ep[0]["instruction"]])
            cols["prev"].append([moves[t - 3 + i] if t - 3 + i >= 0 else -1 for i in range(3)])   # a_{t-3..t-1}
            cols["tS"].append(ep[t]["value"]); cols["mS"].append(ep[t]["value_mask"])
            cols["tD"].append(gamma_d ** (n - 2 - t) if ok and t <= n - 2 else 0.0)
            cols["succ"].append(ok); cols["died"].append(died); cols["ep"].append(e)
        first += n
    N = first
    assert z.shape[0] == N, (z.shape, N)
    ep_first = torch.tensor(cols["first"], device=dev)
    prev = torch.tensor(cols["prev"], device=dev)
    vD, vS = [], []
    with torch.no_grad():
        for s in range(0, N, a.chunk):
            idx = torch.arange(s, min(N, s + a.chunk), device=dev)
            B = idx.shape[0]
            Z = windows(z, ep_first, idx)                                         # [B, 4, 64, 16]
            g = torch.tensor(cols["g"][s:s + B], device=dev); gd = torch.tensor(cols["gd"][s:s + B], device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                emb, mask, _ = model.prefix(Z, pre, post_ids[g], post_mask[g])
                hh = model.backbone(inputs_embeds=emb, attention_mask=mask).last_hidden_state
                vS.append(model.value_head(hh[:, -1]).squeeze(-1).float())
            av = torch.zeros(B, 4, 18, device=dev); av[:, 0, 17] = 1.0
            for j in range(1, 4):                       # slot j holds frame t-3+j, led into by a_{t-4+j} (start flag if padded)
                act = prev[s:s + B, j - 1]
                real = act >= 0
                av[:, j, :17] = F.one_hot(act.clamp_min(0), 17).float() * real[:, None]
                av[:, j, 17] = (~real).float()
            st = torch.full((B, 4), _log2_int(k_max), device=dev, dtype=torch.long)
            sg = torch.full((B, 4), sig, device=dev, dtype=torch.long)
            ag = heads.tokens((B, 4), task_ids=gd.reshape(B, 1).expand(B, 4), device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = dyn(av, st, sg, (1 - tau) * Z, agent_tokens=ag)
            h = flatten_agent_out(out[-1][:, -1].float())
            vD.append(heads.value.mean(heads.value.at(h, 0).float()).float())
    res = dict(vD=torch.cat(vD).cpu().numpy(), vS=torch.cat(vS).cpu().numpy(), gamma_d=gamma_d, gamma_s=gamma_s,
               **{k: np.asarray(v) for k, v in cols.items() if k not in ("prev", "first", "idx")})
    np.savez(a.out, **res)
    m = res["mS"] > 0
    for name, v in (("dreamer", res["vD"]), ("slm", res["vS"])):
        for tn, tgt in (("tD", res["tD"]), ("tS", res["tS"])):
            y, x = tgt[m], v[m]
            print(f"{name} vs {tn}: mse {np.mean((x - y) ** 2):.4f} r2 {1 - np.mean((x - y) ** 2) / np.var(y):.3f} "
                  f"corr {np.corrcoef(x, y)[0, 1]:.3f}", flush=True)
    print("corr(V_D, V_S)", float(np.corrcoef(res["vD"][m], res["vS"][m])[0, 1]), "frames", int(m.sum()), "episodes",
          int(res["ep"].max() + 1), "VALUES_DONE", flush=True)


if __name__ == "__main__":
    main()
