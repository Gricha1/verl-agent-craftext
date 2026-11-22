"""Dreamer 4 next to veRL (Stage-2 RL through veRL, safe_rl_nlp.md rules 18-21): the agent loop's world model.

For every lane of a vector rollout it keeps the last real frames and played actions, and answers
  first(frames)                          new episodes: encode the frames, return the SLM's latent windows [L, 4*64*16]
  step(lanes, actions, frames)           the lanes played these actions and see these frames: their new latent windows
  score(lanes, plans)                    one row per candidate plan (lanes[i] is the lane of row i): the Stage-1 plan score
in the format of eval_slm_d4_env.py --imagine --score reward (the same code, moved here):
  dream the plan from the real context (windowed_rollout, K denoising steps), read the phase-2 agent heads on every
  imagined window (expected reward r_k, terminal probability p_k), the critic V on every imagined window, and
    S = sum_k gamma^(k-1) A_k r_k + sum_k gamma^k A_k p_k V_k + gamma^L A_{L+1} V_L,   A_k = prod_{j<k} (1 - p_j)
  optionally M dreams per plan: score = mean - pessimism * std.
GRPO in veRL uses S as the reward of each of the n plans of one state. The world model is frozen here (version 1); PPO
never reaches into it.

  python3 dreamer_service.py --dyn D --heads H --tok T --critic C.pt --socket /run/dreamer.sock --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


class Service:
    def __init__(self, a):
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
        from dreamer4.dynamics_eval import windowed_rollout
        from dreamer4.models.agent_heads import build_agent_heads, flatten_agent_out
        from dreamer4.models.dynamics import _log2_int
        from dreamer4.models.tokenizer import FrozenTokenizer
        from dreamer4.models.world_model import WorldModel
        from dreamer4.train.train_dynamics import load_dynamics_checkpoint
        from train_slm_reason_act_d4 import SYSTEM_PLAN, SLMReasonAct
        from hl_critic import attach_for
        self.torch, self.F, self.a = torch, F, a
        self.windowed_rollout, self.flatten_agent_out, self._log2_int = windowed_rollout, flatten_agent_out, _log2_int
        dev = self.dev = torch.device("cuda")
        torch.manual_seed(a.seed)
        self.tok = FrozenTokenizer(a.tok, history=4, device=dev)
        _, dck = load_dynamics_checkpoint(a.dyn, device=dev)
        self.window, self.k_max, self.tau_ctx = (dck["config"]["data"]["seq_len"], dck["config"]["model"]["k_max"],
                                                 dck["config"]["objective"]["tau_ctx"])
        self.ctx = self.window - 1
        hck = torch.load(a.heads, map_location="cpu", weights_only=False)
        world = WorldModel(a.dyn, device=dev, n_agent=int(hck["n_agent"]))
        world.load_dynamics_weights(hck["dyn"])
        self.dyn = world.dyn.eval()
        self.heads = build_agent_heads(hck["agent_cfg"]).to(dev).eval()
        self.heads.load_state_dict(hck["agent"])
        self.cont = hasattr(self.heads, "cont")
        from dreamer4.models.agent_heads import terminal_bins
        self.term_bin = None if self.cont else terminal_bins(self.heads.reward, world.continues_from_reward, dev)
        man = world.dataset.manifest
        self.critics = []
        for cp in [a.critic] + [x for x in a.extra_critics.split(",") if x]:
            cst = torch.load(cp, map_location="cpu", weights_only=False)
            text = PreTrainedTokenizerFast(tokenizer_file=str(Path(a.text_tokenizer or cst["args"]["text_tokenizer"]) / "tokenizer.json"))
            lm = AutoModelForCausalLM.from_pretrained(a.init or cst["args"]["init_checkpoint"], dtype=torch.float32)
            lm.resize_token_embeddings(len(text)); lm.tie_weights()
            cr = attach_for(SLMReasonAct(lm, cst["args"]["readout"]).to(dev), cst)
            cr.load_state_dict(cst["model"]); cr.eval()
            self.critics.append(cr)
            self.tasks = cst["tasks"]
            plan = int(cst["args"].get("plan") or 16)
        enc = lambda s: text.encode(s, add_special_tokens=False)
        self.pre = torch.tensor([enc("System:\n" + SYSTEM_PLAN.format(plan) + "\nUser:\n")], device=dev)
        posts = [enc(f"\nTask: {t}\nAssistant:\n") for t in self.tasks]
        Lp = max(map(len, posts))
        self.post_ids = torch.zeros(len(self.tasks), Lp, dtype=torch.long, device=dev)
        self.post_mask = torch.zeros_like(self.post_ids)
        for i, ids in enumerate(posts):   # left padding, exactly as in Stage 1 (train_slm_reason_act_d4 / eval_slm_d4_env)
            self.post_ids[i, Lp - len(ids):] = torch.tensor(ids); self.post_mask[i, Lp - len(ids):] = 1
        self.head_task = ({g: i for i, g in enumerate(man["task_names"])} if "task_names" in man else {g: 0 for g in self.tasks})
        self.H = a.horizon
        self.lock = threading.Lock()
        self.stats = dict(scored=0, seconds=0.0)
        print(json.dumps(dict(loaded=True, tasks=self.tasks, H=self.H, K=a.K, critics=len(self.critics), cont=self.cont)), flush=True)

    # ---------------- the Stage-1 scorer (eval_slm_d4_env.py) ----------------
    def value_of(self, Z, gi):
        torch, a = self.torch, self.a
        vs = []
        for s in range(0, Z.shape[0], a.chunk):
            vk = []
            for cr in self.critics:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    emb, mask, _ = cr.prefix(Z[s:s + a.chunk], self.pre, self.post_ids[gi[s:s + a.chunk]], self.post_mask[gi[s:s + a.chunk]])
                    hh = cr.backbone(inputs_embeds=emb, attention_mask=mask).last_hidden_state
                    vk.append(cr.value_head(hh[:, -1]).squeeze(-1).float())
            vk = torch.stack(vk)
            vs.append(vk.mean(0) - a.critic_pessimism * vk.std(0) if len(vk) > 1 else vk[0])
        return torch.cat(vs)

    def dream(self, zc, acts):
        torch, a = self.torch, self.a
        outs = []
        for s in range(0, zc.shape[0], a.chunk):
            zf = torch.cat([zc[s:s + a.chunk], zc[s:s + a.chunk, -1:].expand(-1, self.H, -1, -1)], 1)
            oh = self.F.one_hot(acts[s:s + a.chunk], 17).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                d, _, _ = self.windowed_rollout(self.dyn, z_full=zf, actions=oh, window=self.window, ctx=self.ctx, K=a.K,
                                                k_max=self.k_max, tau_ctx=self.tau_ctx)
            outs.append(d.float())
        return torch.cat(outs)

    def agent_scan(self, S, into, fresh, gi, ks):
        torch, a, dev = self.torch, self.a, self.dev
        C = S.shape[0]
        task = torch.tensor([self.head_task[self.tasks[int(g)]] for g in gi.tolist()], device=dev)
        sig = min(int(round((1.0 - self.tau_ctx) * self.k_max)), self.k_max)
        rs, ps = [], []
        for k in ks:
            zw = S[:, k:k + 4]
            av = torch.zeros(C, 4, 18, device=dev)
            av[:, 0, 17] = 1.0
            for j in range(1, 4):
                if fresh[k + j]:
                    av[:, j, :17] = self.F.one_hot(into[:, k + j - 1], 17).float()
                else:
                    av[:, j, 17] = 1.0
            out_r, out_p = [], []
            for s0 in range(0, C, a.chunk):
                sl = slice(s0, s0 + a.chunk)
                B = zw[sl].shape[0]
                st = torch.full((B, 4), self._log2_int(self.k_max), device=dev, dtype=torch.long)
                sg = torch.full((B, 4), sig, device=dev, dtype=torch.long)
                ag = self.heads.tokens((B, 4), task_ids=task[sl].reshape(B, 1).expand(B, 4), device=dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = self.dyn(av[sl], st, sg, (1 - self.tau_ctx) * zw[sl], agent_tokens=ag)
                hh = out[-1][:, -1].float()
                lr_ = self.heads.reward.at(self.flatten_agent_out(hh), 0).float()
                out_r.append(self.heads.reward.mean(lr_))
                if self.cont:
                    out_p.append(torch.sigmoid(self.heads.cont(self.flatten_agent_out(hh)).squeeze(-1).float()))
                else:   # reward bins that end the episode (no continue head), as eval_slm_d4_env.py
                    out_p.append((lr_.softmax(-1) * self.term_bin).sum(-1))
            rs.append(torch.cat(out_r)); ps.append(torch.cat(out_p))
        return torch.stack(rs, 1), torch.stack(ps, 1)

    def score_rows(self, lane_z, lane_played, lane_t, lane_g, plans):
        """lane_z [n, >=4, 64, 16] latent history (last 4 used), lane_played [n, 3] a_{t-3..t-1}, lane_t [n] step index,
        lane_g [n] task id; plans: per row a list of action ids. Returns the score per row."""
        torch, a, dev, H, ctx = self.torch, self.a, self.dev, self.H, self.ctx
        M_ = max(1, a.dream_samples)
        rows = [i for i in range(len(plans)) for _ in range(M_)]
        Zt = lane_z[rows]                                                         # [C, 4, 64, 16]
        tt = lane_t[rows]
        fresh_real = [[False] + [int(t) - 3 + i >= 1 for i in range(1, 4)] for t in tt.tolist()]
        into_real = lane_played[rows]                                             # [C, 3]
        before = lane_played[rows][:, 3 - (ctx - 1):]                             # a_{t-ctx+1..t-1}
        acts, valid = [], []
        for r in rows:
            p = list(plans[r][:H]) or [0]
            acts.append(p + [0] * (H - len(p))); valid.append([k < len(p) for k in range(H)])
        acts = torch.cat([before, torch.tensor(acts, device=dev)], 1)
        valid = torch.tensor(valid, device=dev)
        d = self.dream(Zt[:, 4 - ctx:], acts)                                     # [C, H, 64, 16]
        S = torch.cat([Zt, d], 1)
        Wk = torch.stack([S[:, k:k + 4] for k in range(1, H + 1)], 1)
        into = torch.cat([into_real, acts[:, ctx - 1:]], 1)
        # fresh flags differ per row only at the episode start (t < 3): group rows by their pattern
        rk = torch.zeros(len(rows), H, device=dev); pk = torch.zeros(len(rows), H, device=dev)
        pats = {}
        for i, f in enumerate(fresh_real):
            pats.setdefault(tuple(f), []).append(i)
        g = lane_g[rows]
        for pat, idx in pats.items():
            ix = torch.tensor(idx, device=dev)
            r_, p_ = self.agent_scan(S[ix], into[ix], list(pat) + [True] * H, g[ix], list(range(1, H + 1)))
            rk[ix], pk[ix] = r_, p_
        v = self.value_of(Wk.flatten(0, 1), g.repeat_interleave(H, 0)).view(-1, H)
        last = valid.long().sum(1) - 1
        alive = torch.cumprod(torch.cat([torch.ones_like(pk[:, :1]), 1 - pk], 1), 1)
        disc = a.gamma ** torch.arange(H, device=dev, dtype=torch.float32)
        rsum = (disc * alive[:, :H] * rk * valid).sum(1)
        vl = v.gather(1, last[:, None])[:, 0]
        tail = a.gamma ** (last + 1).float() * alive.gather(1, (last + 1)[:, None])[:, 0] * vl
        if self.cont:
            tail = tail + (a.gamma * disc * alive[:, :H] * pk * v.masked_fill(~valid, 0) * valid).sum(1)
        sc = rsum + tail
        if M_ > 1:
            scm = sc.view(-1, M_)
            sc = scm.mean(1) - a.pessimism * scm.std(1)
        return sc

    # ---------------- one rollout ----------------
    def serve_rollout(self, connection) -> None:
        torch, dev = self.torch, self.dev
        frames_hist, z_hist, played, steps, g = None, None, None, None, None

        def windows(lanes):
            return torch.stack([torch.stack(z_hist[l][-4:]) for l in lanes]).reshape(len(lanes), -1).float().cpu().numpy()

        def encode(lanes):
            """Latent of each lane's newest frame, encoded with the frames before it (up to 4, as eval_slm_d4_env.py:
            xs[max(0, t - 3):t + 1]); lanes with equal history length go in one batch."""
            out = [None] * len(lanes)
            by_len = {}
            for i, l in enumerate(lanes):
                by_len.setdefault(len(frames_hist[l]), []).append(i)
            for n, idx in by_len.items():
                win = torch.stack([torch.stack(frames_hist[lanes[i]]) for i in idx])          # [m, n, 63, 63, 3] uint8
                with torch.no_grad():
                    z = self.tok.encode_frames(win)[:, -1].float()
                for k, i in enumerate(idx):
                    out[i] = z[k]
            return torch.stack(out)

        while True:
            try:
                message = connection.recv()
            except EOFError:
                break
            try:
                name, *rest = message
                if name in ("first", "reset"):   # first(frames, tasks): all lanes; reset(lanes, frames, tasks): new episodes
                    if name == "first":
                        frames, tasks = rest
                        L = len(frames); lanes = list(range(L))
                        frames_hist, z_hist, played, steps, g = [None] * L, [None] * L, [None] * L, [0] * L, [0] * L
                    else:
                        lanes, frames, tasks = rest
                    to = lambda f: torch.from_numpy(np.ascontiguousarray(f)).to(dev)
                    for l, f, t in zip(lanes, frames, tasks):
                        frames_hist[l] = [to(f)]                                             # real frames so far (up to 4)
                        played[l] = [0, 0, 0]; steps[l] = 0
                        g[l] = self.tasks.index(t) if t in self.tasks else 0
                    with self.lock:
                        z0 = encode(lanes)
                    for i, l in enumerate(lanes):
                        z_hist[l] = [z0[i]] * 4
                    connection.send(windows(lanes))
                elif name == "step":
                    lanes, actions, frames = rest
                    for l, a_, f in zip(lanes, actions, frames):
                        frames_hist[l] = frames_hist[l][-3:] + [torch.from_numpy(np.ascontiguousarray(f)).to(dev)]
                        played[l] = played[l][-2:] + [int(a_)]
                        steps[l] += 1
                    with self.lock:
                        zn = encode(lanes)
                    for i, l in enumerate(lanes):
                        z_hist[l] = z_hist[l][-3:] + [zn[i]]
                    connection.send(windows(lanes))
                elif name == "score":
                    lanes, plans = rest
                    t0 = time.time()
                    lz = torch.stack([torch.stack(z_hist[l][-4:]) for l in lanes])
                    lp = torch.tensor([played[l] for l in lanes], device=dev, dtype=torch.long)
                    lt = torch.tensor([steps[l] for l in lanes], device=dev)
                    lg = torch.tensor([g[l] for l in lanes], device=dev)
                    with self.lock, torch.no_grad():
                        sc = self.score_rows(lz, lp, lt, lg, plans)
                    self.stats["scored"] += len(plans); self.stats["seconds"] += time.time() - t0
                    connection.send(sc.float().cpu().numpy())
                else:
                    raise ValueError(f"unknown message {name}")
            except Exception as error:   # the agent loop raises it, so a broken world model stops the run
                connection.send(error)
                raise

    def serve(self) -> None:
        socket = self.a.socket
        socket.unlink(missing_ok=True)
        with Listener(str(socket), family="AF_UNIX", authkey=b"dreamer") as listener:
            (self.a.out / "ready").write_text("ready\n")
            print(json.dumps(dict(ready=True)), flush=True)
            while True:
                connection = listener.accept()
                threading.Thread(target=self.serve_rollout, args=(connection,), daemon=True).start()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dyn", required=True, help="Dreamer 4 dynamics checkpoint (phase 1)")
    p.add_argument("--heads", required=True, help="phase-2 checkpoint (train_heads.py / train_heads_on_dreams.py)")
    p.add_argument("--tok", required=True, help="Dreamer 4 tokenizer checkpoint (the SLM's)")
    p.add_argument("--critic", required=True, help="SLM checkpoint whose value head scores the imagined windows")
    p.add_argument("--extra-critics", default="")
    p.add_argument("--critic-pessimism", type=float, default=0.0)
    p.add_argument("--text-tokenizer", default="", help="override the critic checkpoint's text_tokenizer path")
    p.add_argument("--init", default="", help="override the critic checkpoint's init_checkpoint path")
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--K", type=int, default=1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--dream-samples", type=int, default=1)
    p.add_argument("--pessimism", type=float, default=0.0)
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--socket", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--selftest", action="store_true", help="one fake rollout (random frames, random plans), print the scores")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    service = Service(a)
    if a.selftest:
        class Conn:   # the messages a rollout sends, answers printed
            def __init__(self):
                rng = np.random.default_rng(0)
                fr = lambda n: [rng.integers(0, 255, (63, 63, 3), dtype=np.uint8) for _ in range(n)]
                plans = [list(rng.integers(0, 17, 16)) for _ in range(6 * 8)]
                self.q = [("first", fr(6), [service.tasks[0]] * 6), ("step", [0, 1, 2], [5, 5, 1], fr(3)),
                          ("score", [i // 8 for i in range(48)], plans), ("reset", [4], fr(1), [service.tasks[0]]),
                          ("score", [4] * 8, plans[:8])]
            def recv(self):
                if not self.q:
                    raise EOFError
                return self.q.pop(0)
            def send(self, x):
                if isinstance(x, Exception):
                    raise x
                print("reply", getattr(x, "shape", None), np.round(np.asarray(x).ravel()[:8], 3).tolist(), flush=True)
        t0 = time.time()
        service.serve_rollout(Conn())
        print(json.dumps(dict(selftest="ok", seconds=round(time.time() - t0, 1), **service.stats)), flush=True)
        return
    service.serve()


if __name__ == "__main__":
    main()
