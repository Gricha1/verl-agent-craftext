#!/usr/bin/env python3
"""SLM over frozen Dreamer 4 latents, behaviour cloning of reasoning + action as text in the format of Alexey Skrynnik's
ZeroHero Crafter student (8.78 after SFT, 13.41 after PPO; branch crafter-memory-student-20261106, train_fast.py), plus the
reconstruction of x_t and the value of the task.

ZeroHero student (zerohero/rl/crafter_screen_policy.py memory_prompt_ids, crafter_qwen_student_data.rel_memory_target):
  prompt  "System:\n" + SYSTEM + "\nUser:\n" + <vision_start> 64 image tokens <vision_end> + "\n" + context + "\nAssistant:\n"
  target  "<reasoning>\n{reasoning}\n</reasoning>\n<action>{Action Name}</action>\n<memory>{memory}</memory>" + EOS
  loss    token cross-entropy on the target only (labels -100 on the prompt); AdamW lr 6e-4, betas (0.9, 0.95), weight decay
          0.1, warmup 500, cosine to 5% of the peak, batch 128, gradient clip 1.

Here, the same layout with our data:
  prompt  "System:\n" + SYSTEM + "\nUser:\n" + [adapter(z_{t-3..t}) 256 tokens ; 64 readout tokens] + "\nTask: " + g + "\nAssistant:\n"
  target  "<reasoning>\n{Gemma reasoning}\n</reasoning>\n<action>{Action Name}</action>" + EOS
          (no <memory>: the wood2200 PPO teacher has none; action names as in ZeroHero: Move West, Do, ...)
  heads   reconstruction from the readout tokens -> frozen Dreamer 4 decoder (MSE + 0.2 LPIPS, train_slm_recon_d4.py);
          value V(g, Z) on the last prompt position ("Assistant:\n"), MSE to G_t = gamma^(T-1-t) on success (add_value_targets).
  loss    L = sum_i L_i / sg(EMA_i) over reconstruction, text CE, value (each term divided by its running mean).
Frames without a reasoning label (episodes the teacher failed, not labelled) train reconstruction and value only.

--plan K: a plan of actions instead of one action (user, 2026-11-18): target
  "<reasoning>\n{reasoning}\n</reasoning>\n<plan>{a_t}, {a_t+1}, ..., {a_t+K-1}</plan>" + EOS, the teacher's next K actions of
  the episode (fewer at its end); the system text asks for the next K actions. Evaluation: share of greedy plans whose k-th
  action is the teacher's k-th next action, for every k, and the mean length of the matching prefix.
--accum M: the batch is split into M micro-batches whose gradients are summed (same update, less memory).
--dreamer-h DIR: h_train.pt / h_val.pt of cache_dreamer_h.py, the agent-token outputs h_t (n_agent x d) of a Dreamer 4 phase-2
  model (task as text, reward / value / instruction heads) on the real window ending at frame t. They enter the prompt as
  n_agent more tokens after the latents, e_h = MLP(h) + slot: the SLM reads the world model's own task-aware state.

Reasoning labels: rows of the phase-2 reasoning data dir of wood2200 (Gemma-4, label_reasoning.py --context;
reason_wood2200_h200_20261106/data/{train,valid}.jsonl), cot_target "<reasoning>R</reasoning>\nACTION: X".

  python train_slm_reason_act_d4.py --tok <tok> --init-checkpoint <initial_random> --text-tokenizer <dir with tokenizer.json> \
     --raw-data <wood2200 raw> --reasoning <dir with train.jsonl valid.jsonl> --train <wood2200_v3> --val <wood2200_d4/val> \
     --latent-cache <dir> --readout 64 --out <dir> [--from-run <train_slm_recon_d4 run dir>]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_slm_recon_d4 import OBJECTS, SLMRecon, decode, encode_split, load_split, windows

SYSTEM = ("You play Crafter from the game screen. Think about the best course of action step by step inside "
          "<reasoning></reasoning>, then give the action inside <action></action>")
SYSTEM_PLAN = ("You play Crafter from the game screen. Think about the best course of action step by step inside "
               "<reasoning></reasoning>, then give your next {} actions, comma-separated, inside <plan></plan>")
NAMES = ("Noop", "Move West", "Move East", "Move North", "Move South", "Do", "Sleep", "Place Stone", "Place Table",
         "Place Furnace", "Place Plant", "Make Wood Pickaxe", "Make Stone Pickaxe", "Make Iron Pickaxe",
         "Make Wood Sword", "Make Stone Sword", "Make Iron Sword")   # ZeroHero ACTIONS, the Craftax-Classic order
EOS = 2


def parse_action(text):
    m = re.search(r"<action>(.*?)</action>", text)
    return NAMES.index(m.group(1).strip()) if m and m.group(1).strip() in NAMES else None


def parse_plan(text):
    """List of action indices inside <plan></plan> (None for a name that is not an action), or None without the tags."""
    m = re.search(r"<plan>(.*?)</plan>", text, re.S)
    if m is None:
        return None
    return [NAMES.index(s.strip()) if s.strip() in NAMES else None for s in m.group(1).split(",")]


class SLMReasonAct(SLMRecon):
    def __init__(self, lm, readout, n_h=0, d_h=0, **kw):
        super().__init__(lm, readout, **kw)
        d = lm.config.hidden_size
        self.embed = lm.get_input_embeddings()
        self.lm_head = lm.lm_head
        self.value_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.n_h = n_h
        if n_h:   # Dreamer 4 agent-token outputs h_t as extra prompt tokens (--dreamer-h)
            self.h_adapter = nn.Sequential(nn.Linear(d_h, d), nn.GELU(), nn.Linear(d, d))
            self.h_slot = nn.Parameter(torch.randn(n_h, d) * .02)

    def prefix(self, Z, pre, post, post_mask, Hd=None):
        """Embeddings of the prompt: pre text, latents (+ Dreamer h tokens) + readouts, post text (task, padded)
        -> (embeds, mask, readout slice)."""
        B = Z.shape[0]
        e = self.adapter(Z) + self.slot + self.index
        parts = [self.embed(pre).expand(B, -1, -1), e.reshape(B, -1, e.shape[-1])]
        if self.n_h:
            parts.append(self.h_adapter(Hd.float()) + self.h_slot)
        parts += [self.readout.expand(B, -1, -1), self.embed(post)]
        lengths = [p.shape[1] for p in parts]
        mask = torch.cat([torch.ones(B, sum(lengths[:-1]), dtype=torch.long, device=Z.device), post_mask], 1)
        start = sum(lengths[:-2])
        return torch.cat([p.float() for p in parts], 1), mask, slice(start, start + self.R)

    def recon(self, h, sl):
        zhat = torch.tanh(self.head(h[:, sl]))
        B = h.shape[0]
        if self.R == 1:
            return zhat.view(B, self.w, self.n_lat, self.d_lat)
        return zhat.view(B, self.n_lat, self.w, self.d_lat).transpose(1, 2)

    def forward(self, Z, pre, post, post_mask, target, target_mask, Hd=None):
        emb, mask, sl = self.prefix(Z, pre, post, post_mask, Hd)
        P = emb.shape[1]
        x = torch.cat([emb, self.embed(target).float()], 1)
        mask = torch.cat([mask, target_mask], 1)
        h = self.backbone(inputs_embeds=x, attention_mask=mask).last_hidden_state
        value = self.value_head(h[:, P - 1]).squeeze(-1).float()   # last prompt position, "Assistant:\n"
        logits = self.lm_head(h[:, P - 1:-1]).float()             # predicts target[0..L-1]
        return self.recon(h, sl), logits, value

    @torch.no_grad()
    def generate(self, Z, pre, post, post_mask, max_tokens=64, Hd=None):
        emb, mask, _ = self.prefix(Z, pre, post, post_mask, Hd)
        out = self.backbone(inputs_embeds=emb, attention_mask=mask, use_cache=True)
        past, h = out.past_key_values, out.last_hidden_state[:, -1]
        B = Z.shape[0]
        done = torch.zeros(B, dtype=torch.bool, device=Z.device)
        written = []
        for _ in range(max_tokens):
            tok = self.lm_head(h).argmax(-1)
            tok = torch.where(done, torch.zeros_like(tok), tok)
            written.append(tok)
            done |= tok == EOS
            if bool(done.all()):
                break
            mask = torch.cat([mask, torch.ones(B, 1, dtype=mask.dtype, device=mask.device)], 1)
            out = self.backbone(inputs_embeds=self.embed(tok[:, None]).float(), attention_mask=mask, past_key_values=past,
                                use_cache=True)
            past, h = out.past_key_values, out.last_hidden_state[:, -1]
        return torch.stack(written, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tok", required=True)
    p.add_argument("--init-checkpoint", required=True, help="initial_random of the SLM (architecture and random weights)")
    p.add_argument("--text-tokenizer", required=True, help="dir with the tokenizer of the reasoning data (wood2200 SLM)")
    p.add_argument("--raw-data", required=True)
    p.add_argument("--reasoning", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--from-run", default="")
    p.add_argument("--repo", default="/w/repo")
    p.add_argument("--readout", type=int, default=64)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--final-lr-fraction", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--val-samples", type=int, default=2048)
    p.add_argument("--generate-samples", type=int, default=512)
    p.add_argument("--lpips-weight", type=float, default=0.2)
    p.add_argument("--value-gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=7950)
    p.add_argument("--latent-cache", default="")
    p.add_argument("--plan", type=int, default=0, help="K > 0: target <plan> with the teacher's next K actions instead of <action>")
    p.add_argument("--accum", type=int, default=1, help="micro-batches per optimizer step (batch / accum examples each)")
    p.add_argument("--dreamer-h", default="", help="dir with h_train.pt / h_val.pt (cache_dreamer_h.py): Dreamer 4 h_t tokens")
    a = p.parse_args()
    sys.path.insert(0, f"{a.repo}/vlm_pipeline/phase4_world_model")
    from imagine_eval import classify_tiles, tile_prototypes
    from train_slm_world_model import ACTIONS, add_value_targets, load_episodes
    from dreamer4.models.tokenizer import FrozenTokenizer
    from dreamer4.train.perceptual import LpipsPerceptual
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device("cuda")
    tok = FrozenTokenizer(a.tok, history=4, device=device)
    lpips = LpipsPerceptual(net="alex", up=128).to(device)
    protos, index, names = tile_prototypes()
    protos, index, names = torch.from_numpy(protos).to(device), torch.from_numpy(index).to(device), list(names)
    text = PreTrainedTokenizerFast(tokenizer_file=str(Path(a.text_tokenizer) / "tokenizer.json"))
    enc = lambda s: text.encode(s, add_special_tokens=False)
    unk = text.convert_tokens_to_ids(text.unk_token) if text.unk_token else None

    episodes = load_episodes(Path(a.raw_data), None, None)
    add_value_targets(episodes, a.value_gamma)
    tasks = sorted({ep[0]["instruction"] for ep in episodes.values()})
    reasoning, mismatch = {}, 0
    for split in ("train", "valid"):
        for line in open(Path(a.reasoning) / f"{split}.jsonl"):
            r = json.loads(line)
            m = re.match(r"<reasoning>(.*)</reasoning>\nACTION: (\S+)$", r["cot_target"], re.S)
            key = (int(r["source_episode_id"]), int(r["step"]))
            if m is None or m.group(2) != r["action"]:
                mismatch += 1; continue
            reasoning[key] = (m.group(1).strip(), ACTIONS.index(r["action"]))
    print("reasoning labels", len(reasoning), "unparsed", mismatch, flush=True)

    system = SYSTEM_PLAN.format(a.plan) if a.plan else SYSTEM
    pre = torch.tensor([enc("System:\n" + system + "\nUser:\n")], device=device)
    posts = [enc(f"\nTask: {t}\nAssistant:\n") for t in tasks]
    Lp = max(map(len, posts))
    post_ids = torch.zeros(len(tasks), Lp, dtype=torch.long, device=device)
    post_mask = torch.zeros(len(tasks), Lp, dtype=torch.long, device=device)
    for i, ids in enumerate(posts):   # right-aligned so that "Assistant:\n" is always the last prompt position
        post_ids[i, Lp - len(ids):] = torch.tensor(ids); post_mask[i, Lp - len(ids):] = 1

    def split_data(root, tag):
        man_path = Path(root) / ("manifest.json" if (Path(root) / "manifest.json").exists() else "wood_manifest.json")
        man = json.loads(man_path.read_text())
        task, value, vmask, targets, has, teacher = [], [], [], [], [], []
        for key, n in zip(man["episode_ids"], man["lengths"]):
            ep = episodes[int(key)]
            assert len(ep) == n
            for t, it in enumerate(ep):
                task.append(tasks.index(ep[0]["instruction"])); value.append(it["value"]); vmask.append(it["value_mask"])
                teacher.append(it["action"])
                lab = reasoning.get((int(key), t))
                if lab is not None and lab[1] == it["action"]:
                    if a.plan:   # the teacher's next K actions of this episode (fewer near its end)
                        plan = ", ".join(NAMES[x["action"]] for x in ep[t:t + a.plan])
                        ids = enc(f"<reasoning>\n{lab[0]}\n</reasoning>\n<plan>{plan}</plan>") + [EOS]
                    else:
                        ids = enc(f"<reasoning>\n{lab[0]}\n</reasoning>\n<action>{NAMES[it['action']]}</action>") + [EOS]
                    targets.append(ids); has.append(1)
                else:
                    targets.append([EOS]); has.append(0)
        T = max(map(len, targets))
        tgt = torch.zeros(len(targets), T, dtype=torch.long)
        tmask = torch.zeros(len(targets), T, dtype=torch.long)
        for i, ids in enumerate(targets):
            tgt[i, :len(ids)] = torch.tensor(ids); tmask[i, :len(ids)] = 1
        if unk is not None:
            print(tag, "unknown tokens in targets", int((tgt == unk).sum()), flush=True)
        lens = tmask.sum(1)[torch.tensor(has, dtype=torch.bool)].float()
        print(tag, "frames", len(task), "with reasoning", sum(has), "target tokens mean/max", round(float(lens.mean()), 1),
              int(lens.max()), flush=True)
        return [x.to(device) for x in (torch.tensor(task), torch.tensor(value, dtype=torch.float32),
                                       torch.tensor(vmask, dtype=torch.float32), tgt, tmask,
                                       torch.tensor(has, dtype=torch.float32), torch.tensor(teacher))]

    data = {}
    for tag, root in (("train", a.train), ("val", a.val)):
        frames, starts, lengths = load_split(root)
        cache = Path(a.latent_cache) / f"z_{tag}.pt" if a.latent_cache else None
        z, x, first = encode_split(tok, frames, starts, lengths, device, cache)
        data[tag] = (z, x, first, *split_data(root, tag))
        if tag == "val":   # end (exclusive) of each frame's episode, for the teacher's next actions
            vend = torch.cat([torch.full((n,), s + n, dtype=torch.long) for s, n in
                              zip(np.concatenate(([0], np.cumsum(lengths)[:-1])), lengths)]).to(device)
    zt, xt, ft, gt, vt, mt, Tt, TMt, Ht, At = data["train"]
    zv, xv, fv, gv, vv, mv, Tv, TMv, Hv, Av = data["val"]
    if a.dreamer_h:   # [N, n_agent, d] per frame, same order as the latents
        hT = torch.load(Path(a.dreamer_h) / "h_train.pt").to(device); hV = torch.load(Path(a.dreamer_h) / "h_val.pt").to(device)
        assert len(hT) == len(zt) and len(hV) == len(zv), (hT.shape, zt.shape, hV.shape, zv.shape)
        n_h, d_h = hT.shape[1:]
        print("Dreamer h tokens", tuple(hT.shape), flush=True)
    else:
        hT = hV = None; n_h = d_h = 0
    HD = lambda h, idx: None if h is None else h[idx]
    gen = torch.Generator().manual_seed(a.seed)
    val_idx = torch.randperm(len(zv), generator=gen)[:a.val_samples].to(device)
    vmean = float((vt * mt).sum() / mt.sum())

    lm = AutoModelForCausalLM.from_pretrained(a.init_checkpoint, dtype=torch.float32)
    lm.resize_token_embeddings(len(text))
    nn.init.normal_(lm.get_input_embeddings().weight, std=lm.config.initializer_range)   # new vocabulary, random rows
    lm.tie_weights()
    model = SLMReasonAct(lm, a.readout, n_h=n_h, d_h=d_h).to(device)
    if a.from_run:
        state = torch.load(Path(a.from_run) / "latest.pt", map_location="cpu", weights_only=False)
        sd = {k: v for k, v in state["model"].items() if "embed_tokens" not in k}   # its embeddings were never used
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print("fine-tune from", a.from_run, "step", state["step"], "not loaded:", sorted({k.split(".")[0] for k in missing}),
              "unexpected:", unexpected, flush=True)
    n_par = sum(v.numel() for v in model.parameters())
    print(f"parameters {n_par/1e6:.2f}M, vocabulary {len(text)}, prompt {pre.shape[1]}+{256 + n_h + a.readout}+{Lp} tokens", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    warm = min(a.warmup, max(1, a.steps // 20))
    lr_at = lambda s: a.lr * ((s + 1) / warm if s < warm else
                              a.final_lr_fraction + (1 - a.final_lr_fraction) * .5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm))))

    def to_img(x):
        return x.permute(0, 3, 1, 2).float() / 255

    def batch_inputs(idx, T, TM, g):
        w = int(TM[idx].sum(1).max())
        return post_ids[g[idx]], post_mask[g[idx]], T[idx, :w], TM[idx, :w]

    @torch.no_grad()
    def evaluate(step):
        model.eval()
        s = dict(mse=0.0, lpips=0.0, nll=0.0, tok=0.0, tok_acc=0.0); n = 0
        vse = vn = vconst = 0.0
        hit = {o: 0 for o in OBJECTS}; real_n = {o: 0 for o in OBJECTS}
        for st in range(0, len(val_idx), 128):
            idx = val_idx[st:st + 128]
            Z = windows(zv, fv, idx); x = to_img(xv[idx.cpu()].to(device))
            po, pm, tg, tm = batch_inputs(idx, Tv, TMv, gv)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                zhat, logits, value = model(Z, pre, po, pm, tg, tm, Hd=HD(hV, idx))
                xh = decode(tok, zhat.float()).float()
            b = len(idx); n += b
            s["mse"] += F.mse_loss(xh, x).item() * b; s["lpips"] += lpips(xh, x).item() * b
            lm_mask = tm.bool() & Hv[idx].bool()[:, None]
            ce = F.cross_entropy(logits.transpose(1, 2), tg, reduction="none")
            s["nll"] += float((ce * lm_mask).sum()); s["tok"] += float(lm_mask.sum())
            s["tok_acc"] += float(((logits.argmax(-1) == tg) & lm_mask).sum())
            m = mv[idx]
            vse += float(((value - vv[idx]) ** 2 * m).sum()); vconst += float(((vmean - vv[idx]) ** 2 * m).sum()); vn += float(m.sum())
            real, pred = classify_tiles(x, protos, index)[0], classify_tiles(xh, protos, index)[0]
            for o in OBJECTS:
                c = names.index(o); r = real == c
                real_n[o] += int(r.sum()); hit[o] += int(((pred == c) & r).sum())
        res = dict(mse=s["mse"] / n, lpips=s["lpips"] / n, text_nll=s["nll"] / max(s["tok"], 1), text_token_acc=s["tok_acc"] / max(s["tok"], 1))
        res["psnr"] = float(10 * np.log10(1 / max(res["mse"], 1e-10)))
        res["value_mse"] = vse / max(vn, 1); res["value_mse_const"] = vconst / max(vn, 1)
        for o in OBJECTS:
            res[f"slm_recall_{o}"] = hit[o] / max(real_n[o], 1)
        # greedy generation: is the action inside <action></action> the teacher's?
        gi = val_idx[:a.generate_samples]
        ok = parsed = 0; examples = []
        K = max(a.plan, 1)
        pos_hit, pos_n, prefix = np.zeros(K), np.zeros(K), []
        for st in range(0, len(gi), 128):
            idx = gi[st:st + 128]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ids = model.generate(windows(zv, fv, idx), pre, post_ids[gv[idx]], post_mask[gv[idx]],
                                     max_tokens=80 + 6 * a.plan, Hd=HD(hV, idx))
            for i, row in enumerate(ids.tolist()):
                row = row[:row.index(EOS)] if EOS in row else row
                txt = text.decode(row)
                j = int(idx[i])
                teacher = [int(Av[j + k]) for k in range(K) if j + k < int(vend[j])]
                if a.plan:
                    plan = parse_plan(txt)
                    parsed += plan is not None
                    plan = plan or []
                    good = 0
                    for k, want in enumerate(teacher):
                        pos_n[k] += 1
                        same = k < len(plan) and plan[k] == want
                        pos_hit[k] += same
                        good = good + 1 if same and good == k else good
                    prefix.append(good)
                    ok += bool(plan) and plan[0] == teacher[0]
                else:
                    act = parse_action(txt)
                    parsed += act is not None; ok += act == teacher[0]
                if len(examples) < 6:
                    examples.append(dict(frame=j, task=tasks[int(gv[j])], teacher=", ".join(NAMES[t] for t in teacher), text=txt))
        res["action_acc_greedy"] = ok / len(gi); res["action_parsed"] = parsed / len(gi)
        if a.plan:   # share of plans whose k-th action is the teacher's k-th next action; matching prefix length
            res["plan_acc_by_position"] = (pos_hit / np.maximum(pos_n, 1)).round(4).tolist()
            res["plan_matching_prefix"] = float(np.mean(prefix))
        res["examples"] = examples; res["real_tiles"] = real_n; res["step"] = step
        model.train()
        return res

    log = open(out / "metrics.jsonl", "a")
    if a.from_run:
        res = evaluate(0); log.write(json.dumps(res) + "\n"); log.flush(); print("[val]", json.dumps(res)[:1500], flush=True)
    ema = {}
    step, t0 = 0, time.time()
    run = {k: 0.0 for k in ("rec", "text", "val")}
    model.train()
    best = None
    while step < a.steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):   # a.batch examples per optimizer step, in a.accum micro-batches
            idx = torch.randint(0, len(zt), (a.batch // a.accum,), device=device)
            Z = windows(zt, ft, idx)
            x = to_img(xt[idx.cpu()].to(device, non_blocking=True))
            po, pm, tg, tm = batch_inputs(idx, Tt, TMt, gt)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                zhat, logits, value = model(Z, pre, po, pm, tg, tm, Hd=HD(hT, idx))
                xh = decode(tok, zhat).float()
            lm_mask = (tm.bool() & Ht[idx].bool()[:, None]).float()
            ce = F.cross_entropy(logits.transpose(1, 2), tg, reduction="none")
            terms = dict(rec=F.mse_loss(xh, x) + a.lpips_weight * lpips(xh, x),
                         text=(ce * lm_mask).sum() / lm_mask.sum().clamp_min(1),
                         val=((value - vt[idx]) ** 2 * mt[idx]).sum() / mt[idx].sum().clamp_min(1))
            loss = 0.0
            for k, v in terms.items():
                ema[k] = v.item() if k not in ema else .99 * ema[k] + .01 * v.item()
                loss = loss + v / max(ema[k], 1e-8)
            (loss / a.accum).backward()
            for k in terms:
                run[k] += terms[k].item() / a.accum
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); step += 1
        if step % 100 == 0:
            print(f"step {step} rec {run['rec']/100:.5f} text_ce {run['text']/100:.4f} val_mse {run['val']/100:.5f} "
                  f"lr {lr_at(step):.2e} gnorm {gn.item():.2f} {step/(time.time()-t0):.2f} it/s", flush=True)
            run = {k: 0.0 for k in run}
        if step % a.eval_every == 0 or step == a.steps:
            res = evaluate(step)
            log.write(json.dumps(res) + "\n"); log.flush()
            print("[val]", json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in res.items()})[:1500], flush=True)
            state = dict(model=model.state_dict(), step=step, args=vars(a), val=res, tasks=tasks)
            torch.save(state, out / "latest.pt")
            (out / "checkpoints").mkdir(exist_ok=True)
            torch.save(state, out / "checkpoints" / f"step_{step:06d}.pt")
            if best is None or res["text_nll"] < best:
                best = res["text_nll"]; torch.save(state, out / "best.pt")
    print("TRAIN_DONE", step, flush=True)


if __name__ == "__main__":
    main()
