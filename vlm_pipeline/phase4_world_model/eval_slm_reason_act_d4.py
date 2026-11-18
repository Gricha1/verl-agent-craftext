#!/usr/bin/env python3
"""Action of a train_slm_reason_act_d4.py checkpoint on the validation frames, two ways:

  free  the SLM writes its own reasoning (greedy) and then <action>; the action is parsed from the text;
  gt    the ground-truth (Gemma) reasoning is given: prompt + "<reasoning>\n{gt}\n</reasoning>\n<action>" and the 17 action names
        are scored, p(a) ∝ exp(sum of log-probabilities of the tokens of "{Name}</action>"); predicted = argmax.

Compared with the teacher's action. With --plan K checkpoints the gt mode scores the first plan element
("<plan>{Name}," / "<plan>{Name}</plan>" for the last step): only the first action is checked.

  python eval_slm_reason_act_d4.py --run <run dir> --checkpoint step_006000.pt --out <json> [same data flags as training]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train_slm_reason_act_d4 import EOS, NAMES, SYSTEM, SYSTEM_PLAN, SLMReasonAct, parse_action, parse_plan
from train_slm_recon_d4 import encode_split, load_split, windows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--checkpoint", default="latest.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    run = Path(a.run)
    state = torch.load(run / "run" / ("checkpoints" if a.checkpoint.startswith("step_") else "") / a.checkpoint,
                       map_location="cpu", weights_only=False)
    cfg = state["args"]
    sys.path.insert(0, f"{cfg['repo']}/vlm_pipeline/phase4_world_model")
    from train_slm_world_model import ACTIONS, load_episodes
    from dreamer4.models.tokenizer import FrozenTokenizer
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
    device = torch.device("cuda")
    plan = cfg.get("plan", 0)
    text = PreTrainedTokenizerFast(tokenizer_file=str(Path(cfg["text_tokenizer"]) / "tokenizer.json"))
    enc = lambda s: text.encode(s, add_special_tokens=False)
    lm = AutoModelForCausalLM.from_pretrained(cfg["init_checkpoint"], dtype=torch.float32)
    lm.resize_token_embeddings(len(text)); lm.tie_weights()
    model = SLMReasonAct(lm, cfg["readout"]).to(device)
    model.load_state_dict(state["model"]); model.eval()
    tasks = state["tasks"]
    tok = FrozenTokenizer(cfg["tok"], history=4, device=device)
    frames, starts, lengths = load_split(cfg["val"])
    z, _, first = encode_split(tok, frames, starts, lengths, device, Path(cfg["latent_cache"]) / "z_val.pt")
    episodes = load_episodes(Path(cfg["raw_data"]), None, None)
    reasoning = {}
    for split in ("train", "valid"):
        for line in open(Path(cfg["reasoning"]) / f"{split}.jsonl"):
            r = json.loads(line)
            m = re.match(r"<reasoning>(.*)</reasoning>\nACTION: (\S+)$", r["cot_target"], re.S)
            if m:
                reasoning[(int(r["source_episode_id"]), int(r["step"]))] = m.group(1).strip()
    man_path = Path(cfg["val"]) / ("manifest.json" if (Path(cfg["val"]) / "manifest.json").exists() else "wood_manifest.json")
    man = json.loads(man_path.read_text())
    rows = []   # (frame index, task index, teacher action, gt reasoning, last step of the episode)
    g = 0
    for key, n in zip(man["episode_ids"], man["lengths"]):
        ep = episodes[int(key)]
        for t, it in enumerate(ep):
            gt = reasoning.get((int(key), t))
            if gt is not None:
                rows.append((g, tasks.index(ep[0]["instruction"]), it["action"], gt, t == n - 1))
            g += 1
    if a.limit:
        rows = rows[:a.limit]
    system = SYSTEM_PLAN.format(plan) if plan else SYSTEM
    pre = torch.tensor([enc("System:\n" + system + "\nUser:\n")], device=device)
    posts = [enc(f"\nTask: {t}\nAssistant:\n") for t in tasks]
    Lp = max(map(len, posts))
    post_ids = torch.zeros(len(tasks), Lp, dtype=torch.long, device=device)
    post_mask = torch.zeros(len(tasks), Lp, dtype=torch.long, device=device)
    for i, ids in enumerate(posts):
        post_ids[i, Lp - len(ids):] = torch.tensor(ids); post_mask[i, Lp - len(ids):] = 1
    opener = "<plan>" if plan else "<action>"
    closers = [", ", "</plan>"] if plan else ["</action>"]

    def words(t):
        return re.findall(r"[a-z]+", t.lower())

    def f1(a, b):   # bag-of-words F1 of two texts
        ca, cb = collections.Counter(words(a)), collections.Counter(words(b))
        common = sum((ca & cb).values())
        return 0.0 if common == 0 else 2 * common / (sum(ca.values()) + sum(cb.values()))

    def reasoning_of(t):
        m = re.search(r"<reasoning>(.*?)</reasoning>", t, re.S)
        return m.group(1).strip() if m else ""

    free_ok = gt_ok = 0
    per_frame = []
    confusion_gt, confusion_free = collections.Counter(), collections.Counter()
    examples = []
    with torch.no_grad():
        for st in range(0, len(rows), a.batch):
            part = rows[st:st + a.batch]
            idx = torch.tensor([r[0] for r in part], device=device)
            gi = torch.tensor([r[1] for r in part], device=device)
            Z = windows(z, first, idx)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                gen = model.generate(Z, pre, post_ids[gi], post_mask[gi], max_tokens=80 + 6 * plan)
            free = []
            for row in gen.tolist():
                row = row[:row.index(EOS)] if EOS in row else row
                txt = text.decode(row)
                act = (parse_plan(txt) or [None])[0] if plan else parse_action(txt)
                free.append((act, txt))
            # ground-truth reasoning, the 17 names scored
            emb, mask, _ = model.prefix(Z, pre, post_ids[gi], post_mask[gi])
            scores = torch.full((len(part), len(NAMES)), -1e9, device=device)
            for b, r in enumerate(part):
                closer = closers[1] if (plan and r[4]) else closers[0]
                # whole strings, tokenized as in training; candidates share the tokens up to cp, scored from there on
                seqs = [enc(f"<reasoning>\n{r[3]}\n</reasoning>\n{opener}{name}{closer}") for name in NAMES]
                cp = 0
                while all(len(s) > cp for s in seqs) and len({s[cp] for s in seqs}) == 1:
                    cp += 1
                head = seqs[0][:cp]
                cands = [s[cp:] for s in seqs]
                W = max(map(len, seqs))
                ids = torch.zeros(len(seqs), W, dtype=torch.long, device=device)
                for i, s in enumerate(seqs):
                    ids[i, :len(s)] = torch.tensor(s)
                x = torch.cat([emb[b:b + 1].expand(len(seqs), -1, -1), model.embed(ids).float()], 1)
                m = torch.cat([mask[b:b + 1].expand(len(seqs), -1), (ids != 0).long() | (torch.arange(W, device=device) < len(head)).long()], 1)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    h = model.backbone(inputs_embeds=x, attention_mask=m).last_hidden_state
                P = emb.shape[1]
                logp = model.lm_head(h[:, P - 1:-1]).float().log_softmax(-1)
                tok_lp = logp.gather(-1, ids[..., None])[..., 0]
                for i, c in enumerate(cands):
                    scores[b, i] = tok_lp[i, len(head):len(head) + len(c)].sum()
            prob = scores.softmax(-1)
            pred = scores.argmax(-1).tolist()
            for b, r in enumerate(part):
                teacher = r[2]
                gt_ok += pred[b] == teacher; free_ok += free[b][0] == teacher
                confusion_gt[(NAMES[teacher], NAMES[pred[b]])] += 1
                mine = reasoning_of(free[b][1])
                per_frame.append(dict(frame=r[0], task=tasks[r[1]], teacher=NAMES[teacher], gt_reasoning=r[3], reasoning=mine,
                                      action=NAMES[free[b][0]] if free[b][0] is not None else None, gt_pred=NAMES[pred[b]],
                                      exact=mine == r[3], f1=f1(mine, r[3])))
                confusion_free[(NAMES[teacher], NAMES[free[b][0]] if free[b][0] is not None else "unparsed")] += 1
                if len(examples) < 12:
                    top = prob[b].topk(3)
                    examples.append(dict(frame=r[0], task=tasks[r[1]], teacher=NAMES[teacher], gt_reasoning=r[3],
                                         gt_pred=NAMES[pred[b]], gt_top3=[(NAMES[i], round(float(v), 3)) for v, i in zip(top.values, top.indices)],
                                         free_text=free[b][1]))
    n = len(rows)
    ok = [x for x in per_frame if x["action"] == x["teacher"]]; bad = [x for x in per_frame if x["action"] != x["teacher"]]
    res = dict(run=run.name, step=state["step"], frames=n, free_action_acc=free_ok / n, gt_reasoning_action_acc=gt_ok / n,
               reasoning_exact=float(np.mean([x["exact"] for x in per_frame])), reasoning_f1=float(np.mean([x["f1"] for x in per_frame])),
               reasoning_f1_when_action_right=float(np.mean([x["f1"] for x in ok])) if ok else None,
               reasoning_f1_when_action_wrong=float(np.mean([x["f1"] for x in bad])) if bad else None,
               distinct_reasoning=len({x["reasoning"] for x in per_frame}), distinct_gt_reasoning=len({x["gt_reasoning"] for x in per_frame}),
               teacher_actions=collections.Counter(NAMES[r[2]] for r in rows).most_common(),
               confusion_gt=[[k[0], k[1], v] for k, v in confusion_gt.most_common(30)],
               confusion_free=[[k[0], k[1], v] for k, v in confusion_free.most_common(30)], examples=examples)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    with open(Path(a.out).with_suffix(".frames.jsonl"), "w") as f:
        for x in per_frame:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(json.dumps({k: res[k] for k in ("run", "step", "frames", "free_action_acc", "gt_reasoning_action_acc", "reasoning_exact",
                                          "reasoning_f1", "reasoning_f1_when_action_right", "reasoning_f1_when_action_wrong",
                                          "distinct_reasoning", "distinct_gt_reasoning")}), flush=True)


if __name__ == "__main__":
    main()
