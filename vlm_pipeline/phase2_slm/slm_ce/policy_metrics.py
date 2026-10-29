"""Policy-level metrics of saved CE checkpoints over the 17 action strings, on train or val rows.

The same quantities ``crafter_vision_slm --target teacher_distribution`` logs every epoch on val
(distillation loss, 17-action accuracy vs the teacher argmax, KL(teacher || student),
student / teacher entropy), for checkpoints of any run, on a data dir whose rows carry
``teacher_probs``. ``--split train`` scores a fixed random train sample (seed 0) of the val size,
so train and val curves are directly comparable.

usage: python policy_metrics.py --data DIR --out OUT.json [--split valid|train] CKPT [CKPT ...]
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # vlm_pipeline/phase2_slm
from slm_ce.crafter_vision_slm import VisionEncoder, VisionSoftTokenCausalLM, action_candidates, action_log_probs

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--split", choices=["valid", "train"], default="valid")
parser.add_argument("checkpoints", type=Path, nargs="+")
args = parser.parse_args()

device = torch.device("cuda")
tok = AutoTokenizer.from_pretrained(args.data / "tokenizer")
rows = torch.load(args.data / f"{args.split}_tokens.pt", weights_only=False)
if args.split == "train":
    sample_size = len(torch.load(args.data / "valid_tokens.pt", weights_only=False))
    random.Random(0).shuffle(rows)
    rows = rows[:sample_size]
candidates = action_candidates(tok)
lengths = torch.tensor([len(c) for c in candidates], dtype=torch.float32, device=device)
results = []
for ckpt in args.checkpoints:
    if (ckpt / "tokenizer.json").read_bytes() != (args.data / "tokenizer" / "tokenizer.json").read_bytes():
        raise ValueError(f"{ckpt} was trained with a different tokenizer")
    lm = AutoModelForCausalLM.from_pretrained(ckpt).to(device)
    vision = VisionEncoder(width=lm.config.hidden_size, visual_tokens=4).to(device)
    vision.load_state_dict(torch.load(ckpt / "vision_adapter.pt", map_location=device, weights_only=True))
    model = VisionSoftTokenCausalLM(lm, vision).eval()
    nll = norm = hits = kl = h_s = h_t = 0.0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for offset in range(0, len(rows), 8):
            seq, teacher = action_log_probs(model, rows[offset:offset + 8], candidates, tok.pad_token_id, device, args.data)
            nll += -(teacher * seq).sum().item()
            norm += (teacher * lengths).sum().item()
            student = seq.log_softmax(-1)
            hits += (seq.argmax(-1) == teacher.argmax(-1)).sum().item()
            t_log = teacher.clamp_min(1e-12).log()
            kl += (teacher * (t_log - student)).sum().item()
            h_s -= (student.exp() * student).sum().item()
            h_t -= (teacher * t_log).sum().item()
    n = len(rows)
    metrics = json.loads((ckpt / "checkpoint_complete.json").read_text())
    row = dict(checkpoint=ckpt.name, epoch=metrics.get("epoch"), split=args.split, rows=n,
               distill_loss=nll / norm, action_accuracy=hits / n, kl_teacher_student=kl / n,
               student_entropy=h_s / n, teacher_entropy=h_t / n)
    print(json.dumps(row), flush=True)
    results.append(row)
args.out.write_text(json.dumps(results, indent=1) + "\n")
