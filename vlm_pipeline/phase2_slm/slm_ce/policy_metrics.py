"""Policy-level validation metrics of saved CE checkpoints over the 17 action strings.

The same quantities ``crafter_vision_slm --target teacher_distribution`` logs every epoch
(distillation loss, 17-action accuracy vs the teacher argmax, KL(teacher || student),
student / teacher entropy), for checkpoints of any run, on a data dir whose rows carry
``teacher_probs``.

usage: python policy_metrics.py <data_dir> <out.json> <checkpoint_dir> [<checkpoint_dir> ...]
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # vlm_pipeline/phase2_slm
from slm_ce.crafter_vision_slm import VisionEncoder, VisionSoftTokenCausalLM, action_candidates, action_log_probs

data, out = Path(sys.argv[1]), Path(sys.argv[2])
device = torch.device("cuda")
tok = AutoTokenizer.from_pretrained(data / "tokenizer")
rows = torch.load(data / "valid_tokens.pt", weights_only=False)
candidates = action_candidates(tok)
lengths = torch.tensor([len(c) for c in candidates], dtype=torch.float32, device=device)
results = []
for ckpt in map(Path, sys.argv[3:]):
    if (ckpt / "tokenizer.json").read_bytes() != (data / "tokenizer" / "tokenizer.json").read_bytes():
        raise ValueError(f"{ckpt} was trained with a different tokenizer")
    lm = AutoModelForCausalLM.from_pretrained(ckpt).to(device)
    vision = VisionEncoder(width=lm.config.hidden_size, visual_tokens=4).to(device)
    vision.load_state_dict(torch.load(ckpt / "vision_adapter.pt", map_location=device, weights_only=True))
    model = VisionSoftTokenCausalLM(lm, vision).eval()
    nll = norm = hits = kl = h_s = h_t = 0.0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for offset in range(0, len(rows), 8):
            seq, teacher = action_log_probs(model, rows[offset:offset + 8], candidates, tok.pad_token_id, device, data)
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
    row = dict(checkpoint=ckpt.name, epoch=metrics.get("epoch"), validation_distill_loss=nll / norm,
               validation_action_accuracy=hits / n, validation_kl_teacher_student=kl / n,
               validation_student_entropy=h_s / n, validation_teacher_entropy=h_t / n)
    print(json.dumps(row), flush=True)
    results.append(row)
out.write_text(json.dumps(results, indent=1) + "\n")
