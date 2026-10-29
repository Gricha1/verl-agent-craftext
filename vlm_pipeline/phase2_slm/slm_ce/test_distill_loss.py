"""Check of the teacher-distribution CE on real wood rows and a random 5M model.

1. one-hot teacher_probs on the executed action -> loss equals the argmax CE (model(...).loss);
2. candidate tokenisation reproduces each row's stored ``action`` tokens;
3. a real teacher distribution gives a finite loss and gradients reach both LM and CNN.
"""
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # vlm_pipeline/phase2_slm
from slm_ce.crafter_slm import build_model
from slm_ce.crafter_vision_slm import (VisionEncoder, VisionSoftTokenCausalLM, action_candidates,
                                         action_log_probs, batch_tensors)

data = Path(sys.argv[1])
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(data / "tokenizer")
rows = torch.load(data / "valid_tokens.pt", weights_only=False)[:16]
device = torch.device("cuda")
lm = build_model(tok, "5m").to(device)
vision = VisionEncoder(width=lm.config.hidden_size, visual_tokens=4).to(device)
model = VisionSoftTokenCausalLM(lm, vision)
cands = action_candidates(tok)
lengths = torch.tensor([len(c) for c in cands], dtype=torch.float32, device=device)

index = [cands.index(r["action"]) for r in rows]  # (2): raises if a stored target is not a candidate
print("candidate tokens match stored targets for", len(index), "rows")
one_hot = [dict(r, teacher_probs=[1.0 if k == i else 0.0 for k in range(len(cands))]) for r, i in zip(rows, index)]

model.eval()
with torch.no_grad():
    seq, teacher = action_log_probs(model, one_hot, cands, tok.pad_token_id, device, data)
    distill = (-(teacher * seq).sum() / (teacher * lengths).sum()).item()
    argmax_ce = model(**batch_tensors(rows, "action", tok.pad_token_id, device, data), use_cache=False).loss.item()
print(f"(1) one-hot distillation {distill:.6f} vs argmax CE {argmax_ce:.6f}")
assert abs(distill - argmax_ce) < 1e-4 * max(1.0, argmax_ce), "one-hot distillation must equal argmax CE"

model.train()
seq, teacher = action_log_probs(model, rows, cands, tok.pad_token_id, device, data)
loss = -(teacher * seq).sum() / (teacher * lengths).sum()
loss.backward()
lm_grad = sum(p.grad.abs().sum().item() for p in lm.parameters() if p.grad is not None)
cnn_grad = sum(p.grad.abs().sum().item() for p in vision.parameters() if p.grad is not None)
print(f"(3) teacher-distribution loss {loss.item():.4f}, grad |LM| {lm_grad:.3g}, |CNN| {cnn_grad:.3g}, "
      f"teacher max-prob mean {teacher.max(-1).values.mean().item():.3f}")
assert torch.isfinite(loss) and lm_grad > 0 and cnn_grad > 0
print("OK")
