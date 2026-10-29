"""Train/val token accuracy, loss and whole-action accuracy of saved CE checkpoints.

Same token metric as ``slm_ce.crafter_vision_slm`` validation (argmax of the text logits vs labels);
``action_accuracy`` counts a row correct only if every target token is correct.
"""
import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # vlm_pipeline/phase2_slm
from slm_ce.crafter_vision_slm import VisionEncoder, VisionSoftTokenCausalLM, batch_tensors

run, out = Path(sys.argv[1]), Path(sys.argv[2])
data = run / "data"
tok = AutoTokenizer.from_pretrained(data / "tokenizer")
train_rows = torch.load(data / "train_tokens.pt", weights_only=False)
valid_rows = torch.load(data / "valid_tokens.pt", weights_only=False)
random.Random(0).shuffle(train_rows)
train_rows = train_rows[: len(valid_rows)]  # same-size random train sample
device = torch.device("cuda")


def evaluate(model, rows):
    total, tokens, correct, rows_ok = 0.0, 0, 0, 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for offset in range(0, len(rows), 32):
            inputs = batch_tensors(rows[offset:offset + 32], "action", tok.pad_token_id, device, data)
            output = model(**inputs, use_cache=False)
            labels = inputs["labels"][:, 1:]
            mask = labels.ne(-100)
            count = mask.sum().item()
            total += output.loss.item() * count
            tokens += count
            hit = (output.logits[:, 4:-1].argmax(-1) == labels) | ~mask
            correct += (hit & mask).sum().item()
            rows_ok += hit.all(dim=1).sum().item()
    return dict(loss=total / tokens, token_accuracy=correct / tokens, action_accuracy=rows_ok / len(rows))


results = []
for name, epoch in (("initial_random", 0), ("epoch_02", 2), ("epoch_04", 4), ("epoch_08", 8),
                    ("best_validation", None), ("epoch_12", 12), ("epoch_48", 48)):
    ckpt = run / ("initial_random" if name == "initial_random" else f"run_5m_vision_action/{name}")
    lm = AutoModelForCausalLM.from_pretrained(ckpt).to(device)
    vision = VisionEncoder(width=lm.config.hidden_size, visual_tokens=4).to(device)
    vision.load_state_dict(torch.load(ckpt / "vision_adapter.pt", map_location=device, weights_only=True))
    model = VisionSoftTokenCausalLM(lm, vision).eval()
    if epoch is None:
        epoch = json.loads((ckpt / "checkpoint_complete.json").read_text())["epoch"]
    row = dict(checkpoint=name, epoch=epoch, train=evaluate(model, train_rows), val=evaluate(model, valid_rows))
    print(json.dumps(row), flush=True)
    results.append(row)
out.write_text(json.dumps(results, indent=1) + "\n")
