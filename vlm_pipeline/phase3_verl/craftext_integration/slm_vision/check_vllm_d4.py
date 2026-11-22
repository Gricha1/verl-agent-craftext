"""vLLM rollout vs HF policy for the materialised Dreamer-latent SLM (materialize_d4_checkpoint.py): same prompt, same latent
window, greedy decoding; prints both answers and whether they are identical. Run in the modern veRL image:
    PYTHONPATH=<phase3_verl> python check_vllm_d4.py --model <dir> [--n 4]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

here = Path(__file__).resolve()
sys.path.insert(0, str(here.parents[3] / "phase4_world_model"))
from slm_plan_text import parse_plan, prompt_parts  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=176)
    a = p.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import craftext_integration.slm_vision.vllm_model  # noqa: F401  registers the architecture
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(a.model)
    pol = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=torch.float32).cuda().eval()
    cfg = pol.config
    pre, post = prompt_parts(cfg.crafter_tasks[0], cfg.crafter_plan)
    ids = tok.encode(pre, add_special_tokens=False) + list(cfg.crafter_visual_token_ids) + tok.encode(post, add_special_tokens=False)
    rng = np.random.default_rng(0)
    d = cfg.crafter_dreamer
    Z = [rng.uniform(-1, 1, d["window"] * d["n_lat"] * d["d_lat"]).astype(np.float32) for _ in range(a.n)]
    hf = []
    with torch.no_grad():
        for z in Z:
            cur = torch.tensor([ids], device="cuda")
            for _ in range(a.max_tokens):
                nxt = pol(input_ids=cur, pixel_values=torch.from_numpy(z)[None].cuda()).logits[:, -1].argmax(-1, keepdim=True)
                cur = torch.cat([cur, nxt], 1)
                if int(nxt) == 2:
                    break
            hf.append(cur[0, len(ids):].tolist())
    llm = LLM(model=a.model, trust_remote_code=True, dtype="float32", enforce_eager=True, gpu_memory_utilization=0.3,
              max_model_len=768, limit_mm_per_prompt={"image": 1})
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)
    outs = llm.generate([{"prompt_token_ids": ids, "multi_modal_data": {"image": z[:, None, None]}} for z in Z], sp)
    same = 0
    for z, h, o in zip(Z, hf, outs):
        v = list(o.outputs[0].token_ids)
        ht, vt = tok.decode(h, skip_special_tokens=True), tok.decode(v, skip_special_tokens=True)
        same += int(h == v)
        print("HF  ", len(h), parse_plan(ht), ht[:160].replace("\n", " "))
        print("vLLM", len(v), parse_plan(vt), vt[:160].replace("\n", " "))
    print(f"CHECK-VLLM identical greedy answers {same}/{len(Z)}", flush=True)


if __name__ == "__main__":
    main()
