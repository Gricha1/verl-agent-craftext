"""Plans of the Stage-1 SLM on real Crafter states, three ways (why does the veRL smoke parse only a part of its plans?):
  ref   SLMReasonAct from the training checkpoint, latents from dreamer_service's encoder (eval_slm_d4_env.py way)
  pol   the materialised veRL plugin (HF), same latents
Greedy and top-k sampled answers, parse rate and length.
    python check_d4_plan_real.py --model <materialised> --ckpt <step.pt> --tok T --dyn D --heads H --critic C [--steps 30]
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--ckpt", required=True)
    p.add_argument("--tok", required=True); p.add_argument("--dyn", required=True); p.add_argument("--heads", required=True)
    p.add_argument("--critic", required=True); p.add_argument("--steps", type=int, default=30); p.add_argument("--samples", type=int, default=8)
    a = p.parse_args()
    import crafter
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
    from dreamer_service import Service
    from slm_plan_text import parse_plan, prompt_parts
    from train_slm_reason_act_d4 import SLMReasonAct
    svc = Service(SimpleNamespace(dyn=a.dyn, heads=a.heads, tok=a.tok, critic=a.critic, extra_critics="", critic_pessimism=0.0,
                                  text_tokenizer="", init="", horizon=8, K=1, gamma=0.99, dream_samples=1, pessimism=0.0,
                                  chunk=1024, seed=0))
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    text = PreTrainedTokenizerFast(tokenizer_file=str(Path(st["args"]["text_tokenizer"]) / "tokenizer.json"))
    lm = AutoModelForCausalLM.from_pretrained(st["args"]["init_checkpoint"], dtype=torch.float32)
    lm.resize_token_embeddings(len(text)); lm.tie_weights()
    ref = SLMReasonAct(lm, st["args"]["readout"]).cuda().eval(); ref.load_state_dict(st["model"])
    pol = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=torch.float32).cuda().eval()
    from transformers import AutoTokenizer
    ptok = AutoTokenizer.from_pretrained(a.model)
    pre_s, post_s = prompt_parts(st["tasks"][0], st["args"]["plan"])
    pre = torch.tensor([text.encode(pre_s, add_special_tokens=False)]).cuda()
    post = torch.tensor([text.encode(post_s, add_special_tokens=False)]).cuda()
    ids = ptok.encode(pre_s, add_special_tokens=False) + list(pol.config.crafter_visual_token_ids) + ptok.encode(post_s, add_special_tokens=False)
    env = crafter.Env(size=(64, 64), seed=150000000, length=10 ** 9)
    frames = [np.ascontiguousarray(env.reset()[:63, :63])]
    stats = {k: [0, 0, 0] for k in ("ref_greedy", "ref_topk", "pol_greedy", "pol_topk")}   # parsed, total, tokens
    rng = np.random.default_rng(0)
    for t in range(a.steps):
        win = torch.stack([torch.from_numpy(f).cuda() for f in frames[-4:]])[None]
        with torch.no_grad():
            z = svc.tok.encode_frames(win)[:, -1].float()
        if t == 0:
            zs = [z[0]] * 4
        else:
            zs = zs[-3:] + [z[0]]
        Z = torch.stack(zs)[None]                                         # [1, 4, 64, 16]
        S = a.samples
        with torch.no_grad():
            for mode in ("greedy", "topk"):
                n = 1 if mode == "greedy" else S
                Zn = Z.expand(n, -1, -1, -1)
                # ref: SLMReasonAct generate with optional top-k sampling
                emb, mask, _ = ref.prefix(Zn, pre, post.expand(n, -1), torch.ones(n, post.shape[1], dtype=torch.long).cuda())
                o = ref.backbone(inputs_embeds=emb, attention_mask=mask, use_cache=True)
                past, h = o.past_key_values, o.last_hidden_state[:, -1]
                done = torch.zeros(n, dtype=torch.bool).cuda(); out = []
                for _ in range(176):
                    lg = ref.lm_head(h).float()
                    if mode == "greedy":
                        tk = lg.argmax(-1)
                    else:
                        kth = lg.topk(20, -1).values[:, -1:]; tk = torch.multinomial(lg.masked_fill(lg < kth, -1e9).softmax(-1), 1)[:, 0]
                    tk = torch.where(done, torch.zeros_like(tk), tk); out.append(tk); done |= tk == 2
                    if bool(done.all()):
                        break
                    mask = torch.cat([mask, torch.ones(n, 1, dtype=mask.dtype).cuda()], 1)
                    o = ref.backbone(inputs_embeds=ref.embed(tk[:, None]).float(), attention_mask=mask, past_key_values=past, use_cache=True)
                    past, h = o.past_key_values, o.last_hidden_state[:, -1]
                for row in torch.stack(out, 1).tolist():
                    row = row[:row.index(2)] if 2 in row else row
                    s = stats[f"ref_{mode}"]; s[0] += int(bool(parse_plan(text.decode(row)))); s[1] += 1; s[2] += len(row)
                # pol: HF plugin, full forward each step
                cur = torch.tensor([ids] * n).cuda(); pv = Z.reshape(1, -1).expand(n, -1)
                fin = torch.zeros(n, dtype=torch.bool).cuda(); gen = []
                for _ in range(176):
                    lg = pol(input_ids=cur, pixel_values=pv).logits[:, -1].float()
                    if mode == "greedy":
                        tk = lg.argmax(-1)
                    else:
                        kth = lg.topk(20, -1).values[:, -1:]; tk = torch.multinomial(lg.masked_fill(lg < kth, -1e9).softmax(-1), 1)[:, 0]
                    tk = torch.where(fin, torch.zeros_like(tk), tk); gen.append(tk); fin |= tk == 2
                    cur = torch.cat([cur, tk[:, None]], 1)
                    if bool(fin.all()):
                        break
                for row in torch.stack(gen, 1).tolist():
                    row = row[:row.index(2)] if 2 in row else row
                    s = stats[f"pol_{mode}"]; s[0] += int(bool(parse_plan(ptok.decode(row)))); s[1] += 1; s[2] += len(row)
        o_, r, d, info = env.step(int(rng.integers(0, 17)))
        frames.append(np.ascontiguousarray(o_[:63, :63]))
        if d:
            break
    for k, (ok, n, toks) in stats.items():
        print(f"{k}: parsed {ok}/{n} = {ok / max(n, 1):.2f}, mean tokens {toks / max(n, 1):.0f}", flush=True)


if __name__ == "__main__":
    main()
