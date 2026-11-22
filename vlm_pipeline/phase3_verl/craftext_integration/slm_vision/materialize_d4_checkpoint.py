"""Materialise a phase 4 SLM over Dreamer 4 latents (train_slm_reason_act_d4.py, SLMReasonAct: plan + value) as a veRL
model plugin, for Stage-2 RL through veRL (safe_rl_nlp.md, rules for agents 18-21).

The prompt of SLMReasonAct is   embed(pre) | adapter(Z) + slot + index (window x n_lat tokens) | readout (R) | embed(post)
and the plugin represents the middle part by ``window * n_lat + R`` reserved marker tokens whose embeddings the front end
(DreamerLatentFrontEnd, at ``vision``) replaces; ``pixel_values`` is the latent window flattened to
``window * n_lat * d_lat`` numbers. The text vocabulary (tok2288) stays the first rows of the embedding, so the logits are
the old ones plus the markers. The value head of SLMReasonAct is not part of the policy (GRPO needs no critic; the plan
scorer's critic lives in dreamer_service.py).

    python materialize_d4_checkpoint.py --ckpt <run>/run/checkpoints/step_060000.pt --init <initial_random> \
        --text-tokenizer <tok2288> --output <dir> [--check]
--check compares the materialised policy with SLMReasonAct on random latent windows: prompt logits and greedy plans.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True, help="step_*.pt of train_slm_reason_act_d4.py")
    parser.add_argument("--init", type=Path, required=True, help="initial_random: architecture of the SLM")
    parser.add_argument("--text-tokenizer", type=Path, required=True, help="dir with tokenizer.json (tok2288)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--n-lat", type=int, default=64)
    parser.add_argument("--d-lat", type=int, default=16)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--phase4", type=Path, default=Path(__file__).resolve().parents[3] / "phase4_world_model")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite output checkpoint: {output}")

    import torch
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    st = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state, readout = st["model"], int(st["args"]["readout"])
    n_markers = args.window * args.n_lat + readout
    markers = [f"<|crafter_visual_{i}|>" for i in range(n_markers)]
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(args.text_tokenizer / "tokenizer.json"))
    n_text = len(tokenizer)
    tokenizer.eos_token_id  # noqa: B018
    eos = tokenizer.convert_ids_to_tokens(2)
    tokenizer.add_special_tokens({"eos_token": eos, "pad_token": tokenizer.convert_ids_to_tokens(0)})
    # veRL's Qwen continuous-token builder requires the ChatML boundary tokens; the SLM never saw them, so they come
    # after the markers and are blocked in the logits like the markers (config.crafter_blocked_token_ids)
    chat = ["<|im_start|>", "<|im_end|>"]
    if tokenizer.add_special_tokens({"additional_special_tokens": markers + chat}) != len(markers) + len(chat):
        raise RuntimeError("Expected only new reserved marker tokens")
    lm = AutoModelForCausalLM.from_pretrained(args.init, dtype=torch.float32)
    lm.resize_token_embeddings(n_text)
    lm.tie_weights()
    sd = {}
    for k, v in state.items():
        if k.startswith("backbone."):
            sd["model." + k[len("backbone."):]] = v
        elif k == "lm_head.weight":
            sd[k] = v
    missing, unexpected = lm.load_state_dict(sd, strict=False)
    assert not unexpected and all("embed_tokens" in m or "lm_head" in m for m in missing), (missing, unexpected)
    lm.resize_token_embeddings(len(tokenizer))
    lm.config.eos_token_id, lm.config.pad_token_id = 2, 0
    output.mkdir(parents=True, exist_ok=True)
    lm.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)

    here = Path(__file__).resolve().parent
    shutil.copy2(here / "modeling_crafter_vision_slm.py", output / "modeling_crafter_vision_slm.py")
    sys.path.insert(0, str(here))
    from modeling_crafter_vision_slm import CrafterVisionQwen3ForCausalLM

    config = lm.config
    config.architectures = ["CrafterVisionQwen3ForCausalLM"]
    config.auto_map = {
        "AutoModelForCausalLM": "modeling_crafter_vision_slm.CrafterVisionQwen3ForCausalLM",
        "AutoModelForTokenClassification": "modeling_crafter_vision_slm.CrafterVisionQwen3ForTokenClassification",
    }
    config.crafter_visual_tokens = len(markers)
    config.crafter_visual_token_ids = tokenizer.convert_tokens_to_ids(markers)
    config.crafter_blocked_token_ids = tokenizer.convert_tokens_to_ids(chat)
    assert config.crafter_visual_token_ids == list(range(n_text, n_text + len(markers))), "markers must follow the text vocabulary"
    config.crafter_dreamer = dict(n_lat=args.n_lat, d_lat=args.d_lat, window=args.window, readout=readout)
    config.crafter_freeze_front_end = False
    config.crafter_plan = int(st["args"].get("plan") or 0)
    config.crafter_tasks = list(st["tasks"])
    policy = CrafterVisionQwen3ForCausalLM.from_pretrained(output, config=config)
    fe = {"adapter." + k[len("adapter."):]: v for k, v in state.items() if k.startswith("adapter.")}
    fe.update(slot=state["slot"], index=state["index"], readout=state["readout"])
    policy.vision.load_state_dict(fe)
    policy.save_pretrained(output, safe_serialization=True)
    config.save_pretrained(output)
    manifest = {"ckpt": str(args.ckpt), "ckpt_sha256": sha256(args.ckpt), "text_tokenizer": str(args.text_tokenizer),
                "visual_marker_ids": config.crafter_visual_token_ids, "crafter_dreamer": config.crafter_dreamer,
                "tasks": config.crafter_tasks, "plan": config.crafter_plan,
                "contract": "veRL passes pixel_values [B, window*n_lat*d_lat]: the Dreamer 4 latent window of the state"}
    (output / "verl_visual_slm_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("materialised", output, "markers", len(markers), "vocab", len(tokenizer), flush=True)
    if args.check:
        check(args, st, output, tokenizer, markers)


def check(args, st, output, tokenizer, markers) -> None:
    """Same prompt into SLMReasonAct and the materialised policy: logits over the text vocabulary and greedy plans."""
    import torch
    from transformers import AutoModelForCausalLM
    sys.path.insert(0, str(args.phase4))
    from train_slm_reason_act_d4 import SLMReasonAct, SYSTEM_PLAN

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lm = AutoModelForCausalLM.from_pretrained(args.init, dtype=torch.float32)
    n_text = st["model"]["lm_head.weight"].shape[0]
    lm.resize_token_embeddings(n_text); lm.tie_weights()
    ref = SLMReasonAct(lm, int(st["args"]["readout"])).to(dev)
    ref.load_state_dict(st["model"]); ref.eval()
    pol = AutoModelForCausalLM.from_pretrained(output, trust_remote_code=True, dtype=torch.float32).to(dev).eval()
    plan = int(st["args"].get("plan") or 0)
    pre_txt = "System:\n" + SYSTEM_PLAN.format(plan) + "\nUser:\n"
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
    worst, same = 0.0, 0
    B = 6
    for task in st["tasks"]:
        pre, post = enc(pre_txt), enc(f"\nTask: {task}\nAssistant:\n")
        Z = torch.randn(B, args.window, args.n_lat, args.d_lat, device=dev).clamp(-1, 1)
        with torch.no_grad():
            pre_t = torch.tensor([pre], device=dev); post_t = torch.tensor([post], device=dev)
            emb, mask, _ = ref.prefix(Z, pre_t, post_t.expand(B, -1), torch.ones(B, len(post), dtype=torch.long, device=dev))
            h = ref.backbone(inputs_embeds=emb, attention_mask=mask).last_hidden_state
            l_ref = ref.lm_head(h[:, -1]).float()
            ids = torch.tensor([pre + pol.config.crafter_visual_token_ids + post] * B, device=dev)
            out = pol(input_ids=ids, pixel_values=Z.reshape(B, -1), attention_mask=torch.ones_like(ids))
            l_pol = out.logits[:, -1, :n_text].float()
            worst = max(worst, float((l_ref - l_pol).abs().max()))
            g_ref = ref.generate(Z, pre_t, post_t.expand(B, -1), torch.ones(B, len(post), dtype=torch.long, device=dev),
                                 max_tokens=60)
            cur = ids
            for _ in range(60):   # plain greedy decoding, full forward each step (5M model)
                nxt = pol(input_ids=cur, pixel_values=Z.reshape(B, -1)).logits[:, -1, :n_text].argmax(-1, keepdim=True)
                cur = torch.cat([cur, nxt], 1)
            g_pol = cur[:, ids.shape[1]:]
            for b in range(B):
                a = [t for t in g_ref[b].tolist()]; a = a[:a.index(2) + 1] if 2 in a else a
                c = g_pol[b].tolist(); c = c[:c.index(2) + 1] if 2 in c else c
                same += int(a == c)
    n = B * len(st["tasks"])
    print(f"CHECK max |logit diff| {worst:.2e}, identical greedy plans {same}/{n}", flush=True)
    print("CHECK", "PASS" if worst < 1e-3 and same == n else "FAIL", flush=True)


if __name__ == "__main__":
    main()
