"""Does the veRL form of an RSSM policy compute what BC trained? Compares, on real teacher frames:

  state   h_t, z_t from rssm_service.py with those of episode_latents (the states BC saw);
  logits  the materialised HF policy on markers + pixel_values with the BC SLM on inputs_embeds.

  python3 check_rssm_policy.py --rssm-checkpoint <rssm.pt> --init-checkpoint <initial_random> --checkpoint <bc>/best.pt \
    --materialized <dir> --raw-data <collection>
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import threading
import time
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

from train_slm_action_on_rssm import ACTIONS, HEAD, Policy, episode_latents, load_trainable, world
from train_slm_world_model import load_episodes, read_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rssm-checkpoint", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--materialized", type=Path, required=True)
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument("--h-tokens", type=int, default=50)
    parser.add_argument("--z-tokens", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    import torch
    device = torch.device("cuda")
    rssm, model = world(args, device)
    load_trainable(model, args.checkpoint)
    model.eval()
    policy = Policy(model, device)
    episodes = load_episodes(args.raw_data, None, None)
    chosen = {k: episodes[k] for k in sorted(episodes)[:args.episodes]}
    hs, zs, offset = episode_latents(rssm, args.raw_data, chosen, device)

    # state: the service over its socket, as the agent loop calls it
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from rssm_service import Service
    with tempfile.TemporaryDirectory() as folder:
        socket = Path(folder) / "s.sock"
        options = argparse.Namespace(checkpoint=args.rssm_checkpoint, socket=socket, out=Path(folder) / "out", pack=None,
                                     updates_per_rollout=0, ema=.995, lr=0.0, buffer_frames=1000, save_every=50, seed=1)
        threading.Thread(target=Service(options).serve, daemon=True).start()
        while not (options.out / "ready").exists():
            time.sleep(.1)
        worst = 0.0
        for key, episode in chosen.items():
            connection = Client(str(socket), family="AF_UNIX", authkey=b"rssm")
            connection.send(("begin", False)); connection.recv()
            frames = [np.moveaxis((read_image(args.raw_data, item["image"]) * 255).round().astype(np.uint8), 0, -1) for item in episode]
            connection.send(("first", frames[0][None]))
            states = [connection.recv()[0]]
            for t in range(len(frames) - 1):
                connection.send(("step", [0], [ACTIONS[episode[t]["action"]]], [0.0], [False], frames[t + 1][None]))
                states.append(connection.recv()[0])
            connection.close()
            served = torch.from_numpy(np.stack(states))
            table = torch.cat((hs, zs), dim=-1)[offset[key]:offset[key] + len(frames)]
            worst = max(worst, float((served - table).abs().max()))
        print(json.dumps(dict(state_max_abs_difference=worst, frames_as_uint8=True)), flush=True)

    # logits: one state, the prompt, a target
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf = AutoModelForCausalLM.from_pretrained(args.materialized, trust_remote_code=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.materialized, trust_remote_code=True)
    print(json.dumps(dict(
        trainable_front_end=sum(p.numel() for p in hf.vision.parameters() if p.requires_grad),
        front_end=sum(p.numel() for p in hf.vision.parameters()))), flush=True)
    markers = list(hf.config.crafter_visual_token_ids)
    worst, seen = 0.0, 0
    key = next(iter(chosen))
    with torch.inference_mode():
        for t in range(0, len(chosen[key]), max(len(chosen[key]) // 8, 1)):
            goal = chosen[key][t]["instruction"]
            h, z = hs[offset[key] + t].to(device), zs[offset[key] + t].to(device)
            target = model.ids("ACTION: DO")
            base = policy.prefix(goal, h, z)
            trained = model.lm(inputs_embeds=torch.cat((base, policy.embed(torch.tensor(target, device=device))))[None]).logits[0].float()
            ids = tokenizer.encode(HEAD.format(goal), add_special_tokens=False) + markers + target
            if ids[:-len(target) - len(markers)] != model.ids(HEAD.format(goal)):
                raise ValueError("the prompt is tokenised differently")
            served = hf(input_ids=torch.tensor([ids], device=device), pixel_values=torch.cat((h, z))[None]).logits[0].float()
            common = trained.shape[-1]
            worst = max(worst, float((trained[..., :common] - served[..., :common]).abs().max()))
            seen += 1
    print(json.dumps(dict(logits_max_abs_difference=worst, states=seen, vocabulary=common, with_markers=hf.config.vocab_size)), flush=True)


if __name__ == "__main__":
    main()
