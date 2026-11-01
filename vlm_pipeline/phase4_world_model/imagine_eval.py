#!/usr/bin/env python3
"""Imagination test of a trained SLM world model (train_slm_world_model.py best.pt) on validation episodes.

Only the real frame x_t is seen: h_t = 0 (as at the start of every training chunk), z_t = posterior mode
(gaussian mean / categorical argmax) from x_t. Then, for d = 1..H, h_{t+d} = SLM transition(h, z, a_{t+d-1}
[, reasoning_{t+d-1}]) with actions (and reasoning text) from the dataset, z_{t+d} = prior mode,
x^_{t+d} = decoder(h_{t+d}, z_{t+d}).
No later real frame enters the rollout.

Compared per step d with
  copy-last : x^_{t+d} = x_t (nothing moves)
  mean frame: x^_{t+d} = mean training frame
  recon     : decoder output with z from the posterior on the REAL x_{t+d} (filtered h) - an autoencoder
              bound that sees every real frame, shown only to tell decoder blur from imagination error.

  python imagine_eval.py --checkpoint <run>/best.pt --raw-data <collection> --split-from <CE data dir> \
    --init-checkpoint <initial_random> [--reasoning reasoning.jsonl] --horizon 32 --out <dir>/<name>
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from train_slm_world_model import (PIXELS, SLMWorldModel, load_episodes, read_image, save_json)

CHANGE = 0.05  # a pixel "changed" when some channel differs from x_t by more than this


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument("--split-from", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--reasoning", type=Path)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--stride", type=int, default=4, help="distance between start steps t in an episode")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--examples", type=int, default=8, help="rollouts saved as images")
    parser.add_argument("--seed", type=int, default=7960)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    tokenizer = AutoTokenizer.from_pretrained(args.init_checkpoint)
    lm = AutoModelForCausalLM.from_pretrained(args.init_checkpoint)
    model = SLMWorldModel(lm, tokenizer, config["h_size"], config["z_size"], config["h_tokens"], config["z_tokens"],
                          config.get("z_kind", "gaussian"), config.get("z_classes", 32), config.get("image_net", "small"))
    model.lm.load_state_dict(state["model"]["lm"]); model.modules.load_state_dict(state["model"]["support"])
    model.to(device).eval()

    reasoning = None
    if args.reasoning:
        reasoning = {(int(r["episode_id"]), int(r["step"])): r for r in map(json.loads, args.reasoning.open())}
    episodes = load_episodes(args.raw_data, reasoning)
    valid_ids = {json.loads(line)["source_episode_id"] for line in (args.split_from / "valid.jsonl").open()}
    train = [e for k, e in episodes.items() if k not in valid_ids]
    valid = {k: e for k, e in episodes.items() if k in valid_ids}
    H = args.horizon
    starts = [(k, t) for k, e in sorted(valid.items()) for t in range(0, len(e) - H, args.stride)]
    rng = random.Random(args.seed)
    mean_frame = np.mean([read_image(args.raw_data, item["image"])
                          for item in rng.sample([i for e in train for i in e], 2000)], axis=0)
    mean_frame_t = torch.from_numpy(mean_frame).to(device)
    # examples: half with a reward inside the window, half random
    with_reward = [s for s in starts if any(i["reward"] > 0 for i in valid[s[0]][s[1] + 1:s[1] + H + 1])]
    examples = rng.sample(with_reward, min(len(with_reward), args.examples // 2))
    examples += rng.sample([s for s in starts if s not in examples], args.examples - len(examples))

    keys = ("model", "copy_last", "mean_frame", "recon")
    sq = {k: np.zeros(H) for k in keys}                 # summed per-pixel MSE per step
    sq_changed = {k: np.zeros(H) for k in keys}         # summed squared error over changed pixels
    n_changed = np.zeros(H)
    beats_copy = np.zeros(H)
    reward_rows = []
    saved = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for offset in range(0, len(starts), args.batch_size):
            batch = starts[offset:offset + args.batch_size]
            windows = [valid[k][t:t + H + 1] for k, t in batch]
            images = torch.from_numpy(np.stack([[read_image(args.raw_data, i["image"]) for i in w] for w in windows])).to(device)
            actions = torch.tensor([[i["action"] for i in w[:-1]] for w in windows], device=device)
            texts = [[i["reasoning"] for i in w[:-1]] for w in windows]
            goals = [w[0]["instruction"] for w in windows]
            B = len(batch)
            h = torch.zeros(B, model.h_size, device=device)
            z = model.mode(model.posterior(images[:, 0], h))  # the only real frame used
            hf, zf = h, z                                     # filtered (autoencoder bound) branch
            preds, recons = [], []
            for d in range(H):
                rs = [r[d] for r in texts]
                h, last = model.transition(goals, actions[:, d], rs, h, z, device)
                z = model.mode(model.prior(h))
                preds.append(model.decode(h, z).float())
                reward = (model.modules["reward_head"](last)[:, 0].float() / config["reward_target_scale"]).tolist()
                reward_rows += [(d, reward[i], windows[i][d + 1]["reward"]) for i in range(B)]
                hf, _ = model.transition(goals, actions[:, d], rs, hf, zf, device)
                zf = model.mode(model.posterior(images[:, d + 1], hf))
                recons.append(model.decode(hf, zf).float())
            pred, recon = torch.stack(preds, 1), torch.stack(recons, 1)    # [B, H, 3, 63, 63]
            target, first = images[:, 1:].float(), images[:, :1].float()
            guesses = dict(model=pred, copy_last=first.expand_as(target), recon=recon,
                           mean_frame=mean_frame_t.expand_as(target))
            changed = ((target - first).abs().amax(dim=2, keepdim=True) > CHANGE).float()   # [B, H, 1, 63, 63]
            n_changed += changed.sum(dim=(0, 2, 3, 4)).cpu().numpy() * 3
            for key, guess in guesses.items():
                err = (guess - target).square()
                sq[key] += err.mean(dim=(2, 3, 4)).sum(0).cpu().numpy()
                sq_changed[key] += (err * changed).sum(dim=(0, 2, 3, 4)).cpu().numpy()
            beats_copy += (((pred - target).square().mean(dim=(2, 3, 4))
                            < (first - target).square().mean(dim=(2, 3, 4))).float().sum(0).cpu().numpy())
            for i, s in enumerate(batch):
                if s in examples:
                    saved[f"{s[0]}_{s[1]}"] = dict(real=images[i].float().cpu().numpy(), model=pred[i].cpu().numpy(),
                                                    recon=recon[i].cpu().numpy(),
                                                    actions=[valid[s[0]][s[1] + d]["action"] for d in range(H)],
                                                    rewards=[valid[s[0]][s[1] + d + 1]["reward"] for d in range(H)],
                                                    reasoning=texts[i], goal=goals[i])
    n = len(starts)
    reward_rows = np.asarray(reward_rows)
    events = reward_rows[:, 2] > 0
    result = dict(
        checkpoint=str(args.checkpoint), checkpoint_epoch=state["metrics"]["epoch"],
        checkpoint_validation_world_loss=state["metrics"]["validation_world_loss"],
        reasoning=bool(args.reasoning), horizon=H, training_horizon=config["horizon"], stride=args.stride,
        validation_episodes=len(valid), starts=n, change_threshold=CHANGE,
        mse_by_step={k: (v / n).round(6).tolist() for k, v in sq.items()},
        changed_pixel_fraction_by_step=(n_changed / (n * PIXELS)).round(5).tolist(),
        mse_on_changed_pixels_by_step={k: (v / np.maximum(n_changed, 1)).round(6).tolist() for k, v in sq_changed.items()},
        model_beats_copy_last_fraction_by_step=(beats_copy / n).round(4).tolist(),
        reward=dict(steps_with_reward=int(events.sum()), steps=len(reward_rows),
                    mean_prediction_on_reward_steps=float(reward_rows[events, 1].mean()) if events.any() else None,
                    mean_prediction_on_other_steps=float(reward_rows[~events, 1].mean()),
                    mse=float(np.mean((reward_rows[:, 1] - reward_rows[:, 2]) ** 2)),
                    mse_zero_baseline=float(np.mean(reward_rows[:, 2] ** 2))),
        examples=list(saved),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    save_json(args.out / "imagine_metrics.json", result)
    np.savez_compressed(args.out / "imagine_examples.npz",
                        **{f"{k}__{part}": np.asarray(v[part]) for k, v in saved.items() for part in ("real", "model", "recon", "actions", "rewards")})
    save_json(args.out / "imagine_examples_text.json", {k: dict(goal=v["goal"], reasoning=v["reasoning"]) for k, v in saved.items()})
    print(json.dumps({k: result[k] for k in ("checkpoint_epoch", "starts", "reward")}), flush=True)


if __name__ == "__main__":
    main()
