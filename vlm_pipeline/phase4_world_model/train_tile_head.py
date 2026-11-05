"""Are the small objects in h, z? A tile class head on a frozen RSSM, no gradient into the world model.

h_t, z_t are filtered over the last --context real frames as for the text head. The target of each of the 7 x 9 map
tiles is its class by the nearest texture prototype of the frame itself. Every --weights mode trains its own head on
the train episodes. Scores on the validation episodes: recall and precision per class on real frames (also split by
bright and dark frames), and along a prior rollout with the teacher's actions.

  python3 train_tile_head.py --rssm-checkpoint <run>/epoch_060.pt --raw-data <collection> --split-from <data dir> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from imagine_eval import classify_tiles, tile_prototypes
from train_slm_text_on_rssm import rssm_latents
from train_slm_world_model import build_model, load_episodes, mlp, read_image, save_json

CLASSES = ("zombie", "cow", "tree", "plant", "table", "water")
DARK = .3  # mean brightness of the map view below which a frame counts as night


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rssm-checkpoint", type=Path, required=True)
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument("--split-from", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weights", default="inverse,sqrt", help="comma list of inverse, sqrt, none")
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--rollout", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7950)
    args = parser.parse_args()
    import torch
    import torch.nn.functional as F
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    state = torch.load(args.rssm_checkpoint, map_location="cpu", weights_only=False)
    if state["config"].get("transition") != "gru":
        parser.error("--rssm-checkpoint must be a --transition gru world model")
    rssm = build_model(state["config"], None, device)
    rssm.modules.load_state_dict(state["model"]["support"])
    rssm.eval()
    for parameter in rssm.parameters():
        parameter.requires_grad_(False)
    protos, index, names = tile_prototypes()
    protos, index = torch.from_numpy(protos).to(device), torch.from_numpy(index).to(device)
    names = list(names)
    episodes = load_episodes(args.raw_data, None, None)
    valid_ids = {json.loads(line)["source_episode_id"] for line in (args.split_from / "valid.jsonl").open()}
    started = time.time()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        h, z, offset = rssm_latents(rssm, args.raw_data, episodes, args.context, device)
    latents = torch.cat((h, z), dim=-1).to(device)
    targets, bright = [], []
    with torch.inference_mode():
        for episode in episodes.values():
            frames = torch.from_numpy(np.stack([read_image(args.raw_data, item["image"]) for item in episode])).to(device).float()
            targets.append(classify_tiles(frames, protos, index)[0].to(torch.uint8))
            bright.append(frames[:, :, :49, :].mean(dim=(1, 2, 3)))
    targets, bright = torch.cat(targets), torch.cat(bright)
    mark = torch.zeros(len(latents), dtype=torch.bool, device=device)
    for key, episode in episodes.items():
        if key in valid_ids:
            mark[offset[key]:offset[key] + len(episode)] = True
    train, valid = (~mark).nonzero().flatten(), mark.nonzero().flatten()
    share = targets[train].long().flatten().bincount(minlength=len(names)).float()
    share = share / share.sum()
    print(json.dumps(dict(states=len(latents), train=len(train), valid=len(valid), seconds=time.time() - started,
                          dark_share=float((bright[valid] < DARK).float().mean()))), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    ids = {name: names.index(name) for name in CLASSES}

    def scores(guess, real):
        out = {}
        for name, i in ids.items():
            r, g = real == i, guess == i
            out[name] = dict(recall=round(float((r & g).sum() / r.sum().clamp_min(1)), 3),
                             precision=round(float((r & g).sum() / g.sum().clamp_min(1)), 3), real=int(r.sum()))
        return out

    starts = [(k, t) for k, e in sorted(episodes.items()) if k in valid_ids for t in range(args.context - 1, len(e) - args.rollout - 1, 4)]
    results = dict(checkpoint=str(args.rssm_checkpoint), starts=len(starts), classes=names)
    for mode in args.weights.split(","):
        if mode == "inverse":
            weight = (1 / share.clamp_min(1e-4) / (share > 0).sum()).clamp_max(100.0)
        elif mode == "sqrt":
            weight = 1 / share.clamp_min(1e-4).sqrt()
            weight = (weight / (weight * share).sum()).clamp_max(100.0)
        else:
            weight = torch.ones(len(names), device=device)
        head = mlp(torch.nn, latents.shape[1], rssm.h_size, 2, 63 * len(names)).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=.01)
        predict = lambda x: head(x).view(-1, 7, 9, len(names))
        history = []
        for epoch in range(1, args.epochs + 1):
            head.train()
            order = train[torch.randperm(len(train), device=device)]
            total, count = 0.0, 0
            for part in order.split(args.batch_size):
                loss = F.cross_entropy(predict(latents[part]).flatten(0, 2), targets[part].long().flatten(), weight=weight)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total, count = total + float(loss), count + 1
            head.eval()
            with torch.inference_mode():
                guess = torch.cat([predict(latents[part]).argmax(-1) for part in valid.split(4096)])
            row = dict(epoch=epoch, train_loss=total / count, **{k: v for k, v in scores(guess, targets[valid].long()).items()})
            history.append(row)
            print(mode, json.dumps(dict(epoch=epoch, train_loss=round(total / count, 4), zombie=row["zombie"], cow=row["cow"])), flush=True)
        night = bright[valid] < DARK
        result = dict(history=history, day=scores(guess[~night], targets[valid][~night].long()),
                      night=scores(guess[night], targets[valid][night].long()), real=history[-1])
        rollout = {}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for off in range(0, len(starts), 64):
                batch = starts[off:off + 64]
                rows = torch.tensor([offset[k] + t for k, t in batch], device=device)
                hm, zm = latents[rows, :rssm.h_size], latents[rows, rssm.h_size:]
                actions = torch.tensor([[item["action"] for item in episodes[k][t:t + args.rollout]] for k, t in batch], device=device)
                for d in range(1, args.rollout + 1):
                    hm, _ = rssm.transition(None, actions[:, d - 1], None, hm, zm, device)
                    zm = rssm.mode(rssm.prior(hm))
                    guess_d = predict(torch.cat((hm, zm), dim=-1).float()).argmax(-1)
                    rollout.setdefault(d, ([], []))
                    rollout[d][0].append(guess_d)
                    rollout[d][1].append(targets[rows + d].long())
        result["imagined"] = {d: scores(torch.cat(g), torch.cat(r)) for d, (g, r) in rollout.items()}
        results[mode] = result
        torch.save(dict(model=head.state_dict(), classes=names, mode=mode), args.out / f"head_{mode}.pt")
        save_json(args.out / "tile_head.json", results)
        for name in ("zombie", "cow"):
            print(mode, name, "real", result["real"][name], "day", result["day"][name], "night", result["night"][name],
                  "imagined", {d: result["imagined"][d][name]["recall"] for d in (1, 4, 8, 16)},
                  "precision8", result["imagined"][8][name]["precision"], flush=True)
    print("EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
