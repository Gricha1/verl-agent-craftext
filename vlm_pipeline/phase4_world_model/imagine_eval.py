#!/usr/bin/env python3
"""Imagination test of a trained SLM world model (train_slm_world_model.py best.pt) on validation episodes.

Only the real frame x_t is seen: h_t = 0 (as at the start of every training chunk), z_t = posterior mode
(gaussian mean / categorical argmax) from x_t. Then, for d = 1..H, h_{t+d} = SLM transition(h, z, a_{t+d-1}
[, reasoning_{t+d-1}]) with actions (and reasoning text) from the dataset, z_{t+d} = prior mode,
x^_{t+d} = decoder(h_{t+d}, z_{t+d}).
No later real frame enters the rollout.

--context K (DreamerV3 video predictions use 5): h starts at 0 on x_{t-K+1} and is filtered through the real frames
x_{t-K+1..t} (z = posterior mode, transitions with the dataset actions) before imagining from x_t; K = 1 is the
single-frame start above. --min-start keeps the same starts t for every K (t >= min-start).
--prior-z sample: z_{t+d} is sampled from the prior instead of its mode (--seed sets the sample).

Compared per step d with
  copy-last : x^_{t+d} = x_t (nothing moves)
  mean frame: x^_{t+d} = mean training frame
  recon     : decoder output with z from the posterior on the REAL x_{t+d} (filtered h) - an autoencoder
              bound that sees every real frame, shown only to tell decoder blur from imagination error.
Tile metric (what is drawn where, not pixel MSE): each 7x7 map tile of a frame is assigned the class of its nearest
Craftax-Classic texture prototype (grass, tree, water, sand, stone, player, cow, ...; tile_prototypes); accuracy =
share of tiles whose guessed class equals the real one, also on tiles whose class changed since x_t.

--descriptions state_descriptions.jsonl (models trained with --descriptions): at the steps --description-steps the SLM
writes the text of x_{t+d} from the inputs of the transition into it, on the imagined branch (model) and on the filtered
one, which saw the real x_{t+d-1} (filtered). Compared with Gemma's description of the real x_{t+d} by the facts of
description_facts.py, next to copy_last (Gemma's description of x_t). Every --description-every-th start, as text is slow.

  python imagine_eval.py --checkpoint <run>/best.pt --raw-data <collection> --split-from <CE data dir> \
    --init-checkpoint <initial_random> [--reasoning reasoning.jsonl] --horizon 32 --out <dir>/<name>
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from description_facts import FactScore
from train_slm_world_model import (PIXELS, build_model, load_episodes, read_image, save_json)

CHANGE = 0.05  # a pixel "changed" when some channel differs from x_t by more than this
ASSETS = Path(__file__).resolve().parents[2] / "caged_craftext/Craftax/craftax/craftax_classic/assets"
TILE, ROWS, COLS = 7, 7, 9  # the map view: 7 x 9 blocks of 7 x 7 pixels above the two inventory rows
BLOCKS = dict(grass="grass.png", water="water.png", stone="stone.png", tree="tree.png", wood="wood.png", path="path.png",
              coal="coal.png", iron="iron.png", diamond="diamond.png", table="table.png", furnace="furnace.png",
              sand="sand.png", lava="lava.png", plant="plant_on_grass.png", ripe_plant="ripe_plant_on_grass.png")
OVERLAYS = dict(player=("player-left.png", "player-right.png", "player-up.png", "player-down.png", "player-sleep.png"),
                cow=("cow.png",), zombie=("zombie.png",))
DAYLIGHT = (1.0, 0.8, 0.65, 0.5)  # light levels rendered without the night static (it is drawn only below 0.5)
TILE_RELIABLE = 0.003  # a real tile is "explained" when its per-pixel MSE to the nearest prototype is below this


PLAYER_CLASS = None


def tile_prototypes(assets: Path = ASSETS):
    """7x7 tile prototypes of the Craftax-Classic renderer: every block texture, the player / cow / zombie drawn over
    grass, sand and path, each at the DAYLIGHT levels (night blend of the renderer: desaturate, tint (0, 16, 64)).
    Returns (prototypes [K, 3, 7, 7] in [0, 1], class index [K], class names)."""
    from PIL import Image

    def texture(name):
        image = Image.open(assets / name).convert("RGBA").resize((TILE, TILE), Image.NEAREST)
        rgba = np.asarray(image, dtype=np.float32) / 255
        return rgba[..., :3], rgba[..., 3:]

    names, tiles = [], []
    for name, file in BLOCKS.items():
        names.append(name); tiles.append(texture(file)[0])
    names.append("out_of_bounds"); tiles.append(np.full((TILE, TILE, 3), 128 / 255, np.float32))
    for name, files in OVERLAYS.items():
        for file in files:
            rgb, alpha = texture(file)
            for ground in ("grass", "sand", "path"):
                names.append(name); tiles.append(texture(BLOCKS[ground])[0] * (1 - alpha) + rgb * alpha)
    classes = sorted(set(names), key=names.index)
    protos, index = [], []
    tint = np.array([0, 16, 64], np.float32) / 255
    for name, tile in zip(names, tiles):
        lum = (tile @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
        night = 0.5 * (0.4 * tile + 0.6 * lum) + 0.5 * tint
        for light in DAYLIGHT:
            protos.append(light * tile + (1 - light) * night); index.append(classes.index(name))
    global PLAYER_CLASS
    PLAYER_CLASS = classes.index("player")
    return np.moveaxis(np.stack(protos), -1, 1), np.asarray(index), classes


def classify_tiles(frames, protos, index):
    """frames [..., 3, 63, 63] -> (class [..., 7, 9], per-pixel MSE to the nearest prototype [..., 7, 9]).

    The centre tile is the agent. There are no prototypes for its sleeping texture, which the nearest neighbour called a
    zombie: 637 of the 1253 "zombie" tiles of the validation frames, so the tile is the player whatever it looks like."""
    lead, n = frames.shape[:-3], frames.dim() - 3
    tiles = frames[..., :ROWS * TILE, :].unflatten(-2, (ROWS, TILE)).unflatten(-1, (COLS, TILE))  # [..., 3, R, 7, C, 7]
    tiles = tiles.permute(*range(n), n + 1, n + 3, n, n + 2, n + 4)                               # [..., R, C, 3, 7, 7]
    import torch
    flat = tiles.reshape(-1, 3 * TILE * TILE).float()
    p = protos.reshape(len(protos), -1).float()
    with torch.autocast(frames.device.type, enabled=False):  # bf16 would blur the distances
        dist = flat.square().sum(1, keepdim=True) - 2 * flat @ p.T + p.square().sum(1)[None]
    best, arg = dist.min(1)
    classes = index[arg].reshape(*lead, ROWS, COLS)
    if PLAYER_CLASS is not None:
        classes[..., ROWS // 2, COLS // 2] = PLAYER_CLASS
    return classes, (best.clamp_min(0) / p.shape[1]).reshape(*lead, ROWS, COLS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument("--split-from", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, help="SLM world models: initial_random of the phase-2 run")
    parser.add_argument("--reasoning", type=Path)
    parser.add_argument("--descriptions", type=Path, help="state_descriptions.jsonl: also score the text the SLM writes")
    parser.add_argument("--description-steps", default="1,4,8,16", help="imagined steps d at which the SLM writes text")
    parser.add_argument("--description-every", type=int, default=4, help="write text at every n-th start")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--stride", type=int, default=4, help="distance between start steps t in an episode")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context", type=int, default=1, help="real frames the state is filtered through, x_t last")
    parser.add_argument("--min-start", type=int, default=0, help="first start t (>= context - 1)")
    parser.add_argument("--prior-z", choices=("mode", "sample"), default="mode",
                        help="imagined z: prior mode (argmax / mean) or a sample from the prior, as DreamerV3 imagines")
    parser.add_argument("--examples", type=int, default=8, help="rollouts saved as images")
    parser.add_argument("--example-starts", default="", help="'<episode>_<t>,...': save these rollouts instead")
    parser.add_argument("--seed", type=int, default=7960)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    model = build_model(config, args.init_checkpoint, device)
    if model.lm is not None:
        model.lm.load_state_dict(state["model"]["lm"])
    model.modules.load_state_dict(state["model"]["support"])
    model.eval()

    reasoning = None
    if args.reasoning:
        reasoning = {(int(r["episode_id"]), int(r["step"])): r for r in map(json.loads, args.reasoning.open())}
    descriptions = None
    if args.descriptions:
        if model.lm is None or not config.get("descriptions"):
            parser.error("--descriptions needs a model trained with --descriptions")
        descriptions = {(int(r["episode_id"]), int(r["step"])): r["state_description"]
                        for r in map(json.loads, args.descriptions.open())}
    text_steps = [int(d) for d in args.description_steps.split(",")] if descriptions else []
    text_scores = {d: {k: FactScore() for k in ("model", "filtered", "copy_last")} for d in text_steps}
    text_rows = []
    episodes = load_episodes(args.raw_data, reasoning, descriptions)
    valid_ids = {json.loads(line)["source_episode_id"] for line in (args.split_from / "valid.jsonl").open()}
    train = [e for k, e in episodes.items() if k not in valid_ids]
    valid = {k: e for k, e in episodes.items() if k in valid_ids}
    H, C = args.horizon, args.context - 1  # C: context frames before x_t
    if args.min_start < C:
        parser.error("--min-start must be >= context - 1")
    starts = [(k, t) for k, e in sorted(valid.items()) for t in range(0, len(e) - H, args.stride) if t >= args.min_start]
    rng = random.Random(args.seed)
    mean_frame = np.mean([read_image(args.raw_data, item["image"])
                          for item in rng.sample([i for e in train for i in e], 2000)], axis=0)
    mean_frame_t = torch.from_numpy(mean_frame).to(device)
    # examples: half with a reward inside the window, half random
    with_reward = [s for s in starts if any(i["reward"] > 0 for i in valid[s[0]][s[1] + 1:s[1] + H + 1])]
    if args.example_starts:
        examples = [tuple(map(int, s.split("_"))) for s in args.example_starts.split(",")]
    else:
        examples = rng.sample(with_reward, min(len(with_reward), args.examples // 2))
        examples += rng.sample([s for s in starts if s not in examples], args.examples - len(examples))

    keys = ("model", "copy_last", "mean_frame", "recon")
    sq = {k: np.zeros(H) for k in keys}                 # summed per-pixel MSE per step
    sq_changed = {k: np.zeros(H) for k in keys}         # summed squared error over changed pixels
    n_changed = np.zeros(H)
    beats_copy = np.zeros(H)
    # tile metric: every 7x7 map tile -> nearest Craftax texture class; a guess is right when its class is the real one
    protos, proto_class, tile_classes = tile_prototypes()
    protos, proto_class = torch.from_numpy(protos).to(device), torch.from_numpy(proto_class).to(device)
    tile_keys = ("model", "copy_last", "recon")
    tile_right = {k: np.zeros(H) for k in tile_keys}            # all map tiles
    tile_right_changed = {k: np.zeros(H) for k in tile_keys}    # tiles whose real class differs from x_t's
    tile_right_reliable = {k: np.zeros(H) for k in tile_keys}   # tiles the prototypes explain (real MSE < TILE_RELIABLE)
    n_tiles_changed, n_tiles_reliable = np.zeros(H), np.zeros(H)
    class_total, class_right = np.zeros(len(tile_classes)), np.zeros(len(tile_classes))  # model, over all steps
    reward_rows = []
    saved = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for offset in range(0, len(starts), args.batch_size):
            batch = starts[offset:offset + args.batch_size]
            full = [valid[k][t - C:t + H + 1] for k, t in batch]
            images = torch.from_numpy(np.stack([[read_image(args.raw_data, i["image"]) for i in w] for w in full])).to(device)
            actions = torch.tensor([[i["action"] for i in w[:-1]] for w in full], device=device)
            texts = [[i["reasoning"] for i in w[:-1]] for w in full]
            goals = [w[0]["instruction"] for w in full]
            B = len(batch)
            h = torch.zeros(B, model.h_size, device=device)
            z = model.mode(model.posterior(images[:, 0], h))
            for j in range(C):  # context: filter h through the real frames x_{t-C..t}
                h, _ = model.transition(goals, actions[:, j], [r[j] for r in texts], h, z, device)
                z = model.mode(model.posterior(images[:, j + 1], h))
            # from here on the window starts at x_t, the last real frame used
            images, actions, texts, windows = images[:, C:], actions[:, C:], [r[C:] for r in texts], [w[C:] for w in full]
            hf, zf = h, z                                     # filtered (autoencoder bound) branch
            start_recon = model.decode(h, z).float()          # x_t itself: reconstruction, the rollout starts here
            preds, recons = [], []
            keep = [i for i in range(B) if (offset + i) % args.description_every == 0]
            for d in range(H):
                rs = [r[d] for r in texts]
                if d + 1 in text_steps and keep:  # the text of x_{t+d+1} comes from the inputs of the transition into it
                    guess = {key: model.describe([goals[i] for i in keep], actions[keep, d], [rs[i] for i in keep],
                                                 hi[keep], zi[keep], device)
                             for key, hi, zi in (("model", h, z), ("filtered", hf, zf))}
                    for n, i in enumerate(keep):
                        real, first = windows[i][d + 1]["description"], windows[i][0]["description"]
                        text_scores[d + 1]["model"].add(guess["model"][n], real, first)
                        text_scores[d + 1]["filtered"].add(guess["filtered"][n], real, first)
                        text_scores[d + 1]["copy_last"].add(first, real, first)
                        text_rows.append(dict(episode=batch[i][0], t=batch[i][1], d=d + 1, goal=goals[i], real=real, first=first,
                                              model=guess["model"][n], filtered=guess["filtered"][n]))
                h, last = model.transition(goals, actions[:, d], rs, h, z, device)
                z = model.mode(model.prior(h)) if args.prior_z == "mode" else model.sample(model.prior(h))
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
            real_class, real_err = classify_tiles(target, protos, proto_class)           # [B, H, 7, 9]
            first_class = classify_tiles(first, protos, proto_class)[0]                 # [B, 1, 7, 9]
            moved, reliable = real_class != first_class, real_err < TILE_RELIABLE
            n_tiles_changed += moved.sum(dim=(0, 2, 3)).cpu().numpy()
            n_tiles_reliable += reliable.sum(dim=(0, 2, 3)).cpu().numpy()
            for key in tile_keys:
                guess_class = first_class.expand_as(real_class) if key == "copy_last" else \
                    classify_tiles(guesses[key], protos, proto_class)[0]
                right = guess_class == real_class
                tile_right[key] += right.sum(dim=(0, 2, 3)).cpu().numpy()
                tile_right_changed[key] += (right & moved).sum(dim=(0, 2, 3)).cpu().numpy()
                tile_right_reliable[key] += (right & reliable).sum(dim=(0, 2, 3)).cpu().numpy()
                if key == "model":
                    class_total += torch.bincount(real_class.flatten(), minlength=len(tile_classes)).cpu().numpy()
                    class_right += torch.bincount(real_class[right], minlength=len(tile_classes)).cpu().numpy()
            for i, s in enumerate(batch):
                if s in examples:
                    saved[f"{s[0]}_{s[1]}"] = dict(real=images[i].float().cpu().numpy(), model=pred[i].cpu().numpy(),
                                                    recon=recon[i].cpu().numpy(), start_recon=start_recon[i].cpu().numpy(),
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
        context=args.context, min_start=args.min_start, prior_z=args.prior_z, seed=args.seed,
        validation_episodes=len(valid), starts=n, change_threshold=CHANGE,
        mse_by_step={k: (v / n).round(6).tolist() for k, v in sq.items()},
        changed_pixel_fraction_by_step=(n_changed / (n * PIXELS)).round(5).tolist(),
        mse_on_changed_pixels_by_step={k: (v / np.maximum(n_changed, 1)).round(6).tolist() for k, v in sq_changed.items()},
        model_beats_copy_last_fraction_by_step=(beats_copy / n).round(4).tolist(),
        tile_accuracy_by_step={k: (v / (n * ROWS * COLS)).round(4).tolist() for k, v in tile_right.items()},
        tile_accuracy_on_changed_by_step={k: (v / np.maximum(n_tiles_changed, 1)).round(4).tolist()
                                          for k, v in tile_right_changed.items()},
        tile_accuracy_on_reliable_by_step={k: (v / np.maximum(n_tiles_reliable, 1)).round(4).tolist()
                                           for k, v in tile_right_reliable.items()},
        tile_changed_fraction_by_step=(n_tiles_changed / (n * ROWS * COLS)).round(4).tolist(),
        tile_reliable_fraction_by_step=(n_tiles_reliable / (n * ROWS * COLS)).round(4).tolist(),
        tile_class_recall={c: dict(tiles=int(class_total[i]), recall=round(float(class_right[i] / class_total[i]), 4))
                           for i, c in enumerate(tile_classes) if class_total[i] > 0},
        reward=dict(steps_with_reward=int(events.sum()), steps=len(reward_rows),
                    mean_prediction_on_reward_steps=float(reward_rows[events, 1].mean()) if events.any() else None,
                    mean_prediction_on_other_steps=float(reward_rows[~events, 1].mean()),
                    mse=float(np.mean((reward_rows[:, 1] - reward_rows[:, 2]) ** 2)),
                    mse_zero_baseline=float(np.mean(reward_rows[:, 2] ** 2))),
        examples=list(saved),
    )
    if descriptions:
        result["description_facts_by_step"] = {str(d): {k: v.result() for k, v in scores.items()}
                                               for d, scores in text_scores.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    save_json(args.out / "imagine_metrics.json", result)
    if descriptions:
        with (args.out / "imagine_descriptions.jsonl").open("w") as out:
            out.writelines(json.dumps(row) + "\n" for row in text_rows)
    np.savez_compressed(args.out / "imagine_examples.npz",
                        **{f"{k}__{part}": np.asarray(v[part]) for k, v in saved.items() for part in ("real", "model", "recon", "start_recon", "actions", "rewards")})
    save_json(args.out / "imagine_examples_text.json", {k: dict(goal=v["goal"], reasoning=v["reasoning"]) for k, v in saved.items()})
    print(json.dumps({k: result[k] for k in ("checkpoint_epoch", "starts", "reward")}), flush=True)


if __name__ == "__main__":
    main()
