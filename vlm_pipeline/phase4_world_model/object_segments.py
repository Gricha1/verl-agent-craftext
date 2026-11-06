"""Segments of 17 frames (16 steps) with the object on every frame, whole teacher dataset, at most --cap training segments.

find: writes the segments. run: per checkpoint, step 0 draws the start frame from the posterior, steps 1-16 use only the prior.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from imagine_eval import classify_tiles, tile_prototypes
from train_slm_action_on_rssm import episode_latents
from train_slm_world_model import build_model, load_episodes, read_image

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=("find", "run"), required=True); p.add_argument("--object", required=True); p.add_argument("--cap", type=int, default=5000)
p.add_argument("--checkpoint"); p.add_argument("--name"); p.add_argument("--raw-data", type=Path); p.add_argument("--split-from", type=Path); p.add_argument("--dir", type=Path)
a = p.parse_args()
H = 16
device = torch.device("cuda")
protos, index, names = tile_prototypes()
protos, index = torch.from_numpy(protos).to(device), torch.from_numpy(index).to(device)
ZOMBIE = list(names).index(a.object)
episodes = load_episodes(a.raw_data, None, None)
valid_ids = {json.loads(l)["source_episode_id"] for l in (a.split_from / "valid.jsonl").open()}


def any_zombie(x):
    return (classify_tiles(x.to(device).float()[:, None], protos, index)[0][:, 0] == ZOMBIE).flatten(1).any(1)


if a.mode == "find":
    rows, frames_total, frames_zombie = [], 0, 0
    for done, (key, episode) in enumerate(sorted(episodes.items())):
        x = torch.from_numpy(np.stack([read_image(a.raw_data, i["image"]) for i in episode]))
        has = torch.cat([any_zombie(x[i:i + 512]).cpu() for i in range(0, len(x), 512)]).numpy()
        frames_total += len(has); frames_zombie += int(has.sum())
        run = 0
        for t in range(len(has) - 1, -1, -1):  # run: frames with a zombie from t on
            run = run + 1 if has[t] else 0
            if run >= H + 1:
                rows.append((key, t))
        if done % 200 == 0:
            print(done, len(rows), flush=True)
    a.dir.mkdir(parents=True, exist_ok=True)
    rows.sort()
    found = len(rows)
    shuffle = np.random.default_rng(7950).permutation(found)  # all validation segments, at most --cap of the others
    train = [i for i in shuffle if rows[i][0] not in valid_ids][:a.cap]
    rows = sorted([rows[i] for i in train] + [r for r in rows if r[0] in valid_ids])
    is_valid = [k in valid_ids for k, _ in rows]
    print("segments found", found, "kept", len(rows), flush=True)
    (a.dir / "segments.json").write_text(json.dumps(dict(rows=rows, is_valid=is_valid, found=found, frames=frames_total, frames_with_zombie=frames_zombie)))
    print(dict(episodes=len(episodes), frames=frames_total, frames_with_zombie=frames_zombie, segments=len(rows), segments_valid=sum(is_valid),
               episodes_with_segment=len({k for k, _ in rows}), valid_episodes_with_segment=len({k for (k, _), v in zip(rows, is_valid) if v})), flush=True)
    print("EVAL_DONE", flush=True)
    sys.exit()

info = json.loads((a.dir / "segments.json").read_text())
rows = [tuple(r) for r in info["rows"]]
chosen = {k: episodes[k] for k in {k for k, _ in rows}}
state = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
model = build_model(state["config"], None, device)
model.modules.load_state_dict(state["model"]["support"]); model.eval()
torch.manual_seed(7950)
h_all, z_all, offset = episode_latents(model, a.raw_data, chosen, device)
actions = torch.tensor([[chosen[k][t + d]["action"] for d in range(H)] for k, t in rows], device=device)
starts = torch.tensor([offset[k] + t for k, t in rows])
choose = model.sample if model.prior_modes else model.mode
drawn = np.zeros((len(rows), H + 1), bool)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    for off in range(0, len(rows), 512):
        idx = starts[off:off + 512]
        h, z = h_all[idx].to(device), z_all[idx].to(device)
        drawn[off:off + len(idx), 0] = any_zombie(model.decode(h, z).float()).cpu().numpy()
        for d in range(1, H + 1):
            h, _ = model.transition(None, actions[off:off + len(idx), d - 1], None, h, z, device)
            z = choose(model.prior(h))
            drawn[off:off + len(idx), d] = any_zombie(model.decode(h, z).float()).cpu().numpy()
alive = np.logical_and.accumulate(drawn, axis=1)
is_valid = np.array(info["is_valid"])
report = {}
for tag, mask in (("all", np.ones(len(rows), bool)), ("valid", is_valid), ("train", ~is_valid)):
    report[tag] = dict(segments=int(mask.sum()), survivors=alive[mask].sum(0).tolist(), drawn=drawn[mask].sum(0).tolist())
(a.dir / f"{a.name}.json").write_text(json.dumps(report, indent=1))
print(a.name, report["all"], report["valid"]["survivors"], flush=True)
print("EVAL_DONE", flush=True)
