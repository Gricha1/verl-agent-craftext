"""Replay the fixed Crafter64 plans with the Dreamer weights from a Stage 2 snapshot."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import replay as fixed
from compare_objects import objects, read_json, rollout_metrics, score
from dreamer4.models.agent_heads import build_agent_heads
from dreamer4.models.world_model import WorldModel
from plot_four_variants import benchmark_key
from reconstruct_benchmark_starts import reconstruct


def load_world(args):
    template = torch.load(args.heads, map_location="cpu", weights_only=False, mmap=True)
    snapshot = torch.load(args.online, map_location="cpu", weights_only=False, mmap=True)
    if snapshot["step"] != args.step or snapshot["version"] != args.step:
        raise ValueError("The selected Stage 2 snapshot does not match the requested update")
    world = WorldModel(args.dyn, device="cuda", n_agent=template["n_agent"])
    world.tok.tok.load_state_dict(snapshot["tokenizer"])
    world.dyn.load_state_dict(snapshot["dynamics"])
    world.tok.tok.eval()
    world.dyn.eval()
    heads = build_agent_heads(template["agent_cfg"]).cuda().eval()
    heads.load_state_dict(snapshot["heads"])
    print(json.dumps(dict(stage2_step=args.step, world_version=snapshot["version"],
                          replay_size=len(snapshot["replay"]))), flush=True)
    return world, heads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixed", "dyn", "heads", "online", "out", "reference", "reference-objects", "context"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--reuse-replay", action="store_true", help="score the saved full replay again")
    args = parser.parse_args()
    if args.limit and args.reuse_replay:
        parser.error("--reuse-replay requires a full evaluation")
    torch.set_num_threads(4)
    torch.manual_seed(7950)
    args.out.mkdir(parents=True, exist_ok=True)
    signature = fixed.frozen_signature(args.fixed)
    audit = fixed.auditor(args.fixed)
    label = f"stage2-scalar-{args.step}"
    benchmark = read_json(args.reference_objects)
    if signature != benchmark["benchmark_sha256"]:
        raise ValueError("The shared reference uses a different fixed bank")
    world, heads = load_world(args)
    sprites = args.out / "replay"
    reconstruction = np.zeros((len(benchmark["objects"]), 2), dtype=int)
    error, frame_count = 0., 0
    for recording in fixed.RECORDINGS:
        source = args.fixed / "episode_recordings" / recording
        cached = args.context / signature / f"{recording}.npy"
        if not cached.is_file():
            raise FileNotFoundError(cached)
        if not args.reuse_replay:
            fixed.replay(world, heads, source, sprites / recording, args.limit, audit, None, cached)
        frames, _, _ = fixed.observed_context(source, audit, None, cached)
        frames = frames[:args.limit] if args.limit else frames
        predictions = reconstruct(world.tok, frames, batch=12)
        for t, (frame, prediction) in enumerate(zip(frames, predictions, strict=True)):
            truth = objects(frame, t)
            pred = objects(prediction, t)
            for o, (expected, guessed) in enumerate(zip(truth, pred, strict=True)):
                reconstruction[o] += score(expected, guessed)
            error += float(np.square((prediction.astype(np.float32) - frame) / 255).mean())
        frame_count += len(frames)
        print(f"reconstructed {recording}: {len(frames)} frames", flush=True)
    if args.limit:
        print(f"Smoke completed with {frame_count} starting frames", flush=True)
        return

    report = read_json(args.reference / "errors.json")
    _, plans, transitions, worlds = benchmark_key(report)
    if report["benchmark_sha256"] != signature:
        raise ValueError("The original replay uses a different fixed bank")
    fixed.measure(audit, sprites, signature, label, args.out / "errors.json")
    stats = rollout_metrics((signature, worlds, (args.reference / "replay", sprites)),
                            model_ids=("original", label))
    if (plans, transitions) != (benchmark["plans"], benchmark["transitions"]):
        raise ValueError("The original replay has an unexpected size")
    if frame_count != benchmark["paired_frames"][0]:
        raise ValueError("The number of reconstructed frames differs")
    if [rows[1:] for rows in stats["eligible"]] != [rows[1:] for rows in benchmark["eligible"]]:
        raise ValueError("The replay uses different reference objects")

    stats["paired_frames"][0] = frame_count
    for o in range(len(reconstruction)):
        stats["eligible"][o][0] = benchmark["eligible"][o][0]
        stats["hits"][0][o][0] = benchmark["hits"][0][o][0]
        stats["hits"][1][o][0] = reconstruction[o].tolist()
    stats["pixel_mse"][0][0] = benchmark["pixel_mse"][0][0]
    stats["pixel_mse"][1][0] = error / frame_count
    (args.out / "objects.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
