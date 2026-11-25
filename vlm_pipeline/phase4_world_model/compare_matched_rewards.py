"""Compare two reward heads at the same detected events in fixed Crafter replays."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sheet(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def trace(audit, observation, sprite, row, plan, origin):
    start, start_scene, start_pose, inventory_at_start, unlocked, prior_health = observation
    earned = set(unlocked)
    before, previous_scene, pose = inventory_at_start, start_scene, start_pose
    previous_frame, facing = start, start_pose
    base_y = (2 * plan + 1) * 63
    for k in range(row["plan_len"][plan]):
        action = row["full_actions"][plan][k]
        predicted = row["dream_step_reward"][plan][k]
        frame = sprite[base_y:base_y + 63, (k + 1) * 63:(k + 2) * 63]
        after = audit.inventory(frame)
        next_scene = audit.scene(frame, origin + k + 1)
        next_pose = audit.player_pose(frame, origin + k + 1)
        result = audit.diagnose(action, before, after, previous_scene, next_scene, facing, pose, next_pose,
                                earned, predicted)
        events = result["achievements"]
        if events and all(event in earned for event in events):
            limit = min(result["expected"], .1) if result["expected"] is not None else .1
            repeated_reward = predicted > audit.SCORE_EPSILON and predicted - limit >= audit.REPEATED_REWARD_MARGIN
        else:
            repeated_reward = False

        asleep = pose == "sleep" and before["energy"] is not None and before["energy"] < 9
        sleep_error = asleep and audit.sleep_transition_error(action, before, after, prior_health, next_pose,
                                                             previous_frame, frame, events)
        event_info = None
        if (len(events) == 1 and result["expected"] is not None and not result["impossible"]
                and not result["state_error"] and not sleep_error):
            event = events[0]
            status = "repeat_observed" if event in unlocked else "repeat_in_plan" if event in earned else "first"
            nominal = float(status == "first")
            corrected = predicted - (result["expected"] - nominal)
            event_info = event, status, corrected
        yield event_info, repeated_reward

        earned.update(events)
        prior_health, before = before["health"], after
        previous_frame, previous_scene, pose = frame, next_scene, next_pose
        if next_pose in audit.MOVES:
            facing = next_pose
        elif action in audit.MOVES and next_pose != "sleep":
            facing = action


def compare_world(task):
    fixed, reference, scalar, world, size = task
    sys.path.insert(0, str(fixed / "page_files"))
    import benchmark_audit as audit

    counts = Counter()
    paired, unpaired = [], [Counter(), Counter()]
    recordings = [replay / world for replay in (reference, scalar)]
    manifests = [read_json(path / "manifest.json") for path in recordings]
    if manifests[0]["steps"] != manifests[1]["steps"] or manifests[0]["steps"] * 8 != size["plans"]:
        raise ValueError(f"Different plan counts for {world}")

    observed_prior_health = None
    for t in range(manifests[0]["steps"]):
        paths = [path / f"{t:04d}" for path in recordings]
        base, candidate = rows = [read_json(path.with_suffix(".json")) for path in paths]
        if any(base[key] != candidate[key] for key in ("before", "plan_len", "plan_actions", "full_actions")):
            raise ValueError(f"Mismatched saved inputs at {world}:{t + 1}")
        sprites = [sheet(path.with_suffix(".png")) for path in paths]
        start = sprites[0][:63, :63]
        if not np.array_equal(start, sprites[1][:63, :63]):
            raise ValueError(f"Mismatched start frame at {world}:{t + 1}")
        state = base["before"]
        start_scene, start_pose = audit.scene(start, t), audit.player_pose(start, t)
        observation = (start, start_scene, start_pose, state["inventory"], set(state["achievements"]), observed_prior_health)
        counts["plans"] += len(base["plan_len"])
        for plan, _ in enumerate(base["plan_len"]):
            traces = [trace(audit, observation, sprite, row, plan, t)
                      for sprite, row in zip(sprites, rows, strict=True)]
            for k, ((old, old_repeat), (new, new_repeat)) in enumerate(zip(*traces, strict=True)):
                counts["transitions"] += 1
                counts["original_repeat_flags"] += old_repeat
                counts["scalar_repeat_flags"] += new_repeat
                if old is not None and new is not None and old[:2] == new[:2]:
                    event, status, old_reward = old
                    paired.append([world, t + 1, plan + 1, k + 1, event, status, old_reward, new[2]])
                    counts["original_repeat_flags_matched"] += old_repeat
                    counts["scalar_repeat_flags_matched"] += new_repeat
                else:
                    for info, by_event in zip((old, new), unpaired, strict=True):
                        if info is not None:
                            by_event[info[:2]] += 1
        observed_prior_health = state["inventory"]["health"]

    if (counts["plans"], counts["transitions"]) != (size["plans"], size["transitions"]):
        raise ValueError(f"Unexpected replay size for {world}")
    print(world, f"matched {len(paired)} / {counts['transitions']} transitions", flush=True)
    return world, counts, paired, unpaired


def summarize(rows, first):
    rewards = np.asarray([row[6:] for row in rows], dtype=np.float64)
    difference = rewards[:, 1] - rewards[:, 0]
    return dict(n=len(rows), starts=len({tuple(row[:2]) for row in rows}),
                mean=rewards.mean(axis=0).tolist(), median=np.median(rewards, axis=0).tolist(),
                change_mean=float(difference.mean()), change_median=float(np.median(difference)),
                threshold=.5 if first else .1,
                count_at_threshold=(rewards >= (.5 if first else .1)).sum(axis=0).tolist(),
                count_at_half=(rewards >= .5).sum(axis=0).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixed", "reference", "scalar", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    old_report = read_json(args.reference / "errors.json")
    new_report = read_json(args.scalar / "errors.json")
    signature = hashlib.sha256((args.fixed / "sha256.json").read_bytes()).hexdigest()
    if old_report["benchmark_sha256"] != signature or new_report["benchmark_sha256"] != signature:
        raise ValueError("The two replays and fixed inputs must share the same benchmark")
    worlds = old_report["worlds"]
    if worlds.keys() != new_report["worlds"].keys():
        raise ValueError("The two replays must cover the same worlds")
    tasks = [(args.fixed, args.reference / "replay", args.scalar / "replay", w, size) for w, size in worlds.items()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(compare_world, tasks))

    totals = Counter()
    missed = [Counter(), Counter()]
    rows, groups, by_achievement = [], defaultdict(list), defaultdict(lambda: defaultdict(list))
    for _, counts, pairs, unpaired in results:
        totals.update(counts)
        rows.extend(pairs)
        for model_counts, world_counts in zip(missed, unpaired, strict=True):
            model_counts.update(world_counts)
        for row in pairs:
            event, status = row[4:6]
            groups[status].append(row)
            by_achievement[event][status].append(row)
    expected = old_report["totals"]
    if (totals["plans"], totals["transitions"]) != (expected["plans"], expected["transitions"]):
        raise ValueError("Unexpected pooled benchmark size")
    if (totals["original_repeat_flags"], totals["scalar_repeat_flags"]) != (expected["repeat"], new_report["totals"]["repeat"]):
        raise ValueError("Repeat flags differ from the published audits")
    report = dict(
        benchmark_sha256=signature, model_order=[old_report["variant"], new_report["variant"]], counts=totals,
        by_status={status: summarize(items, status == "first") for status, items in groups.items()},
        by_achievement={event: {status: summarize(items, status == "first") for status, items in statuses.items()}
                        for event, statuses in sorted(by_achievement.items())},
        by_world={world: {status: summarize(items, status == "first") for status in groups
                          if (items := [row for row in pairs if row[5] == status])}
                  for world, _, pairs, _ in results},
        unpaired_usable_events=[{event: {status: counts.get((event, status), 0)
                                         for status in ("first", "repeat_observed", "repeat_in_plan")}
                                 for event in sorted({key[0] for key in counts})} for counts in missed],
        pair_columns=["recording", "step", "plan", "offset", "achievement", "status",
                      "original_reward_minus_health", "scalar_reward_minus_health"],
        pairs=rows,
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["by_status"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
