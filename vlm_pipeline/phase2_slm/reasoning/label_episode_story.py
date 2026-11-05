#!/usr/bin/env python3
"""Rank the teacher episodes by total reward and let the LLM tell what happened on each.

Every episode is cut into segments of --segment steps. The LLM reads, per step, the action, the state description and
the reward signal, and writes what happened in the segment. Then it reads the segments of the episode and writes the
story of the whole episode, with the measured totals given, so that it does not have to guess them.

  python label_episode_story.py --raw-data <collection> --descriptions state_descriptions.jsonl --out <dir> \
    --url http://127.0.0.1:8400/v1

Output in <dir>: ranking.jsonl (worst first), segment_stories.jsonl, episode_stories.jsonl; requests and raw answers in
*.requests.jsonl.gz. Finished rows are skipped on a rerun.
"""
from __future__ import annotations

import argparse
import gzip
import http.client
import json
import threading
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ACHIEVEMENT = 0.015  # the scaled reward of one new achievement is 0.02; health changes are about 0.002 per point

SEGMENT_RULES = """You watch a short part of an episode of an agent in Crafter, a 2D survival and crafting game.
You get, step by step, the action taken, a description of the state, and the reward events.
Return a JSON object. what_happened: one or two sentences about what the agent did and what changed in these steps.
events: short snake_case names of what happened, e.g. collect_wood, health_loss, zombie_nearby, moved_to_tree.
Use only what the steps show. Do not invent objects, causes or events, and do not give advice."""

EPISODE_RULES = """You read the parts of one episode of an agent in Crafter, a 2D survival and crafting game, in order.
Return a JSON object. what_happened: three to five sentences telling the whole episode: what the agent did, what went
right or wrong and how it ended. key_events: the most important events in order, as short snake_case names.
cause: one sentence with the main reason for the result, based only on what the parts show.
The totals at the top are measured, use them as they are. Do not invent anything and do not give advice."""

SEGMENT_SCHEMA = dict(type="object", additionalProperties=False, required=["what_happened", "events"], properties=dict(
    what_happened=dict(type="string", maxLength=320),
    events=dict(type="array", maxItems=6, items=dict(type="string", maxLength=40))))
EPISODE_SCHEMA = dict(type="object", additionalProperties=False, required=["what_happened", "key_events", "cause"], properties=dict(
    what_happened=dict(type="string", maxLength=800),
    key_events=dict(type="array", maxItems=10, items=dict(type="string", maxLength=40)),
    cause=dict(type="string", maxLength=240)))


def load(raw_data: Path, descriptions: Path) -> dict:
    import pyarrow.parquet as pq
    text = {(int(r["episode_id"]), int(r["step"])): r["state_description"] for r in map(json.loads, descriptions.open())}
    rows = pq.read_table(raw_data / "train.parquet").to_pylist() + pq.read_table(raw_data / "val.parquet").to_pylist()
    episodes = defaultdict(list)
    for row in rows:
        extra = row["extra_info"]
        key = (int(extra["episode_id"]), int(extra["step"]))
        episodes[key[0]].append(dict(step=key[1], action=str(extra["teacher_action"]), reward=float(extra["reward"]),
                                     done=bool(extra["done"]), success=bool(extra["success"]),
                                     instruction=str(extra["instruction"]), description=text[key]))
    for episode in episodes.values():
        episode.sort(key=lambda r: r["step"])
    return dict(episodes)


def summary(episode_id: int, episode: list) -> dict:
    rewards = [r["reward"] for r in episode]
    success = any(r["success"] for r in episode)
    ended = episode[-1]["done"]
    return dict(episode_id=episode_id, instruction=episode[0]["instruction"], steps=len(episode), reward=sum(rewards),
                achievements=sum(r > ACHIEVEMENT for r in rewards), negative_reward=sum(r for r in rewards if r < 0),
                success=success, ending="task done" if success else "died or lost" if ended else "cut by the step limit")


def event(reward: float) -> str:
    if reward > ACHIEVEMENT:
        return "new achievement"
    if reward < 0:
        return "lost health"
    if reward > 0:
        return "regained health"
    return "none"


def segment_prompt(meta: dict, part: list) -> list:
    lines = [f"step {r['step']}: action {r['action']}; reward event: {event(r['reward'])}; state: {r['description']}" for r in part]
    return [dict(role="system", content=SEGMENT_RULES),
            dict(role="user", content=f"TASK: {meta['instruction']}\n\n" + "\n".join(lines))]


def episode_prompt(meta: dict, parts: list) -> list:
    head = (f"TASK: {meta['instruction']}\nTotals: {meta['steps']} steps, {meta['achievements']} new achievements, "
            f"result: {meta['ending']}.")
    lines = [f"steps {p['first']}-{p['last']}: {p['what_happened']}" for p in parts]
    return [dict(role="system", content=EPISODE_RULES), dict(role="user", content=head + "\n\n" + "\n".join(lines))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data", required=True, type=Path)
    parser.add_argument("--descriptions", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8400/v1")
    parser.add_argument("--model", default="craftax-gemma")
    parser.add_argument("--concurrency", type=int, default=60)
    parser.add_argument("--segment", type=int, default=16)
    parser.add_argument("--max-parts", type=int, default=40, help="an episode with more segments shows these, evenly spaced")
    parser.add_argument("--extremes", type=int, default=0, help="smoke: only the N worst and N best episodes")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    episodes = load(args.raw_data, args.descriptions)
    ranking = sorted((summary(k, e) for k, e in episodes.items()), key=lambda m: (m["reward"], m["episode_id"]))
    with (args.out / "ranking.jsonl").open("w") as stream:
        for rank, meta in enumerate(ranking):
            stream.write(json.dumps(dict(rank=rank, **meta)) + "\n")
    chosen = ranking[:args.extremes] + ranking[-args.extremes:] if args.extremes else ranking
    print(json.dumps(dict(episodes=len(ranking), chosen=len(chosen), worst=ranking[0]["reward"], best=ranking[-1]["reward"])), flush=True)

    url = args.url.rstrip("/") + "/chat/completions"
    lock = threading.Lock()
    totals = dict(requests=0, prompt_tokens=0, output_tokens=0, retries=0, dropped=0)
    started = time.monotonic()

    def post(body: dict) -> dict:
        for attempt in range(40):
            request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    return json.load(response)
            except (OSError, http.client.HTTPException) as error:
                with lock:
                    totals["dropped"] += 1
                if attempt == 39:
                    raise
                print(json.dumps(dict(dropped=repr(error), attempt=attempt)), flush=True)
                time.sleep(min(60, 2 ** attempt))

    def ask(messages: list, schema: dict, name: str, log_path: Path, key: list, max_tokens: int) -> dict:
        body = dict(model=args.model, messages=messages, temperature=0.0, top_p=1.0, max_tokens=max_tokens,
                    response_format=dict(type="json_schema", json_schema=dict(name=name, strict=True, schema=schema)))
        for _ in range(3):
            result = post(body)
            choice = result["choices"][0]
            try:
                answer = json.loads(choice["message"]["content"])
                ok = choice.get("finish_reason") == "stop" and bool(answer.get("what_happened", "").strip())
            except (json.JSONDecodeError, TypeError, AttributeError):
                ok = False
            with lock:
                totals["requests"] += 1
                totals["prompt_tokens"] += result.get("usage", {}).get("prompt_tokens", 0)
                totals["output_tokens"] += result.get("usage", {}).get("completion_tokens", 0)
                with gzip.open(str(log_path) + ".requests.jsonl.gz", "at") as log:
                    log.write(json.dumps(dict(key=key, request=body, response=result)) + "\n")
            if ok:
                return answer
            with lock:
                totals["retries"] += 1
        raise RuntimeError(f"no valid answer for {key}")

    segment_path, episode_path = args.out / "segment_stories.jsonl", args.out / "episode_stories.jsonl"
    done = {}
    if segment_path.exists():
        done = {(r["episode_id"], r["first"]): r for r in map(json.loads, segment_path.open())}
    todo = []
    for meta in chosen:
        episode = episodes[meta["episode_id"]]
        for start in range(0, len(episode), args.segment):
            if (meta["episode_id"], episode[start]["step"]) not in done:
                todo.append((meta, episode[start:start + args.segment]))
    print(json.dumps(dict(segments_todo=len(todo), segments_done=len(done))), flush=True)

    def tell_segment(item) -> dict:
        meta, part = item
        answer = ask(segment_prompt(meta, part), SEGMENT_SCHEMA, "segment_story", segment_path,
                     [meta["episode_id"], part[0]["step"]], 200)
        return dict(episode_id=meta["episode_id"], first=part[0]["step"], last=part[-1]["step"],
                    what_happened=" ".join(answer["what_happened"].split()), events=answer["events"])

    with segment_path.open("a") as out, ThreadPoolExecutor(args.concurrency) as pool:
        for n, result in enumerate(pool.map(tell_segment, todo), 1):
            out.write(json.dumps(result) + "\n")
            done[(result["episode_id"], result["first"])] = result
            if n % 500 == 0 or n == len(todo):
                out.flush()
                print(json.dumps(dict(segments=n, of=len(todo), per_second=round(n / (time.monotonic() - started), 2), **totals)), flush=True)

    finished = set()
    if episode_path.exists():
        finished = {r["episode_id"] for r in map(json.loads, episode_path.open())}
    parts_of = defaultdict(list)
    for (episode_id, _), row in done.items():
        parts_of[episode_id].append(row)
    remaining = [m for m in chosen if m["episode_id"] not in finished]

    def tell_episode(meta: dict) -> dict:
        parts = sorted(parts_of[meta["episode_id"]], key=lambda p: p["first"])
        if len(parts) > args.max_parts:
            parts = [parts[int(i * (len(parts) - 1) / (args.max_parts - 1))] for i in range(args.max_parts)]
        answer = ask(episode_prompt(meta, parts), EPISODE_SCHEMA, "episode_story", episode_path, [meta["episode_id"]], 500)
        return dict(**meta, what_happened=" ".join(answer["what_happened"].split()), key_events=answer["key_events"],
                    cause=" ".join(answer["cause"].split()))

    with episode_path.open("a") as out, ThreadPoolExecutor(args.concurrency) as pool:
        for n, result in enumerate(pool.map(tell_episode, remaining), 1):
            out.write(json.dumps(result) + "\n")
            if n % 200 == 0 or n == len(remaining):
                out.flush()
                print(json.dumps(dict(episodes=n, of=len(remaining), **totals)), flush=True)
    print("STORIES_DONE", flush=True)


if __name__ == "__main__":
    main()
