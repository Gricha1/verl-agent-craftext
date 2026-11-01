#!/usr/bin/env python3
"""Describe the CURRENT state of every transition in natural language with the LLM teacher (Gemma, as reasoning).

Same LLM and serving as label_reasoning.py, but the LLM sees only the public state (task, visible area, inventory,
vitals: observation_text from replay_teacher_states.py), not the expert action and no facts about later steps.
It writes what the agent sees now and what of it matters for the task.

  python label_state_description.py --states states.jsonl --out state_descriptions.jsonl \
    --url http://127.0.0.1:8384/v1 --model craftax-gemma

Output: one row per transition (episode_id, step, state_description); finished rows are skipped on a rerun.
Requests and raw answers go to <out>.requests.jsonl.gz.
"""
from __future__ import annotations

import argparse
import gzip
import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RULES = """You describe the current state of an agent in Crafter, a 2D survival and crafting game.
You see only what the agent sees: the visible area around it, its inventory and vitals.
Return a JSON object with description: one or two short sentences about the CURRENT state only.
Name the objects next to the agent and the nearest important ones with their rough direction (left, right, up,
down), what the agent faces, the relevant inventory, and any low vitals or nearby danger (zombie, skeleton, lava).
Focus on what matters for the task, e.g. "A tree is two cells left and stone is up; the agent faces grass and
has no wood yet."
Do not suggest actions, do not guess what happens next, do not give coordinates. Do not invent objects."""

SCHEMA = dict(type="object", additionalProperties=False,
              properties=dict(description=dict(type="string", maxLength=300)), required=["description"])


def messages(row: dict) -> list[dict]:
    return [dict(role="system", content=RULES),
            dict(role="user", content=f"TASK: {row['instruction']}\n\n{row['observation_text']}")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8384/v1")
    parser.add_argument("--model", default="craftax-gemma")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0, help="label only the first N rows (smoke)")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.states.open()]
    if args.limit:
        rows = rows[: args.limit]
    done = set()
    if args.out.exists():
        done = {(r["episode_id"], r["step"]) for r in map(json.loads, args.out.open())}
    todo = [r for r in rows if (r["episode_id"], r["step"]) not in done]
    print(json.dumps(dict(rows=len(rows), done=len(done), todo=len(todo))), flush=True)

    lock = threading.Lock()
    url = args.url.rstrip("/") + "/chat/completions"
    totals = dict(requests=0, prompt_tokens=0, output_tokens=0, retries=0)
    started = time.monotonic()

    def label(row: dict) -> dict:
        body = dict(model=args.model, messages=messages(row), temperature=0.0, top_p=1.0, max_tokens=args.max_tokens,
                    response_format=dict(type="json_schema",
                                         json_schema=dict(name="state_description", strict=True, schema=SCHEMA)))
        for attempt in range(3):
            request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=300) as response:
                result = json.load(response)
            choice = result["choices"][0]
            try:
                text = " ".join(json.loads(choice["message"]["content"])["description"].split())
                ok = choice.get("finish_reason") == "stop" and bool(text)
            except (json.JSONDecodeError, KeyError, TypeError):
                ok = False
            with lock:
                totals["requests"] += 1
                totals["prompt_tokens"] += result.get("usage", {}).get("prompt_tokens", 0)
                totals["output_tokens"] += result.get("usage", {}).get("completion_tokens", 0)
                with gzip.open(str(args.out) + ".requests.jsonl.gz", "at") as log:
                    log.write(json.dumps(dict(key=[row["episode_id"], row["step"]], request=body, response=result)) + "\n")
            if ok:
                return dict(episode_id=row["episode_id"], step=row["step"], state_description=text, attempts=attempt + 1)
            with lock:
                totals["retries"] += 1
        raise RuntimeError(f"no valid answer for episode {row['episode_id']} step {row['step']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as out, ThreadPoolExecutor(args.concurrency) as pool:
        for n, result in enumerate(pool.map(label, todo), 1):
            out.write(json.dumps(result) + "\n")
            if n % 500 == 0 or n == len(todo):
                out.flush()
                elapsed = time.monotonic() - started
                print(json.dumps(dict(labelled=n, of=len(todo), per_second=round(n / elapsed, 2), **totals)), flush=True)


if __name__ == "__main__":
    main()
