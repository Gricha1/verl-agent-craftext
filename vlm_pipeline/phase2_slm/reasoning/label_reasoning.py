"""Write a reasoning sentence for every teacher action with an LLM teacher (ZeroHero recipe).

ZeroHero labels its Craftax data with Gemma-4-26B-A4B-it behind vLLM: the LLM sees the
same public observation as the learner and answers a strict JSON schema
{reasoning, action}; the SFT target is the reasoning followed by the action.  Here the
action comes from the PPO teacher A (one task set, one teacher), so the schema allows
only that action and the LLM writes the reasoning that leads to it.

  python label_reasoning.py --states states.jsonl --out reasoning.jsonl \
    --url http://127.0.0.1:8383/v1 --model craftax-gemma

Input rows come from replay_teacher_states.py.  Output: one row per transition
(episode_id, step, action, reasoning, effective); finished rows are skipped on a rerun.
Requests and raw answers go to <out>.requests.jsonl.gz.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RULES = """You explain the moves of an expert agent in Crafter, a 2D survival and crafting game.
You see only what the learner sees: the visible area around the agent, its inventory and vitals.
The expert has already chosen the next action. Write the reasoning that leads to this action.
Return a JSON object with reasoning (one short sentence) and action (the expert's action).
The reasoning must check the CURRENT observation: name what is visible and how this action makes
progress on the task, e.g. "No wood yet and a tree is two cells left, so move left toward it."
Moves are absolute map directions: UP toward the top row, DOWN, LEFT, RIGHT; moving also turns
the agent to face that way. DO interacts with the faced cell: it chops a tree, mines, drinks
water, attacks or eats. Crafting needs a crafting table nearby and enough wood.
Do not mention the expert, hints, future moves or coordinates. Do not invent objects."""
HINDSIGHT = 8
INEFFECTIVE = ("Private note: this action leaves the world and inventory unchanged; explain it as "
               "waiting or a wasted move without inventing an effect.")

SCHEMA = dict(type="object", additionalProperties=False,
              properties=dict(reasoning=dict(type="string", maxLength=240),
                              action=dict(type="string", enum=[])),
              required=["reasoning", "action"])


def effective(row: dict, next_row: dict | None) -> bool:
    """Did the action change anything the learner can see (map, inventory, achievements, facing)?"""
    if next_row is None:
        return True  # the last step ends the episode (success or time limit)
    a, b = row["view"], next_row["view"]
    strip = lambda grid: "".join("." if ch in "ZCSa" else ch for ch in grid)  # mobs move by themselves
    return (a["inventory"] != b["inventory"] or a["achieved"] != b["achieved"] or a["facing"] != b["facing"]
            or a["sleeping"] != b["sleeping"] or strip(a["grid"]) != strip(b["grid"]))


MOVES = {"LEFT": (0, -1), "RIGHT": (0, 1), "UP": (-1, 0), "DOWN": (1, 0)}
WORDS = re.compile(r"(\d+) (left|right|up|down)")
MOVE_CLAIM = re.compile(r"\b(?:move|moving|go|going|step|head|heading|walk|walking)\s+(?:one cell\s+|a cell\s+)?"
                        r"(left|right|up|down)\b", re.I)


def offset(words: str) -> tuple[int, int]:
    d_row = d_col = 0
    for n, side in WORDS.findall(words):
        d_col += {"left": -int(n), "right": int(n)}.get(side, 0)
        d_row += {"up": -int(n), "down": int(n)}.get(side, 0)
    return d_row, d_col


def action_facts(row: dict, is_effective: bool, future: list[dict]) -> str:
    """Computed facts about this action, so the LLM does not have to read the grid itself."""
    view, action = row["view"], row["teacher_action"]
    facts = []
    if action in MOVES:
        target = view["neighbours"][action]
        facts.append(f"Moving {action} enters a {target} cell." if target in ("grass", "sand", "path")
                     else f"{action} is blocked by {target}: the agent only turns to face it.")
        step = MOVES[action]
        for name, words in list(view["nearest"].items())[:3]:
            d_row, d_col = offset(words)
            before, after = abs(d_row) + abs(d_col), abs(d_row - step[0]) + abs(d_col - step[1])
            facts.append(f"The nearest {name} gets {'closer' if after < before else 'farther'} ({before} -> {after} cells).")
    elif action == "DO":
        facts.append(f"DO acts on the faced cell, which is {view['facing_block']}.")
    if not is_effective:
        facts.append("This action changes nothing in the world or the inventory.")
    gained = []
    for k, later in enumerate(future, 1):
        for item, count in later["view"]["inventory"].items():
            if count > (row["view"]["inventory"].get(item, 0)) and item not in gained:
                gained.append(item)
                facts.append(f"{k} steps later the agent has more {item.replace('_', ' ')}.")
    return "Private facts, do not quote: " + " ".join(facts)


def messages(row: dict, is_effective: bool, future: list[dict]) -> list[dict]:
    user = (f"TASK: {row['instruction']}\n\n{row['observation_text']}\n\n"
            f"Expert action: {row['teacher_action']}\n{action_facts(row, is_effective, future)}"
            + ("" if is_effective else "\n" + INEFFECTIVE))
    return [dict(role="system", content=RULES), dict(role="user", content=user)]


def inconsistency(row: dict, reasoning: str) -> str | None:
    """Rule check of the sentence against the action and the faced cell (ZeroHero-style repair)."""
    action = row["teacher_action"]
    claims = {m.lower() for m in MOVE_CLAIM.findall(reasoning)}
    if action in MOVES and claims - {action.lower()}:
        return f"The action is {action}, but the reasoning says to move {', '.join(sorted(claims))}."
    if action not in MOVES and claims:
        return f"The action is {action}, not a move, but the reasoning says to move {', '.join(sorted(claims))}."
    if action == "DO" and row["view"]["facing_block"] != "tree" and re.search(r"\b(chop|harvest)", reasoning, re.I):
        return f"The faced cell is {row['view']['facing_block']}, not a tree: DO cannot chop anything here."
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8383/v1")
    parser.add_argument("--model", default="craftax-gemma")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--limit", type=int, default=0, help="label only the first N rows (smoke)")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.states.open()]
    if args.limit:
        rows = rows[: args.limit]
    done = set()
    if args.out.exists():
        done = {(r["episode_id"], r["step"]) for r in map(json.loads, args.out.open())}
    following = {(r["episode_id"], r["step"]): r for r in rows}
    todo = [r for r in rows if (r["episode_id"], r["step"]) not in done]
    print(json.dumps(dict(rows=len(rows), done=len(done), todo=len(todo))), flush=True)

    lock = threading.Lock()
    url = args.url.rstrip("/") + "/chat/completions"
    totals = dict(requests=0, prompt_tokens=0, output_tokens=0, retries=0)
    started = time.monotonic()

    def label(row: dict) -> dict:
        key = (row["episode_id"], row["step"])
        future = [following[(key[0], key[1] + k)] for k in range(1, HINDSIGHT + 1) if (key[0], key[1] + k) in following]
        is_effective = effective(row, future[0] if future else None)
        schema = json.loads(json.dumps(SCHEMA))
        schema["properties"]["action"]["enum"] = [row["teacher_action"]]
        body = dict(model=args.model, messages=messages(row, is_effective, future), temperature=0.0, top_p=1.0,
                    max_tokens=args.max_tokens,
                    response_format=dict(type="json_schema",
                                         json_schema=dict(name="explained_action", strict=True, schema=schema)))
        problem = None
        for attempt in range(3):
            request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=300) as response:
                result = json.load(response)
            choice = result["choices"][0]
            try:
                item = json.loads(choice["message"]["content"])
                ok = (choice.get("finish_reason") == "stop" and item["action"] == row["teacher_action"]
                      and item["reasoning"].strip())
            except (json.JSONDecodeError, KeyError, TypeError):
                ok = False
            with lock:
                totals["requests"] += 1
                totals["prompt_tokens"] += result.get("usage", {}).get("prompt_tokens", 0)
                totals["output_tokens"] += result.get("usage", {}).get("completion_tokens", 0)
                with gzip.open(str(args.out) + ".requests.jsonl.gz", "at") as log:
                    log.write(json.dumps(dict(key=[row["episode_id"], row["step"]], request=body, response=result)) + "\n")
            if ok:
                reasoning = " ".join(item["reasoning"].split())
                problem = inconsistency(row, reasoning)
                if problem is None or attempt == 2:
                    return dict(episode_id=row["episode_id"], step=row["step"], action=row["teacher_action"],
                                reasoning=reasoning, effective=is_effective, consistent=problem is None,
                                attempts=attempt + 1)
                body["messages"] = body["messages"][:2] + [
                    dict(role="assistant", content=choice["message"]["content"]),
                    dict(role="user", content=f"Correction: {problem} Rewrite the reasoning so that it leads to "
                                              f"{row['teacher_action']} and matches the observation.")]
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
