"""Free-running offline evaluation on the val transitions: the model writes its own answer.

Teacher-forced val metrics give a reasoning model the LLM teacher's sentence, which names the move in ~72%
of the wood val rows, so its action accuracy is inflated.  Here every val row gets only the frame and the
prompt; the materialised model decodes greedily (reasoning, if trained with it, then the action) and is
scored against the PPO teacher's action.  The same code scores action-only models, so B_ce hard / soft /
soft + reasoning are compared on identical rows.

  python eval_offline_reasoning.py --model <materialized dir> --data <phase-2 data dir> \
    [--states <reasoning run>/states.jsonl] --out result.json

Action: accuracy vs the teacher action, unparseable rate, per task.  Reasoning (when the answer has one):
format, word ROUGE-L F1 and exact match vs the Gemma sentence, does the sentence point at the teacher's
action (cue words), does it agree with the model's own action and with the faced cell (the labeller's rule
check, needs --states).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parents[2] / "phase3_verl")]
from reasoning.label_reasoning import inconsistency  # noqa: E402
from craftext_integration.slm_vision.prompt import parse_slm_action  # noqa: E402

# words that point at an action (the leak measured on the teacher-forced val)
CUES = {"LEFT": r"\bleft\b", "RIGHT": r"\bright\b", "UP": r"\bup\b", "DOWN": r"\bdown\b",
        "DO": r"\b(chop|attack|interact|drink|collect|mine|eat|DO)\b", "PLACE_TABLE": r"place.*table",
        "MAKE_WOOD_PICKAXE": r"pickaxe", "MAKE_WOOD_SWORD": r"sword", "PLACE_PLANT": r"(plant|sapling)",
        "NOOP": r"(wait|nothing|no.?op)", "SLEEP": r"sleep"}
REASONING = re.compile(r"<reasoning>(.*?)</reasoning>", re.S)


def rouge_l(a: str, b: str) -> float:
    x, y = a.lower().split(), b.lower().split()
    if not x or not y:
        return 0.0
    dp = [0] * (len(y) + 1)
    for xi in x:
        prev = 0
        for j, yj in enumerate(y, 1):
            prev, dp[j] = dp[j], prev + 1 if xi == yj else max(dp[j], dp[j - 1])
    lcs = dp[-1]
    p, r = lcs / len(x), lcs / len(y)
    return 0.0 if lcs == 0 else 2 * p * r / (p + r)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--states", type=Path, help="states.jsonl of the reasoning run (faced-cell check)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dump", type=Path, help="jsonl with every row's answer (for mismatch inspection)")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True).cuda().eval()
    markers, eos = list(model.config.crafter_visual_token_ids), int(model.config.eos_token_id)
    rows = [json.loads(line) for line in (args.data / "valid.jsonl").open()]
    rows = rows[: args.limit or None]
    states = {}
    if args.states:
        for line in args.states.open():
            s = json.loads(line)
            states[(s["episode_id"], s["step"])] = s["view"]

    def frame(row):
        image = np.load(args.data / row["image_path"])["observation"].astype(np.float32)
        return np.moveaxis(image / 255.0 if image.max() > 1 else image, -1, 0)

    started = time.time()
    answers = [None] * len(rows)
    by_prompt = defaultdict(list)  # equal prompts -> equal lengths, so no padding is needed
    for i, row in enumerate(rows):
        by_prompt[row["prompt"]].append(i)
    with torch.no_grad():
        for prompt, indices in by_prompt.items():
            ids = markers + tok.encode(prompt, add_special_tokens=False)
            for offset in range(0, len(indices), args.batch):
                chunk = indices[offset: offset + args.batch]
                x = torch.tensor([ids] * len(chunk), device="cuda")
                pix = torch.from_numpy(np.stack([frame(rows[i]) for i in chunk])).cuda()
                out = [[] for _ in chunk]
                live = [True] * len(chunk)
                for _ in range(args.max_new_tokens):
                    nxt = model(input_ids=x, pixel_values=pix).logits[:, -1].argmax(-1)
                    for k, t in enumerate(nxt.tolist()):
                        if live[k]:
                            if t == eos:
                                live[k] = False
                            else:
                                out[k].append(t)
                    if not any(live):
                        break
                    x = torch.cat([x, nxt[:, None]], 1)
                for k, i in enumerate(chunk):
                    answers[i] = tok.decode(out[k], skip_special_tokens=True)

    total, hits, invalid = len(rows), 0, 0
    per_task = defaultdict(Counter)
    r = dict(n=0, formatted=0, exact=0, rouge=0.0, cue_teacher=0, cue_any=0, self_consistent=0, faced_ok=0, checked=0)
    samples = []
    dump = args.dump.open("w") if args.dump else None
    for row, answer in zip(rows, answers):
        parsed = parse_slm_action(answer)
        teacher = row["action"]
        task = row["prompt"].split("**TASK:** ")[1].split("\n")[0]
        ok = parsed.valid and parsed.action_name == teacher
        if dump:
            gold_text = REASONING.search(row.get("cot_target", ""))
            dump.write(json.dumps(dict(episode_id=row["source_episode_id"], step=row["step"], task=task,
                                       image_path=row["image_path"], teacher_action=teacher,
                                       model_action=parsed.action_name if parsed.valid else None,
                                       model_answer=answer, gemma=gold_text.group(1).strip() if gold_text else None,
                                       teacher_effective=row.get("reasoning_effective_action"))) + "\n")
        hits += ok
        invalid += not parsed.valid
        per_task[task]["n"] += 1
        per_task[task]["hits"] += ok
        gold = REASONING.search(row["cot_target"])
        if gold is None:
            continue  # action-only data: no reasoning to score
        r["n"] += 1
        own = REASONING.search(answer)
        if own is None:
            continue
        text = own.group(1).strip()
        r["formatted"] += 1
        r["exact"] += text == gold.group(1).strip()
        r["rouge"] += rouge_l(text, gold.group(1))
        cues = [a for a, pattern in CUES.items() if re.search(pattern, text, re.I)]
        r["cue_any"] += bool(cues)
        r["cue_teacher"] += teacher in cues
        view = states.get((row["source_episode_id"], row["step"]))
        if view is not None and parsed.valid:
            r["checked"] += 1
            r["self_consistent"] += inconsistency(dict(teacher_action=parsed.action_name, view=view), text) is None
        if len(samples) < 40:
            samples.append(dict(task=task, step=row["step"], teacher_action=teacher, model_answer=answer,
                                gemma=gold.group(1).strip()))
    if dump:
        dump.close()
    result = dict(model=str(args.model), data=str(args.data), rows=total, seconds=round(time.time() - started, 1),
                  action_accuracy=hits / total, unparseable=invalid / total,
                  per_task={t: dict(n=c["n"], accuracy=c["hits"] / c["n"]) for t, c in per_task.items()})
    if r["n"]:
        f = max(r["formatted"], 1)
        result["reasoning"] = dict(rows=r["n"], formatted=r["formatted"] / r["n"], exact_match=r["exact"] / f,
                                   rouge_l_f1=r["rouge"] / f, names_teacher_action=r["cue_teacher"] / f,
                                   names_some_action=r["cue_any"] / f,
                                   consistent_with_own_action_and_faced_cell=r["self_consistent"] / max(r["checked"], 1),
                                   checked=r["checked"])
        result["samples"] = samples
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
