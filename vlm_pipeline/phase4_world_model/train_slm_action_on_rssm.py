"""Behaviour cloning of the SLM on the frozen RSSM state, continuing from the SLM that writes state descriptions.

The start is the checkpoint of train_slm_text_on_rssm.py: SLM and its h_prefix / z_prefix adapters (the user's choice on
2026-11-06, unlike phase-2 CE from random weights). The state is the one the agent has when it acts: h_0 = 0, z_t the
posterior mode of the real frame, h_{t+1} = RSSM(h_t, z_t, a_t) over the whole episode, with no reset.

Input: "Task: ..." + h tokens + z tokens. Target: "ACTION: X", or "<reasoning>...</reasoning>\nACTION: X" with --mode cot.
Loss: expected cross-entropy under the teacher's 17-action distribution, the soft target of crafter_vision_slm.py,
per expected target token: -sum_a pi(a) log p(target of a | input). Rows come from a phase-2 data dir, same split.

  python3 train_slm_action_on_rssm.py train --rssm-checkpoint <rssm> --text-checkpoint <text run>/best.pt \
    --init-checkpoint <initial_random> --raw-data <collection> --data <phase-2 data dir> --out <dir>
  python3 train_slm_action_on_rssm.py eval --rssm-checkpoint <rssm> --checkpoint <dir>/best.pt ... --seeds 20264000:20264032
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from train_slm_world_model import ACTIONS, build_model, load_episodes, read_image, save_json

ADAPTERS = ("h_prefix", "z_prefix")
HEAD = "Task:\n{}\n\nChoose the next action:\n"


def world(args, device):
    import torch
    state = torch.load(args.rssm_checkpoint, map_location="cpu", weights_only=False)
    rssm = build_model(state["config"], None, device)
    rssm.modules.load_state_dict(state["model"]["support"])
    rssm.eval()
    for parameter in rssm.parameters():
        parameter.requires_grad_(False)
    model = build_model(dict(state["config"], transition="slm", h_tokens=args.h_tokens, z_tokens=args.z_tokens),
                        args.init_checkpoint, device)
    for key in list(model.modules):
        if key not in ADAPTERS:
            del model.modules[key]
    return rssm, model


def load_trainable(model, path):
    import torch
    state = torch.load(path, map_location="cpu", weights_only=False)["model"]
    model.lm.load_state_dict(state["lm"])
    for key in ADAPTERS:
        model.modules[key].load_state_dict(state["adapters"][key])


def trainable_state(model):
    return {"lm": model.lm.state_dict(), "adapters": {k: model.modules[k].state_dict() for k in ADAPTERS}}


def episode_latents(rssm, root, episodes, device, batch=64):
    """h_t, z_t of every state, each episode filtered from its first frame. Returns (h, z, offset)."""
    import torch
    keys = sorted(episodes, key=lambda k: len(episodes[k]))
    offset, total = {}, 0
    for key in keys:
        offset[key] = total
        total += len(episodes[key])
    hs, zs = torch.empty(total, rssm.h_size), torch.empty(total, rssm.z_dim)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(keys), batch):
            group = keys[start:start + batch]
            frames = [torch.from_numpy(np.stack([read_image(root, item["image"]) for item in episodes[k]])) for k in group]
            actions = [torch.tensor([item["action"] for item in episodes[k]]) for k in group]
            h = torch.zeros(len(group), rssm.h_size, device=device)
            z = None
            for t in range(len(frames[-1])):
                alive = [i for i, f in enumerate(frames) if len(f) > t]
                index = torch.tensor(alive, device=device)
                frame = torch.stack([frames[i][t] for i in alive]).to(device)
                if t:
                    previous = torch.stack([actions[i][t - 1] for i in alive]).to(device)
                    h_next, _ = rssm.transition(None, previous, None, h[index], z[index], device)
                else:
                    h_next = h[index]
                z_next = rssm.mode(rssm.posterior(frame, h_next))
                h = h.index_copy(0, index, h_next.to(h.dtype))
                z = z_next if z is None else z.index_copy(0, index, z_next.to(z.dtype))
                rows = torch.tensor([offset[group[i]] + t for i in alive])
                hs[rows], zs[rows] = h_next.float().cpu(), z_next.float().cpu()
    return hs, zs, offset


class Policy:
    """Target scoring and greedy text of the SLM from h, z."""

    def __init__(self, model, device):
        self.model, self.device = model, device
        self.embed = model.lm.get_input_embeddings()
        self.eos = int(model.tok.eos_token_id)
        self.candidates = [model.ids(f"ACTION: {name}") + [self.eos] for name in ACTIONS]

    def prefix(self, goal, h, z):
        torch = self.torch = __import__("torch")
        model = self.model
        head = self.embed(torch.tensor(model.ids(HEAD.format(goal)), device=self.device))
        hs = model.modules["h_prefix"](h[None]).view(model.h_tokens, model.width)
        zs = model.modules["z_prefix"](z[None]).view(model.z_tokens, model.width)
        return torch.cat((head, hs.to(head.dtype), zs.to(head.dtype)), dim=0)

    def log_probs(self, goals, reasonings, h, z):
        """log p(reasoning + ACTION: a <eos> | input), [B][17]."""
        import torch
        count = len(self.candidates)
        pieces, targets, starts = [], [], []
        for i, (goal, reasoning) in enumerate(zip(goals, reasonings)):
            base = self.prefix(goal, h[i], z[i])
            first = self.model.ids(reasoning) if reasoning else []
            for candidate in self.candidates:
                ids = first + candidate
                pieces.append(torch.cat((base, self.embed(torch.tensor(ids, device=self.device))), dim=0))
                targets.append(ids); starts.append(base.shape[0])
        logits = self.model._run_lm(pieces, self.device).logits.float().log_softmax(-1)
        total = logits.new_zeros(len(pieces))
        size = max(len(t) for t in targets)
        rows = torch.arange(len(pieces), device=self.device)
        ids = torch.tensor([t + [0] * (size - len(t)) for t in targets], device=self.device)
        first = torch.tensor(starts, device=self.device) - 1
        for j in range(size):
            mask = torch.tensor([j < len(t) for t in targets], device=self.device)
            total = total + logits[rows, first + j, ids[:, j]] * mask
        return total.view(len(goals), count)

    def act(self, goal, h, z, max_tokens):
        """Greedy text from the input; a key-value cache keeps a long reasoning cheap."""
        import torch
        piece = self.prefix(goal, h, z)[None]
        past, written = None, []
        for _ in range(max_tokens):
            output = self.model.lm(inputs_embeds=piece, past_key_values=past, use_cache=True)
            past = output.past_key_values
            token = int(output.logits[0, -1].float().argmax())
            if token == self.eos:
                break
            written.append(token)
            piece = self.embed(torch.tensor([[token]], device=self.device))
        return self.model.tok.decode(written, skip_special_tokens=True)


def read_rows(path, mode):
    rows = []
    for line in path.open():
        r = json.loads(line)
        reasoning = ""
        if mode == "cot":
            target, action = r["cot_target"], r["action_target"]
            if not target.endswith(action):
                raise ValueError(f"reasoning target does not end with the action: {target[-40:]}")
            reasoning = target[:-len(action)]
        rows.append(dict(key=(int(r["source_episode_id"]), int(r["step"])), reasoning=reasoning, teacher=r["teacher_probs"]))
    return rows


def train(args):
    import torch
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda")
    rssm, model = world(args, device)
    load_trainable(model, args.text_checkpoint)
    policy = Policy(model, device)
    train_rows, valid_rows = read_rows(args.data / "train.jsonl", args.mode), read_rows(args.data / "valid.jsonl", args.mode)
    keep = {r["key"][0] for r in train_rows + valid_rows}
    episodes = {k: e for k, e in load_episodes(args.raw_data, None, None).items() if k in keep}
    started = time.time()
    h, z, offset = episode_latents(rssm, args.raw_data, episodes, device)
    index = lambda rows: torch.tensor([offset[r["key"][0]] + r["key"][1] for r in rows])
    args.out.mkdir(parents=True, exist_ok=True)
    parameters = list(model.lm.parameters()) + [p for k in ADAPTERS for p in model.modules[k].parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=.01)
    save_json(args.out / "experiment.json", {**{k: str(v) for k, v in vars(args).items()}, "train_rows": len(train_rows),
                                            "valid_rows": len(valid_rows), "latents_seconds": time.time() - started})
    print(json.dumps(dict(train=len(train_rows), valid=len(valid_rows), latents=time.time() - started)), flush=True)
    lengths = torch.tensor([len(c) for c in policy.candidates], device=device, dtype=torch.float32)

    def loss_of(rows):
        at = index(rows)
        goals = [episodes[r["key"][0]][r["key"][1]]["instruction"] for r in rows]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scores = policy.log_probs(goals, [r["reasoning"] for r in rows], h[at].to(device), z[at].to(device))
        teacher = torch.tensor([r["teacher"] for r in rows], device=device, dtype=torch.float32)
        first = torch.tensor([len(model.ids(r["reasoning"])) if r["reasoning"] else 0 for r in rows], device=device, dtype=torch.float32)
        total = -(teacher * scores).sum()
        size = (teacher * (lengths + first[:, None])).sum()
        return total, size, scores, teacher

    best, updates, started = float("inf"), 0, time.time()
    order = list(range(len(train_rows)))
    for epoch in range(1, args.epochs + 1):
        random.shuffle(order)
        model.train()
        sums, count = 0.0, 0
        for start in range(0, len(order), args.batch_size):
            batch = [train_rows[i] for i in order[start:start + args.batch_size]]
            for group in optimizer.param_groups:
                group["lr"] = args.lr * min(1., (updates + 1) / 100)
            optimizer.zero_grad(set_to_none=True)
            total, size, _, _ = loss_of(batch)
            loss = total / size
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            updates += 1; sums += float(loss); count += 1
            if args.max_updates and updates >= args.max_updates:
                break
            if updates % 500 == 0:
                print(json.dumps(dict(epoch=epoch, updates=updates, loss=round(sums / count, 4), seconds=round(time.time() - started))), flush=True)
        model.eval()
        total, size, hits, seen, entropy = 0.0, 0.0, 0, 0, 0.0
        sample = valid_rows[::args.valid_stride]
        with torch.inference_mode():
            for start in range(0, len(sample), args.batch_size):
                a, b, scores, teacher = loss_of(sample[start:start + args.batch_size])
                total, size = total + float(a), size + float(b)
                hits += int((scores.argmax(-1) == teacher.argmax(-1)).sum()); seen += len(teacher)
                p = scores.log_softmax(-1)
                entropy += float(-(p.exp() * p).sum())
        row = dict(epoch=epoch, updates=updates, seconds=time.time() - started, train_loss=sums / count, validation_loss=total / size,
                   validation_action_accuracy=hits / seen, validation_entropy=entropy / seen)
        print(json.dumps(row), flush=True)
        with (args.out / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        state = dict(model=trainable_state(model), metrics=row, mode=args.mode)
        if row["validation_loss"] < best:
            best = row["validation_loss"]
            torch.save(state, args.out / "best.pt")
        if epoch == args.epochs or epoch % args.save_every == 0:
            torch.save(state, args.out / f"epoch_{epoch:03d}.pt")
        if args.max_updates and updates >= args.max_updates:
            break
    save_json(args.out / "complete.json", dict(best_validation_loss=best, updates=updates))
    print("TRAIN_DONE", flush=True)


def evaluate(args):
    """Closed loop in the teacher env, greedy: action from h_t, z_t; real frame; h_{t+1}; posterior z_{t+1}."""
    import torch
    device = torch.device("cuda")
    rssm, model = world(args, device)
    load_trainable(model, args.checkpoint)
    model.eval()
    policy = Policy(model, device)
    sys.path[:0] = [str(args.teacher_root), str(args.teacher_root / "Craftax")]
    sys.path.insert(0, str(args.repo))
    import jax, jax.numpy as jnp
    from craftax.craftax_env import make_craftax_env_from_name
    from craftext.environment.craftext_wrapper import InstructionWrapper
    from craftext_integration.slm_vision.prompt import parse_slm_action
    base = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", False)
    params = base.default_params
    env = InstructionWrapper(base, args.settings)
    tasks = list(env.scenario_handler.scenario_data.instructions_list)
    step = jax.jit(env.step)
    first, last = map(int, args.seeds.split(":"))
    episodes = []
    frame_of = lambda obs: torch.from_numpy(np.moveaxis(np.asarray(obs, dtype=np.float32), -1, 0).copy()).to(device)[None]
    started = time.time()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for task in range(len(tasks)):
            for seed in range(first, last):
                key = jax.random.PRNGKey(seed)
                obs, state = env.reset(key, params, instruction_idx=task)
                h = torch.zeros(1, rssm.h_size, device=device)
                z = rssm.mode(rssm.posterior(frame_of(obs), h))
                actions, responses, success, ret = collections.Counter(), [], False, 0.0
                for t in range(args.max_steps):
                    text = policy.act(tasks[task], h[0], z[0], args.max_tokens)
                    name = parse_slm_action(text).action_name
                    if len(responses) < 3:
                        responses.append(text)
                    actions[name] += 1
                    key, k = jax.random.split(key)
                    number = ACTIONS.index(name)
                    obs, state, r, done, _ = step(k, state, jnp.int32(number), params)
                    ret += float(r)
                    success = bool(state.instruction_done)
                    if bool(done):
                        break
                    h, _ = rssm.transition(None, torch.tensor([number], device=device), None, h, z, device)
                    z = rssm.mode(rssm.posterior(frame_of(obs), h))
                episodes.append(dict(task=tasks[task], seed=seed, steps=t + 1, success=success, ret=ret, top_actions=actions.most_common(4), first_responses=responses))
                print(json.dumps(episodes[-1]), flush=True)
    result = dict(instructions=tasks, episodes=episodes, success_rate=sum(e["success"] for e in episodes) / len(episodes), seconds=time.time() - started)
    args.output.write_text(json.dumps(result, indent=1))
    print(json.dumps(dict(success_rate=result["success_rate"], successes=sum(e["success"] for e in episodes), episodes=len(episodes))), flush=True)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "eval"):
        p = commands.add_parser(name)
        p.add_argument("--rssm-checkpoint", type=Path, required=True)
        p.add_argument("--init-checkpoint", type=Path, required=True)
        p.add_argument("--h-tokens", type=int, default=50)
        p.add_argument("--z-tokens", type=int, default=5)
        p.add_argument("--seed", type=int, default=7101)
    run = commands.choices["train"]
    run.add_argument("--text-checkpoint", type=Path, required=True)
    run.add_argument("--raw-data", type=Path, required=True)
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--mode", choices=("action", "cot"), default="action")
    run.add_argument("--epochs", type=int, default=12)
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--lr", type=float, default=1e-4)
    run.add_argument("--save-every", type=int, default=4)
    run.add_argument("--valid-stride", type=int, default=8)
    run.add_argument("--max-updates", type=int, default=0)
    check = commands.choices["eval"]
    check.add_argument("--checkpoint", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--teacher-root", type=Path, required=True)
    check.add_argument("--repo", type=Path, required=True)
    check.add_argument("--settings", default="achievements_wood")
    check.add_argument("--seeds", default="20264000:20264032")
    check.add_argument("--max-steps", type=int, default=400)
    check.add_argument("--max-tokens", type=int, default=160)
    args = parser.parse_args()
    train(args) if args.command == "train" else evaluate(args)


if __name__ == "__main__":
    main()
