#!/usr/bin/env python3
"""The SLM writes the next state from h_t, z_t of a frozen RSSM world model (phase 4).

The SLM world models keep h and z trained through the SLM, and their reconstructions lose isolated trees, so their text
cannot say where a tree is. Here the RSSM alone sees the world: its encoder, GRU, prior and decoder are frozen. The SLM
is trained from its initial weights, all of it with the h / z adapters, on one loss, the NLL of the Gemma description
of x_{t+1} given the goal, h_t, z_t and a_t:

  [E("Task: g ... after action:"); h tokens; z tokens; E("ACTION: a"); E(eos); "Next state: <description of x_{t+1}><eos>"]

h_t, z_t: the RSSM filters up to --context real frames ending at x_t, as imagine_eval.py does (posterior mode). They are
computed once, nothing here reaches the RSSM. Every --facts-every epochs the text is scored as in imagine_eval.py, by
facts against Gemma's description, on the same starts: "filtered" reads the RSSM state of the real x_{t+d-1}, "model"
the state the RSSM imagined from x_t with the real actions (prior mode).

  python train_slm_text_on_rssm.py --rssm-checkpoint <gru run>/run/epoch_300.pt --init-checkpoint <run>/initial_random \
    --raw-data <teacher collection> --split-from <phase-2 CE data dir> --descriptions state_descriptions.jsonl --out <dir>
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from description_facts import FactScore
from train_slm_world_model import build_model, load_episodes, read_image, save_json

ADAPTERS = ("h_prefix", "z_prefix")


def rssm_latents(rssm, root, episodes, context, device, batch=256):
    """h_t, z_t of every state, each filtered from zero through its last `context` real frames. Returns (h, z, offset)."""
    import torch
    offset, hs, zs, total = {}, [], [], 0
    for key, episode in episodes.items():
        offset[key] = total
        total += len(episode)
        frames = torch.from_numpy(np.stack([read_image(root, item["image"]) for item in episode]))
        actions = torch.tensor([item["action"] for item in episode])
        h_out, z_out = torch.zeros(len(episode), rssm.h_size), torch.zeros(len(episode), rssm.z_dim)
        for length in range(1, min(context, len(episode)) + 1):  # the first states have a shorter history
            steps = list(range(length - 1, len(episode))) if length == context else [length - 1]
            for start in range(0, len(steps), batch):
                ends = torch.tensor(steps[start:start + batch])
                h = torch.zeros(len(ends), rssm.h_size, device=device)
                z = rssm.mode(rssm.posterior(frames[ends - length + 1].to(device), h))
                for back in range(length - 2, -1, -1):
                    h, _ = rssm.transition(None, actions[ends - back - 1].to(device), None, h, z, device)
                    z = rssm.mode(rssm.posterior(frames[ends - back].to(device), h))
                h_out[ends], z_out[ends] = h.float().cpu(), z.float().cpu()
        hs.append(h_out)
        zs.append(z_out)
    return torch.cat(hs), torch.cat(zs), offset


class Text:
    """Batches of (episode, t): the inputs of the transition out of x_t and the description of x_{t+1}."""

    def __init__(self, model, episodes, h, z, offset, device):
        self.model, self.episodes, self.h, self.z, self.offset, self.device = model, episodes, h, z, offset, device

    def inputs(self, batch):
        import torch
        index = torch.tensor([self.offset[k] + t for k, t in batch])
        goals = [self.episodes[k][t]["instruction"] for k, t in batch]
        actions = torch.tensor([self.episodes[k][t]["action"] for k, t in batch], device=self.device)
        return goals, actions, [None] * len(batch), self.h[index].to(self.device), self.z[index].to(self.device)

    def nll(self, batch):
        """Summed token NLL per sample and the number of tokens."""
        goals, actions, reasoning, h, z = self.inputs(batch)
        targets = [self.episodes[k][t + 1]["description"] for k, t in batch]
        return self.model.transition(goals, actions, reasoning, h, z, self.device, targets=targets)[2:]


def validation_nll(text, samples, batch_size):
    import torch
    nll, tokens = 0.0, 0.0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(samples), batch_size):
            sums, lengths = text.nll(samples[start:start + batch_size])
            nll, tokens = nll + float(sums.sum()), tokens + float(lengths.sum())
    return nll / tokens


def score_facts(text, rssm, starts, steps, batch_size, out):
    """Text of x_{t+d} from the imagined and from the filtered RSSM state, scored by facts against Gemma."""
    import torch
    device, episodes = text.device, text.episodes
    scores = {d: {k: FactScore() for k in ("model", "filtered", "copy_last")} for d in steps}
    rows = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for offset in range(0, len(starts), batch_size):
            batch = starts[offset:offset + batch_size]
            goals, _, none, h, z = text.inputs(batch)
            for d in range(1, max(steps) + 1):
                at = [(k, t + d - 1) for k, t in batch]
                _, actions, _, h_real, z_real = text.inputs(at)
                if d in steps:
                    imagined = text.model.describe(goals, actions, none, h, z, device)
                    filtered = text.model.describe(goals, actions, none, h_real, z_real, device)
                    for (k, t), guess, seen in zip(batch, imagined, filtered):
                        real, first = episodes[k][t + d]["description"], episodes[k][t]["description"]
                        scores[d]["model"].add(guess, real, first)
                        scores[d]["filtered"].add(seen, real, first)
                        scores[d]["copy_last"].add(first, real, first)
                        rows.append(dict(episode=k, t=t, d=d, goal=episodes[k][t]["instruction"], real=real, first=first,
                                         model=guess, filtered=seen))
                h, _ = rssm.transition(None, actions, None, h, z, device)
                z = rssm.mode(rssm.prior(h))
    with out.open("w", encoding="utf-8") as stream:
        stream.writelines(json.dumps(row) + "\n" for row in rows)
    return {str(d): {k: v.result() for k, v in by_who.items()} for d, by_who in scores.items()}


def trainable_state(model):
    return {"lm": model.lm.state_dict(), "adapters": {k: model.modules[k].state_dict() for k in ADAPTERS}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rssm-checkpoint", type=Path, required=True, help="frozen RSSM (--transition gru) world model")
    parser.add_argument("--init-checkpoint", type=Path, required=True, help="initial_random of a phase-2 run")
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument("--split-from", type=Path, required=True, help="its valid.jsonl episodes are the validation episodes")
    parser.add_argument("--descriptions", type=Path, required=True, help="state_descriptions.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--h-tokens", type=int, default=50)
    parser.add_argument("--z-tokens", type=int, default=5)
    parser.add_argument("--context", type=int, default=8, help="real frames the RSSM filters, x_t last")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--facts-every", type=int, default=5, help="epochs between scores of the written text")
    parser.add_argument("--facts-steps", default="1,4,8,16", help="imagined steps d")
    parser.add_argument("--facts-limit", type=int, default=0, help="smoke: score only this many starts")
    parser.add_argument("--max-updates", type=int, default=0, help="smoke: stop after this many updates")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7970)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(args.out)
    import torch
    from torch.utils.tensorboard import SummaryWriter
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda")
    state = torch.load(args.rssm_checkpoint, map_location="cpu", weights_only=False)
    if state["config"].get("transition") != "gru":
        parser.error("--rssm-checkpoint must be a --transition gru world model")
    rssm = build_model(state["config"], None, device)
    rssm.modules.load_state_dict(state["model"]["support"])
    rssm.eval()
    for parameter in rssm.parameters():
        parameter.requires_grad_(False)
    model = build_model(dict(state["config"], transition="slm", h_tokens=args.h_tokens, z_tokens=args.z_tokens),
                        args.init_checkpoint, device)
    for key in list(model.modules):  # the SLM keeps its adapters; h_transition stays only because transition() calls it
        if key not in ADAPTERS + ("h_transition",):
            del model.modules[key]
    model.modules["h_transition"].requires_grad_(False)
    parameters = list(model.lm.parameters()) + [p for k in ADAPTERS for p in model.modules[k].parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=.01)

    descriptions = {(int(r["episode_id"]), int(r["step"])): r["state_description"] for r in map(json.loads, args.descriptions.open())}
    episodes = load_episodes(args.raw_data, None, descriptions)
    valid_ids = {json.loads(line)["source_episode_id"] for line in (args.split_from / "valid.jsonl").open()}
    samples = lambda keep: [(k, t) for k, e in sorted(episodes.items()) if keep(k) for t in range(len(e) - 1)]
    train, valid = samples(lambda k: k not in valid_ids), samples(lambda k: k in valid_ids)
    steps = [int(d) for d in args.facts_steps.split(",")]
    # the starts of imagine_eval.py (horizon 32, stride 4) with its --description-every 4
    starts = [(k, t) for k, e in sorted(episodes.items()) if k in valid_ids for t in range(0, len(e) - 32, 4) if t >= 7][::4]
    starts = starts[:args.facts_limit] if args.facts_limit else starts
    started = time.time()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        h, z, offset = rssm_latents(rssm, args.raw_data, episodes, args.context, device)
    text = Text(model, episodes, h, z, offset, device)
    args.out.mkdir(parents=True, exist_ok=True)
    config = dict({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  architecture="slm_text_on_frozen_rssm", trainable="SLM + h / z adapters", frozen="RSSM",
                  target="description of x_{t+1}", rssm_config=state["config"], rssm_epoch=state["metrics"]["epoch"],
                  lm_parameters=sum(p.numel() for p in model.lm.parameters()), train_samples=len(train),
                  adapter_parameters=sum(p.numel() for k in ADAPTERS for p in model.modules[k].parameters()),
                  validation_samples=len(valid), facts_starts=len(starts), latents_seconds=time.time() - started)
    save_json(args.out / "experiment.json", config)
    print(json.dumps({k: v for k, v in config.items() if k != "rssm_config"}), flush=True)
    writer = SummaryWriter(str(args.out / "tensorboard"))
    best, updates, started = float("inf"), 0, time.time()
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train)
        model.train()
        nll_sum, token_sum, grad_sum, batches = 0.0, 0.0, 0.0, 0
        for start in range(0, len(train), args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sums, lengths = text.nll(train[start:start + args.batch_size])
                loss = sums.sum() / lengths.sum()
            loss.backward()
            grad_sum += float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
            optimizer.step()
            nll_sum, token_sum = nll_sum + float(sums.detach().sum()), token_sum + float(lengths.sum())
            updates, batches = updates + 1, batches + 1
            if args.max_updates and updates >= args.max_updates:
                break
        model.eval()
        metrics = dict(epoch=epoch, updates=updates, seconds=time.time() - started, train_nll_per_token=nll_sum / token_sum,
                       train_grad_norm=grad_sum / batches, validation_nll_per_token=validation_nll(text, valid, args.eval_batch_size))
        stop = bool(args.max_updates and updates >= args.max_updates)
        if epoch % args.facts_every == 0 or stop:
            metrics["facts"] = score_facts(text, rssm, starts, steps, args.eval_batch_size, args.out / f"descriptions_epoch_{epoch:03d}.jsonl")
        with (args.out / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics) + "\n")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(key, value, epoch)
        writer.flush()
        save_json(args.out / "progress.json", metrics)
        print(json.dumps({k: v for k, v in metrics.items() if k != "facts"}), flush=True)
        if metrics["validation_nll_per_token"] < best:
            best = metrics["validation_nll_per_token"]
            torch.save(dict(model=trainable_state(model), config=config, metrics=metrics), args.out / "best.pt")
        if args.save_every and epoch % args.save_every == 0:
            torch.save(dict(model=trainable_state(model), config=config, metrics=metrics), args.out / f"epoch_{epoch:03d}.pt")
        if stop:
            break
    save_json(args.out / "complete.json", dict(best_validation_nll_per_token=best, updates=updates))
    writer.close()


if __name__ == "__main__":
    main()
