"""PPO + DistilBERT + CNN on original Crafter, in the setting of ZeroHero crafter-vision-12m.

Environment: native CUDA Crafter worlds with the original 256x256 screen (ZeroHero rezero_world, pixel-exact renderer).
Input: the screen and the line "Accomplished achievements: ...", encoded by frozen DistilBERT (CLS, 768). No task
instruction. Reward: 1 per newly unlocked achievement. Episodes end at death or after --horizon steps.
Network and PPO settings are those of model A (caged_craftext/baselines/ppo_with_instruction.py, ActorCriticConvWithBERT).

Score: distinct achievements of 22 unlocked in an episode. `eval` plays stock Crafter 1.8.3 on the held-out worlds of
ZeroHero (seeds 150000000-150000127, temperature 1, one Noop after reset, death or 2000 steps).

  python3 train_ppo_crafter.py train --zerohero <ZeroHero checkout> --out <run>
  python3 train_ppo_crafter.py eval --zerohero <ZeroHero checkout> --checkpoint <run>/final.pt --output <json>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIONS = 17
TEXT_SIZE = 768
EVAL_SEED = 150000000
TRAIN_SEED = 300000000  # far from the collection and evaluation worlds of ZeroHero


def layer(module, gain):
    nn.init.orthogonal_(module.weight, gain)
    nn.init.zeros_(module.bias)
    return module


class ActorCritic(nn.Module):
    """ActorCriticConvWithBERT: three (conv 5x5, 32) + max pool 3, then the actor and critic heads on image + text."""

    def __init__(self, width: int):
        super().__init__()
        self.vision = nn.Sequential(*[module for channels in (3, 32, 32) for module in (
            nn.Conv2d(channels, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(3, 3))], nn.Flatten())
        size = 32 * 9 * 9 + TEXT_SIZE
        self.actor = nn.Sequential(layer(nn.Linear(size, width), 2 ** .5), nn.ReLU(), layer(nn.Linear(width, width), 2 ** .5),
                                   nn.ReLU(), layer(nn.Linear(width, ACTIONS), .01))
        self.critic = nn.Sequential(layer(nn.Linear(size, width), 2 ** .5), nn.ReLU(), layer(nn.Linear(width, width), 2 ** .5),
                                    nn.ReLU(), layer(nn.Linear(width, 1), 1.0))

    def forward(self, images, text):
        features = torch.cat((self.vision(images.permute(0, 3, 1, 2).float() / 255), text), dim=-1)
        return self.actor(features), self.critic(features).squeeze(-1)


class Texts:
    """Frozen DistilBERT over the accomplished-achievements line, cached per set of achievements."""

    def __init__(self, names, device, model="distilbert-base-uncased"):
        from transformers import AutoModel, AutoTokenizer
        self.names, self.device = names, device
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).to(device).eval()
        self.index = {}
        self.table = torch.empty((1024, TEXT_SIZE), device=device)
        self.powers = 2 ** torch.arange(len(names), device=device)

    def line(self, mask: int) -> str:
        names = [name for bit, name in enumerate(self.names) if mask >> bit & 1]
        return "Accomplished achievements: " + (", ".join(names) if names else "none")

    @torch.inference_mode()
    def __call__(self, unlocked):
        masks = (unlocked.long() * self.powers).sum(-1)
        unique, inverse = torch.unique(masks, return_inverse=True)
        keys = unique.tolist()
        new = [key for key in keys if key not in self.index]
        if new:
            tokens = self.tokenizer([self.line(key) for key in new], return_tensors="pt", padding=True).to(self.device)
            cls = self.model(**tokens).last_hidden_state[:, 0]
            start = len(self.index)
            while start + len(new) > len(self.table):
                self.table = torch.cat((self.table, torch.empty_like(self.table)))
            self.table[start:start + len(new)] = cls
            self.index.update((key, start + i) for i, key in enumerate(new))
        rows = torch.tensor([self.index[key] for key in keys], device=self.device)
        return self.table[rows[inverse]]


class Worlds:
    """Native worlds that restart on fresh seeds; only the screen and the achievement counters leave the simulator."""

    def __init__(self, count, horizon, device, seed, restart=True):
        import crafter
        from rezero_world.crafter_cuda.batch import CUDACrafterBatch
        from rezero_world.crafter_screen_cuda import CUDAScreenRenderer
        self.count, self.horizon, self.device, self.restart = count, horizon, device, restart
        self.next_seed = seed + count
        self.seeds = torch.arange(seed, seed + count, dtype=torch.int64, device=device)
        self.episodes = torch.ones(count, dtype=torch.int64, device=device)
        self.renderer = CUDAScreenRenderer(Path(crafter.__file__).parent / "assets", device)
        self.batch = CUDACrafterBatch.from_seeds(self.seeds, episodes=self.episodes, size=(256, 256), length=horizon + 1,
                                                 capacity=4096, render_calls=0, consume_render_rng=False)
        self.noise = torch.zeros((count, 252, 196), dtype=torch.float64, device=device)
        self.everyone = torch.ones(count, dtype=torch.bool, device=device)
        self.steps = torch.zeros(count, dtype=torch.int64, device=device)
        self.alive = self.everyone.clone()
        self.noop = torch.zeros(count, dtype=torch.int32, device=device)
        self.batch.step(self.noop, self.everyone)
        self.images = torch.zeros((count, 256, 256, 3), dtype=torch.uint8, device=device)
        self.capture(self.everyone)
        self.unlocked = self.batch.achievements > 0

    def capture(self, mask):
        self.batch.kernel.screen_noise(self.batch.rng, self.batch.metadata, mask, self.noise, 2)
        images = self.renderer.render(self.batch, self.noise)[0]
        self.images[mask] = images[mask]

    def step(self, actions):
        """rewards, dones, scores of the worlds that ended (before their restart), lengths"""
        mask = self.alive
        self.batch.step(actions.to(torch.int32), mask)
        if self.batch.metadata[:, 4].any():
            raise RuntimeError("native Crafter error")
        unlocked = self.batch.achievements > 0
        rewards = (unlocked & ~self.unlocked).sum(-1).float()
        self.unlocked = unlocked
        self.steps += mask.long()
        dones = mask & ((self.batch.inventory[:, 0] <= 0) | (self.steps >= self.horizon))
        ended = dones.nonzero().flatten()
        scores, lengths = unlocked[ended].sum(-1).tolist(), self.steps[ended].tolist()
        names = unlocked[ended].clone()
        if self.restart and len(ended):
            self.seeds[ended] = torch.arange(self.next_seed, self.next_seed + len(ended), device=self.device)
            self.next_seed += len(ended)
            self.batch.reset_from_seeds(self.seeds, self.episodes, dones)
            self.batch.step(self.noop, dones)
            self.steps[ended] = 0
            self.unlocked = self.batch.achievements > 0
        elif len(ended):
            self.alive = mask & ~dones
        self.capture(self.alive)
        return rewards, dones, scores, lengths, names


@torch.inference_mode()
def evaluate(model, texts, names, count, horizon, device):
    """Held-out worlds on the native simulator, sampled at temperature 1 (the final score comes from `eval`)."""
    worlds = Worlds(count, horizon, device, EVAL_SEED, restart=False)
    scores, lengths, unlocked = [], [], Counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        while worlds.alive.any():
            logits, _ = model(worlds.images, texts(worlds.unlocked))
            _, _, done_scores, done_lengths, done_names = worlds.step(torch.distributions.Categorical(logits=logits.float()).sample())
            scores += done_scores
            lengths += done_lengths
            for row in done_names.tolist():
                unlocked.update(name for name, on in zip(names, row) if on)
    return dict(eval_score=float(np.mean(scores)), eval_length=float(np.mean(lengths)),
                eval_unlock={name: unlocked[name] / count for name in names})


def train(args):
    sys.path.insert(0, str(args.zerohero))
    from rezero_world.crafter_cuda.batch import ACHIEVEMENTS
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "experiment.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    names = list(ACHIEVEMENTS)
    texts = Texts(names, device)
    worlds = Worlds(args.num_envs, args.horizon, device, TRAIN_SEED + args.seed)
    model = ActorCritic(args.width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    updates = args.total_steps // (args.num_envs * args.num_steps)
    N, T = args.num_envs, args.num_steps
    images = torch.empty((T, N, 256, 256, 3), dtype=torch.uint8, device=device)
    text = torch.empty((T, N, TEXT_SIZE), device=device)
    actions = torch.empty((T, N), dtype=torch.long, device=device)
    old_log = torch.empty((T, N), device=device)
    values = torch.empty((T, N), device=device)
    rewards = torch.empty((T, N), device=device)
    dones = torch.empty((T, N), device=device)
    recent_scores, recent_lengths = deque(maxlen=512), deque(maxlen=512)
    started, best = time.time(), -1.0
    for update in range(1, updates + 1):
        tick = time.time()
        for group in optimizer.param_groups:
            group["lr"] = args.lr * (1 - (update - 1) / updates)
        model.eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for t in range(T):
                images[t], text[t] = worlds.images, texts(worlds.unlocked)
                logits, value = model(images[t], text[t])
                dist = torch.distributions.Categorical(logits=logits.float())
                actions[t] = dist.sample()
                old_log[t], values[t] = dist.log_prob(actions[t]), value.float()
                rewards[t], done, scores, lengths, _ = worlds.step(actions[t])
                dones[t] = done.float()
                recent_scores.extend(scores)
                recent_lengths.extend(lengths)
            _, last = model(worlds.images, texts(worlds.unlocked))
            last = last.float()
        advantages = torch.zeros_like(rewards)
        running = torch.zeros(N, device=device)
        for t in reversed(range(T)):
            following = last if t == T - 1 else values[t + 1]
            live = 1 - dones[t]
            delta = rewards[t] + args.gamma * following * live - values[t]
            running = delta + args.gamma * args.gae_lambda * live * running
            advantages[t] = running
        returns = advantages + values
        rolled = time.time()
        model.train()
        flat = lambda x: x.reshape(-1, *x.shape[2:])
        I, X, A, L, V, G, D = map(flat, (images, text, actions, old_log, values, returns, advantages))
        logs = Counter()
        for _ in range(args.update_epochs):
            order = torch.randperm(T * N, device=device)
            for part in order.chunk(args.minibatches):
                optimizer.zero_grad(set_to_none=True)
                gains = (D[part] - D[part].mean()) / (D[part].std() + 1e-8)
                # one minibatch of frames at this size overflows cuDNN pooling (> 2^31 elements): accumulate its gradient
                for piece, gain in zip(part.split(args.micro_batch), gains.split(args.micro_batch)):
                    share = len(piece) / len(part)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits, value = model(I[piece], X[piece])
                    logits, value = logits.float(), value.float()
                    dist = torch.distributions.Categorical(logits=logits)
                    log_prob = dist.log_prob(A[piece])
                    ratio = (log_prob - L[piece]).exp()
                    pi_loss = -torch.min(ratio * gain, ratio.clamp(1 - args.clip, 1 + args.clip) * gain).mean()
                    clipped = V[piece] + (value - V[piece]).clamp(-args.clip, args.clip)
                    v_loss = .5 * torch.max((value - G[piece]).square(), (clipped - G[piece]).square()).mean()
                    entropy = dist.entropy().mean()
                    (share * (pi_loss + args.vf * v_loss - args.ent * entropy)).backward()
                    logs.update(pi_loss=share * float(pi_loss), v_loss=share * float(v_loss), entropy=share * float(entropy),
                                kl=share * float((L[piece] - log_prob).mean()))
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                logs.update(batches=1)
        count = logs.pop("batches")
        steps = update * T * N
        row = dict(update=update, env_steps=steps, seconds=time.time() - started, steps_per_second=T * N / (time.time() - tick),
                   rollout_seconds=rolled - tick, learn_seconds=time.time() - rolled,
                   train_score=float(np.mean(recent_scores)) if recent_scores else None,
                   train_length=float(np.mean(recent_lengths)) if recent_lengths else None, texts=len(texts.index),
                   **{k: v / count for k, v in logs.items()})
        if update % args.eval_every == 0 or update == updates:
            model.eval()
            row.update(evaluate(model, texts, names, args.eval_worlds, args.horizon, device))
            if row["eval_score"] > best:
                best = row["eval_score"]
                torch.save(dict(model=model.state_dict(), args=vars(args), metrics=row), args.out / "best.pt")
        if update % args.save_every == 0 or update == updates:
            torch.save(dict(model=model.state_dict(), args=vars(args), metrics=row), args.out / f"update_{update:06d}.pt")
        with (args.out / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "eval_unlock"}), flush=True)
    torch.save(dict(model=model.state_dict(), args=vars(args)), args.out / "final.pt")


@torch.inference_mode()
def stock_eval(args):
    """128 held-out worlds of stock Crafter 1.8.3, temperature 1, as ZeroHero evaluate.py."""
    import crafter
    sys.path.insert(0, str(args.zerohero))
    from rezero_world.crafter_cuda.batch import ACHIEVEMENTS
    device = torch.device("cuda")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ActorCritic(state["args"]["width"]).to(device).eval()
    model.load_state_dict(state["model"])
    names = list(ACHIEVEMENTS)
    texts = Texts(names, device)
    torch.manual_seed(args.sampling_seed)
    seeds = list(range(args.seed_base, args.seed_base + args.worlds))
    envs, screens, done_at, steps = {}, {}, {}, {}
    for seed in seeds:
        env = crafter.Env(area=(64, 64), view=(9, 9), size=(256, 256), reward=True, seed=seed)
        env.reset()
        screens[seed], _, _, info = env.step(0)
        envs[seed], steps[seed], done_at[seed] = env, 0, info
    active, episodes = list(seeds), []
    while active:
        unlocked = torch.tensor([[done_at[s]["achievements"][n] > 0 for n in names] for s in active], device=device)
        batch = torch.as_tensor(np.stack([screens[s] for s in active]), device=device)
        logits = torch.cat([model(i, t)[0] for i, t in zip(batch.split(args.batch), texts(unlocked).split(args.batch))])
        chosen = torch.distributions.Categorical(logits=logits.float()).sample().tolist()
        survivors = []
        for seed, action in zip(active, chosen):
            screens[seed], _, ended, info = envs[seed].step(action)
            done_at[seed] = info
            steps[seed] += 1
            if ended or steps[seed] >= args.horizon:
                got = sorted(n for n, c in info["achievements"].items() if c > 0)
                episodes.append(dict(seed=seed, score=len(got), steps=steps[seed], achievements=got))
            else:
                survivors.append(seed)
        active = survivors
    result = dict(checkpoint=str(args.checkpoint), worlds=args.worlds, score=float(np.mean([e["score"] for e in episodes])),
                  length=float(np.mean([e["steps"] for e in episodes])),
                  unlock={n: sum(n in e["achievements"] for e in episodes) / args.worlds for n in names}, episodes=episodes)
    args.output.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "episodes"}))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("train")
    run.add_argument("--zerohero", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--total-steps", type=lambda x: int(float(x)), default=int(2e8))
    run.add_argument("--num-envs", type=int, default=256)
    run.add_argument("--num-steps", type=int, default=100)
    run.add_argument("--horizon", type=int, default=2000)
    run.add_argument("--lr", type=float, default=2e-4)
    run.add_argument("--update-epochs", type=int, default=4)
    run.add_argument("--minibatches", type=int, default=8)
    run.add_argument("--micro-batch", type=int, default=512, help="frames per forward pass of a minibatch")
    run.add_argument("--gamma", type=float, default=.99)
    run.add_argument("--gae-lambda", type=float, default=.8)
    run.add_argument("--clip", type=float, default=.2)
    run.add_argument("--ent", type=float, default=.01)
    run.add_argument("--vf", type=float, default=.5)
    run.add_argument("--max-grad-norm", type=float, default=1.0)
    run.add_argument("--width", type=int, default=512)
    run.add_argument("--eval-every", type=int, default=100)
    run.add_argument("--eval-worlds", type=int, default=128)
    run.add_argument("--save-every", type=int, default=500)
    run.add_argument("--seed", type=int, default=7950)
    check = commands.add_parser("eval")
    check.add_argument("--zerohero", type=Path, required=True)
    check.add_argument("--checkpoint", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--worlds", type=int, default=128)
    check.add_argument("--seed-base", type=int, default=EVAL_SEED)
    check.add_argument("--horizon", type=int, default=2000)
    check.add_argument("--batch", type=int, default=64)
    check.add_argument("--sampling-seed", type=int, default=1)
    args = parser.parse_args()
    train(args) if args.command == "train" else stock_eval(args)


if __name__ == "__main__":
    main()
