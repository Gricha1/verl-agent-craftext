"""The RSSM next to RL: gives the agent loop h_t, z_t and, with --updates-per-rollout, keeps learning from what RL sees.

The agent acts on the state of the BC loop of train_slm_action_on_rssm.py: h_0 = 0, z_t = posterior mode of the real frame,
h_{t+1} = RSSM(h_t, z_t, a_t). PPO gets h_t, z_t as data, so its gradient never reaches the RSSM.

Three copies of the weights:
  acting   what rollouts use. It is fixed within one rollout and read from the EMA at the start of the next one, so the
           states of one PPO batch come from one encoder.
  learner  gets the gradient of the world-model loss of train_slm_world_model.py, on batches of half RL frames, half teacher.
  ema      theta_ema <- rate * theta_ema + (1 - rate) * theta_learner after each gradient step.

Validation rollouts read the state and add nothing to the buffer. --updates-per-rollout 0 is the frozen RSSM baseline.

  python3 rssm_service.py pack --raw-data <collection> --split-from <dir with valid.jsonl> --out <pack dir>
  python3 rssm_service.py serve --checkpoint <rssm.pt> --pack <pack dir> --socket /workspace/run/rssm.sock --out <dir>
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Listener
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_slm_world_model import ACTION_ID, build_model, load_episodes, loss_for_batch, read_image  # noqa: E402

MEASURED = ("world_loss", "reconstruction_mse_per_pixel", "kl_raw", "overshooting_kl", "reward_loss", "continue_loss")


def pack(args) -> None:
    """Teacher train episodes in one array, so that every RL run shares them through the page cache."""
    valid = {json.loads(line)["source_episode_id"] for line in (args.split_from / "valid.jsonl").open()}
    episodes = {k: e for k, e in load_episodes(args.raw_data, None, None).items() if k not in valid}
    keys = sorted(episodes)
    lengths = np.array([len(episodes[k]) for k in keys])
    starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    args.out.mkdir(parents=True, exist_ok=True)
    frames = np.lib.format.open_memmap(args.out / "frames.npy", "w+", np.uint8, (int(lengths.sum()), 3, 63, 63))
    items = [item for k in keys for item in episodes[k]]

    def load(index):
        frames[index] = (read_image(args.raw_data, items[index]["image"]) * 255).round().astype(np.uint8)

    with ThreadPoolExecutor(32) as pool:
        for _ in pool.map(load, range(len(items)), chunksize=256):
            pass
    frames.flush()
    np.savez(args.out / "episodes.npz", starts=starts, lengths=lengths,
             actions=np.array([i["action"] for i in items], dtype=np.int64),
             rewards=np.array([i["reward"] for i in items], dtype=np.float32),
             terminals=np.array([i["terminal"] for i in items], dtype=bool))
    print(json.dumps(dict(episodes=len(keys), frames=len(items))), flush=True)


class Episodes:
    """Rows as in the teacher data: row t holds the frame before action t, the action, its reward, and whether it ended."""

    def __init__(self, capacity: int = 0):
        self.items = collections.deque()
        self.frames = 0
        self.capacity = capacity

    def add(self, frames, actions, rewards, terminals) -> None:
        self.items.append((frames, actions, rewards, terminals))
        self.frames += len(frames)
        while self.capacity and self.frames > self.capacity:
            self.frames -= len(self.items.popleft()[0])

    def sample(self, count: int, length: int, rng):
        """count chunks of length + 1 rows, uniform over start rows."""
        usable = [e for e in self.items if len(e[0]) > length]
        if not usable:
            raise ValueError(f"no episode longer than {length} rows")
        weights = np.array([len(e[0]) - length for e in usable], dtype=np.float64)
        out = []
        for index in rng.choice(len(usable), size=count, p=weights / weights.sum()):
            frames, actions, rewards, terminals = usable[index]
            start = int(rng.integers(0, len(frames) - length))
            stop = start + length + 1
            out.append((np.asarray(frames[start:stop]), actions[start:stop - 1], rewards[start + 1:stop], terminals[start + 1:stop]))
        return [np.stack(column) for column in zip(*out)]


class Service:
    def __init__(self, args):
        import torch
        self.torch, self.args = torch, args
        self.device = torch.device("cuda")
        self.state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        self.config = self.state["config"]
        self.acting = self.fresh_model()
        self.size = self.acting.h_size + self.acting.z_dim
        self.lock = threading.Lock()
        self.rollouts, self.updates, self.pending = 0, 0, 0
        self.started = time.time()
        args.out.mkdir(parents=True, exist_ok=True)
        self.learning = args.updates_per_rollout > 0
        if self.learning:
            self.learner, self.ema, self.start = self.fresh_model(), self.fresh_model(), self.fresh_model()
            self.learner.train()
            for parameter in self.learner.parameters():
                parameter.requires_grad_(True)
            self.optimizer = torch.optim.AdamW(self.learner.parameters(), lr=args.lr or self.config["lr"], weight_decay=.01)
            self.buffer = Episodes(args.buffer_frames)
            self.teacher = Episodes()
            frames = np.load(args.pack / "frames.npy", mmap_mode="r")
            meta = np.load(args.pack / "episodes.npz")
            for start, length in zip(meta["starts"], meta["lengths"]):
                stop = start + length
                self.teacher.add(frames[start:stop], meta["actions"][start:stop], meta["rewards"][start:stop], meta["terminals"][start:stop])
            self.loss_args = SimpleNamespace(**{**self.config, "overshoot_depth": self.config.get("overshoot_depth")})
            self.rng = np.random.default_rng(args.seed)
            self.wake = threading.Condition()
            self.totals = collections.Counter()
            threading.Thread(target=self.train_forever, daemon=True).start()

    def fresh_model(self):
        model = build_model(self.config, None, self.device)
        model.modules.load_state_dict(self.state["model"]["support"])
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def autocast(self):
        return self.torch.autocast("cuda", dtype=self.torch.bfloat16)

    def tensors(self, frames, actions, rewards, terminals):
        torch, device = self.torch, self.device
        images = torch.from_numpy(frames).to(device).float() / 255
        batch, steps = actions.shape
        return (images, torch.from_numpy(actions).to(device), torch.from_numpy(rewards).to(device),
                torch.from_numpy(1.0 - terminals.astype(np.float32)).to(device), [None] * batch, [[None] * steps] * batch)

    def train_forever(self) -> None:
        try:
            self.train_loop()
        except BaseException:  # a thread dies alone; the socket going away stops the RL run instead of a silent freeze
            traceback.print_exc()
            os._exit(1)

    def train_loop(self) -> None:
        torch, args = self.torch, self.args
        half = self.config["batch_size"] // 2
        length = self.config["horizon"]
        while True:
            with self.wake:
                while self.pending == 0:
                    self.wake.wait()
            with self.lock:
                fresh = self.buffer.sample(half, length, self.rng)
            teacher = self.teacher.sample(self.config["batch_size"] - half, length, self.rng)
            batch = self.tensors(*[np.concatenate(pair) for pair in zip(fresh, teacher)])
            self.optimizer.zero_grad(set_to_none=True)
            with self.autocast():
                loss, parts, _ = loss_for_batch(self.learner, *batch, self.loss_args, self.device)
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(self.learner.parameters(), 1.0))
            self.optimizer.step()
            with self.lock, torch.no_grad():
                for own, new in zip(self.ema.modules.parameters(), self.learner.modules.parameters()):
                    own.lerp_(new, 1 - args.ema)
                self.updates += 1
                self.pending -= 1
                for key in MEASURED:
                    self.totals[key] += parts[key]
                self.totals["grad_norm"] += norm
                self.totals["count"] += 1

    def begin(self, learn: bool) -> None:
        """The acting copy for one rollout; a training rollout has a number, and RL's checkpoint of that step matches it."""
        if not self.learning:
            return
        with self.lock:
            self.acting.modules.load_state_dict(self.ema.modules.state_dict())
            if learn:
                self.rollouts += 1
                if self.rollouts % self.args.save_every == 0:
                    self.save(self.rollouts)

    def save(self, rollout: int) -> None:
        model = dict(support={k: v.cpu() for k, v in self.acting.modules.state_dict().items()}, lm={})
        self.torch.save(dict(model=model, config=self.config, rollout=rollout), self.args.out / f"rssm_rollout_{rollout:06d}.pt")

    def report(self, new: list) -> None:
        """What the EMA says about this rollout's frames before any gradient step on them, then what training did."""
        torch = self.torch
        length = self.config["horizon"]
        row = dict(rollout=self.rollouts, seconds=round(time.time() - self.started), new_frames=sum(len(e[0]) for e in new))
        check = Episodes()
        for episode in new:
            check.add(*episode)
        if any(len(e[0]) > length for e in new):
            frames, actions, rewards, terminals = check.sample(self.config["batch_size"], length, self.rng)
            batch = self.tensors(frames, actions, rewards, terminals)
            with torch.inference_mode(), self.autocast():
                _, parts, _ = loss_for_batch(self.acting, *batch, self.loss_args, self.device)
                # how far the adapters' input has moved: z of the same frames by the first weights and by these, h = 0
                images = batch[0].flatten(0, 1)
                h = torch.zeros(len(images), self.acting.h_size, device=self.device)
                shape = (len(images), self.config["z_size"], self.config["z_classes"])
                codes = [m.mode(m.posterior(images, h)).view(shape).argmax(-1) for m in (self.acting, self.start)]
            row.update({f"new_{k}": parts[k] for k in MEASURED})
            row["codes_same_as_start"] = float((codes[0] == codes[1]).float().mean())
        with self.lock:
            count = max(self.totals["count"], 1)
            row.update({f"trained_{k}": self.totals[k] / count for k in (*MEASURED, "grad_norm")})
            row.update(updates=self.updates, pending=self.pending, buffer_frames=self.buffer.frames)
            self.totals.clear()
        with (self.args.out / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    def serve_rollout(self, connection) -> None:
        torch, device, rssm = self.torch, self.device, self.acting
        learn, h, z, lanes_of, new = False, None, None, None, []
        latents = lambda: torch.cat((h.float(), z.float()), dim=-1).cpu().numpy()
        frames_of = lambda frames: torch.from_numpy(np.moveaxis(frames, -1, 1).copy()).to(device).float() / 255
        while True:
            try:
                message = connection.recv()
            except EOFError:
                break
            try:
                name, *rest = message
                if name == "begin":
                    learn = rest[0] and self.learning
                    self.begin(rest[0])
                    connection.send(None)
                elif name == "first":
                    frames = rest[0]
                    lanes_of = [dict(frames=[f], actions=[], rewards=[], terminals=[]) for f in frames]
                    with torch.inference_mode(), self.autocast():
                        h = torch.zeros(len(frames), rssm.h_size, device=device)
                        z = rssm.mode(rssm.posterior(frames_of(frames), h))
                    connection.send(latents())
                elif name == "step":
                    lanes, actions, rewards, ends, frames = rest
                    index = torch.tensor(lanes, device=device)
                    numbers = torch.tensor([ACTION_ID[a] for a in actions], device=device)
                    with torch.inference_mode(), self.autocast():
                        h_next, _ = rssm.transition(None, numbers, None, h[index], z[index], device)
                        z_next = rssm.mode(rssm.posterior(frames_of(frames), h_next))
                        h = h.index_copy(0, index, h_next.to(h.dtype))
                        z = z.index_copy(0, index, z_next.to(z.dtype))
                    for i, lane in enumerate(lanes):
                        episode = lanes_of[lane]
                        episode["actions"].append(ACTION_ID[actions[i]])
                        episode["rewards"].append(rewards[i])
                        episode["terminals"].append(ends[i])
                        if ends[i]:
                            new.append((np.moveaxis(np.stack(episode["frames"]), -1, 1), np.array(episode["actions"], dtype=np.int64),
                                        np.array(episode["rewards"], dtype=np.float32), np.array(episode["terminals"], dtype=bool)))
                        else:
                            episode["frames"].append(frames[i])
                    connection.send(latents())
                else:
                    raise ValueError(f"unknown message {name}")
            except Exception as error:  # the agent loop raises it, so that a broken RSSM stops the run
                connection.send(error)
                raise
        if learn and new:
            self.report(new)
            with self.lock:
                for episode in new:
                    self.buffer.add(*episode)
            if any(len(e[0]) > self.config["horizon"] for e in self.buffer.items):
                with self.wake:
                    self.pending += self.args.updates_per_rollout
                    self.wake.notify()

    def serve(self) -> None:
        socket = self.args.socket
        socket.unlink(missing_ok=True)
        with Listener(str(socket), family="AF_UNIX", authkey=b"rssm") as listener:
            (self.args.out / "ready").write_text("ready\n")
            print(json.dumps(dict(ready=True, learning=self.learning, state=self.size)), flush=True)
            while True:
                connection = listener.accept()
                threading.Thread(target=self.serve_rollout, args=(connection,), daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("pack")
    p.add_argument("--raw-data", type=Path, required=True)
    p.add_argument("--split-from", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    s = commands.add_parser("serve")
    s.add_argument("--checkpoint", type=Path, required=True)
    s.add_argument("--socket", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--pack", type=Path, help="teacher frames of the pack command; needed with --updates-per-rollout")
    s.add_argument("--updates-per-rollout", type=int, default=0, help="gradient steps per training rollout; 0 keeps the RSSM frozen")
    s.add_argument("--ema", type=float, default=.995)
    s.add_argument("--lr", type=float, default=0.0, help="default: the lr of the RSSM run")
    s.add_argument("--buffer-frames", type=int, default=300000)
    s.add_argument("--save-every", type=int, default=50, help="rollouts between saved acting weights, same as RL's SAVE_FREQ")
    s.add_argument("--seed", type=int, default=7101)
    args = parser.parse_args()
    if args.command == "pack":
        pack(args)
    else:
        if args.updates_per_rollout and not args.pack:
            parser.error("--updates-per-rollout needs --pack")
        Service(args).serve()


if __name__ == "__main__":
    main()
