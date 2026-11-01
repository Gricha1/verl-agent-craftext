#!/usr/bin/env python3
"""World model whose deterministic transition is the visual SLM (phase 4, hypothesis H-WM-GRPO).

Ported from ZeroHero scripts/train_crafter_slm_world_model.py (7a7e504), with the stochastic state z fixed.
The old runs collapsed z (KL ~0.002 nats over 48 dims from epoch 2): pixel MSE was a mean over 11,907
values (~0.03) against a KL summed over 48 dims, there were no free bits, and z_0 had no KL. Now, as
DreamerV3 / Dynalang:

  recon  = 1/2 sum_pixels (x_hat - x)^2            (Gaussian log-likelihood, unit variance)
  kl     = 0.5 max(1, KL(sg q || p)) + 0.1 max(1, KL(q || sg p))   per sample, free bits 1 nat
  z_0    gets the same KL against p(z_0 | h_0 = 0)
  over   = 0.5 max(1, KL(sg q_{tau+d} || p(h~_{tau+d})))            latent overshooting, d = 2..K
  loss   = recon + kl + over + 1.0 MSE(r_hat, 50 r) + 0.1 BCE(c_hat, 1 - done)

Transition: [E("Task: g ... after action:"); 30|50|100 soft h tokens; 5 soft z tokens;
E("ACTION: a" or "<reasoning>r</reasoning>\nACTION: a"); E(eos)] -> SLM final hidden o_t ->
h_{t+1} = MLP(o_t); reward / continuation heads read o_t; decoder reads [h_{t+1}; z_{t+1}].
Every parameter is trained (the SLM included).

--z-kind categorical --z-size 32 --z-classes 32 --image-net dreamer: DreamerV3 latent (32 categoricals of 32
classes, 1% unimix, straight-through one-hot samples, KL per categorical) and DreamerV3 image nets (stride-2
CNN to 4x4x256; transposed-conv decoder 4x4 -> 64x64 cropped to 63x63) instead of the 48-dim gaussian and the
7x7 bilinear decoder, which blur every frame (even the autoencoder recon). Imagination uses the mode (argmax).

Per epoch on validation: world losses, KL per z dimension and the number of active dimensions, the
posterior's dependence on the frame (real vs shuffled frames), reconstruction with z from the prior mean
instead of the posterior, and an open-loop prior rollout decoded and compared with the real frames.

  python train_slm_world_model.py --raw-data <teacher collection> --split-from <phase-2 CE data dir> \
    --init-checkpoint <run>/initial_random --out <run dir> --h-size 512 --h-tokens 30 --horizon 16 \
    [--reasoning reasoning.jsonl]
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ACTIONS = (
    "NOOP", "LEFT", "RIGHT", "UP", "DOWN", "DO", "SLEEP", "PLACE_STONE",
    "PLACE_TABLE", "PLACE_FURNACE", "PLACE_PLANT", "MAKE_WOOD_PICKAXE",
    "MAKE_STONE_PICKAXE", "MAKE_IRON_PICKAXE", "MAKE_WOOD_SWORD",
    "MAKE_STONE_SWORD", "MAKE_IRON_SWORD",
)
ACTION_ID = {name: i for i, name in enumerate(ACTIONS)}
PIXELS = 63 * 63 * 3


def save_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def read_image(root: Path, name: str) -> np.ndarray:
    with np.load(root / "images" / name) as archive:
        image = archive["observation"].astype("float32", copy=False)
    if image.max() > 1:
        image = image / 255.0
    return np.moveaxis(image, -1, 0)


def load_episodes(root: Path, reasoning: dict | None):
    import pyarrow.parquet as pq
    rows = pq.read_table(root / "train.parquet").to_pylist() + pq.read_table(root / "val.parquet").to_pylist()
    episodes = defaultdict(list)
    for row in rows:
        extra = row["extra_info"]
        key = (int(extra["episode_id"]), int(extra["step"]))
        text = None
        if reasoning is not None:
            label = reasoning.get(key)
            if label is None or label["action"] != extra["teacher_action"]:
                raise ValueError(f"no reasoning for the teacher action at episode {key[0]} step {key[1]}")
            text = label["reasoning"]
        episodes[key[0]].append(dict(step=key[1], action=ACTION_ID[str(extra["teacher_action"])], reasoning=text,
                                     image=Path(str(extra["observation_path"])).name,
                                     instruction=str(extra["instruction"]), reward=float(extra["reward"]),
                                     terminal=bool(extra["done"])))
    for episode_id, episode in episodes.items():
        episode.sort(key=lambda r: r["step"])
        if [r["step"] for r in episode] != list(range(len(episode))):
            raise ValueError(f"non-contiguous episode {episode_id}")
        episode[-1]["terminal"] = True  # a collection cutoff also ends the rollout
    return dict(episodes)


def make_chunks(episodes, length: int):
    chunks = []
    for episode in episodes.values():
        maximum = len(episode) - length - 1
        if maximum < 0:
            continue
        starts = list(range(0, maximum + 1, length))
        if starts[-1] != maximum:
            starts.append(maximum)
        chunks += [episode[start:start + length + 1] for start in starts]
    return chunks


def batch_from_chunks(root, chunks, device):
    import torch
    images = np.stack([[read_image(root, item["image"]) for item in chunk] for chunk in chunks])
    actions = np.asarray([[item["action"] for item in chunk[:-1]] for chunk in chunks], dtype=np.int64)
    rewards = np.asarray([[item["reward"] for item in chunk[1:]] for chunk in chunks], dtype=np.float32)
    continuation = np.asarray([[0.0 if item["terminal"] else 1.0 for item in chunk[1:]] for chunk in chunks],
                              dtype=np.float32)
    reasoning = [[item["reasoning"] for item in chunk[:-1]] for chunk in chunks]
    goals = [chunk[0]["instruction"] for chunk in chunks]
    return (torch.from_numpy(images).to(device), torch.from_numpy(actions).to(device),
            torch.from_numpy(rewards).to(device), torch.from_numpy(continuation).to(device), goals, reasoning)


def normal(raw):
    import torch.nn.functional as F
    mu, s = raw.float().chunk(2, dim=-1)  # KL and sampling in fp32 under bf16 autocast
    return mu, F.softplus(s) + 1e-4


def categorical(raw, groups, classes, unimix=0.01):
    """DreamerV3: `groups` categoricals of `classes` classes, 1% uniform mix; returns (probs [B, groups, classes],)."""
    probs = raw.float().view(len(raw), groups, classes).softmax(-1)
    return ((1 - unimix) * probs + unimix / classes,)


def gaussian_kl_dims(q_mu, q_std, p_mu, p_std):
    """KL(q || p) per latent dimension, shape [batch, z]."""
    import torch
    return torch.log(p_std / q_std) + (q_std.square() + (q_mu - p_mu).square()) / (2 * p_std.square()) - .5


def categorical_kl_dims(q_probs, p_probs):
    """KL(q || p) per categorical variable, shape [batch, groups]."""
    return (q_probs * (q_probs.log() - p_probs.log())).sum(-1)


def dreamer_encoder(nn):
    """DreamerV3 CNN: 64x64 (63x63 padded) -> stride-2 convs 32/64/128/256 -> 4x4x256 -> flat 4096."""
    layers, channels = [nn.ZeroPad2d((0, 1, 0, 1))], 3
    for out in (32, 64, 128, 256):
        layers += [nn.Conv2d(channels, out, 4, stride=2, padding=1), nn.GroupNorm(1, out), nn.SiLU()]
        channels = out
    return nn.Sequential(*layers, nn.Flatten())


def dreamer_decoder(nn, size):
    """DreamerV3 CNN decoder: [h; z] -> 4x4x256 -> transposed convs 128/64/32/3 -> 64x64, cropped to 63x63."""
    layers = [nn.Linear(size, 256 * 4 * 4), nn.Unflatten(1, (256, 4, 4))]
    channels = 256
    for out in (128, 64, 32):
        layers += [nn.ConvTranspose2d(channels, out, 4, stride=2, padding=1), nn.GroupNorm(1, out), nn.SiLU()]
        channels = out
    # negative padding crops the 64x64 output back to the 63x63 frame
    layers += [nn.ConvTranspose2d(channels, 3, 4, stride=2, padding=1), nn.ZeroPad2d((0, -1, 0, -1)), nn.Sigmoid()]
    return nn.Sequential(*layers)


class SLMWorldModel:
    """Support modules around one trainable decoder-only SLM transition."""

    def __init__(self, lm, tokenizer, h_size, z_size, h_tokens, z_tokens, z_kind="gaussian", z_classes=32,
                 image_net="small"):
        import torch
        import torch.nn as nn
        self.lm, self.tok, self._torch = lm, tokenizer, torch
        self.width = lm.config.hidden_size
        self.h_size, self.z_size, self.h_tokens, self.z_tokens = h_size, z_size, h_tokens, z_tokens
        # gaussian: z = z_size reals; categorical: z = z_size one-hot variables of z_classes classes (flattened)
        self.z_kind, self.z_classes = z_kind, z_classes
        z_dim = z_size if z_kind == "gaussian" else z_size * z_classes
        z_raw = 2 * z_size if z_kind == "gaussian" else z_size * z_classes
        self.z_dim = z_dim
        if image_net == "small":
            cnn, embed = nn.Sequential(
                nn.Conv2d(3, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(), nn.Linear(32 * 7 * 7, 256), nn.LayerNorm(256), nn.Tanh(),
            ), 256
            decoder = nn.Sequential(
                nn.Linear(h_size + z_dim, 32 * 7 * 7), nn.GELU(), nn.Unflatten(1, (32, 7, 7)),
                nn.Upsample(size=(63, 63), mode="bilinear", align_corners=False),
                nn.Conv2d(32, 32, 5, padding=2), nn.GELU(), nn.Conv2d(32, 3, 3, padding=1), nn.Sigmoid(),
            )
        else:  # "dreamer"
            cnn, embed = dreamer_encoder(nn), 256 * 4 * 4
            decoder = dreamer_decoder(nn, h_size + z_dim)
        self.modules = nn.ModuleDict({
            "cnn": cnn,
            "posterior": nn.Sequential(nn.Linear(h_size + embed, h_size), nn.GELU(), nn.Linear(h_size, z_raw)),
            "h_prefix": nn.Sequential(nn.Linear(h_size, h_size), nn.GELU(), nn.Linear(h_size, h_tokens * self.width)),
            "z_prefix": nn.Sequential(nn.Linear(z_dim, z_dim), nn.GELU(), nn.Linear(z_dim, z_tokens * self.width)),
            "h_transition": nn.Sequential(nn.Linear(self.width, h_size), nn.GELU(), nn.Linear(h_size, h_size)),
            "z_prior": nn.Sequential(nn.Linear(h_size, h_size), nn.GELU(), nn.Linear(h_size, z_raw)),
            "reward_head": nn.Sequential(nn.Linear(self.width, h_size), nn.GELU(), nn.Linear(h_size, 1)),
            "continue_head": nn.Sequential(nn.Linear(self.width, h_size), nn.GELU(), nn.Linear(h_size, 1)),
            "pixel_decoder": decoder,
        })
        self._text_cache = {}

    def to(self, device):
        self.lm.to(device); self.modules.to(device); return self

    def train(self, mode=True):
        self.lm.train(mode); self.modules.train(mode); return self

    def eval(self):
        return self.train(False)

    def parameters(self):
        return list(self.lm.parameters()) + list(self.modules.parameters())

    def state_dict(self):
        return {"support": self.modules.state_dict(), "lm": self.lm.state_dict()}

    def ids(self, text):
        if text not in self._text_cache:
            self._text_cache[text] = self.tok(text, add_special_tokens=False)["input_ids"]
        return self._text_cache[text]

    def _dist(self, raw):
        return normal(raw) if self.z_kind == "gaussian" else categorical(raw, self.z_size, self.z_classes)

    def posterior(self, image, h):
        return self._dist(self.modules["posterior"](self._torch.cat((h, self.modules["cnn"](image)), dim=-1)))

    def prior(self, h):
        return self._dist(self.modules["z_prior"](h))

    def sample(self, dist):
        """Reparameterised (gaussian) or straight-through one-hot (categorical) sample, flat [B, z_dim]."""
        torch = self._torch
        if self.z_kind == "gaussian":
            return dist[0] + dist[1] * torch.randn_like(dist[1])
        probs = dist[0]
        index = torch.multinomial(probs.flatten(0, 1), 1).view(probs.shape[:-1])
        onehot = torch.nn.functional.one_hot(index, self.z_classes).to(probs.dtype)
        return (onehot + probs - probs.detach()).flatten(1)

    def mode(self, dist):
        """Gaussian mean / categorical argmax one-hot, flat [B, z_dim]."""
        if self.z_kind == "gaussian":
            return dist[0]
        probs = dist[0]
        return self._torch.nn.functional.one_hot(probs.argmax(-1), self.z_classes).to(probs.dtype).flatten(1)

    def kl_dims(self, q, p):
        """KL(q || p) per gaussian dimension / per categorical variable."""
        return gaussian_kl_dims(*q, *p) if self.z_kind == "gaussian" else categorical_kl_dims(q[0], p[0])

    def transition(self, goals, actions, reasoning, h, z, device):
        """Run the SLM; h[t+1] and the heads read only its final (EOS) hidden state."""
        torch = self._torch
        embed = self.lm.get_input_embeddings()
        hs = self.modules["h_prefix"](h).view(len(h), self.h_tokens, self.width)
        zs = self.modules["z_prefix"](z).view(len(z), self.z_tokens, self.width)
        pieces = []
        for index, (goal, action, text) in enumerate(zip(goals, actions.tolist(), reasoning)):
            answer = f"ACTION: {ACTIONS[int(action)]}" if text is None else f"<reasoning>{text}</reasoning>\nACTION: {ACTIONS[int(action)]}"
            head = embed(torch.tensor(self.ids(f"Task:\n{goal}\n\nPredict next environment state after action:\n"), device=device))
            tail = embed(torch.tensor(self.ids(answer) + [int(self.tok.eos_token_id)], device=device))
            pieces.append(torch.cat((head, hs[index].to(head.dtype), zs[index].to(head.dtype), tail), dim=0))
        maximum = max(piece.shape[0] for piece in pieces)
        padded = torch.stack([torch.cat((piece, piece.new_zeros(maximum - piece.shape[0], self.width))) for piece in pieces])
        mask = torch.tensor([[1] * piece.shape[0] + [0] * (maximum - piece.shape[0]) for piece in pieces], device=device)
        output = self.lm(inputs_embeds=padded, attention_mask=mask, output_hidden_states=True, use_cache=False)
        last = output.hidden_states[-1][torch.arange(len(pieces), device=device), mask.sum(dim=1) - 1].float()
        return self.modules["h_transition"](last), last

    def decode(self, h, z):
        return self.modules["pixel_decoder"](self._torch.cat((h, z), dim=-1))


def detach(dist):
    return tuple(t.detach() for t in dist)


def balanced_kl(model, q, p, free, dyn_scale, rep_scale):
    """DreamerV3: free bits per sample on each side; returns (loss, raw KL per sample, KL per dim)."""
    dims_dyn = model.kl_dims(detach(q), p)
    dims_rep = model.kl_dims(q, detach(p))
    dyn, rep = dims_dyn.sum(-1), dims_rep.sum(-1)
    loss = dyn_scale * dyn.clamp_min(free).mean() + rep_scale * rep.clamp_min(free).mean()
    return loss, dyn.detach(), dims_dyn.detach()


def loss_for_batch(model, images, actions, rewards, continuation, goals, reasoning, args, device, diagnostics=False):
    import torch
    import torch.nn.functional as F
    batch, steps = actions.shape
    zeros = torch.zeros(batch, model.h_size, device=device)
    h = zeros
    q = model.posterior(images[:, 0], h)
    z = model.sample(q)
    # z_0 is regularised like every other z: against p(z_0 | h_0 = 0)
    kl_loss, kl_raw, kl_dims = balanced_kl(model, q, model.prior(h), args.free_nats, args.kl_dyn, args.kl_rep)
    kl_raws, kl_dims_all = [kl_raw], [kl_dims]
    h_states, q_states = [h], [q]
    recon = torch.zeros((), device=device)
    reward_loss = torch.zeros((), device=device)
    continue_loss = torch.zeros((), device=device)
    for t in range(steps):
        rs = [r[t] for r in reasoning]
        h, last = model.transition(goals, actions[:, t], rs, h, z, device)
        p = model.prior(h)
        q = model.posterior(images[:, t + 1], h)
        z = model.sample(q)
        # Gaussian log-likelihood with unit variance: 1/2 squared error summed over the frame's pixels
        recon = recon + .5 * (model.decode(h, z).float() - images[:, t + 1]).square().flatten(1).sum(-1).mean()
        reward_loss = reward_loss + F.mse_loss(model.modules["reward_head"](last).squeeze(-1).float(),
                                               rewards[:, t] * args.reward_target_scale)
        continue_loss = continue_loss + F.binary_cross_entropy_with_logits(
            model.modules["continue_head"](last).squeeze(-1).float(), continuation[:, t])
        step_kl, step_raw, step_dims = balanced_kl(model, q, p, args.free_nats, args.kl_dyn, args.kl_rep)
        kl_loss = kl_loss + step_kl
        kl_raws.append(step_raw); kl_dims_all.append(step_dims)
        h_states.append(h); q_states.append(q)
    recon, reward_loss, continue_loss = recon / steps, reward_loss / steps, continue_loss / steps
    kl_loss = kl_loss / (steps + 1)
    overshoot, terms = torch.zeros((), device=device), 0
    if args.horizon > 1:
        for anchor in range(steps):
            h_roll, z_roll = h_states[anchor], model.mode(q_states[anchor])
            for distance in range(1, min(args.horizon, steps - anchor) + 1):
                rs = [r[anchor + distance - 1] for r in reasoning]
                h_roll, _ = model.transition(goals, actions[:, anchor + distance - 1], rs, h_roll, z_roll, device)
                p_roll = model.prior(h_roll)
                if distance > 1:
                    raw = model.kl_dims(detach(q_states[anchor + distance]), p_roll).sum(-1)
                    overshoot = overshoot + args.kl_dyn * raw.clamp_min(args.free_nats).mean()
                    terms += 1
                z_roll = model.mode(p_roll)
        overshoot = overshoot / max(terms, 1)
    loss = recon + kl_loss + overshoot + args.reward_loss_scale * reward_loss + args.continue_loss_scale * continue_loss
    kl_raw_all = torch.cat(kl_raws)
    parts = dict(world_loss=float(loss.detach()), reconstruction=float(recon.detach()),
                 reconstruction_mse_per_pixel=float(recon.detach()) * 2 / PIXELS,
                 kl_loss=float(kl_loss.detach()), kl_raw=float(kl_raw_all.mean()),
                 overshooting_kl=float(overshoot.detach()), reward_loss=float(reward_loss.detach()),
                 continue_loss=float(continue_loss.detach()))
    extra = None
    if diagnostics:
        extra = dict(kl_dims=torch.cat(kl_dims_all).mean(0), h_states=h_states, q_states=q_states)
    return loss, parts, extra


def diagnose(model, images, actions, goals, reasoning, extra, args, device):
    """Does z carry the frame? (computed on one validation batch, no gradients)"""
    import torch
    batch, steps = actions.shape
    out = {}
    dims = extra["kl_dims"]
    out["kl_per_dim"] = [round(float(v), 5) for v in dims]
    out["active_dims"] = int((dims > args.active_threshold).sum())
    # posterior on the real frame vs on another sample's frame, same h
    h1, q1 = extra["h_states"][1], extra["q_states"][1]
    shuffled = images[torch.roll(torch.arange(batch, device=device), 1), 1]
    q_shuf = model.posterior(shuffled, h1)
    out["posterior_real_vs_shuffled_kl"] = float(model.kl_dims(q1, q_shuf).sum(-1).mean())
    out["posterior_real_vs_shuffled_mu_l2"] = float((q1[0] - q_shuf[0]).flatten(1).norm(dim=-1).mean())
    # reconstruction of every target frame with z from the posterior vs from the prior mean
    post, prior = [], []
    for t in range(steps):
        h, q = extra["h_states"][t + 1], extra["q_states"][t + 1]
        target = images[:, t + 1]
        post.append((model.decode(h, model.mode(q)) - target).square().mean())
        prior.append((model.decode(h, model.mode(model.prior(h))) - target).square().mean())
    out["recon_mse_posterior_z"] = float(torch.stack(post).mean())
    out["recon_mse_prior_mean_z"] = float(torch.stack(prior).mean())
    # open loop from the first frame: prior means only, no further frames
    h, z = extra["h_states"][0], model.mode(extra["q_states"][0])
    rollout = []
    for t in range(steps):
        h, _ = model.transition(goals, actions[:, t], [r[t] for r in reasoning], h, z, device)
        z = model.mode(model.prior(h))
        rollout.append(float((model.decode(h, z) - images[:, t + 1]).square().mean()))
    out["open_loop_mse_by_step"] = [round(v, 5) for v in rollout]
    return out


def evaluate(model, root, chunks, args, device):
    import torch
    sums, count, diag = defaultdict(float), 0, None
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for offset in range(0, len(chunks), args.batch_size):
            images, actions, rewards, continuation, goals, reasoning = batch_from_chunks(
                root, chunks[offset:offset + args.batch_size], device)
            want = diag is None and len(goals) == args.batch_size
            _, parts, extra = loss_for_batch(model, images, actions, rewards, continuation, goals, reasoning, args,
                                             device, diagnostics=want)
            if want:
                diag = diagnose(model, images, actions, goals, reasoning, extra, args, device)
            count += 1
            for key, value in parts.items():
                sums[key] += value
    return {key: value / count for key, value in sums.items()}, diag or {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data", type=Path, required=True, help="teacher collection (train/val.parquet, images/)")
    parser.add_argument("--split-from", type=Path, required=True,
                        help="phase-2 CE data dir: its valid.jsonl episodes are the validation episodes")
    parser.add_argument("--init-checkpoint", type=Path, required=True, help="initial_random of a phase-2 run")
    parser.add_argument("--reasoning", type=Path, help="reasoning.jsonl: the action text gets its LLM reasoning")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--h-size", type=int, default=512)
    parser.add_argument("--z-size", type=int, default=48, help="gaussian dims, or number of categoricals")
    parser.add_argument("--z-kind", choices=("gaussian", "categorical"), default="gaussian")
    parser.add_argument("--z-classes", type=int, default=32, help="classes per categorical (--z-kind categorical)")
    parser.add_argument("--image-net", choices=("small", "dreamer"), default="small",
                        help="small: 3-conv CNN + 7x7 bilinear decoder; dreamer: DreamerV3 stride-2 CNN / transposed-conv decoder")
    parser.add_argument("--h-tokens", type=int, default=30)
    parser.add_argument("--z-tokens", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="split each batch into this many micro-batches (same objective, less memory)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--free-nats", type=float, default=1.0)
    parser.add_argument("--kl-dyn", type=float, default=0.5)
    parser.add_argument("--kl-rep", type=float, default=0.1)
    parser.add_argument("--reward-loss-scale", type=float, default=1.0)
    parser.add_argument("--continue-loss-scale", type=float, default=0.1)
    parser.add_argument("--reward-target-scale", type=float, default=50.0)
    parser.add_argument("--active-threshold", type=float, default=0.01, help="nats; a z dim above it is active")
    parser.add_argument("--max-updates", type=int, default=0, help="smoke: stop after this many updates")
    parser.add_argument("--seed", type=int, default=7950)
    args = parser.parse_args()
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from transformers import AutoModelForCausalLM, AutoTokenizer
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.init_checkpoint)
    lm = AutoModelForCausalLM.from_pretrained(args.init_checkpoint).to(device)
    model = SLMWorldModel(lm, tokenizer, args.h_size, args.z_size, args.h_tokens, args.z_tokens, args.z_kind,
                          args.z_classes, args.image_net).to(device)
    reasoning = None
    if args.reasoning:
        reasoning = {(int(r["episode_id"]), int(r["step"])): r for r in map(json.loads, args.reasoning.open())}
    episodes = load_episodes(args.raw_data, reasoning)
    valid_ids = {json.loads(line)["source_episode_id"] for line in (args.split_from / "valid.jsonl").open()}
    train_episodes = {k: v for k, v in episodes.items() if k not in valid_ids}
    valid_episodes = {k: v for k, v in episodes.items() if k in valid_ids}
    train_chunks, valid_chunks = make_chunks(train_episodes, args.horizon), make_chunks(valid_episodes, args.horizon)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    config = dict(vars(args), architecture="slm_transition_rssm_v2", trainable="all (SLM + support modules)",
                  reconstruction="0.5 * sum over pixels of squared error (unit-variance Gaussian)",
                  kl="kl_dyn * max(free_nats, KL(sg q || p)) + kl_rep * max(free_nats, KL(q || sg p)), z_0 included",
                  z=(f"{args.z_size} gaussian dims" if args.z_kind == "gaussian" else
                     f"{args.z_size} x {args.z_classes} categorical, 1% unimix, straight-through samples"),
                  train_episodes=len(train_episodes), validation_episodes=len(valid_episodes),
                  train_transitions=sum(len(e) for e in train_episodes.values()),
                  validation_transitions=sum(len(e) for e in valid_episodes.values()),
                  train_chunks=len(train_chunks), validation_chunks=len(valid_chunks),
                  lm_parameters=sum(p.numel() for p in lm.parameters()),
                  support_parameters=sum(p.numel() for p in model.modules.parameters()), started_at=time.time())
    save_json(args.out / "experiment.json", {k: str(v) if isinstance(v, Path) else v for k, v in config.items()})
    writer = SummaryWriter(str(args.out / "tensorboard"))
    best, updates, started = float("inf"), 0, time.time()
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_chunks); model.train(); sums = defaultdict(float); batches = 0
        for offset in range(0, len(train_chunks), args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            batch_chunks = train_chunks[offset:offset + args.batch_size]
            micro = -(-len(batch_chunks) // args.grad_accum)
            parts = defaultdict(float)
            for start in range(0, len(batch_chunks), micro):
                part = batch_chunks[start:start + micro]
                images, actions, rewards, continuation, goals, rs = batch_from_chunks(args.raw_data, part, device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    loss, micro_parts, _ = loss_for_batch(model, images, actions, rewards, continuation, goals, rs,
                                                          args, device)
                # the losses are batch means: weight each micro-batch by its share of the batch
                (loss * len(part) / len(batch_chunks)).backward()
                for key, value in micro_parts.items():
                    parts[key] += value * len(part) / len(batch_chunks)
            parts["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimizer.step()
            updates += 1; batches += 1
            for key, value in parts.items():
                sums[key] += value
            if args.max_updates and updates >= args.max_updates:
                break
        train = {key: value / batches for key, value in sums.items()}
        valid, diag = evaluate(model, args.raw_data, valid_chunks, args, device)
        metrics = {"epoch": epoch, "updates": updates, "seconds": time.time() - started,
                   **{f"train_{k}": v for k, v in train.items()}, **{f"validation_{k}": v for k, v in valid.items()},
                   **{f"diag_{k}": v for k, v in diag.items()}}
        with (args.out / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics) + "\n")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(key, value, epoch)
        writer.flush()
        save_json(args.out / "progress.json", metrics)
        print(json.dumps({k: v for k, v in metrics.items() if k != "diag_kl_per_dim"}), flush=True)
        if valid["world_loss"] < best:
            best = valid["world_loss"]
            torch.save(dict(model=model.state_dict(), config=config, metrics=metrics), args.out / "best.pt")
        if args.max_updates and updates >= args.max_updates:
            break
    save_json(args.out / "complete.json", dict(best_validation_world_loss=best, updates=updates))
    writer.close()


if __name__ == "__main__":
    main()
