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

--descriptions state_descriptions.jsonl: the SLM also WRITES the text of the next state. After the EOS of the
transition input it continues with "Next state: <description of x_{t+1}><eos>" (teacher forcing); the summed token
NLL is added to the loss (x --description-loss-scale). Attention is causal, so h_{t+1} and the heads (EOS state)
never see the target, while the EOS state must carry what the text needs. In imagination the SLM can generate it.
--description-input: the transition also reads "State: <description of x_t>" before the action (dataset text in
training, teacher forcing; in imagination the SLM's own description from the previous step).

--transition gru: RSSM baseline without the SLM, h_{t+1} = GRU(h_t, MLP([z_t; one-hot a_t])) as in DreamerV3
(action only), with everything else unchanged; the SLM's quality is measured against it.

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
import re
import time
from collections import Counter, defaultdict
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
VALUE_TASK_SIZE, VALUE_UNITS, VALUE_LAYERS = 64, 256, 3  # task embedding, MLP of the value head
WORD_SIZE = 1024  # at most this many words per word head, the most frequent of the train text
WORD_NORM = 25.0  # about the words of a description: keeps the summed word loss near a few nats
IMAGINED_WORDS = (1, 2, 4, 8, 16)  # imagined steps scored by the word head


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


def load_episodes(root: Path, reasoning: dict | None, descriptions: dict | None = None):
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
        description = None
        if descriptions is not None:
            if key not in descriptions:
                raise ValueError(f"no state description at episode {key[0]} step {key[1]}")
            description = descriptions[key]
        episodes[key[0]].append(dict(step=key[1], action=ACTION_ID[str(extra["teacher_action"])], reasoning=text,
                                     description=description,
                                     image=Path(str(extra["observation_path"])).name,
                                     instruction=str(extra["instruction"]), reward=float(extra["reward"]),
                                     terminal=bool(extra["done"]), done=bool(extra["done"]),
                                     success=bool(extra["success"])))
    for episode_id, episode in episodes.items():
        episode.sort(key=lambda r: r["step"])
        if [r["step"] for r in episode] != list(range(len(episode))):
            raise ValueError(f"non-contiguous episode {episode_id}")
        episode[-1]["terminal"] = True  # a collection cutoff also ends the rollout
    return dict(episodes)


def add_value_targets(episodes, gamma: float):
    """Return of the task, not of the Craftax reward (which pays any achievement): G_t = gamma^(T-1-t) when the
    instruction is completed on arriving at the last state T, else 0. The last state and episodes cut by the
    collection (no done) have no target."""
    for episode in episodes.values():
        last = len(episode) - 1
        success, ended = episode[-1]["success"], episode[-1]["done"]
        for t, item in enumerate(episode):
            item["value"] = float(success) * gamma ** (last - 1 - t) if t < last else 0.0
            item["value_mask"] = float(ended and t < last)


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


IMAGE_CACHE = None  # --cache-images: name -> uint8 frame (the GRU baseline is otherwise bound by frame loading)


def cached_image(root: Path, name: str) -> np.ndarray:
    if IMAGE_CACHE is None:
        return read_image(root, name)
    if name not in IMAGE_CACHE:
        IMAGE_CACHE[name] = (read_image(root, name) * 255).round().astype(np.uint8)
    return IMAGE_CACHE[name].astype(np.float32) / 255


def next_descriptions(chunks):
    """Description of the state AFTER each transition (the text target of --descriptions), [chunk][step]."""
    return [[item["description"] for item in chunk[1:]] for chunk in chunks]


def current_descriptions(chunks):
    """Description of the state BEFORE each transition (the text input of --description-input), [chunk][step]."""
    return [[item["description"] for item in chunk[:-1]] for chunk in chunks]


def batch_from_chunks(root, chunks, device):
    import torch
    images = np.stack([[cached_image(root, item["image"]) for item in chunk] for chunk in chunks])
    actions = np.asarray([[item["action"] for item in chunk[:-1]] for chunk in chunks], dtype=np.int64)
    rewards = np.asarray([[item["reward"] for item in chunk[1:]] for chunk in chunks], dtype=np.float32)
    continuation = np.asarray([[0.0 if item["terminal"] else 1.0 for item in chunk[1:]] for chunk in chunks],
                              dtype=np.float32)
    reasoning = [[item["reasoning"] for item in chunk[:-1]] for chunk in chunks]
    goals = [chunk[0]["instruction"] for chunk in chunks]
    return (torch.from_numpy(images).to(device), torch.from_numpy(actions).to(device),
            torch.from_numpy(rewards).to(device), torch.from_numpy(continuation).to(device), goals, reasoning)


def value_targets(chunks, device):
    """(G_t, mask), [chunk][state], states 0..horizon."""
    import torch
    rows = torch.tensor([[(item["value"], item["value_mask"]) for item in chunk] for chunk in chunks],
                        dtype=torch.float32, device=device)
    return rows[..., 0], rows[..., 1]


def tile_targets(model, images):
    """Class of every map tile, by the nearest texture prototype of the frame itself: [chunk][state][7][9]."""
    from imagine_eval import classify_tiles
    return classify_tiles(images, *model.tile_tables)[0]


def words_of(text: str) -> set:
    return set(re.findall(r"[a-z]+", text.lower()))


def word_vocabulary(texts) -> tuple:
    """The WORD_SIZE most frequent words of the text, with their share of texts."""
    counts = Counter(word for text in texts for word in words_of(text))
    top = counts.most_common(WORD_SIZE)
    return [word for word, _ in top], [count / max(len(texts), 1) for _, count in top]


def word_targets(chunks, key, size, device):
    """Multi-hot words, [chunk][state][word]."""
    import torch
    out = np.zeros((len(chunks), len(chunks[0]), size), dtype=np.float32)
    for c, chunk in enumerate(chunks):
        for t, item in enumerate(chunk):
            out[c, t, item[key]] = 1.0
    return torch.from_numpy(out).to(device)


def word_bce(logits, targets, weight):
    import torch.nn.functional as F
    return F.binary_cross_entropy_with_logits(logits.float(), targets, pos_weight=weight, reduction="none").sum(-1).mean() / WORD_NORM


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


def dreamer_encoder(nn, depth=32):
    """DreamerV3 CNN: 64x64 (63x63 padded) -> stride-2 convs depth x 1/2/4/8 -> 4x4x(8 depth) -> flat 128 depth
    (depth 32: 32/64/128/256 -> 4096)."""
    layers, channels = [nn.ZeroPad2d((0, 1, 0, 1))], 3
    for out in (depth, 2 * depth, 4 * depth, 8 * depth):
        layers += [nn.Conv2d(channels, out, 4, stride=2, padding=1), nn.GroupNorm(1, out), nn.SiLU()]
        channels = out
    return nn.Sequential(*layers, nn.Flatten())


def dreamer_decoder(nn, size, depth=32):
    """DreamerV3 CNN decoder: [h; z] -> 4x4x(8 depth) -> transposed convs depth x 4/2/1, then 3 -> 64x64, cropped
    to 63x63 (depth 32: 256 -> 128/64/32/3)."""
    layers = [nn.Linear(size, 8 * depth * 4 * 4), nn.Unflatten(1, (8 * depth, 4, 4))]
    channels = 8 * depth
    for out in (4 * depth, 2 * depth, depth):
        layers += [nn.ConvTranspose2d(channels, out, 4, stride=2, padding=1), nn.GroupNorm(1, out), nn.SiLU()]
        channels = out
    # negative padding crops the 64x64 output back to the 63x63 frame
    layers += [nn.ConvTranspose2d(channels, 3, 4, stride=2, padding=1), nn.ZeroPad2d((0, -1, 0, -1)), nn.Sigmoid()]
    return nn.Sequential(*layers)


def grid_nets(nn, torch, h_size, cells, classes, depth):
    """--image-net grid: z is one categorical variable per 7 x 7 patch of the 63 x 63 frame, so a small object has its own
    variable and needs no place in a global code. Patches come from the frame's size alone, nothing here knows the game.
    Returns (encoder, its output size, posterior, decoder); the posterior gets [h; features], the decoder [h; z]."""
    side = round(cells ** .5)
    if side * side != cells or side * 7 != 63:
        raise ValueError("--image-net grid needs --z-size 81: 9 x 9 patches of 7 x 7 pixels")
    channels = 2 * depth

    class Cells(nn.Module):
        def __init__(self):
            super().__init__()
            self.context = nn.Linear(h_size, channels)
            self.out = nn.Sequential(nn.Conv2d(2 * channels, channels, 1), nn.SiLU(), nn.Conv2d(channels, classes, 1))

        def forward(self, x):
            h, features = x[:, :h_size], x[:, h_size:].view(-1, channels, side, side)
            context = self.context(h)[:, :, None, None].expand(-1, -1, side, side)
            return self.out(torch.cat((features, context), 1)).permute(0, 2, 3, 1).flatten(1)

    class Draw(nn.Module):
        def __init__(self):
            super().__init__()
            self.context = nn.Linear(h_size, channels)
            self.net = nn.Sequential(
                nn.Conv2d(classes + channels, 2 * channels, 1), nn.SiLU(), nn.Conv2d(2 * channels, 2 * channels, 3, padding=1), nn.SiLU(),
                nn.ConvTranspose2d(2 * channels, channels, 7, stride=7), nn.SiLU(), nn.Conv2d(channels, depth, 3, padding=1), nn.SiLU(),
                nn.Conv2d(depth, 3, 1), nn.Sigmoid())

        def forward(self, x):
            h, z = x[:, :h_size], x[:, h_size:].view(-1, side, side, classes).permute(0, 3, 1, 2)
            context = self.context(h)[:, :, None, None].expand(-1, -1, side, side)
            return self.net(torch.cat((z, context), 1))

    encoder = nn.Sequential(
        nn.Conv2d(3, depth, 3, padding=1), nn.GroupNorm(1, depth), nn.SiLU(), nn.Conv2d(depth, channels, 7, stride=7),
        nn.GroupNorm(1, channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(1, channels), nn.SiLU(), nn.Flatten())
    return encoder, channels * cells, Cells(), Draw()


def mlp(nn, size, units, layers, out):
    """layers x (Linear, GELU), then Linear to out."""
    modules = []
    for _ in range(layers):
        modules += [nn.Linear(size, units), nn.GELU()]
        size = units
    return nn.Sequential(*modules, nn.Linear(size, out))


class SLMWorldModel:
    """Support modules around one trainable decoder-only SLM transition."""

    def __init__(self, lm, tokenizer, h_size, z_size, h_tokens, z_tokens, z_kind="gaussian", z_classes=32,
                 image_net="small", decoder_input="hz", posterior_input="hx", cnn_depth=32, mlp_units=None, mlp_layers=1):
        import torch
        import torch.nn as nn
        self.lm, self.tok, self._torch = lm, tokenizer, torch
        self.width = h_size if lm is None else lm.config.hidden_size  # lm None: RSSMWorldModel (GRU core)
        self.h_size, self.z_size, self.h_tokens, self.z_tokens = h_size, z_size, h_tokens, z_tokens
        # gaussian: z = z_size reals; categorical: z = z_size one-hot variables of z_classes classes (flattened)
        self.z_kind, self.z_classes = z_kind, z_classes
        z_dim = z_size if z_kind == "gaussian" else z_size * z_classes
        z_raw = 2 * z_size if z_kind == "gaussian" else z_size * z_classes
        self.z_dim = z_dim
        # decoder_input "z" / posterior_input "x": the decoder reads only z / the posterior only the frame, so z is a
        # full code of the frame and h serves only the prior and the heads (tokenizer + dynamics, as STORM / IRIS)
        self.decoder_input, self.posterior_input = decoder_input, posterior_input
        dec_size = z_dim if decoder_input == "z" else h_size + z_dim
        if image_net == "small":
            cnn, embed = nn.Sequential(
                nn.Conv2d(3, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(), nn.Linear(32 * 7 * 7, 256), nn.LayerNorm(256), nn.Tanh(),
            ), 256
            decoder = nn.Sequential(
                nn.Linear(dec_size, 32 * 7 * 7), nn.GELU(), nn.Unflatten(1, (32, 7, 7)),
                nn.Upsample(size=(63, 63), mode="bilinear", align_corners=False),
                nn.Conv2d(32, 32, 5, padding=2), nn.GELU(), nn.Conv2d(32, 3, 3, padding=1), nn.Sigmoid(),
            )
        elif image_net == "grid":
            if z_kind != "categorical" or decoder_input != "hz" or posterior_input != "hx":
                raise ValueError("--image-net grid needs categorical z, a decoder and a posterior that read h")
            cnn, embed, cells, decoder = grid_nets(nn, torch, h_size, z_size, z_classes, cnn_depth)
        else:  # "dreamer"
            cnn, embed = dreamer_encoder(nn, cnn_depth), 8 * cnn_depth * 4 * 4
            decoder = dreamer_decoder(nn, dec_size, cnn_depth)
        # DreamerV3 sizes: hidden width of the GRU input, prior, posterior and heads (default h_size); the heads get
        # mlp_layers hidden layers, prior / posterior keep one as in DreamerV3
        units = mlp_units or h_size
        if lm is None:  # DreamerV3 recurrent core: h_{t+1} = GRU(h_t, MLP([z_t; one-hot a_t]))
            core = {"gru_input": nn.Sequential(nn.Linear(z_dim + len(ACTIONS), units), nn.LayerNorm(units), nn.SiLU()),
                    "gru": nn.GRUCell(units, h_size)}
        else:
            core = {"h_prefix": nn.Sequential(nn.Linear(h_size, h_size), nn.GELU(), nn.Linear(h_size, h_tokens * self.width)),
                    "z_prefix": nn.Sequential(nn.Linear(z_dim, z_dim), nn.GELU(), nn.Linear(z_dim, z_tokens * self.width)),
                    "h_transition": nn.Sequential(nn.Linear(self.width, h_size), nn.GELU(), nn.Linear(h_size, h_size))}
        self.modules = nn.ModuleDict({
            "cnn": cnn,
            "posterior": cells if image_net == "grid" else mlp(nn, (0 if posterior_input == "x" else h_size) + embed, units, 1, z_raw),
            **core,
            "z_prior": mlp(nn, h_size, units, 1, z_raw),
            "reward_head": mlp(nn, self.width, units, mlp_layers, 1),
            "continue_head": mlp(nn, self.width, units, mlp_layers, 1),
            "pixel_decoder": decoder,
        })
        self._text_cache = {}
        self.z_fixed_std, self.prior_skip = None, False  # build_model sets them

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
        if self.z_kind != "gaussian":
            return categorical(raw, self.z_size, self.z_classes)
        mu, std = normal(raw)
        return (mu, std.new_full(std.shape, self.z_fixed_std)) if self.z_fixed_std else (mu, std)

    def posterior(self, image, h):
        embed = self.modules["cnn"](image)
        return self._dist(self.modules["posterior"](embed if self.posterior_input == "x" else self._torch.cat((h, embed), dim=-1)))

    def prior(self, h):
        dist = self._dist(self.modules["z_prior"](h))
        return (h[:, -self.z_dim:].to(dist[0].dtype) + dist[0], dist[1]) if self.prior_skip else dist

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

    def _transition_parts(self, goals, actions, reasoning, h, z, device, states=None):
        """Embeddings of one transition input per sample, up to its EOS: task, h tokens, z tokens, action text."""
        torch = self._torch
        embed = self.lm.get_input_embeddings()
        eos = int(self.tok.eos_token_id)
        hs = self.modules["h_prefix"](h).view(len(h), self.h_tokens, self.width)
        zs = self.modules["z_prefix"](z).view(len(z), self.z_tokens, self.width)
        samples = []
        for index, (goal, action, text) in enumerate(zip(goals, actions.tolist(), reasoning)):
            answer = f"ACTION: {ACTIONS[int(action)]}" if text is None else f"<reasoning>{text}</reasoning>\nACTION: {ACTIONS[int(action)]}"
            if states is not None:
                answer = f"State: {states[index]}\n{answer}"
            head = embed(torch.tensor(self.ids(f"Task:\n{goal}\n\nPredict next environment state after action:\n"), device=device))
            tail = embed(torch.tensor(self.ids(answer) + [eos], device=device))
            samples.append([head, hs[index].to(head.dtype), zs[index].to(head.dtype), tail])
        return samples

    def _run_lm(self, pieces, device, **kwargs):
        """The SLM on embedded sequences of different lengths, padded on the right."""
        torch = self._torch
        maximum = max(piece.shape[0] for piece in pieces)
        padded = torch.stack([torch.cat((piece, piece.new_zeros(maximum - piece.shape[0], self.width))) for piece in pieces])
        mask = torch.tensor([[1] * piece.shape[0] + [0] * (maximum - piece.shape[0]) for piece in pieces], device=device)
        return self.lm(inputs_embeds=padded, attention_mask=mask, use_cache=False, **kwargs)

    def describe(self, goals, actions, reasoning, h, z, device, states=None, max_tokens=200):
        """Greedy text of the next state from the inputs of a transition: what --descriptions trains the SLM to write
        after the EOS. Full forward per token (the prompts differ in length). Returns a string per sample."""
        torch = self._torch
        embed = self.lm.get_input_embeddings()
        eos = int(self.tok.eos_token_id)
        start = torch.tensor(self.ids("Next state:"), device=device)
        pieces = [torch.cat(parts + [embed(start)], dim=0)
                  for parts in self._transition_parts(goals, actions, reasoning, h, z, device, states)]
        written = [[] for _ in pieces]
        active = list(range(len(pieces)))
        for _ in range(max_tokens):
            logits = self._run_lm([pieces[i] for i in active], device).logits
            rows = torch.arange(len(active), device=device)
            ends = torch.tensor([pieces[i].shape[0] - 1 for i in active], device=device)
            tokens = logits[rows, ends].float().argmax(-1).tolist()
            for i, token in zip(active, tokens):
                if token != eos:
                    written[i].append(token)
                    pieces[i] = torch.cat((pieces[i], embed(torch.tensor([token], device=device))), dim=0)
            active = [i for i, token in zip(active, tokens) if token != eos]
            if not active:
                break
        return [self.tok.decode(ids, skip_special_tokens=True).strip() for ids in written]

    def transition(self, goals, actions, reasoning, h, z, device, targets=None, states=None):
        """Run the SLM; h[t+1] and the heads read only its final (EOS) hidden state.

        states (--description-input): text of the CURRENT state per sample, put before the action as "State: ...".

        targets (--descriptions): text of the NEXT state per sample, appended after the EOS and scored as language
        modelling (teacher forcing). Attention is causal, so the EOS state (h[t+1], heads) never sees it. Then also
        returns the summed negative log-likelihood of each target, shape [B]."""
        torch = self._torch
        embed = self.lm.get_input_embeddings()
        eos = int(self.tok.eos_token_id)
        pieces, ends, target_ids = [], [], []
        for index, parts in enumerate(self._transition_parts(goals, actions, reasoning, h, z, device, states)):
            ends.append(sum(part.shape[0] for part in parts) - 1)
            if targets is not None:
                ids = self.ids(f"Next state: {targets[index]}") + [eos]
                target_ids.append(ids)
                parts.append(embed(torch.tensor(ids, device=device)))
            pieces.append(torch.cat(parts, dim=0))
        output = self._run_lm(pieces, device, output_hidden_states=True)
        rows = torch.arange(len(pieces), device=device)
        last = output.hidden_states[-1][rows, torch.tensor(ends, device=device)].float()
        if targets is None:
            return self.modules["h_transition"](last), last
        nll = []
        for index, (end, ids) in enumerate(zip(ends, target_ids)):
            # the logits at position p predict the token at p + 1: the EOS predicts the first target token
            logits = output.logits[index, end:end + len(ids)].float()
            nll.append(torch.nn.functional.cross_entropy(logits, torch.tensor(ids, device=device), reduction="sum"))
        lengths = torch.tensor([len(ids) for ids in target_ids], device=device, dtype=torch.float32)
        return self.modules["h_transition"](last), last, torch.stack(nll), lengths

    def decode(self, h, z):
        return self.modules["pixel_decoder"](z if self.decoder_input == "z" else self._torch.cat((h, z), dim=-1))


class RSSMWorldModel(SLMWorldModel):
    """Baseline without the SLM (--transition gru): DreamerV3 GRU core, action only (no task / reasoning text).
    Latent, image nets, prior / posterior, heads and losses are those of SLMWorldModel; the heads read h_{t+1}."""

    def __init__(self, h_size, z_size, z_kind="gaussian", z_classes=32, image_net="small", decoder_input="hz",
                 posterior_input="hx", cnn_depth=32, mlp_units=None, mlp_layers=1):
        super().__init__(None, None, h_size, z_size, 0, 0, z_kind, z_classes, image_net, decoder_input, posterior_input,
                         cnn_depth, mlp_units, mlp_layers)

    def to(self, device):
        self.modules.to(device); return self

    def train(self, mode=True):
        self.modules.train(mode); return self

    def parameters(self):
        return list(self.modules.parameters())

    def state_dict(self):
        return {"support": self.modules.state_dict(), "lm": {}}

    def transition(self, goals, actions, reasoning, h, z, device, targets=None, states=None):
        if targets is not None or states is not None:
            raise ValueError("the GRU transition reads and writes no text")
        a = self._torch.nn.functional.one_hot(actions, len(ACTIONS)).to(z.dtype)
        core = h[:, :-self.z_dim] if self.prior_skip else h
        h = self.modules["gru"](self.modules["gru_input"](self._torch.cat((z, a), dim=-1)), core)
        if self.prior_skip:
            h = self._torch.cat((h, z.to(h.dtype)), dim=-1)
        return h, h

    def add_prior_skip(self):
        """--prior-skip: h = [GRU state; z_t], the prior predicts z_{t+1} - z_t, so a large z need not pass through the GRU."""
        if self.z_kind != "gaussian":
            raise ValueError("--prior-skip needs a gaussian z")
        self.prior_skip = True
        units = self.modules["gru_input"][0].out_features
        self.modules["gru"] = self._torch.nn.GRUCell(units, self.h_size - self.z_dim)

    def add_value(self, tasks):
        """Value head of --value-loss-scale: V(h_t, z_t, g), the RSSM itself has no task."""
        nn = self._torch.nn
        self.value_tasks = {task: index for index, task in enumerate(tasks)}
        self.modules["value_task"] = nn.Embedding(len(tasks), VALUE_TASK_SIZE)
        self.modules["value_head"] = mlp(nn, self.h_size + self.z_dim + VALUE_TASK_SIZE, VALUE_UNITS, VALUE_LAYERS, 1)

    def value(self, h, z, goals):
        torch = self._torch
        tasks = torch.tensor([self.value_tasks[goal] for goal in goals], device=h.device)
        return self.modules["value_head"](torch.cat((h, z, self.modules["value_task"](tasks)), dim=-1)).squeeze(-1).float()

    def add_words(self, name, size):
        """Word heads (--word-descriptions, --word-stories): which words the text of a state has, from h, z."""
        self.modules[name] = mlp(self._torch.nn, self.h_size + self.z_dim, self.h_size, 2, size)

    def words(self, name, h, z):
        return self.modules[name](self._torch.cat((h, z), dim=-1))

    def add_tiles(self, classes):
        """Tile class head of --tile-loss-scale: the class of each of the 7 x 9 map tiles from h, z."""
        self.tile_shape = (7, 9, len(classes))
        self.modules["tile_head"] = mlp(self._torch.nn, self.h_size + self.z_dim, self.h_size, 2, 7 * 9 * len(classes))

    def tile_logits(self, h, z):
        return self.modules["tile_head"](self._torch.cat((h, z), dim=-1)).float().view(-1, *self.tile_shape)


def build_model(config, init_checkpoint, device):
    """The world model described by a run config (experiment.json / checkpoint config), untrained."""
    z = (config["z_size"], config.get("z_kind", "gaussian"), config.get("z_classes", 32), config.get("image_net", "small"),
         config.get("decoder_input", "hz"), config.get("posterior_input", "hx"), config.get("cnn_depth", 32),
         config.get("mlp_units"), config.get("mlp_layers", 1))
    if config.get("transition", "slm") == "gru":
        model = RSSMWorldModel(config["h_size"], *z)
        if config.get("value_tasks"):
            model.add_value(config["value_tasks"])
        if config.get("tile_classes"):
            model.add_tiles(config["tile_classes"])
        if config.get("word_vocabulary"):
            model.add_words("word_head", len(config["word_vocabulary"]))
        if config.get("story_vocabulary"):
            model.add_words("story_head", len(config["story_vocabulary"]))
        model.z_fixed_std = config.get("z_fixed_std")
        if config.get("prior_skip"):
            model.add_prior_skip()
        return model.to(device)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(init_checkpoint)
    lm = AutoModelForCausalLM.from_pretrained(init_checkpoint).to(device)
    return SLMWorldModel(lm, tokenizer, config["h_size"], z[0], config["h_tokens"], config["z_tokens"], *z[1:]).to(device)


IMAGINED_PIXELS = (2, 4, 8)  # distances of the overshooting rollouts drawn by --prior-recon


def detach(dist):
    return tuple(t.detach() for t in dist)


def prior_pixel_error(model, h, p, image):
    """--prior-recon: .5 * squared error of the frame drawn from the prior mean. It trains the dynamics, not the decoder."""
    decoder = list(model.modules["pixel_decoder"].parameters())
    for v in decoder:
        v.requires_grad_(False)
    out = model.decode(h, model.mode(p)).float()
    for v in decoder:
        v.requires_grad_(True)
    return .5 * (out - image).square().flatten(1).sum(-1).mean()


def balanced_kl(model, q, p, free, dyn_scale, rep_scale):
    """DreamerV3: free bits per sample on each side; returns (loss, raw KL per sample, KL per dim)."""
    dims_dyn = model.kl_dims(detach(q), p)
    dims_rep = model.kl_dims(q, detach(p))
    dyn, rep = dims_dyn.sum(-1), dims_rep.sum(-1)
    loss = dyn_scale * dyn.clamp_min(free).mean() + rep_scale * rep.clamp_min(free).mean()
    return loss, dyn.detach(), dims_dyn.detach()


def loss_for_batch(model, images, actions, rewards, continuation, goals, reasoning, args, device, diagnostics=False,
                   descriptions=None, states=None, values=None, tiles=None, words=None, stories=None):
    """descriptions: [chunk][step] text of the state after each transition (--descriptions), scored by the SLM.
    states: [chunk][step] text of the state before each transition (--description-input), fed to the SLM; like the
    reasoning text it comes from the dataset also in the overshooting rollouts.
    values: (G_t, mask) of value_targets (--value-loss-scale). tiles: tile_targets (--tile-loss-scale).
    words, stories: word_targets of the text of each state and of the story after it."""
    state_at = (lambda t: None) if states is None else (lambda t: [s[t] for s in states])
    import torch
    import torch.nn.functional as F
    batch, steps = actions.shape
    text_nll, text_tokens = torch.zeros((), device=device), torch.zeros((), device=device)
    zeros = torch.zeros(batch, model.h_size, device=device)
    h = zeros
    q = model.posterior(images[:, 0], h)
    z = model.sample(q)
    # z_0 is regularised like every other z: against p(z_0 | h_0 = 0)
    kl_loss, kl_raw, kl_dims = balanced_kl(model, q, model.prior(h), args.free_nats, args.kl_dyn, args.kl_rep)
    kl_raws, kl_dims_all = [kl_raw], [kl_dims]
    h_states, q_states, z_states = [h], [q], [z]
    recon = torch.zeros((), device=device)
    reward_loss = torch.zeros((), device=device)
    continue_loss = torch.zeros((), device=device)
    tile_loss, tile_right = torch.zeros((), device=device), 0.0
    word_loss, story_loss, imagined_loss, imagined_terms = [torch.zeros((), device=device) for _ in range(3)] + [0]
    focus = getattr(args, "recon_focus", 0.0)
    # --free-encoder: the dynamics read z as data, nothing but the reconstruction reaches the encoder
    fed = (lambda t: t.detach()) if getattr(args, "free_encoder", False) else (lambda t: t)
    pixel_scale, pixel_loss, pixel_terms = getattr(args, "prior_recon", 0.0), torch.zeros((), device=device), 0
    for t in range(steps):
        rs = [r[t] for r in reasoning]
        if descriptions is None:
            h, last = model.transition(goals, actions[:, t], rs, h, fed(z), device, states=state_at(t))
        else:
            h, last, nll, lengths = model.transition(goals, actions[:, t], rs, h, fed(z), device,
                                                     targets=[d[t] for d in descriptions], states=state_at(t))
            text_nll, text_tokens = text_nll + nll.mean(), text_tokens + lengths.mean()
        p = model.prior(h)
        q = model.posterior(images[:, t + 1], h)
        z = model.sample(q)
        # Gaussian log-likelihood with unit variance: 1/2 squared error summed over the frame's pixels
        error = (model.decode(h, z).float() - images[:, t + 1]).square()
        if focus:  # the pixels of a tile weigh (its error / the frame's mean error) ** focus, mean weight 1 per frame
            cells = F.avg_pool2d(error.detach().mean(1, keepdim=True), 7)
            weights = (cells / cells.mean((1, 2, 3), keepdim=True).clamp_min(1e-8)) ** focus
            weights = weights / weights.mean((1, 2, 3), keepdim=True).clamp_min(1e-8)
            error = error * F.interpolate(weights, scale_factor=7, mode="nearest")
        recon = recon + .5 * error.flatten(1).sum(-1).mean()
        if pixel_scale:
            pixel_loss = pixel_loss + prior_pixel_error(model, h, p, images[:, t + 1]); pixel_terms += 1
        if tiles is not None:
            logits = model.tile_logits(h, z)
            tile_loss = tile_loss + F.cross_entropy(logits.flatten(0, 2), tiles[:, t + 1].flatten(), weight=model.tile_weight,
                                                    reduction="none").view(batch, -1).sum(-1).mean()
            tile_right = tile_right + float((logits.detach().argmax(-1) == tiles[:, t + 1]).float().mean())
        if words is not None:
            word_loss = word_loss + word_bce(model.words("word_head", h, z), words[:, t + 1], model.word_weight)
        if stories is not None:
            story_loss = story_loss + word_bce(model.words("story_head", h, z), stories[:, t + 1], model.story_weight)
        reward_loss = reward_loss + F.mse_loss(model.modules["reward_head"](last).squeeze(-1).float(),
                                               rewards[:, t] * args.reward_target_scale)
        continue_loss = continue_loss + F.binary_cross_entropy_with_logits(
            model.modules["continue_head"](last).squeeze(-1).float(), continuation[:, t])
        step_kl, step_raw, step_dims = balanced_kl(model, q, p, args.free_nats, args.kl_dyn, args.kl_rep)
        kl_loss = kl_loss + step_kl
        kl_raws.append(step_raw); kl_dims_all.append(step_dims)
        h_states.append(h); q_states.append(q); z_states.append(z)
    recon, reward_loss, continue_loss = recon / steps, reward_loss / steps, continue_loss / steps
    tile_loss = tile_loss / steps
    word_loss, story_loss = word_loss / steps, story_loss / steps
    value_loss = torch.zeros((), device=device)
    value_parts = {}
    if values is not None:
        # states after the burn-in only: h_0 = 0 at a chunk start knows no past, later use filters --context frames
        first = args.value_burn_in
        cut = (lambda t: t.detach()) if args.value_stop_gradient else (lambda t: t)
        predicted = torch.stack([model.value(cut(h_states[t]), cut(z_states[t]), goals)
                                 for t in range(first, steps + 1)], dim=1)
        targets, mask = values[0][:, first:], values[1][:, first:]
        value_loss = ((predicted - targets).square() * mask).sum() / mask.sum().clamp_min(1)
        kept = mask > 0
        pair = torch.stack((predicted.detach()[kept], targets[kept]))
        value_parts = dict(value_loss=float(value_loss.detach()),
                           value_corr=float(torch.nan_to_num(torch.corrcoef(pair)[0, 1])) if pair.shape[1] > 1 else 0.0)
    kl_loss = kl_loss / (steps + 1)
    overshoot, terms = torch.zeros((), device=device), 0
    depth = args.overshoot_depth or args.horizon
    if depth > 1:
        for anchor in range(steps):
            h_roll, z_roll = h_states[anchor], fed(model.mode(q_states[anchor]))
            for distance in range(1, min(depth, steps - anchor) + 1):
                rs = [r[anchor + distance - 1] for r in reasoning]
                h_roll, _ = model.transition(goals, actions[:, anchor + distance - 1], rs, h_roll, z_roll, device,
                                             states=state_at(anchor + distance - 1))
                p_roll = model.prior(h_roll)
                if words is not None and distance in IMAGINED_WORDS:
                    imagined_loss = imagined_loss + word_bce(model.words("word_head", h_roll, model.sample(p_roll)),
                                                             words[:, anchor + distance], model.word_weight)
                    imagined_terms += 1
                if pixel_scale and anchor % 4 == 0 and distance in IMAGINED_PIXELS:
                    pixel_loss = pixel_loss + prior_pixel_error(model, h_roll, p_roll, images[:, anchor + distance]); pixel_terms += 1
                if distance > 1:
                    raw = model.kl_dims(detach(q_states[anchor + distance]), p_roll).sum(-1)
                    overshoot = overshoot + args.kl_dyn * raw.clamp_min(args.free_nats).mean()
                    terms += 1
                z_roll = model.mode(p_roll)
        overshoot = overshoot / max(terms, 1)
    loss = (getattr(args, "recon_scale", 1.0) * recon + kl_loss + overshoot + args.reward_loss_scale * reward_loss
            + args.continue_loss_scale * continue_loss)
    if pixel_scale:
        pixel_loss = pixel_loss / pixel_terms
        loss = loss + pixel_scale * pixel_loss
        value_parts.update(prior_pixel_loss=float(pixel_loss.detach()))
    if values is not None:
        loss = loss + args.value_loss_scale * value_loss
    if words is not None:
        imagined_loss = imagined_loss / max(imagined_terms, 1)
        loss = loss + args.word_loss_scale * (word_loss + imagined_loss)
        value_parts.update(word_loss=float(word_loss.detach()), word_imagined_loss=float(imagined_loss.detach()))
    if stories is not None:
        loss = loss + args.word_loss_scale * story_loss
        value_parts.update(story_word_loss=float(story_loss.detach()))
    if tiles is not None:
        loss = loss + args.tile_loss_scale * tile_loss
        value_parts.update(tile_loss=float(tile_loss.detach()), tile_accuracy=tile_right / steps)
    kl_raw_all = torch.cat(kl_raws)
    parts = dict(world_loss=float(loss.detach()), reconstruction=float(recon.detach()),
                 reconstruction_mse_per_pixel=float(recon.detach()) * 2 / PIXELS,
                 kl_loss=float(kl_loss.detach()), kl_raw=float(kl_raw_all.mean()),
                 overshooting_kl=float(overshoot.detach()), reward_loss=float(reward_loss.detach()),
                 continue_loss=float(continue_loss.detach()), **value_parts)
    if descriptions is not None:
        # log-likelihood of the next-state text, summed over its tokens (like the pixel term), mean over transitions
        text_nll, text_tokens = text_nll / steps, text_tokens / steps
        loss = loss + args.description_loss_scale * text_nll
        parts.update(world_loss=float(loss.detach()), description_nll=float(text_nll.detach()),
                     description_nll_per_token=float(text_nll.detach() / text_tokens))
    extra = None
    if diagnostics:
        extra = dict(kl_dims=torch.cat(kl_dims_all).mean(0), h_states=h_states, q_states=q_states, state_at=state_at)
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
        h, _ = model.transition(goals, actions[:, t], [r[t] for r in reasoning], h, z, device, states=extra["state_at"](t))
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
            part = chunks[offset:offset + args.batch_size]
            images, actions, rewards, continuation, goals, reasoning = batch_from_chunks(root, part, device)
            want = diag is None and len(goals) == args.batch_size
            _, parts, extra = loss_for_batch(model, images, actions, rewards, continuation, goals, reasoning, args,
                                             device, diagnostics=want,
                                             descriptions=next_descriptions(part) if args.descriptions else None,
                                             states=current_descriptions(part) if args.description_input else None,
                                             values=value_targets(part, device) if args.value_loss_scale else None,
                                             tiles=tile_targets(model, images) if args.tile_loss_scale else None,
                                             words=word_targets(part, "word_ids", len(args.word_vocabulary), device) if args.word_descriptions else None,
                                             stories=word_targets(part, "story_ids", len(args.story_vocabulary), device) if args.word_stories else None)
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
    parser.add_argument("--init-checkpoint", type=Path, help="initial_random of a phase-2 run (--transition slm)")
    parser.add_argument("--transition", choices=("slm", "gru"), default="slm",
                        help="gru: DreamerV3 RSSM baseline, GRU core instead of the SLM, action only")
    parser.add_argument("--cache-images", action="store_true", help="keep the frames in memory as uint8")
    parser.add_argument("--reasoning", type=Path, help="reasoning.jsonl: the action text gets its LLM reasoning")
    parser.add_argument("--descriptions", type=Path,
                        help="state_descriptions.jsonl: the SLM also writes the description of the next state")
    parser.add_argument("--description-loss-scale", type=float, default=1.0,
                        help="weight of the next-state text NLL (summed over its tokens)")
    parser.add_argument("--description-input", action="store_true",
                        help="also feed the description of the current state (\"State: ...\" before the action); "
                             "training uses the dataset text, imagination the SLM's own previous description")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--h-size", type=int, default=512)
    parser.add_argument("--z-size", type=int, default=48, help="gaussian dims, or number of categoricals")
    parser.add_argument("--z-kind", choices=("gaussian", "categorical"), default="gaussian")
    parser.add_argument("--z-classes", type=int, default=32, help="classes per categorical (--z-kind categorical)")
    parser.add_argument("--z-fixed-std", type=float,
                        help="--z-kind gaussian --transition gru: all std of q and p are this constant, the encoder cannot drown a dimension "
                             "in noise and the KL is a squared error between the means. A 32 x 32 categorical z loses zombies even with no KL")
    parser.add_argument("--image-net", choices=("small", "dreamer", "grid"), default="small",
                        help="small: 3-conv CNN + 7x7 bilinear decoder; dreamer: DreamerV3 stride-2 CNN / transposed-conv decoder")
    parser.add_argument("--cnn-depth", type=int, default=32,
                        help="--image-net dreamer: channels depth x 1/2/4/8 (DreamerV3 small 32, medium 48)")
    parser.add_argument("--mlp-units", type=int, help="hidden width of GRU input / prior / posterior / heads "
                                                      "(default h-size; DreamerV3 medium 640)")
    parser.add_argument("--mlp-layers", type=int, default=1,
                        help="hidden layers of the reward / continue heads (DreamerV3 medium 3)")
    parser.add_argument("--decoder-input", choices=("hz", "z"), default="hz", help="z: the decoder reads only z")
    parser.add_argument("--posterior-input", choices=("hx", "x"), default="hx", help="x: the posterior reads only the frame")
    parser.add_argument("--h-tokens", type=int, default=30)
    parser.add_argument("--z-tokens", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--overshoot-depth", type=int,
                        help="latent overshooting rolls the prior up to this many steps (default --horizon); "
                             "e.g. --horizon 64 --overshoot-depth 16 keeps the K=16 objective on longer chunks")
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
    parser.add_argument("--save-every", type=int, default=0,
                        help="also keep epoch_<N>.pt every N epochs (best.pt follows the validation world loss only)")
    parser.add_argument("--value-loss-scale", type=float, default=0.0,
                        help="--transition gru: > 0 adds the task value head V(h_t, z_t, g), MSE to the task return")
    parser.add_argument("--value-gamma", type=float, default=0.99)
    parser.add_argument("--value-burn-in", type=int, default=7, help="chunk states before this one get no value loss")
    parser.add_argument("--value-stop-gradient", action="store_true",
                        help="train only the head; by default the value loss also trains the RSSM, as the reward head")
    parser.add_argument("--word-descriptions", type=Path,
                        help="--transition gru: state_descriptions.jsonl; a head finds the words of each state's text, also on imagined states")
    parser.add_argument("--word-stories", type=Path,
                        help="segment_stories.jsonl (needs --word-descriptions): another head finds the words of the story of the next segment")
    parser.add_argument("--word-loss-scale", type=float, default=1.0)
    parser.add_argument("--free-encoder", action="store_true",
                        help="only the reconstruction trains the encoder: use with --kl-rep 0 --value-stop-gradient. The transition reads z detached, "
                             "encoder, posterior and decoder are clipped apart from the rest")
    parser.add_argument("--prior-skip", action="store_true", help="h carries z_t and the prior predicts the change of z, gaussian z. "
                        "--h-size then counts the GRU state and z together: GRU state = h_size - z_size. Give --mlp-units")
    parser.add_argument("--prior-recon", type=float, default=0.0,
                        help="> 0: weight of the pixel error of frames drawn from the prior mean, one step and 2, 4, 8 steps ahead. Trains the dynamics only")
    parser.add_argument("--recon-scale", type=float, default=1.0, help="weight of the reconstruction term against the KL")
    parser.add_argument("--recon-focus", type=float, default=0.0,
                        help="> 0: tiles with a larger reconstruction error weigh more, (error / mean error) ** focus")
    parser.add_argument("--tile-loss-scale", type=float, default=0.0,
                        help="--transition gru: > 0 adds the tile class head, frequency-weighted cross-entropy summed over tiles")
    parser.add_argument("--seed", type=int, default=7950)
    args = parser.parse_args()
    if args.transition == "slm" and args.init_checkpoint is None:
        parser.error("--transition slm needs --init-checkpoint")
    if args.description_input and not args.descriptions:
        parser.error("--description-input needs --descriptions")
    if args.transition == "gru" and (args.reasoning or args.descriptions):
        parser.error("--transition gru reads and writes no text: drop --reasoning / --descriptions")
    if (args.value_loss_scale or args.tile_loss_scale or args.word_descriptions) and args.transition != "gru":
        parser.error("--value-loss-scale, --tile-loss-scale and --word-descriptions need --transition gru")
    if args.word_stories and not args.word_descriptions:
        parser.error("--word-stories needs --word-descriptions")
    import torch
    from torch.utils.tensorboard import SummaryWriter
    global IMAGE_CACHE
    if args.cache_images:
        IMAGE_CACHE = {}
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(vars(args), args.init_checkpoint, device)
    lm = model.lm
    reasoning = None
    if args.reasoning:
        reasoning = {(int(r["episode_id"]), int(r["step"])): r for r in map(json.loads, args.reasoning.open())}
    descriptions = None
    if args.descriptions:
        descriptions = {(int(r["episode_id"]), int(r["step"])): r["state_description"]
                        for r in map(json.loads, args.descriptions.open())}
    episodes = load_episodes(args.raw_data, reasoning, descriptions)
    if args.value_loss_scale:
        add_value_targets(episodes, args.value_gamma)
        args.value_tasks = sorted({episode[0]["instruction"] for episode in episodes.values()})
        model.add_value(args.value_tasks)
        model.to(device)
    valid_ids ={json.loads(line)["source_episode_id"] for line in (args.split_from / "valid.jsonl").open()}
    train_episodes = {k: v for k, v in episodes.items() if k not in valid_ids}
    valid_episodes = {k: v for k, v in episodes.items() if k in valid_ids}
    if args.word_descriptions:
        text = {(int(r["episode_id"]), int(r["step"])): r["state_description"] for r in map(json.loads, args.word_descriptions.open())}
        args.word_vocabulary, share = word_vocabulary([text[k, item["step"]] for k, e in train_episodes.items() for item in e])
        index = {word: i for i, word in enumerate(args.word_vocabulary)}
        for key, episode in episodes.items():
            for item in episode:
                item["word_ids"] = [index[w] for w in words_of(text[key, item["step"]]) if w in index]
        share = torch.tensor(share, device=device).clamp(1e-4, 1 - 1e-4)
        model.word_weight = ((1 - share) / share).sqrt().clamp_max(20.0)
        model.add_words("word_head", len(args.word_vocabulary))
        if args.word_stories:
            stories = [json.loads(line) for line in args.word_stories.open()]
            story_text = {(r["episode_id"], r["first"]): r["what_happened"] + " " + " ".join(r["events"]).replace("_", " ") for r in stories}
            starts = defaultdict(list)
            for episode_id, first in sorted(story_text):
                starts[episode_id].append(first)
            args.story_vocabulary, share = word_vocabulary([v for (k, _), v in story_text.items() if k in train_episodes])
            index = {word: i for i, word in enumerate(args.story_vocabulary)}
            for key, episode in episodes.items():
                for item in episode:  # the story of the segment that holds the next step
                    following = [f for f in starts[key] if f <= item["step"] + 1][-1]
                    item["story_ids"] = [index[w] for w in words_of(story_text[key, following]) if w in index]
            share = torch.tensor(share, device=device).clamp(1e-4, 1 - 1e-4)
            model.story_weight = ((1 - share) / share).sqrt().clamp_max(20.0)
            model.add_words("story_head", len(args.story_vocabulary))
        model.to(device)
        print(json.dumps({"words": len(args.word_vocabulary), "zombie_in_vocabulary": "zombie" in args.word_vocabulary,
                          "cow_in_vocabulary": "cow" in args.word_vocabulary}), flush=True)
    train_chunks, valid_chunks = make_chunks(train_episodes, args.horizon), make_chunks(valid_episodes, args.horizon)
    if args.tile_loss_scale:
        from imagine_eval import classify_tiles, tile_prototypes
        protos, index, names = tile_prototypes()
        model.tile_tables = (torch.from_numpy(protos).to(device), torch.from_numpy(index).to(device))
        args.tile_classes = list(names)
        chooser = random.Random(args.seed)
        frames = [item["image"] for episode in train_episodes.values() for item in episode]
        counts = torch.zeros(len(names), device=device)
        sample = chooser.sample(frames, min(4096, len(frames)))
        for offset in range(0, len(sample), 256):
            batch = np.stack([read_image(args.raw_data, name) for name in sample[offset:offset + 256]])
            counts += classify_tiles(torch.from_numpy(batch).to(device), *model.tile_tables)[0].flatten().bincount(minlength=len(names))
        share = counts / counts.sum()
        model.add_tiles(args.tile_classes)
        model.to(device)
        # inverse frequency over the classes seen, capped: a tile of a very rare class weighs at most 100
        model.tile_weight = (1 / share.clamp_min(1e-4) / (share > 0).sum()).clamp_max(100.0)
        print(json.dumps({"tile_share": {n: round(float(v), 5) for n, v in zip(names, share)},
                          "tile_weight": {n: round(float(v), 2) for n, v in zip(names, model.tile_weight)}}), flush=True)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    config = dict(vars(args), architecture="slm_transition_rssm_v2" if lm is not None else "gru_transition_rssm",
                  trainable="all (SLM + support modules)" if lm is not None else "all (GRU core + support modules)",
                  reconstruction="0.5 * sum over pixels of squared error (unit-variance Gaussian)",
                  kl="kl_dyn * max(free_nats, KL(sg q || p)) + kl_rep * max(free_nats, KL(q || sg p)), z_0 included",
                  z=(f"{args.z_size} gaussian dims" if args.z_kind == "gaussian" else
                     f"{args.z_size} x {args.z_classes} categorical, 1% unimix, straight-through samples"),
                  train_episodes=len(train_episodes), validation_episodes=len(valid_episodes),
                  train_transitions=sum(len(e) for e in train_episodes.values()),
                  validation_transitions=sum(len(e) for e in valid_episodes.values()),
                  train_chunks=len(train_chunks), validation_chunks=len(valid_chunks),
                  lm_parameters=0 if lm is None else sum(p.numel() for p in lm.parameters()),
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
                    loss, micro_parts, _ = loss_for_batch(
                        model, images, actions, rewards, continuation, goals, rs, args, device,
                        descriptions=next_descriptions(part) if args.descriptions else None,
                        states=current_descriptions(part) if args.description_input else None,
                        values=value_targets(part, device) if args.value_loss_scale else None,
                        tiles=tile_targets(model, images) if args.tile_loss_scale else None,
                        words=word_targets(part, "word_ids", len(args.word_vocabulary), device) if args.word_descriptions else None,
                        stories=word_targets(part, "story_ids", len(args.story_vocabulary), device) if args.word_stories else None)
                # the losses are batch means: weight each micro-batch by its share of the batch
                (loss * len(part) / len(batch_chunks)).backward()
                for key, value in micro_parts.items():
                    parts[key] += value * len(part) / len(batch_chunks)
            if args.free_encoder:  # the KL of the dynamics is far larger: one shared clip would stop the image nets
                image_nets = {id(v) for name in ("cnn", "posterior", "pixel_decoder") for v in model.modules[name].parameters()}
                parts["grad_norm_image"] = float(torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if id(v) in image_nets], 1.0))
                parts["grad_norm"] = float(torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if id(v) not in image_nets], 1.0))
            else:
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
        if args.save_every and epoch % args.save_every == 0:
            torch.save(dict(model=model.state_dict(), config=config, metrics=metrics), args.out / f"epoch_{epoch:03d}.pt")
        if args.max_updates and updates >= args.max_updates:
            break
    save_json(args.out / "complete.json", dict(best_validation_world_loss=best, updates=updates))
    writer.close()


if __name__ == "__main__":
    main()
