#!/usr/bin/env python3
"""Offline deterministic RSSM-like latent world model for CrafText trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-commit", default=None,
                        help="Commit of the copied source when the host checkout is intentionally unchanged.")
    parser.add_argument("--smoke-batches", type=int, default=0,
                        help="Run this many optimizer batches and exit after validation.")
    return parser.parse_args()


def episode_split(dataset_dir: str, seed: int):
    rows = [json.loads(line) for line in (Path(dataset_dir) / "transitions.jsonl").open(encoding="utf-8")]
    episode_ids = sorted({int(row["episode_id"]) for row in rows})
    random.Random(seed).shuffle(episode_ids)
    n = len(episode_ids)
    train_ids = set(episode_ids[:int(0.8 * n)])
    val_ids = set(episode_ids[int(0.8 * n):int(0.9 * n)])
    return (
        [row for row in rows if int(row["episode_id"]) in train_ids],
        [row for row in rows if int(row["episode_id"]) in val_ids],
        [row for row in rows if int(row["episode_id"]) not in train_ids | val_ids],
    )


def sequence_chunks(rows, sequence_length: int, stride: int):
    """Make contiguous, within-episode (o_t, a_t, ..., o_{t+T}) chunks."""
    by_episode = defaultdict(list)
    for row in rows:
        by_episode[int(row["episode_id"])].append(row)
    chunks = []
    for episode_rows in by_episode.values():
        episode_rows.sort(key=lambda row: int(row["step"]))
        for start in range(0, len(episode_rows) - sequence_length + 1, stride):
            chunk = episode_rows[start:start + sequence_length]
            steps = [int(row["step"]) for row in chunk]
            if steps != list(range(steps[0], steps[0] + sequence_length)):
                raise ValueError(f"Non-contiguous rows in episode {chunk[0]['episode_id']}: {steps}")
            chunks.append(chunk)
    return chunks


def observation_text(tokenizer, task: str, observation: str):
    prompt = f"Task:\n{task}\n\nCurrent observation:\n{observation}\n\nObservation representation:"
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                         add_generation_prompt=True)


def final_hidden(model, tokenizer, texts, device, max_length):
    batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_length, add_special_tokens=False).to(device)
    with torch.no_grad():
        hidden = model(**batch, output_hidden_states=True, use_cache=False).hidden_states[-1]
    if not batch["attention_mask"][:, -1].bool().all():
        raise RuntimeError("Target encoder expects left padding and a real final token.")
    return hidden[:, -1].float().cpu()


def embedding_cache(rows, model, tokenizer, projection_cpu, device, cfg):
    """Cache frozen E(task, observation) once; this never changes during training."""
    texts = {}
    for row in rows:
        texts.setdefault((str(row["task"]), str(row["observation_t"])), None)
        texts.setdefault((str(row["task"]), str(row["observation_t_plus_1"])), None)
    keys = list(texts)
    batch_size = int(cfg["target_encode_batch_size"])
    for start in range(0, len(keys), batch_size):
        batch_keys = keys[start:start + batch_size]
        hidden = final_hidden(model, tokenizer,
                              [observation_text(tokenizer, task, obs) for task, obs in batch_keys],
                              device, int(cfg["max_sequence_length"]))
        encoded = hidden @ projection_cpu
        for key, value in zip(batch_keys, encoded):
            texts[key] = value.contiguous()
        if (start // batch_size + 1) % 50 == 0 or start + batch_size >= len(keys):
            print(json.dumps({"target_embeddings": min(start + batch_size, len(keys)), "unique_observations": len(keys)}), flush=True)
    return texts


class ChunkDataset(Dataset):
    def __init__(self, chunks, cache):
        self.chunks, self.cache = chunks, cache

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        rows = self.chunks[index]
        task = str(rows[0]["task"])
        observations = [self.cache[(task, str(rows[0]["observation_t"]))]]
        observations.extend(self.cache[(task, str(row["observation_t_plus_1"]))] for row in rows)
        action_ids = [ACTIONS.index(str(row["action_t"])) for row in rows]
        changed = [str(row["observation_t"]) != str(row["observation_t_plus_1"]) for row in rows]
        return {
            "embeddings": torch.stack(observations),  # [T+1, 256]
            "actions": torch.tensor(action_ids, dtype=torch.long),  # [T]
            "changed": torch.tensor(changed, dtype=torch.bool),
        }


class RecurrentWorldModel(nn.Module):
    def __init__(self, latent_dim: int, action_embedding_dim: int, transition_hidden_dim: int):
        super().__init__()
        self.action_embedding = nn.Embedding(len(ACTIONS), action_embedding_dim)
        self.transition = nn.Sequential(
            nn.Linear(latent_dim + action_embedding_dim, transition_hidden_dim), nn.GELU(),
            nn.Linear(transition_hidden_dim, latent_dim),
        )
        self.corrector = nn.GRUCell(latent_dim, latent_dim)

    def prior(self, state, actions):
        return self.transition(torch.cat((state, self.action_embedding(actions)), dim=-1))

    def initial_state(self, target):
        return self.corrector(target, torch.zeros_like(target))


def multi_positive_infonce(prediction, target, temperature):
    prediction = F.normalize(prediction.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    logits = prediction @ target.T / temperature
    same = torch.isclose(target @ target.T, torch.ones_like(logits), atol=1e-6)
    loss = (torch.logsumexp(logits, dim=1) -
            torch.logsumexp(logits.masked_fill(~same, float("-inf")), dim=1)).mean()
    return loss, logits, same


def retrieval_metrics(prediction, target, temperature, mask=None):
    if mask is not None:
        prediction, target = prediction[mask], target[mask]
    if len(prediction) == 0:
        return {"retrieval_top1": float("nan"), "retrieval_top5": float("nan"), "cosine": float("nan")}
    _, logits, same = multi_positive_infonce(prediction, target, temperature)
    ranking = logits.argsort(descending=True, dim=1)
    normalized_prediction = F.normalize(prediction.float(), dim=-1)
    normalized_target = F.normalize(target.float(), dim=-1)
    negative = normalized_prediction @ normalized_target.T
    return {
        "retrieval_top1": float(same.gather(1, ranking[:, :1]).any(1).float().mean()),
        "retrieval_top5": float(same.gather(1, ranking[:, :min(5, len(target))]).any(1).float().mean()),
        "cosine": float(F.cosine_similarity(normalized_prediction, normalized_target).mean()),
        "negative_cosine": float(negative[~torch.eye(len(target), device=target.device, dtype=torch.bool)].mean()) if len(target) > 1 else float("nan"),
    }


def run_teacher_forced(model, embeddings, actions):
    state = model.initial_state(embeddings[:, 0])
    priors, states = [], []
    for t in range(actions.shape[1]):
        prior = model.prior(state, actions[:, t])
        state = model.corrector(embeddings[:, t + 1], prior)
        priors.append(prior)
        states.append(state)
    return torch.stack(priors, dim=1), torch.stack(states, dim=1)


@torch.no_grad()
def validate(model, loader, device, cfg):
    model.eval()
    all_prior, all_target, all_changed, all_state = [], [], [], []
    rollout = {horizon: [[], []] for horizon in (1, 2, 4, 8, 16)}
    action_cos, action_l2 = [], []
    for batch in loader:
        embeddings = batch["embeddings"].to(device)
        actions = batch["actions"].to(device)
        priors, states = run_teacher_forced(model, embeddings, actions)
        all_prior.append(priors.flatten(0, 1))
        all_target.append(embeddings[:, 1:].flatten(0, 1))
        all_changed.append(batch["changed"].to(device).flatten())
        all_state.append(states.flatten(0, 1))
        state = model.initial_state(embeddings[:, 0])
        for horizon in range(1, actions.shape[1] + 1):
            state = model.prior(state, actions[:, horizon - 1])
            if horizon in rollout:
                rollout[horizon][0].append(state)
                rollout[horizon][1].append(embeddings[:, horizon])
        action_ids = torch.arange(len(ACTIONS), device=device).unsqueeze(0).expand(len(state), -1)
        counterfactual = model.prior(state.unsqueeze(1).expand(-1, len(ACTIONS), -1).reshape(-1, state.shape[-1]),
                                     action_ids.reshape(-1)).reshape(-1, len(ACTIONS), state.shape[-1])
        for left in range(len(ACTIONS)):
            for right in range(left + 1, len(ACTIONS)):
                action_cos.append(F.cosine_similarity(counterfactual[:, left], counterfactual[:, right], dim=-1))
                action_l2.append((counterfactual[:, left] - counterfactual[:, right]).norm(dim=-1))
    prior = torch.cat(all_prior)
    target = torch.cat(all_target)
    changed = torch.cat(all_changed)
    state = torch.cat(all_state)
    infonce, _, _ = multi_positive_infonce(prior, target, float(cfg["infonce_temperature"]))
    overall = retrieval_metrics(prior, target, float(cfg["infonce_temperature"]))
    metrics = {
        "val/infonce": float(infonce),
        "val/retrieval_top1": overall["retrieval_top1"],
        "val/retrieval_top5": overall["retrieval_top5"],
        "val/positive_cosine": overall["cosine"],
        "val/negative_cosine": overall["negative_cosine"],
        "latent/prior_norm": float(prior.norm(dim=-1).mean()),
        "latent/prior_std": float(prior.std()),
        "latent/state_norm": float(state.norm(dim=-1).mean()),
        "latent/state_std": float(state.std()),
        "latent/target_norm": float(target.norm(dim=-1).mean()),
        "latent/target_std": float(target.std()),
        "action_sensitivity/mean_pairwise_cosine": float(torch.cat(action_cos).mean()),
        "action_sensitivity/mean_pairwise_l2": float(torch.cat(action_l2).mean()),
    }
    for label, mask in (("changed", changed), ("unchanged", ~changed)):
        grouped = retrieval_metrics(prior, target, float(cfg["infonce_temperature"]), mask)
        metrics.update({f"{label}/{key}": value for key, value in grouped.items() if key != "negative_cosine"})
    for horizon, (predictions, targets) in rollout.items():
        values = retrieval_metrics(torch.cat(predictions), torch.cat(targets), float(cfg["infonce_temperature"]))
        metrics.update({f"rollout_h{horizon}/{key}": value for key, value in values.items() if key != "negative_cosine"})
    return metrics


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(cfg["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    root = Path(cfg["output_root"]) / args.run_name
    root.mkdir(parents=True, exist_ok=False)
    train_rows, val_rows, test_rows = episode_split(cfg["dataset_dir"], int(cfg["split_seed"]))
    all_rows = train_rows + val_rows + test_rows
    train_chunks = sequence_chunks(train_rows, int(cfg["sequence_length"]), int(cfg["chunk_stride"]))
    val_chunks = sequence_chunks(val_rows, int(cfg["sequence_length"]), int(cfg["chunk_stride"]))
    test_chunks = sequence_chunks(test_rows, int(cfg["sequence_length"]), int(cfg["chunk_stride"]))
    if not train_chunks or not val_chunks or not test_chunks:
        raise RuntimeError("One split has no full sequence chunks.")
    commit = args.source_commit or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    resolved = {**cfg, "run_name": args.run_name, "git_commit": commit}
    (root / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8")
    split_info = {"train_transitions": len(train_rows), "val_transitions": len(val_rows), "test_transitions": len(test_rows),
                  "train_sequences": len(train_chunks), "val_sequences": len(val_chunks), "test_sequences": len(test_chunks)}
    (root / "splits.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    tokenizer.padding_side = "left"; tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    target_encoder = AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype=torch.bfloat16,
                                                           trust_remote_code=True).to(device).eval()
    for parameter in target_encoder.parameters(): parameter.requires_grad_(False)
    generator = torch.Generator(device="cpu").manual_seed(int(cfg["target_projection_seed"]))
    projection = torch.randn(target_encoder.config.hidden_size, int(cfg["latent_dim"]), generator=generator,
                             dtype=torch.float32) / math.sqrt(target_encoder.config.hidden_size)
    torch.save(projection, root / "fixed_target_projection.pt")
    # A smoke test deliberately encodes only the batches it will consume; a full
    # run caches every split once, including the held-out test observations.
    if args.smoke_batches:
        smoke_sequences = max(int(cfg["train_batch_size"]), args.smoke_batches)
        cache_rows = [row for chunk in train_chunks[:smoke_sequences] + val_chunks[:int(cfg["train_batch_size"])] for row in chunk]
    else:
        cache_rows = all_rows
    cache = embedding_cache(cache_rows, target_encoder, tokenizer, projection, device, cfg)
    torch.save(cache, root / "frozen_target_embeddings.pt")
    del target_encoder
    torch.cuda.empty_cache()

    active_train_chunks = train_chunks[:max(int(cfg["train_batch_size"]), args.smoke_batches)] if args.smoke_batches else train_chunks
    active_val_chunks = val_chunks[:int(cfg["train_batch_size"])] if args.smoke_batches else val_chunks
    train_loader = DataLoader(ChunkDataset(active_train_chunks, cache), batch_size=int(cfg["train_batch_size"]), shuffle=True)
    val_loader = DataLoader(ChunkDataset(active_val_chunks, cache), batch_size=int(cfg["train_batch_size"]), shuffle=False)
    test_loader = DataLoader(ChunkDataset(test_chunks, cache), batch_size=int(cfg["train_batch_size"]), shuffle=False)
    model = RecurrentWorldModel(int(cfg["latent_dim"]), int(cfg["action_embedding_dim"]), int(cfg["transition_hidden_dim"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    counts = {"trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
              "effective_batch_sequences": int(cfg["train_batch_size"]), "optimizer_steps_per_epoch": len(train_loader)}
    (root / "parameter_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    try:
        from comet_ml import Experiment
        comet = Experiment(workspace=cfg["comet_workspace"], project_name=cfg["comet_project"], auto_output_logging="simple")
        comet.set_name(args.run_name)
        comet.add_tags(["latent_wm", "recurrent", "rssm_like", "offline", "infonce", "no_lora", "no_ppo"])
        comet.log_parameters({**resolved, **split_info, **counts,
                              "dataset_hash": hashlib.sha256((Path(cfg["dataset_dir"]) / "transitions.jsonl").read_bytes()).hexdigest()})
    except Exception as error:
        comet = None
        print(f"[comet disabled] {error}", flush=True)

    def save(name):
        directory = root / name
        directory.mkdir(exist_ok=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": resolved}, directory / "state.pt")

    best = -float("inf")
    global_step = 0
    for epoch in range(int(cfg["epochs"])):
        model.train()
        for batch_index, batch in enumerate(train_loader):
            embeddings = batch["embeddings"].to(device)
            actions = batch["actions"].to(device)
            priors, corrected = run_teacher_forced(model, embeddings, actions)
            loss, _, _ = multi_positive_infonce(priors.flatten(0, 1), embeddings[:, 1:].flatten(0, 1), float(cfg["infonce_temperature"]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["max_grad_norm"])))
            optimizer.step(); global_step += 1
            metrics = {"train/loss": float(loss.detach()), "train/infonce": float(loss.detach()), "train/grad_norm": grad_norm,
                       "latent/prior_norm": float(priors.detach().norm(dim=-1).mean()), "latent/prior_std": float(priors.detach().std()),
                       "latent/state_norm": float(corrected.detach().norm(dim=-1).mean()), "latent/state_std": float(corrected.detach().std()),
                       "latent/target_norm": float(embeddings[:, 1:].norm(dim=-1).mean()), "latent/target_std": float(embeddings[:, 1:].std())}
            if global_step == 1:
                shapes = {"e_t": list(embeddings[:, 0].shape), "z_t": list(corrected[:, 0].shape),
                          "action_embedding": list(model.action_embedding(actions[:, 0]).shape), "prior_t_plus_1": list(priors[:, 0].shape),
                          "corrected_z_t_plus_1": list(corrected[:, 0].shape)}
                print(json.dumps({"shapes": shapes}), flush=True)
                (root / "smoke_shapes.json").write_text(json.dumps(shapes, indent=2), encoding="utf-8")
            print(json.dumps({"step": global_step, **metrics}), flush=True)
            if comet: comet.log_metrics(metrics, step=global_step)
            if args.smoke_batches and global_step >= args.smoke_batches:
                validation = validate(model, val_loader, device, cfg)
                print(json.dumps({"smoke": True, "step": global_step, **validation}), flush=True)
                if comet: comet.log_metrics(validation, step=global_step); comet.end()
                save("smoke")
                return
        validation = validate(model, val_loader, device, cfg)
        print(json.dumps({"epoch": epoch, "step": global_step, **validation}), flush=True)
        if comet: comet.log_metrics(validation, step=global_step, epoch=epoch)
        if validation["val/retrieval_top1"] > best:
            best = validation["val/retrieval_top1"]
            save("best")
    test = validate(model, test_loader, device, cfg)
    print(json.dumps({"final_step": global_step, **test}), flush=True)
    if comet: comet.log_metrics({"test/" + key.split("/", 1)[-1]: value for key, value in test.items()}, step=global_step); comet.end()
    save("final")


if __name__ == "__main__":
    main()
