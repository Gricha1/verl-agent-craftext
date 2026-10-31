"""Train a decoder-only Crafter SLM with a small CNN visual soft-token prefix.

The CNN follows the three Conv(32, 5x5) + ReLU + max-pool stages used by the
CNN+BERT teacher.  It is trained from scratch here; teacher CNN weights are not
silently imported across the JAX/Flax to PyTorch boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


class VisionEncoder:
    """Three-stage teacher-shaped CNN projected to a fixed soft-token prefix."""

    def __init__(self, width: int, visual_tokens: int = 4):
        import torch.nn as nn
        self.module = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, visual_tokens * width),
        )
        self.width = width
        self.visual_tokens = visual_tokens

    def __call__(self, images):
        return self.module(images).view(images.shape[0], self.visual_tokens, self.width)

    def parameters(self):
        return self.module.parameters()

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state):
        return self.module.load_state_dict(state)

    def train(self, mode=True):
        self.module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def to(self, *args, **kwargs):
        self.module.to(*args, **kwargs)
        return self


class VisionSoftTokenCausalLM:
    """Prepend visual embeddings; labels for prefix and text prompt remain ignored."""

    def __init__(self, lm, vision: VisionEncoder):
        self.lm, self.vision = lm, vision

    def train(self):
        self.lm.train()
        self.vision.train()
        return self

    def eval(self):
        self.lm.eval()
        self.vision.eval()
        return self

    def __call__(self, *, input_ids, labels, attention_mask, images, use_cache=False):
        import torch
        image_tokens = self.vision(images)
        text_tokens = self.lm.get_input_embeddings()(input_ids)
        prefix_labels = torch.full((labels.shape[0], image_tokens.shape[1]), -100,
                                   dtype=labels.dtype, device=labels.device)
        prefix_mask = torch.ones((attention_mask.shape[0], image_tokens.shape[1]),
                                 dtype=attention_mask.dtype, device=attention_mask.device)
        return self.lm(
            inputs_embeds=torch.cat((image_tokens, text_tokens), dim=1),
            labels=torch.cat((prefix_labels, labels), dim=1),
            attention_mask=torch.cat((prefix_mask, attention_mask), dim=1),
            use_cache=use_cache,
        )


def load_images(rows, data: Path, device):
    import numpy as np
    import torch
    images = []
    for row in rows:
        relative = row.get("image_path")
        if not relative:
            raise ValueError("Vision training needs image_path in every token row; rerun prepare")
        path = data / relative
        with np.load(path) as archive:
            image = archive["observation"]
        if image.shape != (63, 63, 3):
            raise ValueError(f"Expected 63x63x3 observation, got {image.shape} at {path}")
        if image.dtype.kind not in "fiu":
            raise ValueError(f"Unsupported observation dtype {image.dtype} at {path}")
        image = image.astype("float32", copy=False)
        if image.max() > 1:
            image = image / 255.0
        if image.min() < 0 or image.max() > 1:
            raise ValueError(f"Expected image values in [0, 1] at {path}")
        images.append(np.moveaxis(image, -1, 0))
    return torch.from_numpy(np.stack(images)).to(device=device, non_blocking=True)


def batch_tensors(rows, mode, pad, device, data: Path):
    from slm_ce.crafter_slm import batch_tensors as text_batch
    inputs = text_batch(rows, mode, pad, device)
    inputs["images"] = load_images(rows, data, device)
    return inputs


# Index order of the rows' ``teacher_probs`` (scripts/prepare_offline_teacher_slm_dataset.py).
TEACHER_ACTIONS = (
    "NOOP", "LEFT", "RIGHT", "UP", "DOWN", "DO", "SLEEP", "PLACE_STONE",
    "PLACE_TABLE", "PLACE_FURNACE", "PLACE_PLANT", "MAKE_WOOD_PICKAXE",
    "MAKE_STONE_PICKAXE", "MAKE_IRON_PICKAXE", "MAKE_WOOD_SWORD",
    "MAKE_STONE_SWORD", "MAKE_IRON_SWORD",
)


def action_candidates(tok):
    """Token ids of every action target, built exactly like the rows' ``action`` field."""
    return [tok.encode(f"ACTION: {name}", add_special_tokens=False) + [tok.eos_token_id] for name in TEACHER_ACTIONS]


def reasoning_prefix(row, mode):
    """Target tokens before the action: the reasoning of a cot row, nothing for an action row."""
    if mode == "action":
        return []
    target, action = row[mode], row["action"]
    if target[-len(action):] != action:
        raise ValueError(f"cot target does not end with the action tokens (seed {row.get('seed')}, step {row.get('step')})")
    return target[:-len(action)]


def action_log_probs(model, rows, candidates, pad, device, data: Path, mode="action"):
    """Sequence log-probability of every action target for every row.

    Returns ``(seq_logp [B, A], teacher_probs [B, A])``.  Each row is scored
    against all A candidate targets in one batch of B*A sequences; the CNN runs
    once per row.  ``seq_logp`` is the sum of the target-token log-probs, i.e.
    log p(ACTION: <a><eos> | frame, prompt), so exp(seq_logp) is the student's
    probability of emitting that exact action string.  In cot mode every
    candidate follows the row's reasoning, so ``seq_logp`` is
    log p(<reasoning> | frame, prompt) + log p(ACTION: <a><eos> | frame, prompt, <reasoning>).
    """
    import torch
    from slm_ce.crafter_slm import batch_tensors as text_batch
    if any(row.get("teacher_probs") is None for row in rows):
        raise ValueError("Distillation needs teacher_probs in every token row; rerun prepare")
    count = len(candidates)
    expanded = [dict(prompt=row["prompt"], action=reasoning_prefix(row, mode) + target)
                for row in rows for target in candidates]
    inputs = text_batch(expanded, "action", pad, device)
    image_tokens = model.vision(load_images(rows, data, device)).repeat_interleave(count, dim=0)
    text_tokens = model.lm.get_input_embeddings()(inputs["input_ids"])
    prefix_mask = torch.ones(image_tokens.shape[:2], dtype=inputs["attention_mask"].dtype, device=device)
    logits = model.lm(inputs_embeds=torch.cat((image_tokens, text_tokens), dim=1),
                      attention_mask=torch.cat((prefix_mask, inputs["attention_mask"]), dim=1),
                      use_cache=False).logits
    # As in validation: the logit before text position j predicts text token j.
    labels = inputs["labels"][:, 1:]
    mask = labels.ne(-100)
    token_logp = logits[:, image_tokens.shape[1]:-1].float().log_softmax(-1)
    token_logp = token_logp.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1) * mask
    seq_logp = token_logp.sum(dim=1).view(len(rows), count)
    teacher = torch.tensor([row["teacher_probs"] for row in rows], dtype=torch.float32, device=device)
    return seq_logp, teacher


def save_checkpoint(checkpoint: Path, model: VisionSoftTokenCausalLM, tok, config: dict, metrics: dict):
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.lm.save_pretrained(checkpoint)
    tok.save_pretrained(checkpoint)
    import torch
    torch.save(model.vision.state_dict(), checkpoint / "vision_adapter.pt")
    save(checkpoint / "vision_config.json", config)
    save(checkpoint / "checkpoint_complete.json", metrics)


def train(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from slm_ce.crafter_slm import build_model

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    tok = AutoTokenizer.from_pretrained(args.data / "tokenizer")
    train_rows = torch.load(args.data / "train_tokens.pt", weights_only=False)
    valid_rows = torch.load(args.data / "valid_tokens.pt", weights_only=False)
    if any(not row.get("image_path") for row in train_rows + valid_rows):
        raise ValueError("Dataset has no copied images. Recreate it with prepare_offline_teacher_slm_dataset.py")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but no CUDA device is visible")
    device = torch.device(args.device)
    if args.init_checkpoint is None:
        lm = build_model(tok, args.size).to(device)
        width = lm.config.hidden_size
        vision = VisionEncoder(width=width, visual_tokens=args.visual_tokens).to(device)
        initialization = "random"
    else:
        lm = AutoModelForCausalLM.from_pretrained(args.init_checkpoint).to(device)
        checkpoint_vision = json.loads((args.init_checkpoint / "vision_config.json").read_text(encoding="utf-8"))
        if int(checkpoint_vision["visual_tokens"]) != args.visual_tokens:
            raise ValueError("--visual-tokens must match --init-checkpoint")
        width = lm.config.hidden_size
        if int(checkpoint_vision["hidden_size"]) != width:
            raise ValueError("language and vision hidden sizes disagree in --init-checkpoint")
        vision = VisionEncoder(width=width, visual_tokens=args.visual_tokens).to(device)
        vision.load_state_dict(torch.load(args.init_checkpoint / "vision_adapter.pt", map_location=device,
                                          weights_only=True))
        initialization = f"checkpoint:{args.init_checkpoint.resolve()}"
    model = VisionSoftTokenCausalLM(lm, vision)
    vision_config = dict(architecture="cnn_3x_conv32_k5_relu_maxpool__linear_to_soft_tokens",
                         input_shape=[63, 63, 3], visual_tokens=args.visual_tokens,
                         hidden_size=width, trained_jointly=True,
                         teacher_weights_reused=False)
    if args.initial_checkpoint_out is not None:
        # This is deliberately written before constructing the optimizer or
        # consuming a training batch.  It lets offline CE and online RL start
        # from byte-identical random decoder and CNN weights.
        save_checkpoint(args.initial_checkpoint_out, model, tok, vision_config,
                        dict(epoch=0, step=0, initialization="random", seed=args.seed))
    optimizer = torch.optim.AdamW(list(lm.parameters()) + list(vision.parameters()), lr=2e-4, weight_decay=.01)
    meta = dict(size=args.size, mode=args.mode, target=args.target, lm_parameters=sum(p.numel() for p in lm.parameters()),
                vision_parameters=sum(p.numel() for p in vision.parameters()), seed=args.seed,
                epochs=args.epochs, batch=32, microbatch=8, lr=2e-4, context=4096,
                train_rows=len(train_rows), valid_rows=len(valid_rows), initialization=initialization,
                language_architecture="decoder-only-qwen3", vision=vision_config, started_at=time.time(),
                tokenizer_sha256=hashlib.sha256((args.data / "tokenizer/tokenizer.json").read_bytes()).hexdigest())
    save(args.out / "experiment.json", meta)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(args.out / "tensorboard"))
    step, started, best = 0, time.time(), float("inf")
    candidates = action_candidates(tok)
    candidate_lengths = torch.tensor([len(c) for c in candidates], dtype=torch.float32, device=device)
    if args.target == "teacher_distribution" and any(r.get("teacher_probs") is None for r in train_rows + valid_rows):
        raise ValueError("--target teacher_distribution needs teacher_probs in the token rows; rerun prepare")

    def distillation_loss_terms(rows):
        """-sum_a pi(a) log p(a) and its normaliser sum_a pi(a) len(a), summed over rows.

        Dividing the first by the second is the expected per-token CE of a target
        drawn from the teacher policy; with a one-hot teacher it is exactly the
        argmax CE above, so the two objectives share one scale.  In cot mode a
        target is the row's reasoning followed by the action, so this is CE on
        the reasoning tokens plus the distillation term on the action given the
        reasoning (sum_a pi(a) = 1), normalised by the expected target length.
        """
        seq_logp, teacher = action_log_probs(model, rows, candidates, tok.pad_token_id, device, args.data, args.mode)
        prefix = torch.tensor([len(reasoning_prefix(row, args.mode)) for row in rows], dtype=torch.float32,
                              device=device)
        return (-(teacher * seq_logp).sum(), (teacher * (candidate_lengths + prefix[:, None])).sum(),
                seq_logp, teacher)

    def update(batch):
        nonlocal step
        step += 1
        for group in optimizer.param_groups:
            group["lr"] = 2e-4 * min(1., step / 100)
        optimizer.zero_grad(set_to_none=True)
        if args.target == "teacher_distribution":
            denominator = sum(sum(p * (len(reasoning_prefix(row, args.mode)) + len(c))
                                  for p, c in zip(row["teacher_probs"], candidates)) for row in batch)
        else:
            denominator = sum(len(row[args.mode]) for row in batch)
        loss_sum = 0.0
        for offset in range(0, len(batch), 8):
            chunk = batch[offset:offset + 8]
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                if args.target == "teacher_distribution":
                    loss = distillation_loss_terms(chunk)[0] / denominator
                else:
                    inputs = batch_tensors(chunk, args.mode, tok.pad_token_id, device, args.data)
                    weight = sum(len(row[args.mode]) for row in chunk) / denominator
                    loss = model(**inputs, use_cache=False).loss * weight
            loss.backward()
            loss_sum += loss.item()
        torch.nn.utils.clip_grad_norm_(list(lm.parameters()) + list(vision.parameters()), 1.0)
        optimizer.step()
        return loss_sum

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_rows)
        batches = []
        for offset in range(0, len(train_rows), 32 * 50):
            bucket = sorted(train_rows[offset:offset + 32 * 50], key=lambda r: len(r["prompt"]) + len(r[args.mode]))
            batches.extend(bucket[i:i + 32] for i in range(0, len(bucket), 32))
        random.shuffle(batches)
        losses = [update(batch) for batch in batches]
        model.eval()
        total, tokens, correct = 0.0, 0, 0
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            for offset in range(0, len(valid_rows), 8):
                inputs = batch_tensors(valid_rows[offset:offset + 8], args.mode, tok.pad_token_id, device, args.data)
                output = model(**inputs, use_cache=False)
                labels = inputs["labels"][:, 1:]
                mask = labels.ne(-100)
                count = mask.sum().item()
                total += output.loss.item() * count
                tokens += count
                # Logits before the text begin with the visual prefix.  The
                # first text logit after that prefix predicts text token #1.
                predicted = output.logits[:, args.visual_tokens:-1].argmax(-1)
                correct += ((predicted == labels) & mask).sum().item()
        metrics = dict(epoch=epoch, step=step, target=args.target, train_loss=sum(losses) / len(losses),
                       validation_loss=total / tokens, validation_token_accuracy=correct / tokens)
        if valid_rows[0].get("teacher_probs") is not None:
            # Policy-level validation over the 17 action strings, for either target.
            nll, norm, hits, kl, h_student, h_teacher = 0.0, 0.0, 0, 0.0, 0.0, 0.0
            with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                for offset in range(0, len(valid_rows), 8):
                    a, b, seq_logp, teacher = distillation_loss_terms(valid_rows[offset:offset + 8])
                    nll, norm = nll + a.item(), norm + b.item()
                    student = seq_logp.log_softmax(-1)  # renormalised over the 17 valid actions
                    hits += (seq_logp.argmax(-1) == teacher.argmax(-1)).sum().item()
                    t_log = teacher.clamp_min(1e-12).log()
                    kl += (teacher * (t_log - student)).sum().item()
                    h_student -= (student.exp() * student).sum().item()
                    h_teacher -= (teacher * t_log).sum().item()
            n = len(valid_rows)
            metrics.update(validation_distill_loss=nll / norm, validation_action_accuracy=hits / n,
                           validation_kl_teacher_student=kl / n, validation_student_entropy=h_student / n,
                           validation_teacher_entropy=h_teacher / n)
        metrics["seconds"] = time.time() - started
        # Model selection follows the trained objective.
        selection = metrics["validation_distill_loss"] if args.target == "teacher_distribution" else metrics["validation_loss"]
        with (args.out / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        writer.add_scalar("train/loss", metrics["train_loss"], step)
        writer.add_scalar("valid/validation_loss", metrics["validation_loss"], epoch)
        writer.add_scalar("valid/validation_token_accuracy", metrics["validation_token_accuracy"], epoch)
        if selection < best:
            best = selection
            save_checkpoint(args.out / "best_validation", model, tok, vision_config, metrics)
        if epoch in (2, 4, 8, 12) or epoch == args.epochs:
            save_checkpoint(args.out / f"epoch_{epoch:02d}", model, tok, vision_config, metrics)
        save(args.out / "progress.json", metrics)
        print(json.dumps(metrics), flush=True)
        writer.flush()
    save(args.out / "complete.json", dict(**metrics, best_validation_loss=best,
                                           peak_reserved_bytes=torch.cuda.max_memory_reserved() if device.type == "cuda" else 0))
    writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", choices=["5m", "10m"], default="5m")
    parser.add_argument("--mode", choices=["action", "cot"], default="action")
    parser.add_argument("--target", choices=["argmax", "teacher_distribution"], default="argmax",
                        help="argmax: CE on the executed teacher action; teacher_distribution: expected CE "
                             "under the teacher's full action distribution (policy distillation)")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7101)
    parser.add_argument("--visual-tokens", type=int, default=4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--init-checkpoint", type=Path,
                        help="initialize both decoder and vision adapter from a completed vision-SLM checkpoint")
    parser.add_argument("--initial-checkpoint-out", type=Path,
                        help="write the untrained initialized decoder and vision adapter before CE updates")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
