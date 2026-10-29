"""Frozen-tokenizer preparation and random-init Crafter SLM training."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def prepare(data: Path):
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    if (data / "tokenization.json").exists():
        raise FileExistsError("Tokenizer/data already frozen")
    rows = {s: [json.loads(line) for line in (data / f"{s}.jsonl").open()] for s in ("train", "valid")}
    backend = Tokenizer(models.BPE(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator((r["prompt"] + r["cot_target"] for r in rows["train"]),
        trainers.BpeTrainer(vocab_size=4096, min_frequency=2,
                           initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                           special_tokens=["<pad>", "<unk>", "<eos>"], show_progress=False))
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", unk_token="<unk>", eos_token="<eos>")
    tok.save_pretrained(data / "tokenizer")
    stats = dict(vocab_size=len(tok), context=4096, max_cot_tokens=1024,
                 tokenizer_train_split_only=True, paired_rows=True, splits={})
    for split, examples in rows.items():
        encoded = []
        lengths = []
        excluded = []
        for row in examples:
            prompt = tok.encode(row["prompt"], add_special_tokens=False)
            cot = tok.encode(row["cot_target"], add_special_tokens=False) + [tok.eos_token_id]
            action = tok.encode(row["action_target"], add_special_tokens=False) + [tok.eos_token_id]
            if len(prompt) + len(cot) > 4096 or len(cot) > 1024:
                excluded.append([row["episode_seed"], row["step"], len(prompt), len(cot)])
                continue
            encoded.append(dict(prompt=prompt, action=action, cot=cot,
                                seed=row["episode_seed"], step=row["step"],
                                image_path=row.get("image_path"),
                                teacher_probs=row.get("teacher_probs")))
            lengths.append((len(prompt), len(cot), len(action)))
        torch.save(encoded, data / f"{split}_tokens.pt")
        def quantiles(column):
            values = sorted(v[column] for v in lengths)
            return dict(mean=sum(values)/len(values), p50=values[len(values)//2],
                        p95=values[int(len(values)*.95)], maximum=max(values))
        stats["splits"][split] = dict(rows=len(encoded), excluded=len(excluded),
            prompt_tokens=quantiles(0), cot_tokens=quantiles(1), action_tokens=quantiles(2),
            target_cot_tokens=sum(v[1] for v in lengths), target_action_tokens=sum(v[2] for v in lengths))
        save(data / f"{split}_excluded_lengths.json", excluded)
    stats["input_sha256"] = {name: hashlib.sha256((data/name).read_bytes()).hexdigest()
                              for name in ["train.jsonl", "valid.jsonl", "tokenizer/tokenizer.json", "train_tokens.pt", "valid_tokens.pt"]}
    save(data / "tokenization.json", stats)
    print(json.dumps(stats, indent=2), flush=True)


def build_model(tok, size):
    from transformers import GPT2Config
    from slm_ce.small_lm_architectures import matched_qwen3
    width, heads = (256, 8) if size == "5m" else (384, 6)
    config = GPT2Config(vocab_size=len(tok), n_positions=4096, n_embd=width,
                        n_layer=6, n_head=heads, eos_token_id=tok.eos_token_id,
                        bos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    model = matched_qwen3(config, target_parameters=5_000_000 if size == "5m" else 10_000_000)
    model.config._attn_implementation = "sdpa"
    return model


def batch_tensors(rows, mode, pad, device):
    import torch
    length = (max(len(r["prompt"]) + len(r[mode]) for r in rows)+7)//8*8
    ids = torch.full((len(rows), length), pad, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    for i, row in enumerate(rows):
        p, y = row["prompt"], row[mode]
        ids[i, :len(p)+len(y)] = torch.tensor(p+y)
        labels[i, len(p):len(p)+len(y)] = torch.tensor(y)
    return dict(input_ids=ids.to(device), labels=labels.to(device), attention_mask=ids.ne(pad).to(device))


def train(args):
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    tok = AutoTokenizer.from_pretrained(args.data / "tokenizer")
    train_rows = torch.load(args.data / "train_tokens.pt", weights_only=False)
    valid_rows = torch.load(args.data / "valid_tokens.pt", weights_only=False)
    model = build_model(tok, args.size).cuda()
    model.config.crafter_mode = args.mode
    model.config.crafter_history_size = 4
    model.config.crafter_max_new_tokens = 32 if args.mode == "action" else 1024
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.01)
    meta = dict(size=args.size, mode=args.mode, parameters=sum(p.numel() for p in model.parameters()),
                seed=args.seed, epochs=args.epochs, batch=32, microbatch=8, lr=2e-4,
                context=4096, train_rows=len(train_rows), valid_rows=len(valid_rows),
                initialization="random", architecture="qwen3", started_at=time.time(),
                tokenizer_sha256=hashlib.sha256((args.data/'tokenizer/tokenizer.json').read_bytes()).hexdigest())
    save(args.out / "experiment.json", meta)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(args.out / "tensorboard"))
    step = 0
    step_seconds = []
    started = time.time()
    def update(batch):
        nonlocal step
        step += 1
        for group in optimizer.param_groups:
            group["lr"] = 2e-4 * min(1., step/100)
        optimizer.zero_grad(set_to_none=True)
        denominator = sum(len(r[args.mode]) for r in batch)
        loss_sum = 0.
        for offset in range(0, len(batch), 8):
            chunk = batch[offset:offset+8]
            inputs = batch_tensors(chunk, args.mode, tok.pad_token_id, "cuda")
            weight = sum(len(r[args.mode]) for r in chunk) / denominator
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**inputs, use_cache=False).loss * weight
            loss.backward()
            loss_sum += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        return loss_sum
    for epoch in range(1, args.epochs+1):
        model.train()
        random.shuffle(train_rows)
        batches = []
        for offset in range(0, len(train_rows), 32*50):
            bucket = sorted(train_rows[offset:offset+32*50], key=lambda r: len(r["prompt"])+len(r[args.mode]))
            batches.extend(bucket[i:i+32] for i in range(0, len(bucket), 32))
        random.shuffle(batches)
        losses = []
        for batch in batches:
            step_started = time.time()
            losses.append(update(batch))
            torch.cuda.synchronize()
            step_seconds.append(time.time()-step_started)
            if step % 25 == 0:
                state = dict(epoch=epoch, step=step, loss=sum(losses[-25:])/len(losses[-25:]), seconds=time.time()-started)
                save(args.out / "progress.json", state)
                print(json.dumps(state), flush=True)
                writer.add_scalar("train/loss", state["loss"], step)
            if args.benchmark_steps and step >= args.benchmark_steps:
                torch.cuda.synchronize()
                save(args.out / "benchmark.json", dict(steps=step, seconds=time.time()-started,
                    parameters=meta["parameters"], peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    examples_per_epoch=len(train_rows), warm_step_seconds=step_seconds[5:]))
                checkpoint = args.out / "benchmark_model"
                model.save_pretrained(checkpoint)
                tok.save_pretrained(checkpoint)
                save(checkpoint / "checkpoint_complete.json", dict(epoch=0, step=step, benchmark_only=True))
                writer.close()
                return
        model.eval()
        total, tokens, correct = 0., 0, 0
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for offset in range(0, len(valid_rows), 8):
                inputs = batch_tensors(valid_rows[offset:offset+8], args.mode, tok.pad_token_id, "cuda")
                output = model(**inputs, use_cache=False)
                labels = inputs["labels"][:, 1:]
                mask = labels.ne(-100)
                count = mask.sum().item()
                total += output.loss.item()*count
                tokens += count
                correct += ((output.logits[:, :-1].argmax(-1) == labels) & mask).sum().item()
        metrics = dict(epoch=epoch, step=step, train_loss=sum(losses)/len(losses),
                       validation_loss=total/tokens, validation_token_accuracy=correct/tokens,
                       seconds=time.time()-started)
        with (args.out / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics)+"\n")
        for key in ("validation_loss", "validation_token_accuracy"):
            writer.add_scalar("valid/"+key, metrics[key], epoch)
        if epoch in (2,4,8,12) or epoch == args.epochs:
            checkpoint = args.out / f"epoch_{epoch:02d}"
            model.save_pretrained(checkpoint)
            tok.save_pretrained(checkpoint)
            save(checkpoint / "checkpoint_complete.json", metrics)
        save(args.out / "progress.json", metrics)
        print(json.dumps(metrics), flush=True)
        writer.flush()
    save(args.out / "complete.json", dict(**metrics, peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    writer.close()


def generation_benchmark(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    checkpoint = args.out / "benchmark_model"
    tok = AutoTokenizer.from_pretrained(checkpoint, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa").cuda().eval()
    rows = torch.load(args.data / "train_tokens.pt", weights_only=False)
    rows = sorted(rows, key=lambda r:len(r["prompt"]))
    selected = rows[len(rows)//2:len(rows)//2+16]
    prompts = [tok.decode(row["prompt"]) for row in selected]
    batch = tok(prompts, padding=True, return_tensors="pt", add_special_tokens=False,
                return_token_type_ids=False).to("cuda")
    results = {}
    for length in (8, 128):
        times = []
        for _ in range(3):
            torch.cuda.synchronize()
            started = time.time()
            with torch.inference_mode():
                model.generate(**batch, min_new_tokens=length, max_new_tokens=length,
                               do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
            torch.cuda.synchronize()
            times.append(time.time()-started)
        results[str(length)] = dict(batch_size=16, seconds=times, warm_seconds=sum(times[1:])/2)
    save(args.out / "generation_benchmark.json", results)
    print(json.dumps(results), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "train", "generation-benchmark"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--size", choices=["5m", "10m"], default="5m")
    parser.add_argument("--mode", choices=["action", "cot"], default="action")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7101)
    parser.add_argument("--benchmark-steps", type=int, default=0)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.data)
    else:
        if args.out is None:
            parser.error("--out is required for training")
        if args.command == "train":
            train(args)
        else:
            generation_benchmark(args)


if __name__ == "__main__":
    main()
