#!/usr/bin/env python3
"""Deterministic observation autoencoder through five soft tokens and frozen Qwen.

This is deliberately not an RSSM: samples are independent observations and the
only trainable modules are encoder_head and decoder_state_projector.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import random
import subprocess
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

import train_rssm_qwen_soft_tokens as data


ENCODER_INSTRUCTION = "Encode the following environment observation:\n"
DECODER_INSTRUCTION = "Reconstruct the current environment observation exactly.\n"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke-steps', type=int, default=0)
    parser.add_argument(
        '--source-commit',
        default=None,
        help='Commit identifier to record when the launch host has not checked out this source.',
    )
    return parser.parse_args()


def target_row(row):
    return str(row['task']), str(row['observation_t_plus_1'])


def encoder_prompt(task, observation):
    return f'{ENCODER_INSTRUCTION}Task:\n{task}\n\n{observation}'


def normalise(text):
    return ' '.join(text.strip().split())


class LatentAutoencoder(nn.Module):
    def __init__(self, cfg, d_model):
        super().__init__()
        z_dim, k = int(cfg['latent_z_dim']), int(cfg['soft_tokens'])
        self.encoder_head = nn.Sequential(nn.Linear(d_model, z_dim), nn.GELU(), nn.Linear(z_dim, z_dim))
        self.decoder_state_projector = nn.Sequential(
            nn.Linear(z_dim, z_dim), nn.GELU(), nn.Linear(z_dim, k * d_model)
        )
        self.k, self.d_model = k, d_model
        self.register_buffer('soft_token_target_rms', torch.tensor(float(cfg['soft_token_target_rms'])))
        with torch.no_grad():
            probe = torch.randn(64, z_dim, generator=torch.Generator().manual_seed(0))
            raw_rms = self.decoder_state_projector(probe).float().square().mean().sqrt()
        self.register_buffer('decoder_soft_token_gain', self.soft_token_target_rms / raw_rms.clamp_min(1e-8))

    def soft_tokens(self, z):
        tokens = self.decoder_state_projector(z).reshape(len(z), self.k, self.d_model)
        return tokens * self.decoder_soft_token_gain.to(tokens.dtype)


@torch.no_grad()
def encode_observations(qwen, tokenizer, model, tasks, observations, device, max_length):
    texts = [encoder_prompt(task, observation) for task, observation in zip(tasks, observations)]
    batch = tokenizer(texts, return_tensors='pt', padding=True, truncation=True,
                      max_length=max_length, add_special_tokens=False).to(device)
    hidden = qwen(**batch, output_hidden_states=True, use_cache=False).hidden_states[-1]
    if not batch['attention_mask'][:, -1].bool().all():
        raise RuntimeError('left-padded encoder invariant failed')
    return model.encoder_head(hidden[:, -1].float())


def reconstruction(qwen, tokenizer, model, observations, z, device, max_length):
    """Teacher-forced CE and token accuracy. Target observations never condition prefix."""
    embedding = qwen.get_input_embeddings()
    prefix_ids = tokenizer(DECODER_INSTRUCTION, add_special_tokens=False)['input_ids']
    all_losses, all_correct, all_tokens, shapes = [], [], [], None
    for observation, latent in zip(observations, z):
        target_ids = tokenizer(observation, add_special_tokens=False, truncation=True,
                               max_length=max_length - len(prefix_ids) - model.k)['input_ids']
        if not target_ids:
            target_ids = [tokenizer.eos_token_id]
        prefix = embedding(torch.tensor(prefix_ids, device=device))
        soft = model.soft_tokens(latent[None])[0].to(embedding.weight.dtype)
        target = embedding(torch.tensor(target_ids, device=device))
        inputs = torch.cat((prefix, soft, target), dim=0)[None]
        labels = torch.tensor([-100] * (len(prefix_ids) + model.k) + target_ids, device=device)[None]
        result = qwen(inputs_embeds=inputs, labels=labels, use_cache=False)
        predictions = result.logits[:, :-1].argmax(dim=-1)
        valid = labels[:, 1:] != -100
        all_losses.append(result.loss)
        all_correct.append(((predictions == labels[:, 1:]) & valid).sum())
        all_tokens.append(valid.sum())
        shapes = {'decoder_soft_tokens.shape': list(soft[None].shape), 'logits.shape': list(result.logits.shape)}
    return torch.stack(all_losses).mean(), torch.stack(all_correct).sum(), torch.stack(all_tokens).sum(), shapes


def evaluate_teacher_forced(qwen, tokenizer, model, rows, device, cfg):
    model.eval()
    rows = rows[:int(cfg['validation_examples'])]
    tasks, observations = zip(*(target_row(row) for row in rows))
    with torch.no_grad():
        z = encode_observations(qwen, tokenizer, model, tasks, observations, device, int(cfg['max_sequence_length']))
        shuffle_generator = torch.Generator(device=device).manual_seed(int(cfg['validation_shuffle_seed']))
        shuffled = z[torch.randperm(len(z), generator=shuffle_generator, device=device)]
        zero = torch.zeros_like(z)
    metrics = {}
    for name, latent in [('correct', z), ('shuffled', shuffled), ('zero', zero)]:
        loss, correct, tokens, _ = reconstruction(qwen, tokenizer, model, observations, latent, device, int(cfg['max_sequence_length']))
        ce = float(loss.detach())
        metrics[f'val/recon_ce_{name}'] = ce
        metrics[f'val/recon_ppl_{name}'] = float(math.exp(min(ce, 20.0)))
        metrics[f'val/token_accuracy_{name}'] = float(correct.float() / tokens.clamp_min(1))
    metrics['val/delta_ce_shuffled'] = metrics['val/recon_ce_shuffled'] - metrics['val/recon_ce_correct']
    metrics['val/delta_ce_zero'] = metrics['val/recon_ce_zero'] - metrics['val/recon_ce_correct']
    return metrics


@torch.no_grad()
def generate_examples(qwen, tokenizer, model, rows, device, cfg):
    model.eval()
    rows = rows[:int(cfg['generation_examples'])]
    tasks, targets = zip(*(target_row(row) for row in rows))
    z = encode_observations(qwen, tokenizer, model, tasks, targets, device, int(cfg['max_sequence_length']))
    embedding = qwen.get_input_embeddings()
    instruction = embedding(torch.tensor(tokenizer(DECODER_INSTRUCTION, add_special_tokens=False)['input_ids'], device=device))
    examples, exact, token_match, edit_similarity = [], [], [], []
    for target, latent in zip(targets, z):
        initial = torch.cat((instruction, model.soft_tokens(latent[None])[0].to(embedding.weight.dtype)), dim=0)[None]
        mask = torch.ones(initial.shape[:2], dtype=torch.long, device=device)
        generated_ids = qwen.generate(inputs_embeds=initial, attention_mask=mask, do_sample=False,
                                      max_new_tokens=int(cfg['generation_max_new_tokens']), pad_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        target_ids = tokenizer(target, add_special_tokens=False)['input_ids']
        output_ids = tokenizer(generated, add_special_tokens=False)['input_ids']
        length = max(len(target_ids), len(output_ids), 1)
        matches = sum(a == b for a, b in zip(target_ids, output_ids)) / length
        exact.append(normalise(generated) == normalise(target))
        token_match.append(matches)
        edit_similarity.append(difflib.SequenceMatcher(None, normalise(target), normalise(generated)).ratio())
        examples.append({'target': target, 'generated': generated})
    return {
        'val/gen_exact_match': float(sum(exact) / len(exact)),
        'val/gen_token_match': float(sum(token_match) / len(token_match)),
        'val/gen_normalized_edit_similarity': float(sum(edit_similarity) / len(edit_similarity)),
    }, examples


def grad_norm(module):
    squared = [parameter.grad.detach().float().square().sum() for parameter in module.parameters() if parameter.grad is not None]
    return float(torch.sqrt(sum(squared, torch.zeros((), device=next(module.parameters()).device))))


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg['base_model'], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = 'left'
    qwen = AutoModelForCausalLM.from_pretrained(cfg['base_model'], torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True).to(device).eval()
    for parameter in qwen.parameters():
        parameter.requires_grad_(False)
    cfg['soft_token_target_rms'] = float(qwen.get_input_embeddings().weight.detach().float().square().mean().sqrt())
    train_rows, val_rows, test_rows = data.split(cfg['dataset_dir'], cfg['split_seed'])
    root = Path(cfg['output_root']) / args.run_name
    root.mkdir(parents=True, exist_ok=False)
    commit = args.source_commit or subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True
    ).strip()
    model = LatentAutoencoder(cfg, qwen.config.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['learning_rate']), weight_decay=float(cfg['weight_decay']))
    info = {'train_transitions': len(train_rows), 'val_transitions': len(val_rows), 'test_transitions': len(test_rows),
            'effective_optimizer_batch': int(cfg['train_batch_size']), 'git_commit': commit,
            'dataset_hash': hashlib.sha256((Path(cfg['dataset_dir']) / 'transitions.jsonl').read_bytes()).hexdigest()}
    (root / 'resolved_config.yaml').write_text(yaml.safe_dump({**cfg, **info}, sort_keys=True))
    (root / 'splits.json').write_text(json.dumps(info, indent=2))
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(json.dumps({'trainable_parameters': trainable, **info}), flush=True)
    from comet_ml import Experiment
    comet = Experiment(workspace=cfg['comet_workspace'], project_name=cfg['comet_project'], auto_output_logging='simple')
    comet.set_name(args.run_name); comet.add_tags(['latent_autoencoder', 'frozen_qwen', 'soft_tokens', 'deterministic'])
    comet.log_parameters({**cfg, **info, 'trainable_parameters': trainable})

    def validate(step):
        teacher = evaluate_teacher_forced(qwen, tokenizer, model, val_rows, device, cfg)
        generated, examples = generate_examples(qwen, tokenizer, model, val_rows, device, cfg)
        metrics = {**teacher, **generated}
        (root / f'validation_step_{step}.json').write_text(json.dumps(metrics, indent=2))
        (root / f'generation_step_{step}.json').write_text(json.dumps(examples, indent=2, ensure_ascii=False))
        print(json.dumps({'step': step, **metrics}), flush=True); comet.log_metrics(metrics, step=step)

    validate(0)
    order = list(range(len(train_rows))); step = 0
    while step < (args.smoke_steps or int(cfg['max_optimizer_steps'])):
        random.shuffle(order)
        for index in order:
            row = train_rows[index]; task, observation = target_row(row)
            model.train(); optimizer.zero_grad(set_to_none=True)
            z = encode_observations(qwen, tokenizer, model, [task], [observation], device, int(cfg['max_sequence_length']))
            loss, correct, tokens, shapes = reconstruction(qwen, tokenizer, model, [observation], z, device, int(cfg['max_sequence_length']))
            loss.backward()
            diagnostics = {'encoder_head_gradient_norm': grad_norm(model.encoder_head),
                           'decoder_state_projector_gradient_norm': grad_norm(model.decoder_state_projector),
                           'frozen_qwen_has_grad': any(parameter.grad is not None for parameter in qwen.parameters())}
            if step == 0:
                sanity = {'qwen_hidden.shape': [1, qwen.config.hidden_size], 'z.shape': list(z.shape), **shapes,
                          'observation_is_present_in_encoder_input': observation in encoder_prompt(task, observation),
                          'observation_is_present_in_decoder_conditioning_prompt': observation in DECODER_INSTRUCTION,
                          'observation_is_used_only_as_decoder_target': True, **diagnostics}
                print(json.dumps({'sanity': sanity}), flush=True)
                (root / 'sanity.json').write_text(json.dumps(sanity, indent=2))
                if diagnostics['encoder_head_gradient_norm'] <= 0 or diagnostics['decoder_state_projector_gradient_norm'] <= 0 or diagnostics['frozen_qwen_has_grad']:
                    raise RuntimeError(f'gradient sanity check failed: {diagnostics}')
            gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg['max_grad_norm'])))
            optimizer.step(); step += 1
            metrics = {'train/recon_ce': float(loss.detach()), 'train/recon_ppl': float(math.exp(min(float(loss.detach()), 20.0))),
                       'train/token_accuracy': float(correct.float() / tokens.clamp_min(1)), 'train/grad_norm_preclip': gradient,
                       'train/grad_clip_max': float(cfg['max_grad_norm']), **{f'train/{key}': value for key, value in diagnostics.items()}}
            print(json.dumps({'step': step, **metrics}), flush=True); comet.log_metrics(metrics, step=step)
            if step in (100, 500) or (step > 500 and step % 500 == 0): validate(step)
            if step >= (args.smoke_steps or int(cfg['max_optimizer_steps'])): break
    validate(step)
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step}, root / 'final.pt')
    comet.end()


if __name__ == '__main__':
    main()
