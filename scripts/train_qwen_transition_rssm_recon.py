#!/usr/bin/env python3
"""Frozen-Qwen RSSM with a z-only autoregressive observation reconstruction loss.

This experiment deliberately contains no grounding, reward, actor, or environment
objective.  The only optimized loss is reconstruction CE plus balanced free-nats KL.
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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import train_rssm_qwen_soft_tokens as data


DECODER_INSTRUCTION = "Reconstruct the current environment observation exactly.\n"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke-steps', type=int, default=0)
    parser.add_argument('--source-commit', default=None)
    return parser.parse_args()


def gaussian(x):
    mu, logstd = x.chunk(2, dim=-1)
    return mu, logstd.clamp(-5, 2).exp(), logstd.clamp(-5, 2)


def kl(qm, qs, pm, ps):
    return (torch.log(ps / qs) + (qs.square() + (qm - pm).square()) / (2 * ps.square()) - .5).sum(-1)


def grad_norm(module):
    values = [p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
    return float(torch.sqrt(sum(values, torch.zeros((), device=next(module.parameters()).device))))


def normalise(text):
    return ' '.join(text.strip().split())


class Chunks(Dataset):
    def __init__(self, chunks): self.chunks = chunks
    def __len__(self): return len(self.chunks)
    def __getitem__(self, index): return self.chunks[index]


class QwenTransitionRSSMRecon(nn.Module):
    """Qwen is the transition model; the reconstruction decoder sees only z."""
    def __init__(self, cfg, d_model):
        super().__init__()
        h, z, k = int(cfg['latent_h_dim']), int(cfg['latent_z_dim']), int(cfg['soft_tokens'])
        self.state_projector = nn.Sequential(nn.Linear(h + z, h), nn.GELU(), nn.Linear(h, k * d_model))
        self.posterior_h_projector = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, k * d_model))
        self.transition_head = nn.Sequential(nn.Linear(d_model, h), nn.GELU(), nn.Linear(h, h))
        self.prior_head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, 2 * z))
        self.posterior_head = nn.Sequential(nn.Linear(d_model, h), nn.GELU(), nn.Linear(h, 2 * z))
        # Freshly initialized: no autoencoder weights are transferred here.
        self.decoder_state_projector = nn.Sequential(nn.Linear(z, z), nn.GELU(), nn.Linear(z, k * d_model))
        self.k, self.d_model = k, d_model
        self.register_buffer('soft_token_target_rms', torch.tensor(float(cfg['soft_token_target_rms'])))
        with torch.no_grad():
            generator = torch.Generator(device='cpu').manual_seed(0)
            probe_h = torch.randn(64, h, generator=generator)
            probe_z = torch.randn(64, z, generator=generator)
            state_rms = self.state_projector(torch.cat((probe_h, probe_z), -1)).float().square().mean().sqrt()
            posterior_rms = self.posterior_h_projector(probe_h).float().square().mean().sqrt()
            decoder_rms = self.decoder_state_projector(probe_z).float().square().mean().sqrt()
        self.register_buffer('state_soft_token_gain', self.soft_token_target_rms / state_rms.clamp_min(1e-8))
        self.register_buffer('posterior_soft_token_gain', self.soft_token_target_rms / posterior_rms.clamp_min(1e-8))
        self.register_buffer('decoder_soft_token_gain', self.soft_token_target_rms / decoder_rms.clamp_min(1e-8))

    def state_tokens(self, h, z):
        return self.state_projector(torch.cat((h, z), -1)).reshape(*h.shape[:-1], self.k, self.d_model) * self.state_soft_token_gain.to(h.dtype)

    def posterior_tokens(self, h):
        # The posterior receives h and the real observation only, never prior tensors.
        return self.posterior_h_projector(h).reshape(*h.shape[:-1], self.k, self.d_model) * self.posterior_soft_token_gain.to(h.dtype)

    def decoder_tokens(self, z):
        # The decoder intentionally cannot use h: z must retain observation information.
        return self.decoder_state_projector(z).reshape(*z.shape[:-1], self.k, self.d_model) * self.decoder_soft_token_gain.to(z.dtype)


def eos_hidden(qwen, tokenizer, text, soft, device, max_length):
    ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_length)['input_ids']
    embedding = qwen.get_input_embeddings()
    inputs = torch.cat((embedding(torch.tensor(ids, device=device)), soft.to(embedding.weight.dtype)), 0)[None]
    mask = torch.ones(inputs.shape[:2], device=device, dtype=torch.long)
    return qwen(inputs_embeds=inputs, attention_mask=mask, output_hidden_states=True, use_cache=False).hidden_states[-1][0, -1].float()


def posterior(qwen, tokenizer, model, task, observation, h, device, cfg):
    text = data.obs_text(tokenizer, task, observation)
    hidden = eos_hidden(qwen, tokenizer, text, model.posterior_tokens(h[None])[0], device, int(cfg['max_sequence_length']))
    return gaussian(model.posterior_head(hidden[None])) + (hidden,)


def transition(qwen, tokenizer, model, row, h, z, device, cfg):
    prefix = data.actor_prefix(tokenizer, row)
    full = data.actor_full(tokenizer, row)
    prefix_length = len(tokenizer(prefix, add_special_tokens=False)['input_ids'])
    ids = tokenizer(full, add_special_tokens=False, truncation=True, max_length=int(cfg['max_sequence_length']))['input_ids']
    embedding = qwen.get_input_embeddings()
    state = model.state_tokens(h, z)[0].to(embedding.weight.dtype)
    inputs = torch.cat((embedding(torch.tensor(ids[:prefix_length], device=device)), state,
                        embedding(torch.tensor(ids[prefix_length:], device=device))), 0)[None]
    mask = torch.ones(inputs.shape[:2], device=device, dtype=torch.long)
    hidden = qwen(inputs_embeds=inputs, attention_mask=mask, output_hidden_states=True, use_cache=False).hidden_states[-1][0, -1].float()
    return model.transition_head(hidden[None])


def reconstruction(qwen, tokenizer, model, observations, z, device, max_length, no_soft=False):
    """Teacher-forced CE over observation tokens plus an explicit EOS target."""
    embedding = qwen.get_input_embeddings()
    instruction_ids = tokenizer(DECODER_INSTRUCTION, add_special_tokens=False)['input_ids']
    if tokenizer.eos_token_id is None:
        raise RuntimeError('tokenizer must define eos_token_id for reconstruction')
    pieces, labels_list = [], []
    soft_count = 0 if no_soft else model.k
    for observation, latent in zip(observations, z):
        target_ids = tokenizer(str(observation), add_special_tokens=False, truncation=True,
                               max_length=max_length - len(instruction_ids) - soft_count - 1)['input_ids']
        target_ids.append(tokenizer.eos_token_id)
        prefix = embedding(torch.tensor(instruction_ids, device=device))
        target = embedding(torch.tensor(target_ids, device=device))
        if no_soft:
            piece = torch.cat((prefix, target), 0)
        else:
            soft = model.decoder_tokens(latent[None])[0].to(embedding.weight.dtype)
            piece = torch.cat((prefix, soft, target), 0)
        pieces.append(piece)
        labels_list.append(torch.tensor([-100] * (len(instruction_ids) + soft_count) + target_ids, device=device))
    maximum = max(len(piece) for piece in pieces)
    inputs = torch.stack([torch.cat((piece, torch.zeros(maximum - len(piece), piece.shape[-1], device=device, dtype=piece.dtype))) for piece in pieces])
    labels = torch.stack([torch.cat((label, torch.full((maximum - len(label),), -100, device=device, dtype=torch.long))) for label in labels_list])
    mask = torch.tensor([[1] * len(piece) + [0] * (maximum - len(piece)) for piece in pieces], device=device)
    result = qwen(inputs_embeds=inputs, attention_mask=mask, labels=labels, use_cache=False)
    predicted = result.logits[:, :-1].argmax(dim=-1)
    valid = labels[:, 1:] != -100
    correct = ((predicted == labels[:, 1:]) & valid).sum()
    return result.loss, correct, valid.sum()


@torch.no_grad()
def generate_metrics(qwen, tokenizer, model, observations, z, device, cfg, no_soft=False):
    embedding = qwen.get_input_embeddings()
    instruction = embedding(torch.tensor(tokenizer(DECODER_INSTRUCTION, add_special_tokens=False)['input_ids'], device=device))
    exact, token_match, edit_similarity, examples = [], [], [], []
    for target, latent in zip(observations, z):
        initial = instruction if no_soft else torch.cat((instruction, model.decoder_tokens(latent[None])[0].to(embedding.weight.dtype)), 0)
        generated_ids = qwen.generate(inputs_embeds=initial[None], attention_mask=torch.ones((1, len(initial)), device=device, dtype=torch.long),
                                      do_sample=False, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
                                      max_new_tokens=min(int(cfg['generation_max_new_tokens']), len(tokenizer(str(target), add_special_tokens=False)['input_ids']) + int(cfg['generation_stop_margin'])))
        generated = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        target_ids = tokenizer(str(target), add_special_tokens=False)['input_ids']
        output_ids = tokenizer(generated, add_special_tokens=False)['input_ids']
        token_match.append(sum(a == b for a, b in zip(target_ids, output_ids)) / max(len(target_ids), len(output_ids), 1))
        exact.append(normalise(generated) == normalise(str(target)))
        edit_similarity.append(difflib.SequenceMatcher(None, normalise(str(target)), normalise(generated)).ratio())
        examples.append({'target': str(target), 'generated': generated})
    return {'token_match': float(sum(token_match) / len(token_match)), 'exact_match': float(sum(exact) / len(exact)),
            'edit_similarity': float(sum(edit_similarity) / len(edit_similarity))}, examples


@torch.no_grad()
def validation_states(qwen, tokenizer, model, chunks, device, cfg):
    count = int(cfg['validation_examples'])
    sequence_count = math.ceil(count / int(cfg['sequence_length']))
    selected = random.Random(int(cfg['validation_shuffle_seed'])).sample(chunks, min(sequence_count, len(chunks)))
    states = []
    for rows in selected:
        task = str(rows[0]['task'])
        h = torch.zeros(1, int(cfg['latent_h_dim']), device=device)
        qm, _, _, _ = posterior(qwen, tokenizer, model, task, str(rows[0]['observation_t']), h[0], device, cfg)
        z = qm
        for row in rows:
            h = transition(qwen, tokenizer, model, row, h, z, device, cfg)
            pm, ps, _ = gaussian(model.prior_head(h))
            qm, qs, _ql, _ = posterior(qwen, tokenizer, model, task, str(row['observation_t_plus_1']), h[0], device, cfg)
            states.append({'task': task, 'observation': str(row['observation_t_plus_1']), 'h': h[0], 'prior_mu': pm[0], 'prior_std': ps[0], 'posterior_mu': qm[0], 'posterior_std': qs[0]})
            z = qm
            if len(states) == count:
                return states
    return states


@torch.no_grad()
def validate(qwen, tokenizer, model, chunks, device, cfg):
    model.eval()
    states = validation_states(qwen, tokenizer, model, chunks, device, cfg)
    observations = [state['observation'] for state in states]
    post = torch.stack([state['posterior_mu'] for state in states])
    prior = torch.stack([state['prior_mu'] for state in states])
    posterior_std = torch.stack([state['posterior_std'] for state in states])
    prior_std = torch.stack([state['prior_std'] for state in states])
    permutation = torch.randperm(len(states), generator=torch.Generator(device=device).manual_seed(int(cfg['validation_shuffle_seed'])), device=device)
    shuffled, zero = post[permutation], torch.zeros_like(post)
    metrics = {}
    for name, latent in [('posterior', post), ('posterior_shuffled', shuffled), ('posterior_zero', zero), ('prior', prior)]:
        loss, correct, tokens = reconstruction(qwen, tokenizer, model, observations, latent, device, int(cfg['max_sequence_length']))
        metrics[f'val/recon_ce_{name}'] = float(loss)
        metrics[f'val/recon_token_accuracy_{name}'] = float(correct.float() / tokens.clamp_min(1))
    no_soft_loss, no_soft_correct, no_soft_tokens = reconstruction(qwen, tokenizer, model, observations, post, device, int(cfg['max_sequence_length']), no_soft=True)
    metrics['val/recon_ce_no_soft'] = float(no_soft_loss)
    metrics['val/recon_token_accuracy_no_soft'] = float(no_soft_correct.float() / no_soft_tokens.clamp_min(1))
    metrics['val/delta_recon_ce_shuffled'] = metrics['val/recon_ce_posterior_shuffled'] - metrics['val/recon_ce_posterior']
    metrics['val/delta_recon_ce_zero'] = metrics['val/recon_ce_posterior_zero'] - metrics['val/recon_ce_posterior']
    metrics['val/posterior_real_vs_shuffled_mu_l2'] = float((post - shuffled).norm(dim=-1).mean())
    metrics['val/posterior_real_vs_shuffled_kl'] = float(kl(post, posterior_std, shuffled, posterior_std[permutation]).mean())
    generation_examples = min(int(cfg['generation_examples']), len(states))
    for name, latent in [('posterior', post), ('posterior_shuffled', shuffled), ('posterior_zero', zero), ('prior', prior)]:
        generated, _ = generate_metrics(qwen, tokenizer, model, observations[:generation_examples], latent[:generation_examples], device, cfg)
        for metric, value in generated.items(): metrics[f'val/gen_{metric}_{name}'] = value
    generated, examples = generate_metrics(qwen, tokenizer, model, observations[:generation_examples], post[:generation_examples], device, cfg, no_soft=True)
    for metric, value in generated.items(): metrics[f'val/gen_{metric}_no_soft'] = value
    return metrics, examples, {'posterior_std_mean': float(posterior_std.mean()), 'prior_std_mean': float(prior_std.mean())}


def should_validate(step):
    return step in (100, 500, 1000) or (step > 1000 and step % 1000 == 0)


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    random.seed(int(cfg['seed'])); torch.manual_seed(int(cfg['seed']))
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg['base_model'], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = 'left'
    qwen = AutoModelForCausalLM.from_pretrained(cfg['base_model'], torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
    for parameter in qwen.parameters(): parameter.requires_grad_(False)
    cfg['soft_token_target_rms'] = float(qwen.get_input_embeddings().weight.detach().float().square().mean().sqrt())
    model = QwenTransitionRSSMRecon(cfg, qwen.config.hidden_size).to(device)
    train_rows, val_rows, test_rows = data.split(cfg['dataset_dir'], int(cfg['split_seed']))
    train_chunks = data.chunks(train_rows, int(cfg['sequence_length']), int(cfg['chunk_stride']))
    val_chunks = data.chunks(val_rows, int(cfg['sequence_length']), int(cfg['chunk_stride']))
    loader = DataLoader(Chunks(train_chunks), batch_size=1, shuffle=True, collate_fn=lambda values: values[0])
    root = Path(cfg['output_root']) / args.run_name
    root.mkdir(parents=True, exist_ok=False)
    commit = args.source_commit or subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    info = {'git_commit': commit, 'train_transitions': len(train_rows), 'val_transitions': len(val_rows), 'test_transitions': len(test_rows),
            'train_chunks': len(train_chunks), 'val_chunks': len(val_chunks), 'effective_optimizer_batch': int(cfg['train_batch_size'])}
    (root / 'resolved_config.yaml').write_text(yaml.safe_dump({**cfg, **info, 'run_name': args.run_name}, sort_keys=True))
    (root / 'splits.json').write_text(json.dumps(info, indent=2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['learning_rate']), weight_decay=float(cfg['weight_decay']))
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(json.dumps({'trainable_parameters': trainable, 'posterior_uses_prior': False, 'decoder_uses_h': False, **info}), flush=True)
    from comet_ml import Experiment
    comet = Experiment(workspace=cfg['comet_workspace'], project_name=cfg['comet_project'], auto_output_logging='simple')
    comet.set_name(args.run_name); comet.add_tags(['rssm', 'reconstruction', 'frozen_qwen', 'no_auxiliary_loss'])
    comet.log_parameters({**cfg, **info, 'trainable_parameters': trainable})

    def log_validation(step):
        metrics, examples, diagnostic = validate(qwen, tokenizer, model, val_chunks, device, cfg)
        metrics.update({f'val/{key}': value for key, value in diagnostic.items()})
        (root / f'validation_step_{step}.json').write_text(json.dumps(metrics, indent=2))
        (root / f'generation_step_{step}.json').write_text(json.dumps(examples, indent=2, ensure_ascii=False))
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step}, root / f'checkpoint_step_{step}.pt')
        print(json.dumps({'step': step, **metrics}), flush=True); comet.log_metrics(metrics, step=step)

    log_validation(0)
    step = 0
    limit = args.smoke_steps or int(cfg['max_optimizer_steps'])
    for _epoch in range(int(cfg['epochs'])):
        for rows in loader:
            model.train(); optimizer.zero_grad(set_to_none=True)
            task = str(rows[0]['task'])
            h = torch.zeros(1, int(cfg['latent_h_dim']), device=device)
            qm, qs, _ql, _ = posterior(qwen, tokenizer, model, task, str(rows[0]['observation_t']), h[0], device, cfg)
            z = qm + qs * torch.randn_like(qs)
            raw_values, dyn_values, rep_values, posterior_latents, targets = [], [], [], [], []
            for row in rows:
                h = transition(qwen, tokenizer, model, row, h, z, device, cfg)
                pm, ps, pl = gaussian(model.prior_head(h))
                qm, qs, ql, _ = posterior(qwen, tokenizer, model, task, str(row['observation_t_plus_1']), h[0], device, cfg)
                raw_values.append(kl(qm, qs, pm, ps)); dyn_values.append(kl(qm.detach(), qs.detach(), pm, ps)); rep_values.append(kl(qm, qs, pm.detach(), ps.detach()))
                posterior_latents.append(qm[0]); targets.append(str(row['observation_t_plus_1']))
                z = qm + qs * torch.randn_like(qs)
                if int(cfg['truncate_bptt_steps']) == 1: h, z = h.detach(), z.detach()
            raw_values, dyn_values, rep_values = torch.stack(raw_values), torch.stack(dyn_values), torch.stack(rep_values)
            raw, dyn, rep = raw_values.mean(), dyn_values.mean(), rep_values.mean()
            dyn_used = torch.clamp(dyn_values, min=float(cfg['free_nats'])).mean()
            rep_used = torch.clamp(rep_values, min=float(cfg['free_nats'])).mean()
            used = torch.clamp(raw_values, min=float(cfg['free_nats'])).mean()
            kl_loss = float(cfg['kl_balance']) * dyn_used + (1.0 - float(cfg['kl_balance'])) * rep_used
            recon_ce, correct, token_count = reconstruction(qwen, tokenizer, model, targets, torch.stack(posterior_latents), device, int(cfg['max_sequence_length']))
            total = float(cfg['lambda_recon']) * recon_ce + float(cfg['beta_kl']) * kl_loss
            total.backward()
            components = {name: grad_norm(getattr(model, name)) for name in ('state_projector', 'transition_head', 'prior_head', 'posterior_h_projector', 'posterior_head', 'decoder_state_projector')}
            qwen_has_grad = any(parameter.grad is not None for parameter in qwen.parameters())
            if step == 0:
                sanity = {'gradients': components, 'frozen_qwen_has_grad': qwen_has_grad, 'decoder_input': 'z_post_only',
                          'observation_in_decoder_conditioning': False, 'observation_is_autoregressive_target': True}
                (root / 'sanity.json').write_text(json.dumps(sanity, indent=2))
                print(json.dumps({'sanity': sanity}), flush=True)
                if any(value <= 0 for value in components.values()) or qwen_has_grad:
                    raise RuntimeError(f'gradient sanity check failed: {sanity}')
            grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg['max_grad_norm'])))
            optimizer.step(); step += 1
            metrics = {'train/recon_ce': float(recon_ce.detach()), 'train/token_accuracy': float(correct.float() / token_count.clamp_min(1)),
                       'train/total_loss': float(total.detach()), 'train/kl_raw': float(raw), 'train/kl_used': float(used),
                       'train/kl_dyn': float(dyn), 'train/kl_rep': float(rep), 'train/kl_loss': float(kl_loss),
                       'train/grad_norm_preclip': grad, 'train/grad_clip_max': float(cfg['max_grad_norm']),
                       **{f'train/grad_preclip/{name}': value for name, value in components.items()}}
            if step == 1 or step % int(cfg['log_every_steps']) == 0:
                print(json.dumps({'step': step, **metrics}), flush=True); comet.log_metrics(metrics, step=step)
            if should_validate(step): log_validation(step)
            if step >= limit: break
        if step >= limit: break
    if not should_validate(step): log_validation(step)
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step}, root / 'final.pt')
    comet.end()


if __name__ == '__main__':
    main()
