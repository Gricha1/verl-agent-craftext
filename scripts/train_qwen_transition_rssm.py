#!/usr/bin/env python3
"""Qwen-transition RSSM: q(z_next | h_next, real_observation_next), never prior-conditioned."""
# Implementation is intentionally separate from the GRU Gaussian RSSM experiment.
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import train_rssm_qwen_soft_tokens as data


def gaussian(x):
    mu, logstd = x.chunk(2, dim=-1)
    return mu, logstd.clamp(-5, 2).exp(), logstd.clamp(-5, 2)


class QwenTransitionRSSM(nn.Module):
    """Only Qwen forwards perform state transition / observation correction."""
    def __init__(self, cfg, d_model):
        super().__init__()
        h, z, k = cfg['latent_h_dim'], cfg['latent_z_dim'], cfg['soft_tokens']
        self.state_projector = nn.Sequential(nn.Linear(h + z, h), nn.GELU(), nn.Linear(h, k * d_model))
        self.posterior_h_projector = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, k * d_model))
        self.transition_head = nn.Sequential(nn.Linear(d_model, h), nn.GELU(), nn.Linear(h, h))
        self.prior_head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, 2 * z))
        self.posterior_head = nn.Sequential(nn.Linear(d_model, h), nn.GELU(), nn.Linear(h, 2 * z))
        self.k, self.d_model = k, d_model

    def state_tokens(self, h, z):
        return self.state_projector(torch.cat((h, z), -1)).reshape(*h.shape[:-1], self.k, self.d_model)

    def posterior_tokens(self, h):
        # Architectural invariant: this module receives h only, never prior z/mu/logstd.
        return self.posterior_h_projector(h).reshape(*h.shape[:-1], self.k, self.d_model)


def kl(qm, qs, pm, ps):
    return (torch.log(ps / qs) + (qs.square() + (qm - pm).square()) / (2 * ps.square()) - .5).sum(-1)


def module_grad_norm(module):
    values=[p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
    return float(torch.sqrt(sum(values, torch.zeros((), device=next(module.parameters()).device))))


class Chunks(Dataset):
    def __init__(self, chunks): self.chunks = chunks
    def __len__(self): return len(self.chunks)
    def __getitem__(self, index): return self.chunks[index]


def eos_hidden(qwen, tokenizer, text, soft, device, max_len):
    """Inject soft tokens after the real prompt and return the final non-PAD token state."""
    ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_len)['input_ids']
    emb = qwen.get_input_embeddings()
    x = torch.cat((emb(torch.tensor(ids, device=device)), soft.to(emb.weight.dtype)), 0)[None]
    mask = torch.ones((1, x.shape[1]), device=device, dtype=torch.long)
    return qwen(inputs_embeds=x, attention_mask=mask, output_hidden_states=True, use_cache=False).hidden_states[-1][0, -1].float()


def posterior(qwen, tokenizer, model, task, observation, h, device, cfg):
    # Only h and real observation enter this branch.  Prior tensors are absent by construction.
    text = data.obs_text(tokenizer, task, observation)
    r = eos_hidden(qwen, tokenizer, text, model.posterior_tokens(h[None])[0], device, int(cfg['max_sequence_length']))
    return gaussian(model.posterior_head(r[None])) + (r,)


def transition(qwen, tokenizer, model, row, h, z, device, cfg):
    pre = data.actor_prefix(tokenizer, row)
    full = data.actor_full(tokenizer, row)
    prefix_len = len(tokenizer(pre, add_special_tokens=False)['input_ids'])
    ids = tokenizer(full, add_special_tokens=False, truncation=True, max_length=int(cfg['max_sequence_length']))['input_ids']
    emb = qwen.get_input_embeddings(); state = model.state_tokens(h, z)[0].to(emb.weight.dtype)
    x = torch.cat((emb(torch.tensor(ids[:prefix_len], device=device)), state, emb(torch.tensor(ids[prefix_len:], device=device))), 0)[None]
    out = qwen(inputs_embeds=x, attention_mask=torch.ones((1, x.shape[1]), device=device, dtype=torch.long), output_hidden_states=True, use_cache=False).hidden_states[-1][0, -1].float()
    return model.transition_head(out[None])


@torch.no_grad()
def posterior_dependence(qwen, tokenizer, model, rows, device, cfg):
    """Hold h fixed and replace only o_next: direct anti-collapse diagnostic."""
    task=str(rows[0]['task']); h=torch.zeros(1,int(cfg['latent_h_dim']),device=device); z=torch.zeros(1,int(cfg['latent_z_dim']),device=device)
    qm,qs,_,_=posterior(qwen,tokenizer,model,task,str(rows[0]['observation_t']),h[0],device,cfg); z=qm
    hs=[]; real=[]; shuffled=[]
    for row in rows:
        h=transition(qwen,tokenizer,model,row,h,z,device,cfg); hs.append(h[0]); real.append(str(row['observation_t_plus_1']))
        qm,qs,_,_=posterior(qwen,tokenizer,model,task,real[-1],h[0],device,cfg); z=qm
    shuffled=real[1:]+real[:1]
    q_real=[posterior(qwen,tokenizer,model,task,ob,h,device,cfg) for h,ob in zip(hs,real)]
    q_shuf=[posterior(qwen,tokenizer,model,task,ob,h,device,cfg) for h,ob in zip(hs,shuffled)]
    rmu=torch.cat([x[0] for x in q_real]); rst=torch.cat([x[1] for x in q_real]); smu=torch.cat([x[0] for x in q_shuf]); sst=torch.cat([x[1] for x in q_shuf])
    return {'val/posterior_real_vs_shuffled_mu_l2':float((rmu-smu).norm(dim=-1).mean()),'val/posterior_real_vs_shuffled_kl':float(kl(rmu,rst,smu,sst).mean()),'posterior/mu_std':float(rmu.std()),'posterior/std_mean':float(rst.mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke-steps', type=int, default=0)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    tokenizer = AutoTokenizer.from_pretrained(cfg['base_model'], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    qwen = AutoModelForCausalLM.from_pretrained(cfg['base_model'], torch_dtype=torch.bfloat16, trust_remote_code=True).to(args.device).eval()
    for p in qwen.parameters(): p.requires_grad_(False)
    model = QwenTransitionRSSM(cfg, qwen.config.hidden_size).to(args.device)
    train, val, test = data.split(cfg['dataset_dir'], cfg['split_seed'])
    train_chunks = data.chunks(train, cfg['sequence_length'], cfg['chunk_stride']); val_chunks=data.chunks(val, cfg['sequence_length'], cfg['chunk_stride'])
    loader = DataLoader(Chunks(train_chunks), batch_size=1, shuffle=True, collate_fn=lambda x: x[0])
    root = Path(cfg['output_root']) / args.run_name; root.mkdir(parents=True, exist_ok=False)
    (root/'resolved_config.yaml').write_text(yaml.safe_dump(cfg)); (root/'splits.json').write_text(json.dumps({'train_transitions':len(train),'val_transitions':len(val),'test_transitions':len(test)}))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    print(json.dumps({'trainable_parameters': sum(p.numel() for p in model.parameters()), 'posterior_uses_prior': False, 'posterior_uses_h_next': True, 'posterior_uses_real_observation': True, 'steps_per_epoch':len(loader)}), flush=True)
    try:
        from comet_ml import Experiment
        comet=Experiment(workspace=cfg['comet_workspace'],project_name=cfg['comet_project'],auto_output_logging='simple'); comet.set_name(args.run_name); comet.log_parameters(cfg)
    except Exception as exc: comet=None; print(f'Comet disabled: {exc}', flush=True)
    step=0; device=torch.device(args.device)
    for epoch in range(int(cfg['epochs'])):
        model.train()
        for rows in loader:
            task=str(rows[0]['task']); h=torch.zeros(1,int(cfg['latent_h_dim']),device=device); z=torch.zeros(1,int(cfg['latent_z_dim']),device=device)
            qm,qs,ql,_=posterior(qwen,tokenizer,model,task,str(rows[0]['observation_t']),h[0],device,cfg); z=qm+qs*torch.randn_like(qs)
            raw=[]; dyn=[]; rep=[]
            for row in rows:
                h=transition(qwen,tokenizer,model,row,h,z,device,cfg); pm,ps,pl=gaussian(model.prior_head(h)); qm,qs,ql,_=posterior(qwen,tokenizer,model,task,str(row['observation_t_plus_1']),h[0],device,cfg)
                raw.append(kl(qm,qs,pm,ps)); dyn.append(kl(qm.detach(),qs.detach(),pm,ps)); rep.append(kl(qm,qs,pm.detach(),ps.detach())); z=qm+qs*torch.randn_like(qs)
            raw_values=torch.stack(raw); dyn_values=torch.stack(dyn); rep_values=torch.stack(rep)
            # Free-nats must be part of the objective, per state/sample, not a display-only metric.
            raw=raw_values.mean(); dyn=dyn_values.mean(); rep=rep_values.mean(); dyn_used=torch.clamp(dyn_values,min=float(cfg['free_nats'])).mean(); rep_used=torch.clamp(rep_values,min=float(cfg['free_nats'])).mean(); used=torch.clamp(raw_values,min=float(cfg['free_nats'])).mean(); loss=float(cfg['kl_balance'])*dyn_used+(1-float(cfg['kl_balance']))*rep_used
            opt.zero_grad(); loss.backward(); components={f'train/grad_preclip/{name}':module_grad_norm(getattr(model,name)) for name in ('state_projector','posterior_h_projector','transition_head','prior_head','posterior_head')}; grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg['max_grad_norm']))); opt.step(); step+=1
            metrics={'train/loss':float(loss),'train/kl_raw':float(raw),'train/kl_dyn':float(dyn),'train/kl_rep':float(rep),'train/kl_used':float(used),'train/grad_norm_preclip':grad,'train/grad_clip_max':float(cfg['max_grad_norm']),'latent/h_absmax':float(h.abs().max()),'prior/logstd_absmax':float(pl.abs().max()),'posterior/logstd_absmax':float(ql.abs().max()),**components}; print(json.dumps({'step':step,**metrics}),flush=True); comet and comet.log_metrics(metrics,step=step)
            if step == 1 or step % 500 == 0:
                val_metrics=posterior_dependence(qwen,tokenizer,model,val_chunks[0],device,cfg); print(json.dumps({'step':step,**val_metrics}),flush=True); comet and comet.log_metrics(val_metrics,step=step)
            if step==1: print(json.dumps({'h_next.shape':[1,512],'posterior_soft_tokens.shape':[1,int(cfg['soft_tokens']),qwen.config.hidden_size],'posterior_qwen_eos.shape':[1,qwen.config.hidden_size],'mu_post.shape':list(qm.shape),'logstd_post.shape':list(ql.shape),'posterior_uses_prior':False}),flush=True)
            if step >= (args.smoke_steps or int(cfg['max_optimizer_steps'])): torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step},root/'final.pt'); comet and comet.end(); return


if __name__ == '__main__':
    main()
