#!/usr/bin/env python3
"""Gaussian RSSM + frozen-Qwen soft-token world model (offline)."""
from __future__ import annotations
import argparse, json, math, random, subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import train_rssm_qwen_soft_tokens as base
from train_latent_wm import LocalVector, env_cfg
from agent_system.environments.env_manager import CagedCraftextEnvironmentManager
from agent_system.environments.env_package.caged_craftext.projection import craftext_projection

def args():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--run-name',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke-steps',type=int,default=0);p.add_argument('--source-commit',default=None);return p.parse_args()
def dist(x):
 mu,raw=x.chunk(2,-1);return mu,raw.clamp(-5,2).exp(),raw.clamp(-5,2)
def kl(qmu,qstd,pmu,pstd):return (torch.log(pstd/qstd)+(qstd.square()+(qmu-pmu).square())/(2*pstd.square())-.5).sum(-1)
class GaussianRSSM(nn.Module):
 def __init__(self,c,d):
  super().__init__();H,Z,E,A=c['latent_h_dim'],c['latent_z_dim'],c['observation_dim'],c['action_embedding_dim'];self.act=nn.Embedding(4,A);self.transition=nn.GRUCell(Z+A,H);self.prior=nn.Sequential(nn.Linear(H,H),nn.GELU(),nn.Linear(H,2*Z));self.posterior=nn.Sequential(nn.Linear(H+E,H),nn.GELU(),nn.Linear(H,2*Z));self.decoder=nn.Sequential(nn.Linear(H+Z,H),nn.GELU(),nn.Linear(H,E));self.projector=nn.Sequential(nn.Linear(H+Z,H),nn.GELU(),nn.Linear(H,c['soft_tokens']*d));self.qhead=nn.Sequential(nn.Linear(d,H),nn.GELU(),nn.Linear(H,2*Z))
 def init(self,e,sample):
  h=torch.zeros(len(e),self.transition.hidden_size,device=e.device);mu,std,ls=dist(self.posterior(torch.cat((h,e),-1)));z=mu+std*torch.randn_like(std) if sample else mu;return h,z,(mu,std,ls)
 def step(self,h,z,a,e,sample):
  hn=self.transition(torch.cat((z,self.act(a)),-1),h);pm,ps,pl=dist(self.prior(hn));qm,qs,ql=dist(self.posterior(torch.cat((hn,e),-1)));z=qm+qs*torch.randn_like(qs) if sample else qm;return hn,z,(pm,ps,pl),(qm,qs,ql)
 def tokens(self,h,z):return self.projector(torch.cat((h,z),-1))
 def decode(self,h,z):return self.decoder(torch.cat((h,z),-1))
def rollout(m,e,a,sample):
 h,z,q=m.init(e[:,0],sample);hs=[h];zs=[z];qs=[q];ps=[]
 for t in range(a.shape[1]):h,z,p,q=m.step(h,z,a[:,t],e[:,t+1],sample);hs.append(h);zs.append(z);ps.append(p);qs.append(q)
 return torch.stack(hs,1),torch.stack(zs,1),ps,qs
def qwen_dist(qwen,tok,m,rows,h,z,dev,c,mode='learned'):
 B,T=h.shape[:2];out=[];embed=qwen.get_input_embeddings();K=c['soft_tokens']
 for t in range(T):
  st=m.tokens(h[:,t],z[:,t]).reshape(B,K,-1);st=torch.zeros_like(st) if mode=='zero' else st[torch.randperm(B,device=dev)] if mode=='shuffled' else st;xs=[];ls=[]
  for b in range(B):
   r=rows[b][t];pre=tok(base.actor_prefix(tok,r),add_special_tokens=False)['input_ids'];full=tok(base.actor_full(tok,r),add_special_tokens=False)['input_ids'];ids=torch.tensor(full,device=dev);x=torch.cat((embed(ids[:len(pre)]),st[b].to(embed.weight.dtype),embed(ids[len(pre):])),0);xs.append(x);ls.append(len(x))
  L=max(ls);x=torch.stack([torch.cat((v,torch.zeros(L-len(v),v.shape[-1],device=dev,dtype=v.dtype))) for v in xs]);mask=torch.tensor([[1]*n+[0]*(L-n) for n in ls],device=dev);hidden=qwen(inputs_embeds=x,attention_mask=mask,output_hidden_states=True,use_cache=False).hidden_states[-1];out.append(dist(m.qhead(hidden[torch.arange(B,device=dev),torch.tensor(ls,device=dev)-1].float())))
 return out

def obs_embedding(qwen,tok,proj,obs,dev,c):
 """Frozen observation encoder used after each *real* environment transition."""
 texts=[base.obs_text(tok,"Production CrafText state",str(x)) for x in obs]
 return base.final_hidden(qwen,tok,texts,dev,int(c['max_sequence_length'])).to(dev) @ proj

def env_metrics(out):
 mean=lambda k:float(np.mean([x[k] for x in out])); rewards=[x['reward'] for x in out]
 return {'success_rate':mean('success'),'mean_episode_reward':mean('reward'),'median_episode_reward':float(np.median(rewards)),'std_episode_reward':float(np.std(rewards)),'mean_episode_length':mean('length'),'valid_action_rate':mean('valid'),'parse_error_rate':mean('errors'),'repeated_action_rate':mean('repeated')}

@torch.no_grad()
def production_env_eval(qwen,tok,m,proj,dev,c,seeds,mode):
 """Production 8x8 actor loop.  State is strictly pre-action; zero/base never leak future observations."""
 cfg=env_cfg(int(seeds[0]),int(c['eval_max_steps'])); vec=LocalVector(cfg,seeds); env=CagedCraftextEnvironmentManager(vec,craftext_projection,cfg)
 obs,_=env.reset({}); n=len(seeds); active=np.ones(n,dtype=bool); rewards=np.zeros(n); valid=np.zeros(n); errors=np.zeros(n); repeated=np.zeros(n); lengths=np.zeros(n,dtype=int); wins=np.zeros(n); history=[[] for _ in range(n)]; traces=[[] for _ in range(n)]
 e=obs_embedding(qwen,tok,proj,obs['text'],dev,c); h,z,_=m.init(e,False)
 try:
  for t in range(int(c['eval_max_steps'])):
   idx=np.flatnonzero(active)
   if not len(idx): break
   texts=[tok.apply_chat_template([{'role':'user','content':obs['text'][i]}],tokenize=False,add_generation_prompt=True) for i in idx]
   batch=tok(texts,return_tensors='pt',padding=True,add_special_tokens=False).to(dev)
   kwargs=dict(do_sample=True,temperature=float(c['eval_temperature']),top_p=1.,top_k=0,max_new_tokens=int(c['eval_max_response_length']),pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
   torch.manual_seed(int(seeds[0])*1000+t)
   if mode=='base': y=qwen.generate(**batch,**kwargs)
   else:
    B=len(idx); emb=qwen.get_input_embeddings(); states=m.tokens(h[idx],z[idx]).reshape(B,int(c['soft_tokens']),-1)
    if mode=='zero': states=torch.zeros_like(states)
    lens=batch['attention_mask'].sum(-1).tolist(); pieces=[]
    for j,L in enumerate(lens):
     # left padding is removed: soft tokens are appended after the real prompt.
     ids=batch['input_ids'][j,-int(L):]; pieces.append(torch.cat((emb(ids),states[j].to(emb.weight.dtype)),0))
    maxl=max(len(x) for x in pieces); inp=torch.stack([torch.cat((x,torch.zeros(maxl-len(x),x.shape[-1],device=dev,dtype=x.dtype))) for x in pieces]); mask=torch.tensor([[1]*len(x)+[0]*(maxl-len(x)) for x in pieces],device=dev)
    y=qwen.generate(inputs_embeds=inp,attention_mask=mask,**kwargs)
   # HF returns either generated-only (inputs_embeds) or prompt+generated (base).
   decoded=tok.batch_decode(y if y.shape[1]<=int(c['eval_max_response_length']) else y[:,-int(c['eval_max_response_length']):],skip_special_tokens=True)
   responses=['<action>NOOP</action>']*n
   for i,r in zip(idx,decoded): responses[i]=r
   obs_next,rs,dones,infos=env.step(responses)
   action_ids=torch.zeros(n,dtype=torch.long,device=dev)
   for i in idx:
    info=infos[i]; action=str(info.get('action_name','NOOP')); action_ids[i]=base.ACTIONS.index(action) if action in base.ACTIONS else 0
    rewards[i]+=float(rs[i]); valid[i]+=int(bool(info.get('is_action_valid',False))); errors[i]+=int(not bool(info.get('is_action_valid',False))); repeated[i]+=int(bool(history[i]) and history[i][-1]==action); history[i].append(action); lengths[i]+=1; wins[i]=float(bool(info.get('won',False))); traces[i].append({'step':t,'response':responses[i],'action':action})
   # Update only with the actual post-action environment observation, using posterior mean.
   en=obs_embedding(qwen,tok,proj,obs_next['text'],dev,c); hn,zn,_,_=m.step(h[idx],z[idx],action_ids[idx],en[idx],False); h[idx]=hn; z[idx]=zn; obs=obs_next; active[idx] &= ~np.asarray(dones,dtype=bool)[idx]
 finally: vec.close()
 rows=[{'seed':int(seed),'success':float(wins[i]),'reward':float(rewards[i]),'length':int(lengths[i]),'valid':float(valid[i]/max(1,lengths[i])),'errors':float(errors[i]/max(1,lengths[i])),'repeated':float(repeated[i]/max(1,lengths[i])),'trace':traces[i]} for i,seed in enumerate(seeds)]
 return {**env_metrics(rows),'episodes':rows}

@torch.no_grad()
def validate(m,qwen,tok,loader,dev,c):
 """Offline Gaussian diagnostics: KL decomposition, grounding and open-loop prediction."""
 m.eval(); allm={k:[] for k in ('obs_mse','obs_cosine','kl_raw','kl_dyn','kl_rep','kl_used','qwen_kl','qwen_zero_kl','qwen_shuffled_kl','h_norm','h_std','z_norm','z_std')}; openp={H:[] for H in (1,2,4,8,16)}
 for b in loader:
  e,a=b['e'].to(dev),b['a'].to(dev); h,z,ps,qs=rollout(m,e,a,False); pm,psd=torch.cat([x[0] for x in ps]),torch.cat([x[1] for x in ps]); qm,qsd=torch.cat([x[0] for x in qs[1:]]),torch.cat([x[1] for x in qs[1:]])
  raw=kl(qm,qsd,pm,psd); allm['obs_mse'].append(F.mse_loss(m.decode(h,z),e)); allm['obs_cosine'].append(F.cosine_similarity(m.decode(h,z),e,dim=-1).mean()); allm['kl_raw'].append(raw.mean()); allm['kl_dyn'].append(kl(qm.detach(),qsd.detach(),pm,psd).mean()); allm['kl_rep'].append(kl(qm,qsd,pm.detach(),psd.detach()).mean()); allm['kl_used'].append(torch.clamp(raw,min=float(c['free_nats'])).mean())
  for mode,key in [('learned','qwen_kl'),('zero','qwen_zero_kl'),('shuffled','qwen_shuffled_kl')]:
   d=qwen_dist(qwen,tok,m,b['rows'],h[:,:-1],z[:,:-1],dev,c,mode); allm[key].append(kl(qm.detach(),qsd.detach(),torch.cat([x[0] for x in d]),torch.cat([x[1] for x in d])).mean())
  allm['h_norm'].append(h.norm(dim=-1).mean()); allm['h_std'].append(h.std()); allm['z_norm'].append(z.norm(dim=-1).mean()); allm['z_std'].append(z.std())
  for H in openp:
   hi,zi,_=m.init(e[:,0],False)
   for t in range(H): hi,zi,_,_=m.step(hi,zi,a[:,t],e[:,t+1],False); zi=m.prior(hi)[0] if False else dist(m.prior(hi))[0]
   pred=m.decode(hi,zi); openp[H].append((F.mse_loss(pred,e[:,H]),F.cosine_similarity(pred,e[:,H],dim=-1).mean()))
 out={'val/'+k:float(torch.stack(v).mean()) for k,v in allm.items()}; out['val/kl_balance']=float(c['kl_balance']); out['val/free_nats']=float(c['free_nats'])
 for H,v in openp.items(): out[f'open_loop/h{H}_mse']=float(torch.stack([x[0] for x in v]).mean()); out[f'open_loop/h{H}_cosine']=float(torch.stack([x[1] for x in v]).mean())
 return out
def main():
 a=args();c=yaml.safe_load(Path(a.config).read_text());random.seed(c['seed']);torch.manual_seed(c['seed']);dev=torch.device(a.device);root=Path(c['output_root'])/a.run_name;root.mkdir(parents=True,exist_ok=False);tr,va,te=base.split(c['dataset_dir'],c['split_seed']);tc,vc,ec=[base.chunks(x,c['sequence_length'],c['chunk_stride']) for x in (tr,va,te)];tok=AutoTokenizer.from_pretrained(c['base_model'],trust_remote_code=True);tok.padding_side='left';tok.pad_token=tok.pad_token or tok.eos_token;qwen=AutoModelForCausalLM.from_pretrained(c['base_model'],torch_dtype=torch.bfloat16,trust_remote_code=True).to(dev).eval();[p.requires_grad_(False) for p in qwen.parameters()];g=torch.Generator().manual_seed(c['target_projection_seed']);proj_cpu=torch.randn(qwen.config.hidden_size,c['observation_dim'],generator=g)/math.sqrt(qwen.config.hidden_size);cache=base.cache_embeddings(tr+va+te,qwen,tok,proj_cpu,dev,c);proj=proj_cpu.to(dev);m=GaussianRSSM(c,qwen.config.hidden_size).to(dev);opt=torch.optim.AdamW(m.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay']);tl=DataLoader(base.Data(tc,cache),batch_size=c['train_batch_size'],shuffle=True,collate_fn=base.collate);vl=DataLoader(base.Data(vc[:c['eval_batch_size']] if a.smoke_steps else vc,cache),batch_size=c['eval_batch_size'],collate_fn=base.collate);commit=a.source_commit or subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();seeds=json.loads(Path(c['eval_seeds_path']).read_text())['seeds'];(root/'resolved_config.yaml').write_text(yaml.safe_dump({**c,'git_commit':commit}));(root/'splits.json').write_text(json.dumps({'train_transitions':len(tr),'val_transitions':len(va),'test_transitions':len(te),'steps_per_epoch':len(tl)},indent=2));print(json.dumps({'trainable_parameters':sum(p.numel() for p in m.parameters()),'steps_per_epoch':len(tl)}),flush=True)
 try:
  from comet_ml import Experiment;comet=Experiment(workspace=c['comet_workspace'],project_name=c['comet_project'],auto_output_logging='simple');comet.set_name(a.run_name);comet.log_parameters({**c,'git_commit':commit})
 except Exception:comet=None
 def log_validation(s):
  met=validate(m,qwen,tok,vl,dev,c);(root/f'val_step_{s}.json').write_text(json.dumps(met,indent=2));print(json.dumps({'step':s,**met}),flush=True);comet and comet.log_metrics(met,step=s)
 def log_env(s,mode):
  prefix={'base':'env_base','zero':'env_zero_latent','learned':'env_eval'}[mode]
  result=production_env_eval(qwen,tok,m,proj,dev,c,seeds,mode);(root/f'{prefix}_step_{s}.json').write_text(json.dumps(result,indent=2));met={prefix+'/'+k:v for k,v in result.items() if k!='episodes'};print(json.dumps({'step':s,**met}),flush=True);comet and comet.log_metrics(met,step=s)
 # Same fixed 32 seeds, parser and prompt: base once, then zero/learned at every cadence.
 log_env(0,'base');log_env(0,'zero');log_env(0,'learned');log_validation(0)
 step=0
 for epoch in range(c['epochs']):
  for b in tl:
   e,act=b['e'].to(dev),b['a'].to(dev);h,z,ps,qs=rollout(m,e,act,True);obs=F.mse_loss(m.decode(h,z),e);pm,psd,pl=torch.cat([x[0] for x in ps]),torch.cat([x[1] for x in ps]),torch.cat([x[2] for x in ps]);qm,qsd,ql=torch.cat([x[0] for x in qs[1:]]),torch.cat([x[1] for x in qs[1:]]),torch.cat([x[2] for x in qs[1:]]);raw=kl(qm,qsd,pm,psd);kd=kl(qm.detach(),qsd.detach(),pm,psd).mean();kr=kl(qm,qsd,pm.detach(),psd.detach()).mean();used=torch.clamp(raw,min=c['free_nats']).mean();rssm=c['kl_balance']*kd+(1-c['kl_balance'])*kr;qd=qwen_dist(qwen,tok,m,b['rows'],h[:,:-1],z[:,:-1],dev,c);qmu,qstd= torch.cat([x[0] for x in qd]),torch.cat([x[1] for x in qd]);qkl=kl(qm.detach(),qsd.detach(),qmu,qstd).mean();loss=c['obs_weight']*obs+c['rssm_weight']*rssm+c['qwen_weight']*qkl;opt.zero_grad();loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),c['max_grad_norm']));opt.step();step+=1
   if step==1:
    sh={'mu_prior':list(pm[:1].shape),'logstd_prior':list(pl[:1].shape),'mu_post':list(qm[:1].shape),'logstd_post':list(ql[:1].shape),'z_sample':list(z[:,0].shape),'mu_qwen':list(qmu[:1].shape),'logstd_qwen':list(qd[0][2][:1].shape),'soft_tokens':list(m.tokens(h[:,0],z[:,0]).reshape(len(e),c['soft_tokens'],-1).shape)};print(json.dumps({'shapes':sh,'kl_q_q':float(kl(qm,qsd,qm,qsd).mean()),'qwen_frozen_grad':any(p.grad is not None for p in qwen.parameters())}),flush=True)
   met={'train/loss':float(loss),'train/obs_mse':float(obs),'train/kl_raw':float(raw.mean()),'train/kl_dyn':float(kd),'train/kl_rep':float(kr),'train/kl_used':float(used),'train/kl_qwen':float(qkl),'train/grad_norm':gn};print(json.dumps({'step':step,**met}),flush=True);comet and comet.log_metrics(met,step=step)
   if not a.smoke_steps and step%int(c['env_eval_every_steps'])==0:
    log_env(step,'zero');log_env(step,'learned');log_validation(step)
   if (a.smoke_steps and step>=a.smoke_steps) or step>=c['max_optimizer_steps']:
    log_validation(step);torch.save({'model':m.state_dict(),'optimizer':opt.state_dict()},root/'final.pt');comet and comet.end();return
if __name__=='__main__':main()
