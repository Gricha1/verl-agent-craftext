#!/usr/bin/env python3
"""Offline deterministic RSSM (h,z) with frozen-Qwen world-state soft tokens."""
from __future__ import annotations
import argparse, hashlib, json, math, random, subprocess
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

ACTIONS=("UP","DOWN","LEFT","RIGHT")
def cli():
 p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--run-name",required=True); p.add_argument("--device",default="cuda:0"); p.add_argument("--smoke-steps",type=int,default=0); p.add_argument("--source-commit",default=None); return p.parse_args()
def split(path,seed):
 rows=[json.loads(x) for x in (Path(path)/"transitions.jsonl").open(encoding="utf-8")]; ids=sorted({int(x["episode_id"]) for x in rows}); random.Random(seed).shuffle(ids); n=len(ids); tr=set(ids[:int(.8*n)]); va=set(ids[int(.8*n):int(.9*n)]); return ([x for x in rows if int(x["episode_id"]) in tr],[x for x in rows if int(x["episode_id"]) in va],[x for x in rows if int(x["episode_id"]) not in tr|va])
def chunks(rows,T,stride):
 d=defaultdict(list)
 for r in rows:d[int(r["episode_id"])].append(r)
 out=[]
 for rs in d.values():
  rs.sort(key=lambda x:int(x["step"]))
  for i in range(0,len(rs)-T+1,stride):
   c=rs[i:i+T]; s=[int(x["step"]) for x in c]
   if s!=list(range(s[0],s[0]+T)):raise ValueError("non-contiguous episode chunk")
   out.append(c)
 return out
def obs_text(tok,task,obs): return tok.apply_chat_template([{"role":"user","content":f"Task:\n{task}\n\nCurrent observation:\n{obs}\n\nObservation representation:"}],tokenize=False,add_generation_prompt=True)
def actor_prefix(tok,r): return tok.apply_chat_template([{"role":"user","content":str(r["actor_prompt_t"])}],tokenize=False,add_generation_prompt=True)
def actor_full(tok,r): return tok.apply_chat_template([{"role":"user","content":str(r["actor_prompt_t"])},{"role":"assistant","content":f"<action>{r['action_t']}</action>"}],tokenize=False,add_generation_prompt=False)
def final_hidden(model,tok,texts,dev,mx):
 b=tok(texts,return_tensors="pt",padding=True,truncation=True,max_length=mx,add_special_tokens=False).to(dev)
 with torch.no_grad(): h=model(**b,output_hidden_states=True,use_cache=False).hidden_states[-1]
 if not b["attention_mask"][:,-1].bool().all():raise RuntimeError("left-padding final token invariant failed")
 return h[:,-1].float().cpu()
def nce(p,z,temp,instance=False):
 p=F.normalize(p.float(),dim=-1); z=F.normalize(z.float(),dim=-1); logits=p@z.T/temp
 same=torch.eye(len(z),device=z.device,dtype=torch.bool) if instance else torch.isclose(z@z.T,torch.ones_like(logits),atol=1e-6)
 return (torch.logsumexp(logits,1)-torch.logsumexp(logits.masked_fill(~same,float("-inf")),1)).mean(),logits,same
def retrieval(p,z,temp):
 loss,l,s=nce(p,z,temp); rank=l.argsort(descending=True,dim=1); pn=F.normalize(p.float(),dim=-1); zn=F.normalize(z.float(),dim=-1); mat=pn@zn.T
 return {"infonce":float(loss),"retrieval_top1":float(s.gather(1,rank[:,:1]).any(1).float().mean()),"retrieval_top5":float(s.gather(1,rank[:,:min(5,len(z))]).any(1).float().mean()),"positive_cosine":float(F.cosine_similarity(pn,zn).mean()),"negative_cosine":float(mat[~torch.eye(len(z),device=z.device,dtype=torch.bool)].mean()) if len(z)>1 else float("nan")}
def cache_embeddings(rows,model,tok,proj,dev,cfg):
 keys={}
 for r in rows: keys.setdefault((str(r["task"]),str(r["observation_t"])),None); keys.setdefault((str(r["task"]),str(r["observation_t_plus_1"])),None)
 ks=list(keys); bs=int(cfg["target_encode_batch_size"])
 for i in range(0,len(ks),bs):
  part=ks[i:i+bs]; values=final_hidden(model,tok,[obs_text(tok,a,b) for a,b in part],dev,int(cfg["max_sequence_length"]))@proj
  keys.update(zip(part,values))
  if i+bs>=len(ks):print(json.dumps({"unique_frozen_observations":len(ks)}),flush=True)
 return keys
class Data(Dataset):
 def __init__(self,cs,cache):self.cs,self.cache=cs,cache
 def __len__(self):return len(self.cs)
 def __getitem__(self,i):
  rs=self.cs[i]; task=str(rs[0]["task"]); es=[self.cache[(task,str(rs[0]["observation_t"]))]]+[self.cache[(task,str(r["observation_t_plus_1"]))] for r in rs]
  return {"e":torch.stack(es),"a":torch.tensor([ACTIONS.index(str(r["action_t"])) for r in rs]),"changed":torch.tensor([str(r["observation_t"])!=str(r["observation_t_plus_1"]) for r in rs]),"rows":rs}
def collate(xs):return {"e":torch.stack([x["e"] for x in xs]),"a":torch.stack([x["a"] for x in xs]),"changed":torch.stack([x["changed"] for x in xs]),"rows":[x["rows"] for x in xs]}
class RSSM(nn.Module):
 def __init__(self,cfg,dmodel):
  super().__init__(); H,Z,E,A=int(cfg["latent_h_dim"]),int(cfg["latent_z_dim"]),int(cfg["observation_dim"]),int(cfg["action_embedding_dim"]); self.act=nn.Embedding(4,A); self.transition=nn.GRUCell(Z+A,H); self.prior=nn.Sequential(nn.Linear(H,Z),nn.GELU(),nn.Linear(Z,Z)); self.posterior=nn.Sequential(nn.Linear(H+E,Z),nn.GELU(),nn.Linear(Z,Z)); self.decoder=nn.Sequential(nn.Linear(H+Z,H),nn.GELU(),nn.Linear(H,E)); self.projector=nn.Sequential(nn.Linear(H+Z,H),nn.GELU(),nn.Linear(H,int(cfg["soft_tokens"])*dmodel)); self.qhead=nn.Sequential(nn.Linear(dmodel,H),nn.GELU(),nn.Linear(H,Z))
 def initial(self,e):
  h=torch.zeros(len(e),self.transition.hidden_size,device=e.device,dtype=e.dtype); return h,self.posterior(torch.cat((h,e),-1))
 def step(self,h,z,a,e):
  hn=self.transition(torch.cat((z,self.act(a)),-1),h); zp=self.prior(hn); zq=self.posterior(torch.cat((hn,e),-1)); return hn,zp,zq
 def decode(self,h,z):return self.decoder(torch.cat((h,z),-1))
 def tokens(self,h,z):return self.projector(torch.cat((h,z),-1))
def observe(m,e,a):
 h,z=m.initial(e[:,0]); hs=[h]; zs=[z]; ps=[]
 for t in range(a.shape[1]):h,p,z=m.step(h,z,a[:,t],e[:,t+1]); hs.append(h); ps.append(p); zs.append(z)
 return torch.stack(hs,1),torch.stack(ps,1),torch.stack(zs,1)
def qwen_forward(qwen,tok,m,rows,h,z,dev,cfg,mode="learned"):
 # One frozen-Qwen forward per sequence time; action remains strictly after the soft tokens.
 B,T=h.shape[:2]; answers=[]; emb_layer=qwen.get_input_embeddings(); K=int(cfg["soft_tokens"])
 for t in range(T):
  pieces=[]; lens=[]
  states=m.tokens(h[:,t],z[:,t]).reshape(B,K,-1)
  if mode=="zero":states=torch.zeros_like(states)
  elif mode=="shuffled":states=states[torch.randperm(B,device=dev)]
  for b in range(B):
   r=rows[b][t]; pre=tok(actor_prefix(tok,r),add_special_tokens=False)["input_ids"]; full=tok(actor_full(tok,r),add_special_tokens=False)["input_ids"]
   if full[:len(pre)]!=pre:raise RuntimeError("production action prefix mismatch")
   ids=torch.tensor(pre+full[len(pre):],device=dev); pe=emb_layer(ids[:len(pre)]); ae=emb_layer(ids[len(pre):]); x=torch.cat((pe,states[b].to(pe.dtype),ae),0); pieces.append(x); lens.append(len(x))
  L=max(lens); batch=torch.stack([torch.cat((x,torch.zeros(L-len(x),x.shape[-1],device=dev,dtype=x.dtype)),0) for x in pieces]); mask=torch.tensor([[1]*n+[0]*(L-n) for n in lens],device=dev)
  out=qwen(inputs_embeds=batch,attention_mask=mask,output_hidden_states=True,use_cache=False).hidden_states[-1]; answers.append(m.qhead(out[torch.arange(B,device=dev),torch.tensor(lens,device=dev)-1].float()))
 return torch.stack(answers,1)
def grad_norm(module):return float(torch.sqrt(sum((p.grad.detach().float().norm()**2 for p in module.parameters() if p.grad is not None),torch.zeros((),device=next(module.parameters()).device))))
@torch.no_grad()
def validate(m,qwen,tok,loader,dev,cfg):
 m.eval(); values={k:[] for k in ("obs","dyn","qwen","qwen_zero","qwen_shuffled")}; openp={x:[] for x in (1,2,4,8,16)}; opent={x:[] for x in (1,2,4,8,16)}; changed=[]; hvals=[]; zvals=[]; pvals=[]; toks=[]
 for b in loader:
  e,a=b["e"].to(dev),b["a"].to(dev); h,p,z=observe(m,e,a); de=m.decode(h,z); op=de[:,1:].flatten(0,1); ot=e[:,1:].flatten(0,1); values["obs"].append((op,ot)); values["dyn"].append((p.flatten(0,1),z[:,1:].flatten(0,1))); q=qwen_forward(qwen,tok,m,b["rows"],h[:,:-1],z[:,:-1],dev,cfg); values["qwen"].append((q.flatten(0,1),z[:,1:].flatten(0,1))); values["qwen_zero"].append((qwen_forward(qwen,tok,m,b["rows"],h[:,:-1],z[:,:-1],dev,cfg,"zero").flatten(0,1),z[:,1:].flatten(0,1))); values["qwen_shuffled"].append((qwen_forward(qwen,tok,m,b["rows"],h[:,:-1],z[:,:-1],dev,cfg,"shuffled").flatten(0,1),z[:,1:].flatten(0,1))); changed.append(b["changed"].to(dev).flatten()); hvals.append(h[:,1:].flatten(0,1)); zvals.append(z[:,1:].flatten(0,1)); pvals.append(p.flatten(0,1)); toks.append(m.tokens(h[:,:-1],z[:,:-1]).flatten(0,1))
  hi,zi=m.initial(e[:,0])
  for t in range(a.shape[1]): hi= m.transition(torch.cat((zi,m.act(a[:,t])),-1),hi); zi=m.prior(hi); H=t+1
  # Need each requested horizon from the same initial state.
  for H in openp:
   hi,zi=m.initial(e[:,0])
   for t in range(H):hi=m.transition(torch.cat((zi,m.act(a[:,t])),-1),hi); zi=m.prior(hi)
   openp[H].append(m.decode(hi,zi)); opent[H].append(e[:,H])
 out={}
 for name,pairs in values.items():
  x,y=torch.cat([v[0] for v in pairs]),torch.cat([v[1] for v in pairs]); pref="val/"+name.replace("qwen_","qwen_"); out.update({pref+"_"+k:v for k,v in retrieval(x,y,float(cfg["infonce_temperature"])).items()})
 # Canonical requested names are the learned branch.
 for k,v in list(out.items()):
  if k.startswith("val/qwen_") and not k.startswith("val/qwen_zero") and not k.startswith("val/qwen_shuffled"):out[k.replace("val/qwen_","val/qwen_")]=v
 for H in openp:out.update({f"open_loop_h{H}/"+k:v for k,v in retrieval(torch.cat(openp[H]),torch.cat(opent[H]),float(cfg["infonce_temperature"])).items() if k!="infonce"})
 H=torch.cat(hvals); Z=torch.cat(zvals); P=torch.cat(pvals); S=torch.cat(toks); out.update({"latent/h_norm":float(H.norm(dim=-1).mean()),"latent/h_std":float(H.std()),"latent/posterior_norm":float(Z.norm(dim=-1).mean()),"latent/posterior_std":float(Z.std()),"latent/prior_norm":float(P.norm(dim=-1).mean()),"latent/prior_std":float(P.std()),"soft_tokens/norm":float(S.norm(dim=-1).mean()),"soft_tokens/std":float(S.std())})
 return out
def main():
 a=cli(); c=yaml.safe_load(Path(a.config).read_text()); seed=int(c["seed"]); random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed); dev=torch.device(a.device); root=Path(c["output_root"])/a.run_name;root.mkdir(parents=True,exist_ok=False)
 tr,va,te=split(c["dataset_dir"],int(c["split_seed"])); tc,vc,ec=chunks(tr,int(c["sequence_length"]),int(c["chunk_stride"])),chunks(va,int(c["sequence_length"]),int(c["chunk_stride"])),chunks(te,int(c["sequence_length"]),int(c["chunk_stride"])); commit=a.source_commit or subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(); info={"train_transitions":len(tr),"val_transitions":len(va),"test_transitions":len(te),"train_sequences":len(tc),"val_sequences":len(vc),"test_sequences":len(ec)};(root/"splits.json").write_text(json.dumps(info,indent=2));(root/"resolved_config.yaml").write_text(yaml.safe_dump({**c,"git_commit":commit,"run_name":a.run_name},sort_keys=True))
 tok=AutoTokenizer.from_pretrained(c["base_model"],trust_remote_code=True);tok.padding_side="left";tok.pad_token=tok.pad_token or tok.eos_token;qwen=AutoModelForCausalLM.from_pretrained(c["base_model"],torch_dtype=torch.bfloat16,trust_remote_code=True).to(dev).eval();[p.requires_grad_(False) for p in qwen.parameters()]; g=torch.Generator(device="cpu").manual_seed(int(c["target_projection_seed"]));proj=torch.randn(qwen.config.hidden_size,int(c["observation_dim"]),generator=g)/math.sqrt(qwen.config.hidden_size);torch.save(proj,root/"fixed_target_projection.pt"); cache=cache_embeddings(tr+va+te,qwen,tok,proj,dev,c);torch.save(cache,root/"frozen_target_embeddings.pt");m=RSSM(c,qwen.config.hidden_size).to(dev);opt=torch.optim.AdamW(m.parameters(),lr=float(c["learning_rate"]),weight_decay=float(c["weight_decay"]));tl=DataLoader(Data(tc,cache),batch_size=int(c["train_batch_size"]),shuffle=True,collate_fn=collate);eval_bs=int(c.get("eval_batch_size",c["train_batch_size"]));active_vc=vc[:eval_bs] if a.smoke_steps else vc;vl=DataLoader(Data(active_vc,cache),batch_size=eval_bs,collate_fn=collate);el=DataLoader(Data(ec,cache),batch_size=eval_bs,collate_fn=collate); params={"trainable_parameters":sum(p.numel() for p in m.parameters()),"steps_per_epoch":len(tl),"effective_batch_sequences":int(c["train_batch_size"]),"eval_batch_sequences":eval_bs};(root/"parameter_counts.json").write_text(json.dumps(params,indent=2))
 try:
  from comet_ml import Experiment; comet=Experiment(workspace=c["comet_workspace"],project_name=c["comet_project"],auto_output_logging="simple");comet.set_name(a.run_name);comet.add_tags(["rssm","soft_tokens","frozen_qwen","offline","infonce"]);comet.log_parameters({**c,**info,**params,"git_commit":commit,"dataset_hash":hashlib.sha256((Path(c["dataset_dir"])/"transitions.jsonl").read_bytes()).hexdigest()})
 except Exception as e:comet=None;print(f"[comet disabled] {e}",flush=True)
 def save(name):d=root/name;d.mkdir(exist_ok=True);torch.save({"rssm":m.state_dict(),"optimizer":opt.state_dict(),"config":c},d/"state.pt")
 step=0;bestq=bestopen=-float("inf");stop=False
 for epoch in range(int(c["epochs"])):
  m.train()
  for b in tl:
   e,act=b["e"].to(dev),b["a"].to(dev);h,p,z=observe(m,e,act); op=m.decode(h,z);obs_loss,_,_=nce(op.flatten(0,1),e.flatten(0,1),float(c["infonce_temperature"]));dyn_loss,_,_=nce(p.flatten(0,1),z[:,1:].detach().flatten(0,1),float(c["infonce_temperature"]),instance=True);q=qwen_forward(qwen,tok,m,b["rows"],h[:,:-1],z[:,:-1],dev,c);q_loss,_,_=nce(q.flatten(0,1),z[:,1:].detach().flatten(0,1),float(c["infonce_temperature"]),instance=True);loss=float(c["obs_weight"])*obs_loss+float(c["dyn_weight"])*dyn_loss+float(c["qwen_weight"])*q_loss;opt.zero_grad(set_to_none=True);loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),float(c["max_grad_norm"])));opt.step();step+=1
   if step==1:
    shapes={"e_t":list(e.shape),"h_t":list(h[:,0].shape),"z_post":list(z[:,0].shape),"z_prior":list(p[:,0].shape),"state_concat":[len(e),1024],"projected_tokens":list(m.tokens(h[:,0],z[:,0]).reshape(len(e),int(c["soft_tokens"]),-1).shape),"qwen_predicted_z":list(q[:,0].shape)}; grads={x:grad_norm(getattr(m,x)) for x in ("transition","prior","posterior","decoder","projector","qhead")};grads["frozen_qwen_has_grad"]=any(x.grad is not None for x in qwen.parameters());print(json.dumps({"shapes":shapes,"gradients":grads}),flush=True);(root/"smoke.json").write_text(json.dumps({"shapes":shapes,"gradients":grads},indent=2))
   metrics={"train/loss":float(loss.detach()),"train/obs_infonce":float(obs_loss.detach()),"train/dyn_infonce":float(dyn_loss.detach()),"train/qwen_infonce":float(q_loss.detach()),"train/grad_norm":gn};print(json.dumps({"step":step,**metrics}),flush=True);comet and comet.log_metrics(metrics,step=step)
   if a.smoke_steps and step>=a.smoke_steps:stop=True;break
   if step>=int(c["max_optimizer_steps"]):stop=True;break
  val=validate(m,qwen,tok,vl,dev,c);print(json.dumps({"epoch":epoch,"step":step,**val}),flush=True);comet and comet.log_metrics(val,step=step,epoch=epoch)
  if val["val/qwen_retrieval_top1"]>bestq:bestq=val["val/qwen_retrieval_top1"];save("best_qwen")
  if val["open_loop_h4/retrieval_top1"]>bestopen:bestopen=val["open_loop_h4/retrieval_top1"];save("best_open_loop_h4")
  if stop:break
 test=validate(m,qwen,tok,el,dev,c);save("final");comet and comet.log_metrics({"test/"+k:v for k,v in test.items()},step=step);comet and comet.end();print(json.dumps({"final_step":step,**test}),flush=True)
if __name__=="__main__":main()
