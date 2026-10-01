#!/usr/bin/env python3
"""Matched reward-free latent dynamics pretraining: MSE or multi-positive InfoNCE."""
from __future__ import annotations

import argparse, hashlib, json, math, os, random, subprocess, time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


class Transitions(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--loss-type", choices=("mse","infonce"), required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def load_rows(dataset_dir: Path, seed: int):
    rows=[json.loads(line) for line in (dataset_dir/"transitions.jsonl").open(encoding="utf-8")]
    episodes=sorted({int(r["episode_id"]) for r in rows})
    rng=random.Random(seed); rng.shuffle(episodes)
    n=len(episodes); train=set(episodes[:int(.8*n)]); val=set(episodes[int(.8*n):int(.9*n)])
    return [r for r in rows if r["episode_id"] in train], [r for r in rows if r["episode_id"] in val], [r for r in rows if r["episode_id"] not in train|val]


def target_prompt(row):
    # Target deliberately excludes action, reward and hidden simulator state.
    return "Task:\n"+str(row["task"])+"\n\nCurrent observation:\n"+str(row["observation_t_plus_1"])+"\n\nObservation representation:"


def actor_with_forced_action(tok, row):
    # actor_prompt_t was built by the production CagedCraftextEnvironmentManager.
    return tok.apply_chat_template([
        {"role":"user","content":str(row["actor_prompt_t"])},
        {"role":"assistant","content":f"<action>{row['action_t']}</action>"},
    ], tokenize=False, add_generation_prompt=False)


def target_text(tok, row):
    return tok.apply_chat_template([{"role":"user","content":target_prompt(row)}], tokenize=False, add_generation_prompt=True)


def last_hidden(model, tok, texts, device, grad):
    batch=tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=3072, add_special_tokens=False).to(device)
    ctx=torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        out=model(**batch, output_hidden_states=True, use_cache=False)
        h=out.hidden_states[-1]
        idx=batch["attention_mask"].sum(1)-1
        return h[torch.arange(h.shape[0],device=device),idx]


def make_projection(hidden, latent, seed, path, device):
    g=torch.Generator(device="cpu").manual_seed(seed)
    matrix=torch.randn(hidden,latent,generator=g,dtype=torch.float32)/math.sqrt(hidden)
    torch.save(matrix,path)
    return matrix.to(device=device,dtype=torch.bfloat16)


def multi_positive_infonce(pred,target,temp):
    p=F.normalize(pred.float(),dim=-1); z=F.normalize(target.float(),dim=-1)
    logits=p@z.T/temp
    # Identical target vectors are positives, never false negatives.
    same=torch.isclose((z@z.T), torch.ones_like(logits), atol=1e-6)
    logden=torch.logsumexp(logits,dim=1)
    logpos=torch.logsumexp(logits.masked_fill(~same,float("-inf")),dim=1)
    return (logden-logpos).mean(), logits, same


def evaluate(loader, online, frozen, tok, head, proj, loss_type, temp, device):
    online.eval(); head.eval(); losses=[]; cos=[]; top1=[]; top5=[]
    with torch.no_grad():
      for rows in loader:
        pred=head(last_hidden(online,tok,[actor_with_forced_action(tok,r) for r in rows],device,False).float())
        target=last_hidden(frozen,tok,[target_text(tok,r) for r in rows],device,False).to(proj.dtype)@proj
        mse=F.mse_loss(pred.float(),target.float()); c=F.cosine_similarity(pred.float(),target.float(),dim=-1).mean()
        _, logits, same=multi_positive_infonce(pred,target,temp); ranks=logits.argsort(descending=True)
        top1.append(same.gather(1,ranks[:,:1]).any(1).float().mean()); top5.append(same.gather(1,ranks[:,:5]).any(1).float().mean())
        losses.append(mse); cos.append(c)
    return {"mse":float(torch.stack(losses).mean()),"cosine":float(torch.stack(cos).mean()),"retrieval_top1":float(torch.stack(top1).mean()),"retrieval_top5":float(torch.stack(top5).mean())}


def main():
    args=parse_args(); cfg=yaml.safe_load(Path(args.config).read_text())
    seed=int(cfg["seed"]); random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device=torch.device(args.device); root=Path(cfg["output_root"])/args.run_name; root.mkdir(parents=True,exist_ok=False)
    (root/"resolved_config.yaml").write_text(yaml.safe_dump({**cfg,"loss_type":args.loss_type,"run_name":args.run_name},sort_keys=True))
    commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    train,val,test=load_rows(Path(cfg["dataset_dir"]),seed)
    (root/"splits.json").write_text(json.dumps({"train":len(train),"val":len(val),"test":len(test)}))
    tok=AutoTokenizer.from_pretrained(cfg["base_model"],trust_remote_code=True); tok.padding_side="left"; tok.pad_token=tok.pad_token or tok.eos_token
    frozen=AutoModelForCausalLM.from_pretrained(cfg["base_model"],torch_dtype=torch.bfloat16,trust_remote_code=True).to(device).eval()
    for p in frozen.parameters(): p.requires_grad=False
    base=AutoModelForCausalLM.from_pretrained(cfg["base_model"],torch_dtype=torch.bfloat16,trust_remote_code=True)
    online=get_peft_model(base,LoraConfig(r=int(cfg["lora_rank"]),lora_alpha=int(cfg["lora_alpha"]),lora_dropout=float(cfg["lora_dropout"]),task_type=TaskType.CAUSAL_LM,target_modules=["q_proj","k_proj","v_proj","o_proj"])) .to(device)
    head=nn.Sequential(nn.Linear(online.config.hidden_size,int(cfg["predictor_hidden"])),nn.GELU(),nn.Linear(int(cfg["predictor_hidden"]),int(cfg["latent_dim"]))).to(device)
    proj=make_projection(online.config.hidden_size,int(cfg["latent_dim"]),int(cfg["target_projection_seed"]),root/"fixed_target_projection.pt",device)
    proj_hash=hashlib.sha256((root/"fixed_target_projection.pt").read_bytes()).hexdigest()
    opt=torch.optim.AdamW(list(online.parameters())+list(head.parameters()),lr=float(cfg["learning_rate"]),weight_decay=float(cfg["weight_decay"]))
    train_loader=DataLoader(Transitions(train),batch_size=int(cfg["train_batch_size"]),shuffle=True,collate_fn=lambda x:x)
    val_loader=DataLoader(Transitions(val),batch_size=int(cfg["train_batch_size"]),shuffle=False,collate_fn=lambda x:x)
    try:
      from comet_ml import Experiment
      comet=Experiment(workspace=cfg["comet_workspace"],project_name=cfg["comet_project"],auto_output_logging="simple")
      comet.set_name(args.run_name); comet.add_tags(["latent_wm","reward_free","random_transitions",args.loss_type]); comet.log_parameters({**cfg,"loss_type":args.loss_type,"git_commit":commit,"projection_hash":proj_hash,"dataset_size":len(train)+len(val)+len(test)})
    except Exception as exc: comet=None; print(f"[comet disabled] {exc}",flush=True)
    best=float("inf"); step=0; accum=int(cfg["gradient_accumulation_steps"]); online.train(); head.train()
    for epoch in range(int(cfg["epochs"])):
      opt.zero_grad(set_to_none=True)
      for batch_i,rows in enumerate(train_loader):
        pred=head(last_hidden(online,tok,[actor_with_forced_action(tok,r) for r in rows],device,True).float())
        with torch.no_grad(): target=last_hidden(frozen,tok,[target_text(tok,r) for r in rows],device,False).to(proj.dtype)@proj
        if args.loss_type=="mse": loss=F.mse_loss(pred.float(),target.float()); extra={"train/mse":float(loss.detach())}
        else:
          loss,logits,_=multi_positive_infonce(pred,target,float(cfg["infonce_temperature"])); extra={"train/infonce":float(loss.detach()),"latent/positive_logits":float(logits.diag().mean()),"latent/negative_logits":float((logits.sum()-logits.diag().sum())/(logits.numel()-len(rows)))}
        (loss/accum).backward()
        if (batch_i+1)%accum==0 or batch_i+1==len(train_loader):
          grad=float(torch.nn.utils.clip_grad_norm_(list(online.parameters())+list(head.parameters()),1.0)); opt.step(); opt.zero_grad(set_to_none=True); step+=1
          metrics={"train/loss":float(loss.detach()),"learning_rate":opt.param_groups[0]["lr"],"grad_norm":grad,"latent/z_pred_std":float(pred.float().std()),"latent/z_target_std":float(target.float().std()),**extra}
          if comet: comet.log_metrics(metrics,step=step)
          if step%int(cfg["checkpoint_every_steps"])==0:
            d=root/f"checkpoint_{step}"; d.mkdir(); online.save_pretrained(d/"adapter"); torch.save({"predictor":head.state_dict(),"optimizer":opt.state_dict(),"step":step},d/"state.pt")
      metrics=evaluate(val_loader,online,frozen,tok,head,proj,args.loss_type,float(cfg["infonce_temperature"]),device); (root/f"val_epoch_{epoch}.json").write_text(json.dumps(metrics,indent=2));
      if comet: comet.log_metrics({"val/"+k:v for k,v in metrics.items()},step=step)
      if metrics["mse"]<best: best=metrics["mse"]; d=root/"best"; d.mkdir(exist_ok=True); online.save_pretrained(d/"adapter"); torch.save({"predictor":head.state_dict(),"metrics":metrics},d/"state.pt")
    test_metrics=evaluate(DataLoader(Transitions(test),batch_size=int(cfg["train_batch_size"]),collate_fn=lambda x:x),online,frozen,tok,head,proj,args.loss_type,float(cfg["infonce_temperature"]),device)
    (root/"test_metrics.json").write_text(json.dumps(test_metrics,indent=2)); online.save_pretrained(root/"final_adapter"); torch.save({"predictor":head.state_dict(),"optimizer":opt.state_dict(),"step":step},root/"final_state.pt")
    if comet: comet.log_metrics({"test/"+k:v for k,v in test_metrics.items()},step=step); comet.end()
    print(json.dumps({"run_dir":str(root),"steps":step,"test":test_metrics}),flush=True)

if __name__=="__main__": main()
