#!/usr/bin/env python3
"""Reward-free latent WM pretraining with periodic production-actor evaluation."""
from __future__ import annotations
import argparse, hashlib, json, math, random, subprocess, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from agent_system.environments.env_manager import CagedCraftextEnvironmentManager
from agent_system.environments.env_package.caged_craftext.envs import CagedCraftextWorker
from agent_system.environments.env_package.caged_craftext.projection import craftext_projection

class Rows(Dataset):
    def __init__(self, rows): self.rows=rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

class LocalVector:
    """Production CrafText worker/manager API without Ray in an offline trainer."""
    def __init__(self, cfg, seed):
        self.worker=CagedCraftextWorker(seed=seed, env_kwargs={"config_name":cfg.env.craftext_settings,"use_debug_square_map":True,"observation_type":"ascii","encode_form":"embedding"})
        self.rng=np.random.RandomState(seed)
    def reset(self):
        o,i=self.worker.reset(scenario_idx=int(self.rng.choice(np.arange(3))), return_render=False); return [o],[i]
    def step(self, actions):
        o,r,d,i=self.worker.step(int(actions[0]), return_render=False); return [o],np.asarray([r]),np.asarray([d]),[i]
    def close(self):
        if hasattr(self.worker,"close"): self.worker.close()

def args():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--loss-type",choices=("mse","infonce"),required=True); p.add_argument("--run-name",required=True); p.add_argument("--device",default="cuda:0"); return p.parse_args()

def split(dataset, seed):
    rows=[json.loads(x) for x in (Path(dataset)/"transitions.jsonl").open(encoding="utf-8")]; ids=sorted({int(x["episode_id"]) for x in rows}); random.Random(seed).shuffle(ids); n=len(ids); tr=set(ids[:int(.8*n)]); va=set(ids[int(.8*n):int(.9*n)]); return [x for x in rows if x["episode_id"] in tr],[x for x in rows if x["episode_id"] in va],[x for x in rows if x["episode_id"] not in tr|va]

def actor_text(tok,row):
    return tok.apply_chat_template([{"role":"user","content":str(row["actor_prompt_t"])},{"role":"assistant","content":f"<action>{row['action_t']}</action>"}],tokenize=False,add_generation_prompt=False)
def target_text(tok,row):
    prompt=f"Task:\n{row['task']}\n\nCurrent observation:\n{row['observation_t_plus_1']}\n\nObservation representation:"
    return tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True)

def final_hidden(model,tok,texts,device,grad,max_len,require_eos):
    b=tok(texts,return_tensors="pt",padding=True,truncation=True,max_length=max_len,add_special_tokens=False).to(device)
    with (torch.enable_grad() if grad else torch.no_grad()): h=model(**b,output_hidden_states=True,use_cache=False).hidden_states[-1]
    mask=b["attention_mask"].bool(); idx=[]
    for i in range(len(texts)):
        if require_eos:
            hits=torch.where((b["input_ids"][i]==tok.eos_token_id)&mask[i])[0]
            if len(hits)==0: raise RuntimeError("Teacher-forced action sequence lacks EOS/<|im_end|>")
            idx.append(hits[-1])
        else:
            # tokenizer.padding_side is left: final position must be a real token, never PAD.
            if not mask[i,-1]: raise RuntimeError("Target final hidden state is PAD")
            idx.append(torch.tensor(mask.shape[1]-1,device=device))
    return h[torch.arange(len(texts),device=device),torch.stack(idx)]

def infonce(p,z,t):
    p=F.normalize(p.float(),dim=-1); z=F.normalize(z.float(),dim=-1); logits=p@z.T/t; same=torch.isclose(z@z.T,torch.ones_like(logits),atol=1e-6); return (torch.logsumexp(logits,1)-torch.logsumexp(logits.masked_fill(~same,float("-inf")),1)).mean(),logits,same

def env_cfg(seed):
    return OmegaConf.create({"env":{"env_name":"caged_craftext/CagedCraftextEnv","craftext_settings":"debug_square_8x8","seed":seed,"max_steps":50,"history_length":50,"reasoning_history_length":3,"enable_reasoning":True,"prompt_template_type":"single_token_action_reasoning","store_raw_reasoning_on_missing_action_tag":True,"observation_type":"ascii","auto_reset":False,"rollout":{"n":1},"resources_per_worker":{"num_cpus":0.03},"use_jax_gpu":False,"jax_gpu_fraction":0.0,"use_optimistic_parallel":True,"optimistic_reset_ratio":8,"use_ray_text_render_workers":False},"data":{"train_batch_size":1,"val_batch_size":1}})

def actor_eval(model,tok,device,seeds,max_new):
    """No-gradient shuffled 8x8 evaluation using production prompt, parser and memory."""
    model.eval(); out=[]
    for seed in seeds:
        cfg=env_cfg(int(seed)); vec=LocalVector(cfg,int(seed)); env=CagedCraftextEnvironmentManager(vec,craftext_projection,cfg); obs,_=env.reset({}); reward=0.; valid=errors=repeated=0; acts=[]; trace=[]
        try:
            for step in range(50):
                text=tok.apply_chat_template([{"role":"user","content":obs["text"][0]}],tokenize=False,add_generation_prompt=True); x=tok(text,return_tensors="pt",add_special_tokens=False).to(device); g=torch.Generator(device=device).manual_seed(int(seed)*1000+step)
                with torch.no_grad(): y=model.generate(**x,do_sample=True,temperature=1.,top_p=1.,top_k=0,max_new_tokens=max_new,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,generator=g)
                response=tok.decode(y[0,x["input_ids"].shape[1]:],skip_special_tokens=True); obs,rs,dones,infos=env.step([response]); info=infos[0]; action=str(info.get("action_name","NOOP")); reward+=float(rs[0]); valid+=int(bool(info.get("is_action_valid",False))); errors+=int(not bool(info.get("is_action_valid",False))); repeated+=int(bool(acts) and acts[-1]==action); acts.append(action); trace.append({"step":step,"response":response,"action":action})
                if bool(dones[0]): break
            out.append({"seed":int(seed),"success":float(bool(infos[0].get("won",False))),"reward":reward,"length":len(acts),"valid":valid/max(1,len(acts)),"errors":errors/max(1,len(acts)),"repeated":repeated/max(1,len(acts)),"trace":trace})
        finally: vec.close()
    mean=lambda k:float(np.mean([x[k] for x in out])); rewards=[x["reward"] for x in out]
    return {"success_rate":mean("success"),"mean_episode_reward":mean("reward"),"median_episode_reward":float(np.median(rewards)),"std_episode_reward":float(np.std(rewards)),"mean_episode_length":mean("length"),"valid_action_rate":mean("valid"),"parse_error_rate":mean("errors"),"repeated_action_rate":mean("repeated"),"episodes":out}

def offline(loader,online,frozen,tok,head,proj,device,cfg):
    online.eval(); head.eval(); allm=[]
    with torch.no_grad():
        for rows in loader:
            p=head(final_hidden(online,tok,[actor_text(tok,r) for r in rows],device,False,int(cfg["max_sequence_length"]),True).float()); z=final_hidden(frozen,tok,[target_text(tok,r) for r in rows],device,False,int(cfg["max_sequence_length"]),False).to(proj.dtype)@proj; _,logits,same=infonce(p,z,float(cfg["infonce_temperature"])); rank=logits.argsort(descending=True); allm.append((F.mse_loss(p.float(),z.float()),F.cosine_similarity(p.float(),z.float(),dim=-1).mean(),same.gather(1,rank[:,:1]).any(1).float().mean(),same.gather(1,rank[:,:5]).any(1).float().mean()))
    return dict(zip(("mse","cosine","retrieval_top1","retrieval_top5"),[float(torch.stack([m[i] for m in allm]).mean()) for i in range(4)]))

def main():
    a=args(); cfg=yaml.safe_load(Path(a.config).read_text()); seed=int(cfg["seed"]); random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); device=torch.device(a.device); root=Path(cfg["output_root"])/a.run_name; root.mkdir(parents=True,exist_ok=False)
    train,val,test=split(cfg["dataset_dir"],seed); seeds=json.loads(Path(cfg["eval_seeds_path"]).read_text())["seeds"]; commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(); (root/"resolved_config.yaml").write_text(yaml.safe_dump({**cfg,"loss_type":a.loss_type,"run_name":a.run_name},sort_keys=True)); (root/"splits.json").write_text(json.dumps({"train":len(train),"val":len(val),"test":len(test)}))
    tok=AutoTokenizer.from_pretrained(cfg["base_model"],trust_remote_code=True); tok.padding_side="left"; tok.pad_token=tok.pad_token or tok.eos_token
    frozen=AutoModelForCausalLM.from_pretrained(cfg["base_model"],torch_dtype=torch.bfloat16,trust_remote_code=True).to(device).eval(); [setattr(p,"requires_grad",False) for p in frozen.parameters()]
    online=get_peft_model(AutoModelForCausalLM.from_pretrained(cfg["base_model"],torch_dtype=torch.bfloat16,trust_remote_code=True),LoraConfig(r=int(cfg["lora_rank"]),lora_alpha=int(cfg["lora_alpha"]),lora_dropout=float(cfg["lora_dropout"]),task_type=TaskType.CAUSAL_LM,target_modules=["q_proj","k_proj","v_proj","o_proj"])).to(device); head=nn.Sequential(nn.Linear(online.config.hidden_size,int(cfg["predictor_hidden"])),nn.GELU(),nn.Linear(int(cfg["predictor_hidden"]),int(cfg["latent_dim"]))).to(device)
    g=torch.Generator(device="cpu").manual_seed(int(cfg["target_projection_seed"])); proj=torch.randn(online.config.hidden_size,int(cfg["latent_dim"]),generator=g,dtype=torch.float32)/math.sqrt(online.config.hidden_size); torch.save(proj,root/"fixed_target_projection.pt"); proj=proj.to(device,dtype=torch.bfloat16); counts={"total_model_params":sum(p.numel() for p in online.parameters())+sum(p.numel() for p in head.parameters()),"lora_trainable_params":sum(p.numel() for n,p in online.named_parameters() if p.requires_grad and "lora_" in n),"predictor_trainable_params":sum(p.numel() for p in head.parameters()),"total_trainable_params":sum(p.numel() for p in online.parameters() if p.requires_grad)+sum(p.numel() for p in head.parameters()),"frozen_params":sum(p.numel() for p in online.parameters() if not p.requires_grad)}; (root/"parameter_counts.json").write_text(json.dumps(counts,indent=2)); print(json.dumps(counts),flush=True)
    opt=torch.optim.AdamW(list(online.parameters())+list(head.parameters()),lr=float(cfg["learning_rate"]),weight_decay=float(cfg["weight_decay"])); bs=int(cfg["train_batch_size"]); accum=int(cfg["gradient_accumulation_steps"]); tr=DataLoader(Rows(train),batch_size=bs,shuffle=True,collate_fn=lambda x:x); va=DataLoader(Rows(val),batch_size=bs,collate_fn=lambda x:x)
    try:
        from comet_ml import Experiment
        comet=Experiment(workspace=cfg["comet_workspace"],project_name=cfg["comet_project"],auto_output_logging="simple"); comet.set_name(a.run_name); comet.add_tags(["latent_wm","reward_free","random_transitions","actor_matched_optimization","periodic_env_eval",a.loss_type]); comet.log_parameters({**cfg,**counts,"loss_type":a.loss_type,"git_commit":commit,"effective_optimizer_batch":bs*accum,"dataset_hash":hashlib.sha256((Path(cfg["dataset_dir"])/"transitions.jsonl").read_bytes()).hexdigest()})
    except Exception as e: comet=None; print(f"[comet disabled] {e}",flush=True)
    def logeval(step):
        r=actor_eval(online,tok,device,seeds,int(cfg["eval_max_response_length"])); (root/f"env_eval_step_{step}.json").write_text(json.dumps(r,indent=2)); metrics={"env_eval/"+k:v for k,v in r.items() if k!="episodes"}; print(json.dumps({"step":step,**metrics}),flush=True); comet and comet.log_metrics(metrics,step=step)
    logeval(0); step=0; best=float("inf"); opt.zero_grad(set_to_none=True)
    for epoch in range(int(cfg["epochs"])):
        online.train(); head.train()
        for i,rows in enumerate(tr):
            p=head(final_hidden(online,tok,[actor_text(tok,r) for r in rows],device,True,int(cfg["max_sequence_length"]),True).float())
            with torch.no_grad(): z=final_hidden(frozen,tok,[target_text(tok,r) for r in rows],device,False,int(cfg["max_sequence_length"]),False).to(proj.dtype)@proj
            if a.loss_type=="mse": loss=F.mse_loss(p.float(),z.float()); extra={"train/mse":float(loss.detach())}
            else: loss,logits,_=infonce(p,z,float(cfg["infonce_temperature"])); extra={"train/infonce":float(loss.detach()),"latent/positive_logits":float(logits.diag().mean())}
            (loss/accum).backward()
            if (i+1)%accum==0 or i+1==len(tr):
                grad=float(torch.nn.utils.clip_grad_norm_(list(online.parameters())+list(head.parameters()),1.0)); opt.step(); opt.zero_grad(set_to_none=True); step+=1; metrics={"train/loss":float(loss.detach()),"train/lr":opt.param_groups[0]["lr"],"train/grad_norm":grad,"latent/z_pred_std":float(p.float().std()),"latent/z_target_std":float(z.float().std()),"latent/z_pred_norm":float(p.float().norm(dim=-1).mean()),"latent/z_target_norm":float(z.float().norm(dim=-1).mean()),**extra}; comet and comet.log_metrics(metrics,step=step)
                if step%int(cfg["checkpoint_every_steps"])==0: d=root/f"checkpoint_{step}"; d.mkdir(); online.save_pretrained(d/"adapter"); torch.save({"predictor":head.state_dict(),"optimizer":opt.state_dict(),"step":step},d/"state.pt")
                if step%int(cfg["eval_every_steps"])==0: logeval(step)
        m=offline(va,online,frozen,tok,head,proj,device,cfg); (root/f"val_epoch_{epoch}.json").write_text(json.dumps(m,indent=2)); comet and comet.log_metrics({"val/"+k:v for k,v in m.items()},step=step)
        if m["mse"]<best: best=m["mse"]; d=root/"best"; d.mkdir(exist_ok=True); online.save_pretrained(d/"adapter"); torch.save({"predictor":head.state_dict(),"step":step,"metrics":m},d/"state.pt")
    logeval(step); m=offline(DataLoader(Rows(test),batch_size=bs,collate_fn=lambda x:x),online,frozen,tok,head,proj,device,cfg); (root/"test_metrics.json").write_text(json.dumps(m,indent=2)); online.save_pretrained(root/"final_adapter"); torch.save({"predictor":head.state_dict(),"optimizer":opt.state_dict(),"step":step},root/"final_state.pt"); comet and comet.log_metrics({"test/"+k:v for k,v in m.items()},step=step); comet and comet.end(); print(json.dumps({"run_dir":str(root),"steps":step,"test":m}),flush=True)
if __name__=="__main__": main()
