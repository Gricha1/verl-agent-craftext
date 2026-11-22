#!/usr/bin/env python3
"""Patch a 4ku/dreamer4 checkout (after patch_dreamer4_heads_init.py) so the phase-2 heads finetune of a Dyna loop keeps its
training state across iterations (safe_rl_nlp.md, rules for agents 15-17, 2026-11-22). Before, every iteration's
train_heads.py --init restarted AdamW, the cosine schedule with the dynamics warmup and the RMS loss normalisers.

train_heads.py
  --state_in <previous iteration's checkpoints/train_state.pt>: restores AdamW (both groups: heads, dynamics), the global
      step and the LossCombiner normalisers; the parameter groups and shapes must match.
  --total_steps N: the cosine (and the dynamics warmup) run over the global step, N steps over all loop iterations
      (default --steps), so the iterations form one schedule.
  writes <out>/checkpoints/train_state.pt at the end.

  python patch_dreamer4_heads_state.py <dreamer4 checkout>/dreamer4
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])


def patch(path, edits):
    src = path.read_text()
    for old, new in edits:
        if new in src:
            continue
        assert src.count(old) == 1, (path.name, "not found once", old[:70])
        src = src.replace(old, new)
    path.write_text(src)
    print("patched", path)


patch(root / "train" / "train_heads.py", [
    ("""    ap.add_argument("--init", default="", help="patched: phase-2 checkpoint (heads + finetuned dynamics) to continue from")
""", """    ap.add_argument("--init", default="", help="patched: phase-2 checkpoint (heads + finetuned dynamics) to continue from")
    ap.add_argument("--state_in", default="", help="patched: train_state.pt of the previous Dyna iteration (optimiser, global "
                    "step, loss normalisers)")
    ap.add_argument("--total_steps", type=int, default=0, help="patched: cosine length over all Dyna iterations (default --steps)")
"""),
    ("""    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)

    def cosine(s):
        return 0.5 * (1.0 + math.cos(math.pi * min(s, cfg.steps) / cfg.steps))
""", """    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    st_in = None   # patched: the training state of the previous Dyna iteration
    g0 = 0
    if cfg.state_in:
        st_in = torch.load(cfg.state_in, map_location="cpu", weights_only=False)
        sizes = [len(g["params"]) for g in st_in["opt"]["param_groups"]]
        assert sizes == [len(g["params"]) for g in opt.param_groups], ("--state_in param groups differ", sizes)
        for i, p in enumerate(q for g in opt.param_groups for q in g["params"]):
            s_ = st_in["opt"]["state"].get(i)
            if s_ is not None and "exp_avg" in s_:
                assert tuple(s_["exp_avg"].shape) == tuple(p.shape), ("--state_in shape differs", i)
        opt.load_state_dict(st_in["opt"])
        g0 = int(st_in["step"])
    total = cfg.total_steps or cfg.steps

    def cosine(s):   # patched: over the global step g0 + s of the whole Dyna loop
        return 0.5 * (1.0 + math.cos(math.pi * min(g0 + s, total) / total))
"""),
    ("""        lambdas.append(lambda s: cosine(s) * min(1.0, (s + 1) / max(1, cfg.dyn_warmup)))
""", """        lambdas.append(lambda s: cosine(s) * min(1.0, (g0 + s + 1) / max(1, cfg.dyn_warmup)))
"""),
    ("""                            normalize=bool(cfg.loss_norm), device=dev)
""", """                            normalize=bool(cfg.loss_norm), device=dev)
    if st_in is not None:   # patched
        combiner.load_state_dict(st_in["extra"]["loss_norm"])
        print(f"TRAIN-STATE heads: optimiser + loss_norm restored from {cfg.state_in}, global step {g0}", flush=True)
    else:
        print("TRAIN-STATE heads: fresh optimiser + loss_norm (no --state_in)", flush=True)
"""),
    ("""    run.finish({"demonstrations": len(demos)})
""", """    torch.save({"opt": opt.state_dict(), "step": g0 + cfg.steps, "extra": {"loss_norm": combiner.state_dict()}},
               out / "checkpoints" / "train_state.pt")   # patched
    print(f"TRAIN-STATE saved {out / 'checkpoints' / 'train_state.pt'} at global step {g0 + cfg.steps}", flush=True)
    run.finish({"demonstrations": len(demos)})
"""),
])
