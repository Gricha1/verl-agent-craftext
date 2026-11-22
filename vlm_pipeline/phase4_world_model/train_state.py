"""Training state carried across the iterations of a Dyna loop (safe_rl_nlp.md, rules for agents 15-17, 2026-11-22).

Every Stage-2 trainer of crafter_dyna_loop.sh / dyna_loop.sh runs as its own process in every iteration. Before this module
each of them saved only weights, so every iteration silently restarted AdamW, the warmup + cosine schedule and the RMS loss
normalisers ("очевидно что нам нельзя обучать так"). Now a trainer
  --state-in  <previous iteration's train_state.pt>   restores the optimiser (moments, step counts, param groups), the
                                                      global step and any extra state (loss normalisers)
  --total-steps N                                     the length of its lr schedule over ALL iterations (default: --steps)
and writes <out>/train_state.pt at the end. The lr multiplier is a function of the global step, so the iterations of a loop
form one continuous schedule: one warmup at the very start, one cosine over the whole run.

Iteration 0 of a loop starts fresh on purpose: it is a new training phase (Stage 2 objective and data) after Stage 1, whose
optimisers belong to other losses. The loop fails if a later iteration has no state of the previous one.
"""
import math
from pathlib import Path

import torch


def lr_mult(g, total, warmup=0, floor=0.0):
    """lr multiplier at global step g (1-based): linear warmup over the first `warmup` global steps, then cosine from 1 to
    `floor` over `total` global steps."""
    w = min(1.0, g / warmup) if warmup > 0 else 1.0
    return w * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, g / max(1, total)))))


def load(path, opt, extra=None, who=""):
    """Restore the optimiser and the extra objects ({name: object with load_state_dict}) from save(); return the global
    step the previous run ended at. Without a path: fresh, step 0."""
    if not path:
        print(f"TRAIN-STATE {who}: fresh optimiser{' + ' + ', '.join(extra) if extra else ''} (no --state-in)", flush=True)
        return 0
    st = torch.load(path, map_location="cpu", weights_only=False)
    params = [p for g in opt.param_groups for p in g["params"]]
    sizes = [len(g["params"]) for g in st["opt"]["param_groups"]]
    assert sizes == [len(g["params"]) for g in opt.param_groups], ("param groups differ", path, sizes)
    for i, p in enumerate(params):   # same shapes, so the moments belong to the same tensors
        s = st["opt"]["state"].get(i)
        if s is not None and "exp_avg" in s:
            assert tuple(s["exp_avg"].shape) == tuple(p.shape), ("optimiser state shape differs", path, i, s["exp_avg"].shape, p.shape)
    opt.load_state_dict(st["opt"])
    for k, obj in (extra or {}).items():
        assert k in st["extra"], ("no saved state for", k, path)
        obj.load_state_dict(st["extra"][k])
    print(f"TRAIN-STATE {who}: optimiser{' + ' + ', '.join(extra) if extra else ''} restored from {path}, "
          f"global step {st['step']}", flush=True)
    return int(st["step"])


def save(path, opt, step, extra=None):
    """Optimiser state, global step and the extra objects' state_dict() -> path."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(opt=opt.state_dict(), step=int(step), extra={k: o.state_dict() for k, o in (extra or {}).items()}), path)
    print(f"TRAIN-STATE saved {path} at global step {step}", flush=True)


def add_args(p):
    p.add_argument("--state-in", default="", help="train_state.pt of this trainer in the previous loop iteration: restore the "
                   "optimiser, global step and loss normalisers (train_state.py)")
    p.add_argument("--total-steps", type=int, default=0, help="lr schedule length over all loop iterations (default --steps)")
