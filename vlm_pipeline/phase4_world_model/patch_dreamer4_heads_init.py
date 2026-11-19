#!/usr/bin/env python3
"""Patch a 4ku/dreamer4 (3489e13) checkout (after patch_dreamer4_heads_continue.py) so phase 2 can CONTINUE from a phase-2
checkpoint: the online Dyna loop (dyna_loop.sh, 2026-11-19) finetunes the world model on the episodes the planner has
just played, starting from the world model the planner used, not from the phase-1 dynamics again.

train_heads.py
  --init <phase-2 checkpoint>: after the heads are built, their weights and the finetuned dynamics weights are loaded from
  it (agent_cfg must match: same task list, n_agent, cont head). The optimiser, the cosine schedule and the RMS loss
  normalisers start fresh, so use a smaller --lr / --dyn_lr and fewer --steps than a phase 2 from scratch.
  A checkpoint without a continue head may start a run with --cont_head 1 (Qwen Crafter heads, crafter_dyna_loop.sh): the
  new continue head starts from its random init, every other weight is loaded.

  python patch_dreamer4_heads_init.py <dreamer4 checkout>/dreamer4
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
    ("""    ap.add_argument("--cont_weight", type=float, default=1.0)
""", """    ap.add_argument("--cont_weight", type=float, default=1.0)
    ap.add_argument("--init", default="", help="patched: phase-2 checkpoint (heads + finetuned dynamics) to continue from")
"""),
    ("""    heads = build_agent_heads(agent_cfg).to(dev)
""", """    heads = build_agent_heads(agent_cfg).to(dev)
    if cfg.init:   # patched: continue from a phase-2 checkpoint
        ck = torch.load(cfg.init, map_location="cpu", weights_only=False)
        for k in ("n_tasks", "n_agent", "mtp", "d_hidden", "task_text_ids"):
            assert ck["agent_cfg"].get(k) == agent_cfg.get(k), ("--init agent_cfg differs", k, ck["agent_cfg"].get(k), agent_cfg.get(k))
        new_cont = bool(agent_cfg.get("cont_head")) and not ck["agent_cfg"].get("cont_head")
        assert new_cont or bool(ck["agent_cfg"].get("cont_head")) == bool(agent_cfg.get("cont_head")), "--init has a continue head, this run not"
        missing, unexpected = heads.load_state_dict(ck["agent"], strict=not new_cont)
        assert not unexpected and all(k.startswith("cont") for k in missing), (missing, unexpected)
        if new_cont:
            print("init: new continue head", missing, flush=True)
        world.load_dynamics_weights(ck["dyn"])
        print(f"init from {cfg.init}: heads and finetuned dynamics loaded", flush=True)
"""),
])
