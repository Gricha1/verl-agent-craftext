#!/usr/bin/env python3
"""Patch a 4ku/dreamer4 (3489e13) checkout so that phase 2 (train_heads.py) runs on a world model trained WITHOUT a proprio
stream (our Crafter frames have none). The dynamics model and its losses already take proprio=None; only the clip sampler,
agent_forward and a guard in main() assumed one. Nothing else changes.

  python patch_dreamer4_heads_noproprio.py <dreamer4 checkout>/dreamer4/train/train_heads.py
"""
import sys
from pathlib import Path

path = Path(sys.argv[1])
src = path.read_text()
edits = [
    ('    if ds.proprio_dim is None:\n        raise ValueError("this world model was trained without a proprio stream; the agent reads "\n'
     '                         "one (open the dataset with a proprio mode and retrain phase 1)")\n',
     '    if ds.proprio_dim is None:   # patched: no proprio stream, the agent reads latents only\n'
     '        print("no proprio stream: agent tokens read the latents only", flush=True)\n'),
    ('    d_prop = episodes[int(pick[0])]["proprio"].shape[-1]\n',
     '    has_p = "proprio" in episodes[int(pick[0])]   # patched\n'
     '    d_prop = episodes[int(pick[0])]["proprio"].shape[-1] if has_p else 0\n'),
    ('        proprio[b] = ep["proprio"][s]\n',
     '        if has_p:\n            proprio[b] = ep["proprio"][s]\n'),
    ('    return {"z": z, "proprio": as_t(proprio), "actions"',
     '    return {"z": z, "proprio": as_t(proprio) if has_p else None, "actions"'),
    ('    p_in = (1 - tau_ctx) * prop + tau_ctx * torch.randn_like(prop)\n',
     '    p_in = None if prop is None else (1 - tau_ctx) * prop + tau_ctx * torch.randn_like(prop)\n'),
    ('        clip = {k: torch.cat([rel[k], uni[k]], 0) for k in rel}\n',
     '        clip = {k: torch.cat([rel[k], uni[k]], 0) if rel[k] is not None else None for k in rel}   # patched\n'),
]
for old, new in edits:
    if new in src:
        continue
    assert src.count(old) == 1, ("not found once", old[:60])
    src = src.replace(old, new)
path.write_text(src)
print("patched", path)
