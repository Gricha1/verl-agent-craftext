#!/usr/bin/env bash
# Archive the two legacy single-task teacher checkpoints without mutating them.
set -euo pipefail

source_root=/home/gorbov_gv/craftext_cnn_ppo_expert_v1/runs
archive_root=/home/gorbov_gv/craftext_teacher_checkpoint_archive
runs=(
  cnn_ppo_debug_square16_stone_a100gpu0_20261015_r2
  cnn_ppo_wood_pickaxe_a100gpu1_20261015_r2
)

for run in "${runs[@]}"; do
  src="$source_root/$run/checkpoints"
  dst="$archive_root/$run"
  mkdir -p "$dst"
  for stem in teacher_init teacher_latest teacher_best; do
    test -s "$src/$stem.pt"
    test -s "$src/$stem.pt.json"
    cp -a "$src/$stem.pt" "$src/$stem.pt.json" "$dst/"
  done
  sha256sum "$src"/*.pt "$dst"/*.pt >"$dst/SHA256SUMS.txt"
  python3 - "$src" "$dst" <<'PY'
import hashlib, json, sys
from pathlib import Path
src, dst = map(Path, sys.argv[1:])
for stem in ("teacher_init", "teacher_latest", "teacher_best"):
    a, b = src / f"{stem}.pt", dst / f"{stem}.pt"
    assert a.read_bytes() == b.read_bytes(), stem
    metadata = json.loads((b.with_suffix(b.suffix + ".json")).read_text())
    assert "architecture" in metadata and "config" in metadata, stem
    print(f"ARCHIVE_OK {stem} bytes={b.stat().st_size} architecture={metadata['architecture'].get('name')}")
PY
done

# Deserialize every archived checkpoint into a freshly-created Flax template and
# run one policy forward pass.  CPU avoids perturbing either active GPU run.
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONPATH=/home/gorbov_gv/craftext_cnn_ppo_expert_v1/src/caged_craftext:/home/gorbov_gv/craftext_cnn_ppo_expert_v1/src/caged_craftext/Craftax/craftax /home/gorbov_gv/craftext_cnn_ppo_expert_v1/venv/bin/python - "$archive_root" <<'PY'
import importlib.util, json, sys
from pathlib import Path
import jax, jax.numpy as jnp, optax
from flax.training.train_state import TrainState

root = Path(sys.argv[1])
module_path = Path('/home/gorbov_gv/craftext_cnn_ppo_expert_v1/train_pixel_ppo.py')
spec = importlib.util.spec_from_file_location('legacy_teacher', module_path)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
for checkpoint in sorted(root.glob('*/teacher_*.pt')):
    meta = json.loads(checkpoint.with_suffix(checkpoint.suffix + '.json').read_text())
    cfg = mod.Config(**meta['config'])
    mod.WOOD_PICKAXE_TASK = cfg.task == 'MAKE_WOOD_PICKAXE'
    env = mod.make_env()
    obs, _ = env.reset(jax.random.PRNGKey(0), env.default_params)
    action_dim = 17 if mod.WOOD_PICKAXE_TASK else 5
    net = mod.PixelActorCritic(action_dim, cfg.layer_width)
    params = net.init(jax.random.PRNGKey(1), obs[None, ...])
    state = TrainState.create(apply_fn=net.apply, params=params,
                              tx=optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.lr, eps=1e-5)))
    loaded = mod.restore_checkpoint(checkpoint, state)
    logits, value = net.apply(loaded['params'], obs[None, ...])
    assert logits.shape == (1, action_dim) and value.shape == (1,)
    assert bool(jnp.isfinite(logits).all()) and bool(jnp.isfinite(value).all())
    print(f'RELOAD_OK {checkpoint} logits={tuple(logits.shape)} value={tuple(value.shape)} greedy={int(jnp.argmax(logits[0]))}')
PY
