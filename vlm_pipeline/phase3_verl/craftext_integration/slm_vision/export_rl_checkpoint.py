"""Export a veRL actor checkpoint of the visual SLM as a standalone HF checkpoint (same format as the
materialised B_ce), so the RL policy can be evaluated by slm_ce/closed_loop_eval.py like B_ce.

    python export_rl_checkpoint.py --actor RUN/checkpoints/global_step_100/actor \
        --start PHASE2_RUN/materialized_epoch_48_v2 --output RUN/export_global_step_100

--start is the checkpoint RL started from: it supplies config, tokenizer and modeling code (RL changes
weights only). Every actor parameter must match a start parameter by name and shape; nothing is
silently dropped. Single-GPU FSDP checkpoints (model_world_size_1_rank_0.pt) only.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    shards = sorted(args.actor.glob("model_world_size_*_rank_*.pt"))
    if [s.name for s in shards] != ["model_world_size_1_rank_0.pt"]:
        raise ValueError(f"expected one single-GPU shard, got {[s.name for s in shards]}")

    import torch
    import torch.distributed as dist

    # DTensor shards unpickle only inside a process group; one rank holds the whole tensor.
    if not dist.is_initialized():
        dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29533", rank=0, world_size=1)
    state = torch.load(shards[0], map_location="cpu", weights_only=False)
    state = {k.removeprefix("_fsdp_wrapped_module."): (v.full_tensor() if hasattr(v, "full_tensor") else v)
             for k, v in state.items()}

    args.output.mkdir(parents=True, exist_ok=True)
    for path in args.start.iterdir():
        if path.is_file() and path.suffix != ".safetensors":
            shutil.copy2(path, args.output / path.name)
    import sys
    sys.dont_write_bytecode = True  # leave the start checkpoint untouched (no __pycache__)
    sys.path.insert(0, str(args.start))
    from modeling_crafter_vision_slm import CrafterVisionQwen3ForCausalLM

    model = CrafterVisionQwen3ForCausalLM.from_pretrained(args.start, trust_remote_code=True)
    reference = model.state_dict()
    missing = sorted(set(reference) - set(state))
    unexpected = sorted(set(state) - set(reference))
    # tied embeddings may be stored once
    missing = [k for k in missing if not (k == "lm_head.weight" and getattr(model.config, "tie_word_embeddings", False))]
    if missing or unexpected:
        raise KeyError(f"missing {missing[:10]} ({len(missing)}), unexpected {unexpected[:10]} ({len(unexpected)})")
    for key, value in state.items():
        if value.shape != reference[key].shape:
            raise ValueError(f"{key}: {tuple(value.shape)} vs {tuple(reference[key].shape)}")
    changed = sum(int(not torch.equal(value.to(reference[key].dtype), reference[key])) for key, value in state.items())
    model.load_state_dict({k: v.to(reference[k].dtype) for k, v in state.items()}, strict=False)
    model.save_pretrained(args.output, safe_serialization=True)
    info = {"actor": str(args.actor), "start": str(args.start), "tensors": len(state), "tensors_changed_by_rl": changed}
    (args.output / "rl_export.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
