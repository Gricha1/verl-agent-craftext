"""Materialise a phase 4 BC checkpoint (SLM + h_prefix, z_prefix on the RSSM state) as a veRL model plugin.

The output is the format of materialize_ce_checkpoint.py with ``h_tokens + z_tokens`` reserved marker tokens instead of
four and the adapters at ``vision``: the prompt is "Task: ..." + markers, and the agent loop passes h_t, z_t in place of a
frame. The vocabulary of the BC SLM stays the first rows of the embedding, so its logits are the old ones plus markers.

    python materialize_rssm_checkpoint.py --bc <bc run>/best.pt --init initial_random --rssm rssm.pt --output <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc", type=Path, required=True, help="best.pt or epoch_*.pt of train_slm_action_on_rssm.py")
    parser.add_argument("--init", type=Path, required=True, help="initial_random: architecture and tokenizer of the SLM")
    parser.add_argument("--rssm", type=Path, required=True, help="RSSM checkpoint: h and z sizes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--h-tokens", type=int, default=50)
    parser.add_argument("--z-tokens", type=int, default=5)
    parser.add_argument("--freeze-adapters", action="store_true", help="PPO keeps h_prefix and z_prefix as they are")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite output checkpoint: {output}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rssm = torch.load(args.rssm, map_location="cpu", weights_only=False)["config"]
    z_size = int(rssm["z_size"]) * int(rssm["z_classes"]) if rssm["z_kind"] == "categorical" else int(rssm["z_size"])
    state = torch.load(args.bc, map_location="cpu", weights_only=False)["model"]
    markers = [f"<|crafter_visual_{index}|>" for index in range(args.h_tokens + args.z_tokens)]
    tokenizer = AutoTokenizer.from_pretrained(args.init, trust_remote_code=True)
    if tokenizer.add_special_tokens({"additional_special_tokens": markers}) != len(markers):
        raise RuntimeError("Expected only new reserved marker tokens")
    model = AutoModelForCausalLM.from_pretrained(args.init, trust_remote_code=True)
    model.load_state_dict(state["lm"])
    model.resize_token_embeddings(len(tokenizer))
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)

    shutil.copy2(Path(__file__).with_name("modeling_crafter_vision_slm.py"), output / "modeling_crafter_vision_slm.py")
    from modeling_crafter_vision_slm import CrafterVisionQwen3ForCausalLM

    config = model.config
    config.architectures = ["CrafterVisionQwen3ForCausalLM"]
    config.auto_map = {
        "AutoModelForCausalLM": "modeling_crafter_vision_slm.CrafterVisionQwen3ForCausalLM",
        "AutoModelForTokenClassification": "modeling_crafter_vision_slm.CrafterVisionQwen3ForTokenClassification",
    }
    config.crafter_visual_tokens = len(markers)
    config.crafter_visual_token_ids = tokenizer.convert_tokens_to_ids(markers)
    config.crafter_rssm = dict(h_size=int(rssm["h_size"]), z_size=z_size, h_tokens=args.h_tokens, z_tokens=args.z_tokens)
    config.crafter_freeze_front_end = bool(args.freeze_adapters)
    policy = CrafterVisionQwen3ForCausalLM.from_pretrained(output, config=config, trust_remote_code=True)
    policy.vision.h_prefix.load_state_dict(state["adapters"]["h_prefix"])
    policy.vision.z_prefix.load_state_dict(state["adapters"]["z_prefix"])
    policy.save_pretrained(output, safe_serialization=True)
    config.save_pretrained(output)

    manifest = {
        "bc": str(args.bc),
        "bc_sha256": sha256(args.bc),
        "rssm": str(args.rssm),
        "rssm_sha256": sha256(args.rssm),
        "visual_marker_ids": config.crafter_visual_token_ids,
        "freeze_adapters": bool(args.freeze_adapters),
        "contract": "veRL must pass pixel_values [B, h + z]; missing states are an error",
    }
    (output / "verl_visual_slm_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for path in output.iterdir():
        if path.is_file():
            path.chmod(0o644)


if __name__ == "__main__":
    main()
