"""Materialise the immutable CE visual-SLM checkpoint as a veRL model plugin.

This script never mutates SOURCE.  It writes a fresh HF checkpoint that combines
the source language weights, the separate ``vision_adapter.pt``, and four new
reserved visual marker tokens.  It is a preparation operation, not RL training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


VISUAL_TOKENS = ["<|crafter_visual_0|>", "<|crafter_visual_1|>", "<|crafter_visual_2|>", "<|crafter_visual_3|>"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite output checkpoint: {output}")
    if not (source / "vision_adapter.pt").is_file() or not (source / "vision_config.json").is_file():
        raise FileNotFoundError("Source is not a completed CE visual-SLM checkpoint")

    import torch
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    added = tokenizer.add_special_tokens({"additional_special_tokens": VISUAL_TOKENS})
    if added != len(VISUAL_TOKENS):
        raise RuntimeError("Expected four newly-added reserved visual tokens")
    model = Qwen3ForCausalLM.from_pretrained(source, trust_remote_code=True)
    model.resize_token_embeddings(len(tokenizer))
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)

    # Load through the plugin class after the language checkpoint is safely
    # materialised, then add the CNN state under registered module names.
    shutil.copy2(Path(__file__).with_name("modeling_crafter_vision_slm.py"), output / "modeling_crafter_vision_slm.py")
    from modeling_crafter_vision_slm import CrafterVisionQwen3ForCausalLM

    config = model.config
    config.architectures = ["CrafterVisionQwen3ForCausalLM"]
    # The critic is loaded by veRL through AutoModelForTokenClassification; map
    # it to the visual critic so it sees the same frame as the policy.
    config.auto_map = {
        "AutoModelForCausalLM": "modeling_crafter_vision_slm.CrafterVisionQwen3ForCausalLM",
        "AutoModelForTokenClassification": "modeling_crafter_vision_slm.CrafterVisionQwen3ForTokenClassification",
    }
    config.crafter_visual_tokens = len(VISUAL_TOKENS)
    config.crafter_visual_token_ids = tokenizer.convert_tokens_to_ids(VISUAL_TOKENS)
    visual_model = CrafterVisionQwen3ForCausalLM.from_pretrained(output, config=config, trust_remote_code=True)
    adapter = torch.load(source / "vision_adapter.pt", map_location="cpu", weights_only=True)
    visual_model.vision.load_state_dict(adapter)
    visual_model.save_pretrained(output, safe_serialization=True)
    config.save_pretrained(output)

    manifest = {
        "source": str(source),
        "source_model_sha256": sha256(source / "model.safetensors"),
        "source_vision_sha256": sha256(source / "vision_adapter.pt"),
        "visual_marker_tokens": VISUAL_TOKENS,
        "visual_marker_ids": config.crafter_visual_token_ids,
        "contract": "veRL must pass pixel_values [B,3,63,63]; missing frames are an error",
    }
    (output / "verl_visual_slm_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    # Written by container root onto root-squashed NFS, the files would
    # otherwise be private to ``nobody`` and unreadable by the training user.
    for path in output.iterdir():
        path.chmod(0o644)


if __name__ == "__main__":
    main()
