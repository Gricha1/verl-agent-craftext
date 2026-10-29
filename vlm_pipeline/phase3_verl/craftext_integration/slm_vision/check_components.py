"""Read-only component check for the veRL visual-SLM plugin (no training).

Run inside the modern-veRL image with the repo and ``vllm_plugin_site`` on
PYTHONPATH.  It verifies, on one GPU:

1. the actor (``AutoModelForCausalLM``) loads the CE CNN and uses the frame;
2. the critic, built by veRL's own ``load_valuehead_model``, is the visual
   token classifier with SDPA attention and the same CNN;
3. vLLM resolves the registered visual model in its engine processes and its
   greedy next-token log-probs match the HF actor on the same frame.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="CE checkpoint with vision_adapter.pt")
    parser.add_argument("--skip-vllm", action="store_true")
    args = parser.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    report: dict[str, object] = {}
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    markers = [int(x) for x in config.crafter_visual_token_ids]
    text_ids = tokenizer.encode("Task: collect wood. Observation: tree to the left.", add_special_tokens=False)
    prompt_ids = markers + text_ids
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(2, 63, 63, 3), dtype=np.uint8)
    pixels = torch.from_numpy(frames.transpose(0, 3, 1, 2).copy()).float().div_(255.0)

    # 1. Actor.
    actor = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, attn_implementation="sdpa").eval()
    adapter = torch.load(args.source / "vision_adapter.pt", map_location="cpu", weights_only=True)
    report["actor_class"] = type(actor).__name__
    report["actor_vision_matches_ce"] = all(
        torch.equal(actor.vision.state_dict()[key], value) for key, value in adapter.items()
    )
    ids = torch.tensor([prompt_ids, prompt_ids])
    with torch.no_grad():
        logits = actor(input_ids=ids, pixel_values=pixels).logits[:, -1].float()
        try:
            actor(input_ids=ids)
            report["actor_rejects_missing_frame"] = False
        except ValueError:
            report["actor_rejects_missing_frame"] = True
    report["actor_frame_changes_logits"] = float((logits[0] - logits[1]).abs().max())
    hf_logprobs = torch.log_softmax(logits[0], dim=-1)

    # 2. Critic through veRL's loader, configured as HFModelConfig does it.
    from verl.utils.model import load_valuehead_model

    critic_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True, attn_implementation="sdpa")
    critic_config.num_labels = 1
    critic_config.classifier_dropout = 0.0
    critic_config.hidden_dropout = "0"
    critic_config.summary_dropout_prob = 0.0
    critic = load_valuehead_model(
        local_path=str(args.model), torch_dtype=torch.float32, model_config=critic_config, trust_remote_code=True
    ).eval()
    report["critic_class"] = type(critic).__name__
    report["critic_attn"] = critic.config._attn_implementation
    report["critic_vision_matches_ce"] = all(
        torch.equal(critic.vision.state_dict()[key], value) for key, value in adapter.items()
    )
    with torch.no_grad():
        values = critic(input_ids=ids, pixel_values=pixels).logits
    report["critic_values_shape"] = list(values.shape)
    report["critic_frame_changes_value"] = float((values[0, -1] - values[1, -1]).abs().max())

    # 3. vLLM with the general plugin.
    if not args.skip_vllm:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=str(args.model),
            trust_remote_code=True,
            enforce_eager=True,
            gpu_memory_utilization=0.15,
            max_model_len=1024,
            dtype="float32",
            limit_mm_per_prompt={"image": 1},
        )
        outputs = llm.generate(
            [{"prompt_token_ids": prompt_ids, "multi_modal_data": {"image": frames[0]}}],
            SamplingParams(max_tokens=4, temperature=0.0, logprobs=5),
        )
        # Several tokens exercise the decode path (empty multimodal embeddings).
        report["vllm_generated_tokens"] = len(outputs[0].outputs[0].token_ids)
        step = outputs[0].outputs[0].logprobs[0]
        report["vllm_model_class"] = llm.llm_engine.model_config.architectures
        diffs = {int(token): abs(float(item.logprob) - float(hf_logprobs[int(token)])) for token, item in step.items()}
        report["vllm_vs_hf_top5_max_abs_logprob_diff"] = max(diffs.values())
        report["vllm_greedy_token_equals_hf"] = int(outputs[0].outputs[0].token_ids[0]) == int(hf_logprobs.argmax())

    print("COMPONENT_REPORT " + json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
