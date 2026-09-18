"""Wrap a vision-language backbone with a per-token scalar value head (PPO critic)."""
from __future__ import annotations

import torch
from torch import nn
from transformers.modeling_outputs import TokenClassifierOutput


def vl_hidden_size(config) -> int:
    if getattr(config, "hidden_size", None):
        return int(config.hidden_size)
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is not None and getattr(text_cfg, "hidden_size", None):
        return int(text_cfg.hidden_size)
    raise ValueError(f"Cannot infer hidden_size from config type={type(config)}")


class VLForTokenClassification(nn.Module):
    """Token-level value head on top of a VL causal backbone (matches critic dp_critic API)."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        self.num_labels = 1
        hidden = vl_hidden_size(self.config)
        self.dropout = nn.Dropout(0.0)
        self.score = nn.Linear(hidden, 1, bias=False)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        use_cache=False,
        **kwargs,
    ):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
            **kwargs,
        )
        hidden = outputs.hidden_states[-1]
        logits = self.score(self.dropout(hidden))
        return TokenClassifierOutput(logits=logits)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        fn = getattr(self.backbone, "gradient_checkpointing_enable", None)
        if fn is None:
            return
        if gradient_checkpointing_kwargs is None:
            fn()
        else:
            fn(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        fn = getattr(self.backbone, "gradient_checkpointing_disable", None)
        if fn is not None:
            fn()

    def enable_input_require_grads(self):
        fn = getattr(self.backbone, "enable_input_require_grads", None)
        if fn is not None:
            fn()


def wrap_vl_backbone_for_critic(backbone: nn.Module) -> VLForTokenClassification:
    return VLForTokenClassification(backbone)
