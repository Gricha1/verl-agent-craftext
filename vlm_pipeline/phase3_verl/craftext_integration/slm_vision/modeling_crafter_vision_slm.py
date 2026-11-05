"""veRL model plugin for the CE-trained Crafter visual 5M Qwen3 SLM.

The CE checkpoint represented a frame as four CNN-produced *soft* tokens
prepended to the text embeddings.  For veRL all tokens, including a rollout
response, need an input-id-aligned logit.  The materialised checkpoint therefore
inserts four reserved ids at the beginning of every prompt and this model
replaces only their embeddings with the CNN output.  Text positions begin at
four exactly as in CE, while logits stay aligned with veRL's response mask.

Two heads share this visual front end:

* ``CrafterVisionQwen3ForCausalLM`` -- the actor/reference policy;
* ``CrafterVisionQwen3ForTokenClassification`` -- the PPO critic.  veRL's
  value engine reads ``logits[..., 0]`` of a ``num_labels=1`` token classifier,
  so the critic sees the same frame as the actor instead of a text-only prompt.

With ``config.crafter_rssm`` the same slots hold the world-model state instead of a frame: phase 4's ``h_prefix`` and
``z_prefix`` map h_t, z_t to ``h_tokens + z_tokens`` soft tokens, and ``pixel_values`` is ``[batch, h + z]``.  The agent
loop gets h_t, z_t from phase4_world_model/rssm_service.py; they are data here, PPO never reaches into the RSSM.
"""
from __future__ import annotations

import importlib.util
import logging
from typing import Optional

import torch
from torch import nn
from transformers import Qwen3ForCausalLM, Qwen3ForTokenClassification

logger = logging.getLogger(__name__)


def _sdpa_if_flash_attn_missing(config) -> None:
    """veRL 0.9.1 loads every critic with a hard-coded FlashAttention2 request.

    The training image has no ``flash_attn`` wheel; this 5M model (head_dim 32)
    is served equally well by SDPA, so fall back instead of failing.
    """
    if config._attn_implementation == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        logger.warning("flash_attn is not installed; using SDPA attention for the Crafter visual SLM")
        config._attn_implementation = "sdpa"


class LatentAdapters(nn.Module):
    """h_prefix, z_prefix of phase 4: [B, h + z] to [B, (h_tokens + z_tokens) * width], h tokens first."""

    def __init__(self, h_size: int, z_size: int, h_tokens: int, z_tokens: int, width: int):
        super().__init__()
        self.sizes = (h_size, z_size)
        self.h_prefix = nn.Sequential(nn.Linear(h_size, h_size), nn.GELU(), nn.Linear(h_size, h_tokens * width))
        self.z_prefix = nn.Sequential(nn.Linear(z_size, z_size), nn.GELU(), nn.Linear(z_size, z_tokens * width))

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        h, z = latents.split(self.sizes, dim=-1)
        return torch.cat((self.h_prefix(h), self.z_prefix(z)), dim=-1)


def build_front_end(config) -> nn.Module:
    """What turns one observation into the soft tokens: the CE CNN, or the adapters of ``config.crafter_rssm``."""
    width = int(config.hidden_size)
    rssm = getattr(config, "crafter_rssm", None)
    if rssm:
        return LatentAdapters(int(rssm["h_size"]), int(rssm["z_size"]), int(rssm["h_tokens"]), int(rssm["z_tokens"]), width)
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 7 * 7, int(getattr(config, "crafter_visual_tokens", 4)) * width),
    )


def front_end_input(config) -> tuple:
    """Shape of one observation at ``pixel_values``."""
    rssm = getattr(config, "crafter_rssm", None)
    return (int(rssm["h_size"]) + int(rssm["z_size"]),) if rssm else (3, 63, 63)


class _CrafterVisionMixin:
    """CNN soft-token injection shared by the policy and the critic.

    ``pixel_values`` is ``[frames, 3, 63, 63]`` in the [0, 1] range, or ``[states, h + z]`` with
    ``config.crafter_rssm``.  It is deliberately required whenever the reserved visual ids occur: a veRL
    integration must fail loudly rather than silently running a text-only SLM.
    Frames are matched, in order, to the rows that contain the markers; each
    such row must contain each marker exactly once.  Rows without markers are
    plain text: veRL's ``upsample_batch_to_divisible_size`` appends synthetic
    EOS-only padding rows with empty ``multi_modal_inputs`` and zero masks.
    Packed (remove-padding) rows are rejected because the SDPA backbone cannot
    keep packed samples from attending to each other.
    """

    def _init_crafter_vision(self, config) -> None:
        visual_tokens = int(getattr(config, "crafter_visual_tokens", 4))
        self.crafter_visual_token_ids = tuple(int(x) for x in config.crafter_visual_token_ids)
        if len(self.crafter_visual_token_ids) != visual_tokens:
            raise ValueError("crafter_visual_token_ids must have crafter_visual_tokens entries")
        self.vision = build_front_end(config)
        self.crafter_visual_tokens = visual_tokens
        self.crafter_input_shape = front_end_input(config)
        if getattr(config, "crafter_freeze_front_end", False):
            # Buffers, not requires_grad=False: from_pretrained makes loaded parameters trainable again, and FSDP with
            # use_orig_params=False refuses a mix of frozen and trainable tensors. The checkpoint keys stay the same.
            for module in self.vision.modules():
                for name, parameter in list(module.named_parameters(recurse=False)):
                    del module._parameters[name]
                    module.register_buffer(name, parameter.data)

    def _has_visual_ids(self, input_ids: Optional[torch.LongTensor]) -> bool:
        return input_ids is not None and any(
            torch.any(input_ids.eq(token_id)) for token_id in self.crafter_visual_token_ids
        )

    def _inject_visual_tokens(self, input_ids: torch.LongTensor, pixel_values: torch.Tensor) -> torch.Tensor:
        if tuple(pixel_values.shape[1:]) != self.crafter_input_shape:
            raise ValueError(f"Expected pixel_values [batch,{self.crafter_input_shape}], got {tuple(pixel_values.shape)}")
        marked = input_ids.eq(self.crafter_visual_token_ids[0]).any(dim=1)
        if pixel_values.shape[0] != int(marked.sum()):
            raise ValueError(
                f"pixel_values has {pixel_values.shape[0]} frames for {int(marked.sum())} rows with visual markers "
                f"({input_ids.shape[0]} rows in total); packed remove-padding batches are not supported "
                "by the Crafter visual SLM"
            )
        token_embeddings = self.get_input_embeddings()(input_ids)
        visual_embeddings = self.vision(pixel_values.to(dtype=token_embeddings.dtype))
        visual_embeddings = visual_embeddings.view(pixel_values.shape[0], self.crafter_visual_tokens, -1)
        for offset, token_id in enumerate(self.crafter_visual_token_ids):
            positions = input_ids.eq(token_id)
            counts = positions.sum(dim=1)
            if not (torch.all(counts[marked] == 1) and torch.all(counts[~marked] == 0)):
                raise ValueError("Every visual row must contain each reserved Crafter visual id exactly once")
            token_embeddings = token_embeddings.masked_scatter(
                positions.unsqueeze(-1), visual_embeddings[:, offset : offset + 1, :]
            )
        return token_embeddings

    def _visual_inputs(self, input_ids, pixel_values, inputs_embeds):
        if not self._has_visual_ids(input_ids):
            return input_ids, inputs_embeds
        if pixel_values is None:
            raise ValueError("Crafter visual marker ids require pixel_values; refusing text-only forward")
        if inputs_embeds is not None:
            raise ValueError("Pass input_ids, not inputs_embeds, with Crafter visual markers")
        return None, self._inject_visual_tokens(input_ids, pixel_values)


class CrafterVisionQwen3ForCausalLM(_CrafterVisionMixin, Qwen3ForCausalLM):
    """Qwen3 policy with CE-compatible CNN soft visual tokens."""

    def __init__(self, config):
        _sdpa_if_flash_attn_missing(config)
        super().__init__(config)
        self._init_crafter_vision(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        input_ids, inputs_embeds = self._visual_inputs(input_ids, pixel_values, inputs_embeds)
        output = super().forward(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
        if output.logits is not None:
            # The visual markers are input-only: give them zero probability, exactly as the
            # vLLM rollout model does, so a response can never contain a frame slot.
            # -1e4 rather than -inf keeps p * log p (entropy) finite, also after a
            # temperature division; exp(-1e4 - max) is exactly 0 in fp32.
            marker_ids = torch.tensor(self.crafter_visual_token_ids, device=output.logits.device)
            output.logits = output.logits.index_fill(-1, marker_ids, -1e4)
        return output

    def prepare_inputs_for_generation(self, *args, pixel_values=None, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        # On decode steps the cached prefill already contains the injected
        # visual embeddings; retaining this argument is harmless and makes
        # transformers generation deterministic across model versions.
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
        return model_inputs


class CrafterVisionQwen3ForTokenClassification(_CrafterVisionMixin, Qwen3ForTokenClassification):
    """Visual Qwen3 critic: same backbone and CNN, a fresh ``score`` value head."""

    def __init__(self, config):
        _sdpa_if_flash_attn_missing(config)
        super().__init__(config)
        self._init_crafter_vision(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        input_ids, inputs_embeds = self._visual_inputs(input_ids, pixel_values, inputs_embeds)
        return super().forward(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
