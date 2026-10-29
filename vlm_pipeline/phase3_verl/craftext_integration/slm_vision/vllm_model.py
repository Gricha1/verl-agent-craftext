"""vLLM model registration for the CE Crafter visual SLM.

This is an extension of vLLM's transformers backend, not a rollout or PPO
implementation.  It keeps veRL Agents responsible for scheduling and PPO, but
teaches vLLM how to own/load the CE CNN parameters and replace the four visual
marker embeddings at prefill.
"""
from __future__ import annotations

import torch
import numpy as np
from torch import nn

from vllm import ModelRegistry
from vllm.inputs import mm_input
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.transformers import TransformersForCausalLM
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargsItems
from vllm.multimodal.inputs import MultiModalFieldConfig, PlaceholderRange
from vllm.multimodal.parse import ImageProcessorItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
)
from vllm.sequence import IntermediateTensors


_MARKERS = tuple(f"<|crafter_visual_{i}|>" for i in range(4))


class CrafterVisionProcessingInfo(BaseProcessingInfo):
    """vLLM-side declaration: one CrafText frame becomes four soft tokens."""

    def get_supported_mm_limits(self):
        return {"image": 1}

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        return {"image": 4}


class CrafterVisionDummyInputsBuilder(BaseDummyInputsBuilder):
    def get_dummy_text(self, mm_counts):
        return "".join(_MARKERS) * mm_counts.get("image", 0)

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options):
        return {"image": self._get_dummy_images(width=63, height=63, num_images=mm_counts.get("image", 0))}


class CrafterVisionMultiModalProcessor(BaseMultiModalProcessor):
    """Keep caller token ids intact; convert an RGB CrafText frame to CHW float."""

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        return {"pixel_values": MultiModalFieldConfig.batched("image")}

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        return []

    def apply(self, inputs, timing_ctx):
        prompt_ids = inputs.prompt
        if isinstance(prompt_ids, str):
            prompt_ids = self.info.get_tokenizer().encode(prompt_ids, add_special_tokens=False)
        images = inputs.mm_data_items.get_items("image", ImageProcessorItems)
        pixels = []
        for index in range(images.get_count()):
            image = images.get(index)
            # The vLLM profiler supplies PIL images; real CrafText rollouts
            # supply uint8 numpy arrays.  Normalise both at this boundary.
            pixel = torch.as_tensor(np.asarray(image))
            if pixel.ndim != 3:
                raise ValueError(f"Expected CrafText RGB frame, got {tuple(pixel.shape)}")
            if pixel.shape[-1] in (3, 4):
                pixel = pixel[..., :3].permute(2, 0, 1)
            if tuple(pixel.shape) != (3, 63, 63):
                raise ValueError(f"Expected CrafText RGB [63,63,3], got {tuple(pixel.shape)}")
            pixels.append(pixel.contiguous().to(dtype=torch.float32).div_(255.0))
        marker_ids = [self.info.get_tokenizer().convert_tokens_to_ids(marker) for marker in _MARKERS]
        ranges = []
        cursor = 0
        for _ in pixels:
            try:
                offset = next(
                    pos for pos in range(cursor, len(prompt_ids) - len(marker_ids) + 1)
                    if prompt_ids[pos : pos + len(marker_ids)] == marker_ids
                )
            except StopIteration as exc:
                raise ValueError("CrafText image supplied without four visual marker ids") from exc
            ranges.append(PlaceholderRange(offset=offset, length=4, is_embed=torch.ones(4, dtype=torch.bool)))
            cursor = offset + 4
        mm_kwargs = MultiModalKwargsItems.from_hf_inputs(
            {"pixel_values": torch.stack(pixels)}, self._get_mm_fields_config({}, {})
        )
        return mm_input(
            prompt_token_ids=prompt_ids,
            mm_kwargs=mm_kwargs,
            mm_hashes=inputs.get_mm_hashes(self.info.model_id),
            mm_placeholders={"image": ranges},
        )


@MULTIMODAL_REGISTRY.register_processor(
    CrafterVisionMultiModalProcessor,
    info=CrafterVisionProcessingInfo,
    dummy_inputs=CrafterVisionDummyInputsBuilder,
)
class CrafterVisionQwen3ForCausalLM(TransformersForCausalLM, SupportsMultiModal):
    """vLLM transformer backend with the CE CNN registered as ``vision``."""

    def get_language_model(self):
        # ``TransformersForCausalLM`` itself is the decoder and exposes
        # ``embed_input_ids``.  The generic multimodal protocol otherwise
        # searches for a nested language-model child, which this backend does
        # not have.
        return self

    def embed_multimodal(self, *, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode frames in vLLM, so its scheduler owns the visual prefill."""
        if pixel_values.ndim != 4 or tuple(pixel_values.shape[1:]) != (3, 63, 63):
            raise ValueError(f"Expected CrafText pixels [B,3,63,63], got {tuple(pixel_values.shape)}")
        visual = self.vision(pixel_values.to(dtype=next(self.vision.parameters()).dtype))
        return visual.view(-1, self.visual_tokens, visual.shape[-1] // self.visual_tokens)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: list[torch.Tensor] | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        text_embeddings = super().embed_input_ids(input_ids)
        # Decode steps (and text-only chunks) arrive with an empty list.
        if not multimodal_embeddings:
            if is_multimodal is not None and bool(is_multimodal.any()):
                raise ValueError("CrafText visual marker positions arrived without visual embeddings")
            return text_embeddings
        if is_multimodal is None:
            raise ValueError("Visual embeddings supplied without their CrafText marker mask")
        visual_embeddings = torch.cat(list(multimodal_embeddings), dim=0).to(text_embeddings.dtype)
        if int(is_multimodal.sum()) != visual_embeddings.shape[0]:
            raise ValueError("CrafText visual embeddings and marker positions are misaligned")
        text_embeddings[is_multimodal] = visual_embeddings
        return text_embeddings

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = self.config
        self.visual_token_ids = tuple(int(x) for x in config.crafter_visual_token_ids)
        self.visual_tokens = len(self.visual_token_ids)
        # Do not create this small extension on ``meta``.  Unlike the
        # Transformer tree, it has ordinary PyTorch parameters and vLLM's
        # generic loader fills the allocated tensors directly.
        self.vision = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, self.visual_tokens * int(config.hidden_size)),
        )

    def _visual_embeds(self, input_ids: torch.Tensor, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 4 or tuple(pixel_values.shape[1:]) != (3, 63, 63):
            raise ValueError(f"Expected Crafter pixels [B,3,63,63], got {tuple(pixel_values.shape)}")
        visual = self.vision(pixel_values.to(dtype=next(self.vision.parameters()).dtype))
        visual = visual.view(-1, self.visual_tokens, visual.shape[-1] // self.visual_tokens)
        markers = torch.stack([input_ids.eq(token_id) for token_id in self.visual_token_ids], dim=1)
        if markers.sum().item() != visual.shape[0] * self.visual_tokens:
            raise ValueError("CrafText visual markers and pixel frames are misaligned")
        return visual

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        **kwargs,
    ):
        if not self.pp_group.is_first_rank:
            assert intermediate_tensors is not None
            input_ids = None
            inputs_embeds = intermediate_tensors["hidden_states"]
        elif input_ids is not None and any(torch.any(input_ids.eq(token_id)) for token_id in self.visual_token_ids):
            if pixel_values is None:
                raise ValueError("Crafter visual marker ids require pixel_values; refusing a text-only rollout")
            inputs_embeds = self.embed_input_ids(input_ids)
            visual = self._visual_embeds(input_ids, pixel_values)
            for index, token_id in enumerate(self.visual_token_ids):
                mask = input_ids.eq(token_id)
                inputs_embeds = inputs_embeds.masked_scatter(mask.unsqueeze(-1), visual[:, index : index + 1, :])
            input_ids = None
        elif self.embed_scale is not None and input_ids is not None and inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
            input_ids = None

        if input_ids is not None and input_ids.ndim == 1:
            input_ids = input_ids[None, ...]
        if inputs_embeds is not None and inputs_embeds.ndim == 2:
            inputs_embeds = inputs_embeds[None, ...]
        if positions.ndim == 1:
            positions = positions[None, ...]
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            position_ids=positions,
            attention_instances=self.attention_instances,
            return_dict=False,
            **self._output_aux_hidden_states_kwargs,
            **kwargs,
        )
        hidden_states = outputs[0][0, ...]
        if not self.pp_group.is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        return hidden_states


def register() -> None:
    """Register by the architecture persisted in the materialised checkpoint."""
    ModelRegistry.register_model("CrafterVisionQwen3ForCausalLM", CrafterVisionQwen3ForCausalLM)


register()
