# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from io import BytesIO
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
try:
    from qwen_vl_utils import fetch_image, fetch_video
except ImportError:
    fetch_image = None
    fetch_video = None


def obs_array_to_pil_rgb(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    elif hasattr(image, "__array__") and not isinstance(image, np.ndarray):
        image = np.asarray(image)
    arr = np.asarray(image)
    if arr.size > 0 and arr.max() < 1:
        arr = arr * 255.0
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def qwen_vl_pil_for_mm(image, *, max_pixels: int, min_pixels: int) -> Image.Image:
    """PIL resized like Qwen-VL (smart_resize via qwen_vl_utils)."""
    pil = obs_array_to_pil_rgb(image)
    return fetch_image({"image": pil, "max_pixels": max_pixels, "min_pixels": min_pixels})


def process_image(image: Union[dict, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if "bytes" in image:
        assert "image" not in image, "Cannot have both `bytes` and `image`"
        image["image"] = BytesIO(image["bytes"])

    return fetch_image(image)


def qwen_vl_image_processor_call(processor, images, *, max_pixels: int, min_pixels: int):
    """HF image_processor with the same pixel budget as vLLM ``mm_processor_kwargs``."""
    return processor.image_processor(
        images,
        return_tensors="pt",
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )


def resolve_qwen_image_pad_token_id(tokenizer, processor=None) -> int | None:
    ids = resolve_qwen_image_pad_token_ids(tokenizer, processor=processor)
    return ids[0] if ids else None


def resolve_qwen_image_pad_token_ids(tokenizer, processor=None) -> list[int]:
    """All token ids that represent Qwen-VL image placeholder pads."""
    ids: set[int] = set()
    if processor is not None:
        proc_id = getattr(processor, "image_token_id", None)
        if proc_id is not None:
            try:
                ids.add(int(proc_id))
            except (TypeError, ValueError):
                pass
        image_token = getattr(processor, "image_token", None)
        if image_token and tokenizer is not None:
            try:
                ids.update(int(x) for x in tokenizer.encode(image_token, add_special_tokens=False))
            except Exception:
                pass
    if tokenizer is not None:
        tid = getattr(tokenizer, "image_token_id", None)
        if tid is not None:
            try:
                ids.add(int(tid))
            except (TypeError, ValueError):
                pass
        unk_id = getattr(tokenizer, "unk_token_id", None)
        for tok in ("<|image_pad|>", "<|video_pad|>"):
            try:
                cand = tokenizer.convert_tokens_to_ids(tok)
            except Exception:
                continue
            if cand is None or cand < 0 or cand == unk_id:
                continue
            ids.add(int(cand))
    return sorted(ids)


def qwen_vl_dedup_image_pad_tokens(prompt_ids, image_token_id: int | None = None, *, pad_token_ids=None) -> list[int]:
    """Collapse consecutive ``<|image_pad|>`` tokens (vLLM expands one pad per image)."""
    if prompt_ids is None:
        return []
    pad_ids = set(pad_token_ids or [])
    if image_token_id is not None:
        pad_ids.add(int(image_token_id))
    if not pad_ids:
        return list(prompt_ids) if not isinstance(prompt_ids, list) else prompt_ids
    if isinstance(prompt_ids, torch.Tensor):
        prompt_ids = prompt_ids.tolist()
    elif isinstance(prompt_ids, np.ndarray):
        prompt_ids = prompt_ids.tolist()
    else:
        prompt_ids = list(prompt_ids)

    arr = np.asarray(prompt_ids, dtype=np.int64)
    if arr.size == 0:
        return []
    mask = np.ones(len(arr), dtype=bool)
    is_pad = np.isin(arr, list(pad_ids))
    mask[1:] &= ~(is_pad[1:] & is_pad[:-1])
    return arr[mask].tolist()


def qwen_vl_collapse_vision_span_pads(
    prompt_ids,
    tokenizer,
    *,
    processor=None,
    pad_token_ids=None,
) -> list[int]:
    """Keep exactly one image_pad between each vision_start/vision_end span (vLLM expands it)."""
    if prompt_ids is None:
        return []
    if isinstance(prompt_ids, torch.Tensor):
        prompt_ids = prompt_ids.tolist()
    elif isinstance(prompt_ids, np.ndarray):
        prompt_ids = prompt_ids.tolist()
    else:
        prompt_ids = list(prompt_ids)

    pad_ids = set(pad_token_ids or resolve_qwen_image_pad_token_ids(tokenizer, processor=processor))
    if not pad_ids or tokenizer is None:
        return qwen_vl_dedup_image_pad_tokens(prompt_ids, pad_token_ids=pad_ids)

    unk_id = getattr(tokenizer, "unk_token_id", None)
    vs_id = ve_id = None
    for tok, which in (("<|vision_start|>", "vs"), ("<|vision_end|>", "ve")):
        try:
            tid = int(tokenizer.convert_tokens_to_ids(tok))
        except Exception:
            continue
        if tid is None or tid < 0 or tid == unk_id:
            continue
        if which == "vs":
            vs_id = tid
        else:
            ve_id = tid
    if vs_id is None or ve_id is None:
        return qwen_vl_dedup_image_pad_tokens(prompt_ids, pad_token_ids=pad_ids)

    out: list[int] = []
    i = 0
    n = len(prompt_ids)
    while i < n:
        if prompt_ids[i] == vs_id:
            out.append(vs_id)
            i += 1
            kept_pad = False
            while i < n and prompt_ids[i] != ve_id:
                tok = prompt_ids[i]
                if tok in pad_ids:
                    if not kept_pad:
                        out.append(tok)
                        kept_pad = True
                else:
                    out.append(tok)
                i += 1
            if i < n and prompt_ids[i] == ve_id:
                out.append(ve_id)
                i += 1
        else:
            out.append(prompt_ids[i])
            i += 1
    return out


def qwen_vl_prepare_prompt_ids_for_vllm(
    prompt_ids, tokenizer, *, image_token_id: int | None = None, processor=None
) -> list[int]:
    """Normalize Qwen-VL prompt token ids for vLLM rollout.

    vLLM expects one ``<|image_pad|>`` per image; it expands vision features itself.
    """
    pad_token_ids = resolve_qwen_image_pad_token_ids(tokenizer, processor=processor)
    if image_token_id is not None and int(image_token_id) not in pad_token_ids:
        pad_token_ids = [int(image_token_id), *pad_token_ids]
    collapsed = qwen_vl_dedup_image_pad_tokens(prompt_ids, pad_token_ids=pad_token_ids)
    return qwen_vl_collapse_vision_span_pads(
        collapsed, tokenizer, processor=processor, pad_token_ids=pad_token_ids
    )


def qwen_vl_dedup_image_pad_tokens_legacy(prompt_ids, tokenizer) -> list[int]:
    """Backward-compatible alias."""
    return qwen_vl_prepare_prompt_ids_for_vllm(prompt_ids, tokenizer)


VIDEO_FORMAT_HELP = """Currently, we only support the video formats introduced in qwen2-vl.
Refer to https://github.com/QwenLM/Qwen2.5-VL?tab=readme-ov-file#using---transformers-to-chat.

eg.
{
    "type": "video",
    "video": [
        "file:///path/to/frame1.jpg",
        "file:///path/to/frame2.jpg"
    ]
}

{
    "type": "video",
    "video": "file:///path/to/video.mp4"
}
# Defaults to fps=2, min_frames=4, max_frames=768

{
    "type": "video",
    "video": "file:///path/to/video.mp4",
    "fps": 2,
    "min_frames": 1,
    "max_frames": 32
}
"""


def process_video(
    video: dict,
    nframes: Optional[int] = None,
    fps: Optional[float] = None,
    fps_min_frames: Optional[int] = None,
    fps_max_frames: Optional[int] = None,
) -> torch.Tensor:
    """Converts a video dict into a [n_frames, 3, H, W] tensor

    Add video sample FPS in a future MR
    """

    if not isinstance(video, dict) or "video" not in video:
        raise NotImplementedError(VIDEO_FORMAT_HELP)
    assert nframes is None or fps is None, "Can't use both `nframes` or `fps`"

    # Shallow copy... since we might want to add some keys
    video = dict(video)

    contains_sampling_rules = "nframes" in video or "fps" in video
    if not contains_sampling_rules:
        if nframes is not None:
            video["nframes"] = nframes
        elif fps is not None:
            video["fps"] = fps
            if fps_min_frames is not None:
                video["min_frames"] = fps_min_frames
            if fps_max_frames is not None:
                video["max_frames"] = fps_max_frames

    return fetch_video(video)
