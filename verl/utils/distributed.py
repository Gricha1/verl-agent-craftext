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
"""Utilities for distributed training."""

import os
import tempfile
from verl.utils.device import is_cuda_available, get_torch_device


def initialize_global_process_group(timeout_second=36000):
    from datetime import timedelta

    import torch.distributed

    # Support single-process runs (no torchrun) for lightweight scripts.
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        # Some components (e.g. DeviceMesh) still require an initialized process group.
        # Initialize a trivial 1-rank group with an explicit init_method (no env:// rendezvous).
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        backend = "nccl" if is_cuda_available else "gloo"
        init_file = os.path.join(tempfile.gettempdir(), "verl_single_process_pg")
        init_method = f"file://{init_file}"
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=backend,
                init_method=init_method,
                rank=0,
                world_size=1,
                timeout=timedelta(seconds=timeout_second),
            )
        if torch.distributed.is_initialized():
            get_torch_device().set_device(local_rank)
        return local_rank, 0, 1

    backend = "nccl" if is_cuda_available else "hccl"
    # CPU-only fallback (no CUDA/NPU): use gloo.
    if not is_cuda_available and backend == "hccl":
        backend = "gloo"

    torch.distributed.init_process_group(
        backend,
        timeout=timedelta(seconds=timeout_second),
    )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        get_torch_device().set_device(local_rank)
    return local_rank, rank, world_size
