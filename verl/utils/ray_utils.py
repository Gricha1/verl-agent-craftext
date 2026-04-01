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
"""
Contains commonly used utilities for ray
"""

import concurrent.futures
import logging
from typing import Any, List, Optional

import ray


class _SuppressRayDiskUtilizationWarnings(logging.Filter):
    """Drop Ray's repeated WARNINGs when session /tmp disk usage crosses ~95% (floods the terminal)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR:
            return True
        try:
            msg = record.getMessage().lower()
        except Exception:
            return True
        if "95" not in msg and "0.95" not in msg:
            return True
        if any(k in msg for k in ("disk", "space", "usage", "utilization", "filesystem", "capacity", "full")):
            return False
        return True


def silence_ray_disk_usage_warnings() -> None:
    """Attach filters to Ray loggers before ``ray.init`` (driver process)."""
    filt = _SuppressRayDiskUtilizationWarnings()
    for name in (
        "ray",
        "ray._private",
        "ray._private.worker",
        "ray._private.services",
        "ray._private.ray_logging",
        "ray._private.process_watcher",
    ):
        logging.getLogger(name).addFilter(filt)


def ray_local_fs_capacity_system_config(config) -> dict:
    """
    Raylet logs ERROR every ~10s from file_system_monitor.cc when (1 - free/capacity) >= threshold.
    Default threshold is 0.95; large volumes that are ~95% full but still have hundreds of GB free
    spam the terminal — raise via ``local_fs_capacity_threshold`` (RayConfig).

    This does not affect Python logging; it configures the raylet C++ monitor.
    """
    try:
        from omegaconf import OmegaConf

        lfc = OmegaConf.select(config, "ray_init.local_fs_capacity_threshold", default=0.99)
    except Exception:
        lfc = 0.99
    if lfc is None:
        lfc = 0.99
    return {"local_fs_capacity_threshold": float(lfc)}


def parallel_put(data_list: List[Any], max_workers: Optional[int] = None):
    """
    Puts a list of data into the Ray object store in parallel using a thread pool.

    Args:
        data_list (List[Any]): A list of Python objects to be put into the Ray object store.
        max_workers (int, optional): The maximum number of worker threads to use.
                                     Defaults to min(len(data_list), 16).

    Returns:
        List[ray.ObjectRef]: A list of Ray object references corresponding to the input data_list,
                             maintaining the original order.
    """
    assert len(data_list) > 0, "data_list must not be empty"

    def put_data(index, data):
        return index, ray.put(data)

    if max_workers is None:
        max_workers = min(len(data_list), 16)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        data_list_f = [executor.submit(put_data, i, data) for i, data in enumerate(data_list)]
        res_lst = []
        for future in concurrent.futures.as_completed(data_list_f):
            res_lst.append(future.result())

        # reorder based on index
        output = [None for _ in range(len(data_list))]
        for res in res_lst:
            index, data_ref = res
            output[index] = data_ref

    return output
