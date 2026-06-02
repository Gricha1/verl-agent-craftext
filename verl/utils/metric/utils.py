# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
Metrics utils.
"""

from typing import Any, Dict, List

import numpy as np


def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean, max, or min of each list.
    The reduce operation is determined by the key name:
    - If the key contains "max", np.max is used
    - If the key contains "min", np.min is used
    - Otherwise, np.mean is used

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its reduced value.

    Example:
        >>> metrics = {
        ...     "loss": [1.0, 2.0, 3.0],
        ...     "accuracy": [0.8, 0.9, 0.7],
        ...     "max_reward": [5.0, 8.0, 6.0],
        ...     "min_error": [0.1, 0.05, 0.2]
        ... }
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8, "max_reward": 8.0, "min_error": 0.05}
    """
    for key, val in list(metrics.items()):
        if not isinstance(val, (list, tuple, np.ndarray)):
            val = [val]
        if "max" in key:
            metrics[key] = np.max(val)
        elif "min" in key:
            metrics[key] = np.min(val)
        else:
            metrics[key] = np.mean(val)
    return metrics


def extract_metrics_from_dataproto(output) -> Dict[str, Any]:
    """Read worker metrics from DataProto.meta_info (handles list or scalar values)."""
    if output is None:
        return {}
    meta_info = getattr(output, "meta_info", None) or {}
    metrics = meta_info.get("metrics")
    if metrics is None:
        return {}
    if not isinstance(metrics, dict):
        return {}
    return dict(metrics)


def scalarize_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Convert reduced worker metrics to plain Python floats for experiment loggers."""
    out: Dict[str, float] = {}
    for key, val in metrics.items():
        try:
            if isinstance(val, (list, tuple, np.ndarray)):
                if len(val) == 0:
                    continue
                if "max" in key:
                    val = np.max(val)
                elif "min" in key:
                    val = np.min(val)
                else:
                    val = np.mean(val)
            scalar = float(val.item() if hasattr(val, "item") and not isinstance(val, (list, tuple, np.ndarray)) else val)
        except (TypeError, ValueError):
            continue
        out[key] = scalar
        if key == "world_model/loss":
            out["loss/world_model"] = scalar
        if key == "world_model/reward_loss":
            out["loss/world_model_reward"] = scalar
    return out


def normalize_worker_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Reduce (multi-GPU lists) and scalarize worker metrics for logging."""
    if not metrics:
        return {}
    return scalarize_metrics(reduce_metrics(dict(metrics)))
