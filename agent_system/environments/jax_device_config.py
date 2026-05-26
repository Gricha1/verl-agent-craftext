"""JAX backend selection for Craftax / caged_craftext (CPU by default)."""
from __future__ import annotations

import os
from typing import Any


def configure_craftext_jax_backend(
    use_jax_gpu: bool,
    *,
    gpu_mem_fraction: float | None = None,
) -> None:
    """
    Must run before JAX is imported in this process (call from TaskRunner.run / make_envs early).

    When use_jax_gpu is True:
      - drop JAX_PLATFORMS=cpu so JAX can use CUDA
      - optionally cap XLA client memory (share GPU with vLLM)
    """
    if use_jax_gpu:
        os.environ.pop("JAX_PLATFORMS", None)
        if gpu_mem_fraction is not None and gpu_mem_fraction > 0:
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(gpu_mem_fraction)
    else:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")


def configure_craftext_jax_backend_with_fallback(
    use_jax_gpu: bool,
    *,
    gpu_mem_fraction: float | None = None,
) -> bool:
    """Configure JAX; if GPU was requested but CUDA is not visible, fall back to CPU."""
    configure_craftext_jax_backend(use_jax_gpu, gpu_mem_fraction=gpu_mem_fraction)
    if not use_jax_gpu:
        return False
    try:
        import jax

        devices = jax.devices()
        if not devices:
            raise RuntimeError("jax.devices() returned empty list")
        return True
    except Exception as exc:
        print(
            f"[WARN] Craftext JAX GPU requested but unavailable ({exc}); "
            "falling back to JAX_PLATFORMS=cpu."
        )
        configure_craftext_jax_backend(False)
        return False


def task_runner_cuda_visible_devices(config: Any) -> str:
    """GPU ids for Craftext TaskRunner (Ray CPU actor — needs explicit CUDA_VISIBLE_DEVICES)."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        return os.environ["CUDA_VISIBLE_DEVICES"]
    n = int(getattr(config.trainer, "n_gpus_per_node", 1) or 1)
    return ",".join(str(i) for i in range(n))


def task_runner_jax_runtime_env(
    base_env_vars: dict,
    *,
    use_jax_gpu: bool,
    jax_gpu_fraction: float,
    config: Any,
) -> dict:
    """Runtime env for TaskRunner only: expose GPUs to JAX without Ray num_gpus reservation."""
    env_vars = dict(base_env_vars)
    if use_jax_gpu:
        env_vars.pop("JAX_PLATFORMS", None)
        env_vars["CUDA_VISIBLE_DEVICES"] = task_runner_cuda_visible_devices(config)
        if jax_gpu_fraction > 0:
            env_vars["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(jax_gpu_fraction)
    else:
        env_vars.setdefault("JAX_PLATFORMS", "cpu")
    return env_vars


def craftext_jax_device_summary() -> str:
    try:
        import jax

        return f"JAX devices: {jax.devices()}"
    except Exception as exc:
        return f"JAX devices: unavailable ({exc})"


def apply_jax_gpu_resources(
    resources_per_worker: dict,
    use_jax_gpu: bool,
    jax_gpu_fraction: float,
) -> dict:
    """Ray resource dict for per-env workers (multi-process mode)."""
    out = dict(resources_per_worker)
    if use_jax_gpu and jax_gpu_fraction > 0:
        out["num_gpus"] = jax_gpu_fraction
    return out


def read_jax_gpu_settings(config: Any) -> tuple[bool, float]:
    use = bool(getattr(config.env, "use_jax_gpu", False))
    frac = float(getattr(config.env, "jax_gpu_fraction", 0.15))
    return use, frac
