"""One-shot entry point for the saved r4 Docker container.

The container has a historically working GPU device request.  Its normal command
is ``ppo_with_instruction.py``; this wrapper is swapped in only while starting
the collector, then the original file is restored on the mounted host volume.
"""

from __future__ import annotations

import runpy
import sys


COLLECTOR = "/mnt/run/legacy/caged_craftext/baselines/collect_ppo_achievements_all_trajectories.py"

sys.argv = [
    COLLECTOR,
    "--legacy-root", "/mnt/run/legacy/caged_craftext",
    "--checkpoint-dir", "/mnt/run/run/ppo_mlp_distilbert_cnn_achievements_all_gpu3_20261019_r4/wandb/offline-run-20261019_095252-q8gdkxjc/files/checkpoint_restart_2",
    "--checkpoint-step", "500000000",
    # The historical r4 run directory is world-writable by its container UID;
    # the parent ``/mnt/run/run`` is not.
    "--output-dir", "/mnt/run/run/ppo_mlp_distilbert_cnn_achievements_all_gpu3_20261019_r4/ppo_achievements_all_collection_20261024_r1",
    "--seeds", "20261300:20261312",
    "--task-indices", "all",
    "--max-steps", "1000",
]
runpy.run_path(COLLECTOR, run_name="__main__")
