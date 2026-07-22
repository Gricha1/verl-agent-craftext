#!/usr/bin/env python3
"""Create Comet ML experiment early and write its key for GPU profiler attachment."""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=os.environ.get("COMET_PROJECT_NAME", "verl_agent_alfworld"))
    parser.add_argument("--experiment", default=os.environ.get("RUN_NAME", "verl_run"))
    parser.add_argument(
        "--key-file",
        default=os.environ.get(
            "COMET_EXPERIMENT_KEY_FILE",
            os.path.join(os.environ.get("RAY_TEMP_DIR", "/tmp/ray_temp"), "comet_experiment_key.txt"),
        ),
    )
    args = parser.parse_args()

    api_key = os.environ.get("COMET_API_KEY")
    if not api_key:
        print("[comet_init] COMET_API_KEY not set, skipping early experiment init", flush=True)
        return 0

    try:
        from comet_ml import Experiment
    except ImportError:
        print("[comet_init] comet_ml not installed", file=sys.stderr)
        return 1

    workspace = os.environ.get("COMET_WORKSPACE")
    experiment = Experiment(
        api_key=api_key,
        workspace=workspace,
        project_name=args.project,
        experiment_name=args.experiment,
        auto_param_logging=False,
        auto_metric_logging=False,
        display_summary_level=0,
    )
    experiment.set_name(args.experiment)

    key = experiment.get_key()
    os.makedirs(os.path.dirname(args.key_file) or ".", exist_ok=True)
    with open(args.key_file, "w", encoding="utf-8") as f:
        f.write(key)

    url = ""
    try:
        url = experiment.url
    except Exception:
        pass

    print(
        f"[comet_init] experiment ready: project={args.project} name={args.experiment} "
        f"key_file={args.key_file}" + (f" url={url}" if url else ""),
        flush=True,
    )
    # Keep experiment open; training will attach via ExistingExperiment.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
