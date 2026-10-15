#!/usr/bin/env python3
"""Mirror TensorBoard scalar events into one live Comet experiment.

This is intentionally a sidecar: it can be added to an already-running trainer
without importing, restarting, or modifying the training process.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from comet_ml import Experiment, ExistingExperiment
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output", required=True, help="Path to write the Comet URL metadata")
    parser.add_argument("--experiment-key", help="Resume this existing Comet experiment")
    parser.add_argument("--poll-seconds", type=int, default=15)
    args = parser.parse_args()

    common = {"workspace": "gregory-gorbov", "project_name": "verl-agent-caged-craftext"}
    experiment = (ExistingExperiment(previous_experiment=args.experiment_key, **common)
                  if args.experiment_key else Experiment(auto_output_logging="simple", **common))
    experiment.set_name(args.run_name)
    experiment.log_parameters({"metrics_source": "TensorBoard sidecar", "tensorboard_logdir": args.logdir})
    key = experiment.get_key()
    info = {"name": args.run_name, "key": key,
            "url": f"https://www.comet.com/gregory-gorbov/verl-agent-caged-craftext/{key}",
            "logdir": args.logdir}
    Path(args.output).write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps({"event": "comet_bridge_ready", **info}), flush=True)

    sent: set[tuple[str, int, float]] = set()
    while True:
        try:
            accumulator = EventAccumulator(args.logdir, size_guidance={"scalars": 0})
            accumulator.Reload()
            for tag in accumulator.Tags().get("scalars", []):
                for event in accumulator.Scalars(tag):
                    marker = (tag, event.step, event.wall_time)
                    if marker not in sent:
                        experiment.log_metric(tag, event.value, step=event.step)
                        sent.add(marker)
            experiment.flush()
        except Exception as exc:  # TensorBoard may rotate an event file while it is read.
            print(json.dumps({"event": "comet_bridge_retry", "error": str(exc)}), flush=True)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
