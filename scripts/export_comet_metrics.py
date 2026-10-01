"""Export every scalar metric of one Comet experiment to a portable CSV file."""

import argparse
import csv
from pathlib import Path

from comet_ml import API


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_key", help="Comet experiment key from its URL")
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()

    experiment = API().get_experiment_by_key(args.experiment_key)
    metrics = experiment.get_metrics() or []
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "metric_name",
        "metric_value",
        "step",
        "epoch",
        "timestamp_ms",
        "run_context",
        "offset",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metric in metrics:
            writer.writerow(
                {
                    "metric_name": metric.get("metricName"),
                    "metric_value": metric.get("metricValue"),
                    "step": metric.get("step"),
                    "epoch": metric.get("epoch"),
                    "timestamp_ms": metric.get("timestamp"),
                    "run_context": metric.get("runContext"),
                    "offset": metric.get("offset"),
                }
            )
    print(f"Exported {len(metrics)} metric points to {args.output_csv}")


if __name__ == "__main__":
    main()
