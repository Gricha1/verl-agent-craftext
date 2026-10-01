"""Export every scalar metric of one Comet experiment to a portable CSV file."""

import argparse
import csv
import json
import time
from pathlib import Path

from comet_ml import API


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_key", help="Comet experiment key from its URL")
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--offline-message-log",
        type=Path,
        help="Comet messages.json spool; exports locally queued metrics without network access.",
    )
    args = parser.parse_args()

    failed = []
    if args.offline_message_log:
        metrics = []
        with args.offline_message_log.open(encoding="utf-8") as handle:
            for line in handle:
                message = json.loads(line)
                if message.get("type") != "metric_msg":
                    continue
                metric = message["payload"]["metric"]
                metrics.append(
                    {
                        "metricName": metric.get("metricName"),
                        "metricValue": metric.get("metricValue"),
                        "step": metric.get("step"),
                        "epoch": metric.get("epoch"),
                        "timestamp": message["payload"].get("local_timestamp"),
                        "runContext": None,
                        "offset": message["payload"].get("message_id"),
                    }
                )
    else:
        experiment = API().get_experiment_by_key(args.experiment_key)
        summaries = experiment.get_metrics_summary() or []
        metric_names = sorted(summary["name"] for summary in summaries)
        metrics = []
        for metric_name in metric_names:
            for attempt in range(1, args.retries + 1):
                try:
                    metrics.extend(experiment.get_metrics(metric_name) or [])
                    break
                except Exception as error:  # Comet can intermittently time out on A3.
                    if attempt == args.retries:
                        failed.append({"metric_name": metric_name, "error": repr(error)})
                    else:
                        time.sleep(2**(attempt - 1))
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
    if failed:
        failed_path = args.output_csv.with_suffix(".failed_metrics.json")
        failed_path.write_text(json.dumps(failed, indent=2), encoding="utf-8")
        raise RuntimeError(f"Could not export {len(failed)} metrics; details: {failed_path}")
    print(f"Exported {len(metrics)} metric points to {args.output_csv}")


if __name__ == "__main__":
    main()
