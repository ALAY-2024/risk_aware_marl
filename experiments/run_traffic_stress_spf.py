from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

from experiments.traffic_stress_env import TrafficStressRoutingEnv
from sat_net.solver.spf import SPF
from sat_net.util import NamedDict


HOTSPOT = ["East China", "North China", "South China"]
SCENARIOS = [
    {
        "name": "normal",
        "hotspot_regions": [],
        "multiplier": 1.0,
        "start_ms": 1000,
        "duration_ms": 3000,
    },
    {
        "name": "china_1p5x",
        "hotspot_regions": HOTSPOT,
        "multiplier": 1.5,
        "start_ms": 1000,
        "duration_ms": 3000,
    },
    {
        "name": "china_2p0x",
        "hotspot_regions": HOTSPOT,
        "multiplier": 2.0,
        "start_ms": 1000,
        "duration_ms": 3000,
    },
]

METRIC_FIELDS = [
    "generated",
    "delivered",
    "dropped",
    "delivery_rate",
    "drop_rate",
    "throughput",
    "service_rate",
    "e2e_delay_mean",
    "queue_delay_mean",
    "transmission_delay_mean",
    "propagation_delay_mean",
    "cost_mean",
]


def run_one(config_path: str, scenario: dict, seed: int) -> dict:
    config = NamedDict.load(config_path)
    config.verbose = False
    env = TrafficStressRoutingEnv(config=config, stress_config=scenario)
    solver = SPF()
    env.reset(seed=seed)

    start = time.perf_counter()
    env.run(solver)
    elapsed = time.perf_counter() - start
    metrics = env.calc_metrics()

    row = {
        "scenario": scenario["name"],
        "seed": seed,
        "multiplier": scenario["multiplier"],
        "runtime_s": elapsed,
    }
    for key in METRIC_FIELDS:
        row[key] = getattr(metrics, key)
    return row


def summarize(rows: list[dict]) -> list[dict]:
    output = []
    for scenario in [item["name"] for item in SCENARIOS]:
        subset = [row for row in rows if row["scenario"] == scenario]
        summary = {"scenario": scenario, "n": len(subset)}
        for key in METRIC_FIELDS + ["runtime_s"]:
            values = [float(row[key]) for row in subset]
            summary[f"{key}_mean"] = statistics.fmean(values)
            summary[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(summary)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/traffic_stress_poc.json",
        help="Routing environment configuration.",
    )
    parser.add_argument(
        "--seeds",
        default="11,22,33",
        help="Comma-separated random seeds.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/traffic_stress_poc",
        help="Directory for raw and aggregate results.",
    )
    args = parser.parse_args()

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    out_dir = Path(args.out_dir)

    rows = []
    for seed in seeds:
        for scenario in SCENARIOS:
            row = run_one(args.config, scenario, seed)
            rows.append(row)
            print(json.dumps(row, sort_keys=True))

    summary = summarize(rows)
    write_csv(out_dir / "raw.csv", rows)
    write_csv(out_dir / "summary.csv", summary)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    normal = next(item for item in summary if item["scenario"] == "normal")
    print("\n=== Aggregate traffic-stress sensitivity ===")
    for item in summary:
        delay_delta = (
            item["e2e_delay_mean_mean"] / normal["e2e_delay_mean_mean"] - 1.0
            if normal["e2e_delay_mean_mean"] > 0
            else 0.0
        )
        queue_delta = (
            item["queue_delay_mean_mean"] / normal["queue_delay_mean_mean"] - 1.0
            if normal["queue_delay_mean_mean"] > 0
            else 0.0
        )
        print(
            f"{item['scenario']:>12s}: "
            f"delay={item['e2e_delay_mean_mean']:.3f} ms ({delay_delta:+.1%}), "
            f"queue={item['queue_delay_mean_mean']:.3f} ms ({queue_delta:+.1%}), "
            f"drop={item['drop_rate_mean']:.3%}, "
            f"throughput={item['throughput_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
