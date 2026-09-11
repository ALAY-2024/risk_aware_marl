from __future__ import annotations

import argparse
import csv
import json
import statistics
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
        "name": "china_2p0x",
        "hotspot_regions": HOTSPOT,
        "multiplier": 2.0,
        "start_ms": 1000,
        "duration_ms": 3000,
    },
]


def run_one(config_path: str, packet_rate: float, scenario: dict, seed: int) -> dict:
    config = NamedDict.load(config_path)
    config.verbose = False
    config.traffic.packet_rate_per_ms = packet_rate
    env = TrafficStressRoutingEnv(config=config, stress_config=scenario)
    env.reset(seed=seed)
    env.run(SPF())
    m = env.calc_metrics()
    return {
        "packet_rate_per_ms": packet_rate,
        "scenario": scenario["name"],
        "seed": seed,
        "generated": m.generated,
        "delivered": m.delivered,
        "dropped": m.dropped,
        "delivery_rate": m.delivery_rate,
        "drop_rate": m.drop_rate,
        "throughput": m.throughput,
        "e2e_delay_mean": m.e2e_delay_mean,
        "queue_delay_mean": m.queue_delay_mean,
    }


def aggregate(rows: list[dict]) -> list[dict]:
    output = []
    rates = sorted({float(row["packet_rate_per_ms"]) for row in rows})
    for rate in rates:
        for scenario in [item["name"] for item in SCENARIOS]:
            subset = [
                row
                for row in rows
                if float(row["packet_rate_per_ms"]) == rate
                and row["scenario"] == scenario
            ]
            item = {"packet_rate_per_ms": rate, "scenario": scenario, "n": len(subset)}
            for key in [
                "generated",
                "delivered",
                "delivery_rate",
                "drop_rate",
                "throughput",
                "e2e_delay_mean",
                "queue_delay_mean",
            ]:
                values = [float(row[key]) for row in subset]
                item[f"{key}_mean"] = statistics.fmean(values)
                item[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            output.append(item)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/traffic_stress_poc.json")
    parser.add_argument("--rates", default="0.10,0.25,0.50,1.00")
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--out-dir", default="results/load_sweep_spf")
    args = parser.parse_args()

    rates = [float(x.strip()) for x in args.rates.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    rows = []
    for rate in rates:
        for seed in seeds:
            for scenario in SCENARIOS:
                row = run_one(args.config, rate, scenario, seed)
                rows.append(row)
                print(json.dumps(row, sort_keys=True))

    summary = aggregate(rows)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "raw.csv", rows)
    write_csv(out_dir / "summary.csv", summary)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("\n=== Offered-load calibration ===")
    for rate in rates:
        normal = next(
            item
            for item in summary
            if item["packet_rate_per_ms"] == rate and item["scenario"] == "normal"
        )
        stress = next(
            item
            for item in summary
            if item["packet_rate_per_ms"] == rate and item["scenario"] == "china_2p0x"
        )
        print(
            f"rate={rate:.2f}/ms | "
            f"normal drop={normal['drop_rate_mean']:.2%}, delay={normal['e2e_delay_mean_mean']:.2f} ms, "
            f"stress drop={stress['drop_rate_mean']:.2%}, delay={stress['e2e_delay_mean_mean']:.2f} ms"
        )


if __name__ == "__main__":
    main()
