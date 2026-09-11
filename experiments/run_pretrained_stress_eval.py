from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np

from experiments.traffic_stress_env import TrafficStressRoutingEnv
from sat_net.solver import create_solver
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
    {
        "name": "china_4p0x",
        "hotspot_regions": HOTSPOT,
        "multiplier": 4.0,
        "start_ms": 1000,
        "duration_ms": 3000,
    },
]

MODEL_PATHS = {
    "madqn": "saved_models/madqn",
    "primal_avg": "saved_models/primal_avg",
    "primal_cvar": "saved_models/primal_cvar",
}

METRICS = [
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

EXTRA_METRICS = [
    "queue_delay_p95",
    "queue_delay_p99",
    "e2e_delay_p95",
    "e2e_delay_p99",
    "hotspot_generated",
    "hotspot_delivered",
    "hotspot_dropped",
    "hotspot_delivery_fraction",
    "hotspot_queue_delay_mean",
    "hotspot_queue_delay_p95",
    "hotspot_e2e_delay_mean",
]


def load_solver(env, model_path: str):
    solver_config = NamedDict.load(f"{model_path}/solver_config.json")
    solver_config.device = "cpu"
    solver = create_solver(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        solver_config=solver_config,
        tf_writer=None,
    )
    solver.load_models(f"{model_path}/models/best_model")
    solver.set_eval()
    return solver


def drop_reason_counts(env) -> dict[str, int]:
    counts: dict[str, int] = {}
    for block in env.dropped_packets:
        reason = getattr(block, "drop_reason", None)
        reason_name = getattr(reason, "name", str(reason))
        weight = int(getattr(block, "packet_count", 1))
        counts[reason_name] = counts.get(reason_name, 0) + weight
    return counts


def weighted_mean(blocks, attribute: str) -> float:
    if not blocks:
        return 0.0
    values = np.asarray([float(getattr(block, attribute)) for block in blocks], dtype=float)
    weights = np.asarray([int(getattr(block, "packet_count", 1)) for block in blocks], dtype=float)
    if weights.sum() <= 0:
        return 0.0
    return float(np.average(values, weights=weights))


def weighted_quantile(blocks, attribute: str, q: float) -> float:
    if not blocks:
        return 0.0
    values = np.asarray([float(getattr(block, attribute)) for block in blocks], dtype=float)
    weights = np.asarray([int(getattr(block, "packet_count", 1)) for block in blocks], dtype=float)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    threshold = q * cumulative[-1]
    idx = int(np.searchsorted(cumulative, threshold, side="left"))
    return float(values[min(idx, len(values) - 1)])


def packet_weight(blocks) -> int:
    return int(sum(int(getattr(block, "packet_count", 1)) for block in blocks))


def extra_metrics(env) -> dict[str, float]:
    delivered = list(env.delivered_packets)
    hotspot_ids = {
        region.id for region in env.traffic_model.regions if region.name in set(HOTSPOT)
    }
    hotspot_generated_blocks = [
        block for block in env.generated_packets if block.source_region_id in hotspot_ids
    ]
    hotspot_delivered_blocks = [
        block for block in env.delivered_packets if block.source_region_id in hotspot_ids
    ]
    hotspot_dropped_blocks = [
        block for block in env.dropped_packets if block.source_region_id in hotspot_ids
    ]

    hotspot_generated = packet_weight(hotspot_generated_blocks)
    hotspot_delivered = packet_weight(hotspot_delivered_blocks)
    hotspot_dropped = packet_weight(hotspot_dropped_blocks)

    return {
        "queue_delay_p95": weighted_quantile(delivered, "queue_delay", 0.95),
        "queue_delay_p99": weighted_quantile(delivered, "queue_delay", 0.99),
        "e2e_delay_p95": weighted_quantile(delivered, "e2e_delay", 0.95),
        "e2e_delay_p99": weighted_quantile(delivered, "e2e_delay", 0.99),
        "hotspot_generated": hotspot_generated,
        "hotspot_delivered": hotspot_delivered,
        "hotspot_dropped": hotspot_dropped,
        "hotspot_delivery_fraction": (
            hotspot_delivered / hotspot_generated if hotspot_generated > 0 else 0.0
        ),
        "hotspot_queue_delay_mean": weighted_mean(
            hotspot_delivered_blocks, "queue_delay"
        ),
        "hotspot_queue_delay_p95": weighted_quantile(
            hotspot_delivered_blocks, "queue_delay", 0.95
        ),
        "hotspot_e2e_delay_mean": weighted_mean(
            hotspot_delivered_blocks, "e2e_delay"
        ),
    }


def run_one(config_path: str, model_key: str, scenario: dict, seed: int, packet_rate: float) -> dict:
    config = NamedDict.load(config_path)
    config.verbose = False
    config.traffic.packet_rate_per_ms = packet_rate
    env = TrafficStressRoutingEnv(config=config, stress_config=scenario)
    solver = load_solver(env, MODEL_PATHS[model_key])
    env.reset(seed=seed, start_time=0)
    env.run(solver)
    metrics = env.calc_metrics()

    row = {
        "solver": solver.name,
        "model_key": model_key,
        "scenario": scenario["name"],
        "seed": seed,
        "packet_rate_per_ms": packet_rate,
        "multiplier": scenario["multiplier"],
    }
    for key in METRICS:
        row[key] = getattr(metrics, key)
    row.update(extra_metrics(env))
    row["drop_reasons_json"] = json.dumps(drop_reason_counts(env), sort_keys=True)
    return row


def aggregate(rows: list[dict]) -> list[dict]:
    result = []
    groups = sorted({(row["model_key"], row["scenario"]) for row in rows})
    for model_key, scenario in groups:
        subset = [r for r in rows if r["model_key"] == model_key and r["scenario"] == scenario]
        item = {
            "model_key": model_key,
            "solver": subset[0]["solver"],
            "scenario": scenario,
            "n": len(subset),
        }
        for key in METRICS + EXTRA_METRICS:
            values = [float(row[key]) for row in subset]
            item[f"{key}_mean"] = statistics.fmean(values)
            item[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        result.append(item)
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/traffic_stress_full_poc.json")
    parser.add_argument("--solvers", default="primal_cvar,primal_avg,madqn")
    parser.add_argument("--seeds", default="11")
    parser.add_argument("--packet-rate", type=float, default=5.0)
    parser.add_argument("--out-dir", default="results/pretrained_stress_eval")
    args = parser.parse_args()

    solver_keys = [x.strip() for x in args.solvers.split(",") if x.strip()]
    unknown = [key for key in solver_keys if key not in MODEL_PATHS]
    if unknown:
        raise ValueError(f"unknown solver key(s): {unknown}")
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    rows = []
    for key in solver_keys:
        for seed in seeds:
            for scenario in SCENARIOS:
                row = run_one(args.config, key, scenario, seed, args.packet_rate)
                rows.append(row)
                print(json.dumps(row, sort_keys=True))

    summary = aggregate(rows)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "raw.csv", rows)
    write_csv(out_dir / "summary.csv", summary)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("\n=== Pretrained policy traffic-stress sensitivity ===")
    for key in solver_keys:
        normal = next(x for x in summary if x["model_key"] == key and x["scenario"] == "normal")
        for scenario_name in ("china_2p0x", "china_4p0x"):
            stress = next(x for x in summary if x["model_key"] == key and x["scenario"] == scenario_name)
            q_delta = (
                stress["queue_delay_mean_mean"] / normal["queue_delay_mean_mean"] - 1.0
                if normal["queue_delay_mean_mean"] > 0 else 0.0
            )
            q95_delta = (
                stress["queue_delay_p95_mean"] / normal["queue_delay_p95_mean"] - 1.0
                if normal["queue_delay_p95_mean"] > 0 else 0.0
            )
            print(
                f"{key:>12s} {scenario_name:>12s}: "
                f"queue={stress['queue_delay_mean_mean']:.2f} ms ({q_delta:+.1%}), "
                f"queue-p95={stress['queue_delay_p95_mean']:.2f} ms ({q95_delta:+.1%}), "
                f"hotspot-queue={stress['hotspot_queue_delay_mean_mean']:.2f} ms, "
                f"e2e={stress['e2e_delay_mean_mean']:.2f} ms"
            )


if __name__ == "__main__":
    main()
