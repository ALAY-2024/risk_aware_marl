from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

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


def load_solver(env, model_path: str):
    solver_config = NamedDict.load(f"{model_path}/solver_config.json")
    # Upstream checkpoints were trained on Apple MPS.  CPU keeps CI and
    # reproducibility independent of local accelerator availability.
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
    reasons = drop_reason_counts(env)
    row["drop_reasons_json"] = json.dumps(reasons, sort_keys=True)
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
        for key in METRICS:
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
        stress = next(x for x in summary if x["model_key"] == key and x["scenario"] == "china_2p0x")
        delay_delta = (
            stress["e2e_delay_mean_mean"] / normal["e2e_delay_mean_mean"] - 1.0
            if normal["e2e_delay_mean_mean"] > 0 else 0.0
        )
        drop_delta = stress["drop_rate_mean"] - normal["drop_rate_mean"]
        print(
            f"{key:>12s}: normal delay={normal['e2e_delay_mean_mean']:.2f} ms, "
            f"2x delay={stress['e2e_delay_mean_mean']:.2f} ms ({delay_delta:+.1%}), "
            f"drop change={drop_delta:+.2%}"
        )


if __name__ == "__main__":
    main()
