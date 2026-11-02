"""Evaluate supplied checkpoints against repository and paper-like traffic settings.

Read-only evaluation: no model/source changes. Outputs explicit packet-weighted
metrics, completion fractions and empirical CVaR for verification.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from sat_net.routing_env import RoutingEnvAsync
from sat_net.solver import SPF, create_solver
from sat_net.util import NamedDict

MODEL_PATHS = {
    "madqn": ("saved_models/madqn", "MaDQN"),
    "primal_avg": ("saved_models/primal_avg", "PrimalAvg"),
    "primal_cvar": ("saved_models/primal_cvar", "PrimalCVaR"),
}
THRESHOLD_MS = 10.0


def weighted_upper_tail(values, counts, tail_fraction=0.25):
    if len(values) == 0:
        return None
    idx = np.argsort(values)[::-1]
    v = np.asarray(values, dtype=np.float64)[idx]
    w = np.asarray(counts, dtype=np.float64)[idx]
    remaining = w.sum() * tail_fraction
    weighted_sum = 0.0
    for val, weight in zip(v, w):
        take = min(remaining, weight)
        weighted_sum += val * take
        remaining -= take
        if remaining <= 0:
            break
    return float(weighted_sum / (w.sum() * tail_fraction))


def compute_metrics(env, seed, solver_name, mode, elapsed):
    metrics = json.loads(env.calc_metrics().to_json())
    delivered = env.delivered_packets
    counts = np.asarray([x.packet_count for x in delivered], dtype=np.int64)
    q = np.asarray([x.queue_delay for x in delivered], dtype=np.float64)
    n_delivered = int(counts.sum())
    n_violate = int(counts[q > THRESHOLD_MS].sum()) if len(q) else 0
    completed = metrics["delivered"] + metrics["dropped"]
    metrics.update({
        "solver": solver_name,
        "scenario": mode,
        "seed": seed,
        "simulation_seconds": elapsed,
        "flowlets_generated": len(env.generated_packets),
        "flowlets_delivered": len(delivered),
        "flowlets_dropped": len(env.dropped_packets),
        "unresolved_original_packets": metrics["generated"] - completed,
        "completed_original_packets": completed,
        "completed_fraction": completed / metrics["generated"] if metrics["generated"] else None,
        "queue_cvar25_delivered_ms": weighted_upper_tail(q, counts),
        "queue_violation_delivered_fraction": n_violate / n_delivered if n_delivered else None,
        "queue_violation_completed_including_drops_fraction": (
            (n_violate + metrics["dropped"]) / completed if completed else None
        ),
        "queue_violation_generated_including_drops_fraction": (
            (n_violate + metrics["dropped"]) / metrics["generated"] if metrics["generated"] else None
        ),
        "risk_threshold_ms": THRESHOLD_MS,
        "metric_note": "Per-delivered packet, flowlet weighted; incomplete packets are separately counted.",
    })
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--solver", choices=["spf", *MODEL_PATHS], required=True)
    p.add_argument("--mode", choices=["repository", "paper_three_city_approx"], default="repository")
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--base-seed", type=int, default=3333)
    p.add_argument("--output-root", default="paper_eval_results")
    args = p.parse_args()

    torch.set_num_threads(2)
    env_cfg = NamedDict.load("configs/starlink_dvbs2_test.json")
    env_cfg.time_limit_seconds = args.seconds
    env_cfg.verbose = False
    if args.mode == "paper_three_city_approx":
        # Equal weights and coordinates for the three *named cities* in the paper.
        # Not identical to the paper's actual ground-station/GSL implementation:
        # this branch's source code always uses nearest-satellite regional access.
        env_cfg.traffic.region_file = None
        env_cfg.traffic.population_map_path = None
        env_cfg.traffic.regions = [
            {"name": "Luxembourg", "latitude": 49.6116, "longitude": 6.1319, "weight": 1.0},
            {"name": "Dubai", "latitude": 25.2048, "longitude": 55.2708, "weight": 1.0},
            {"name": "Beijing", "latitude": 39.9042, "longitude": 116.4074, "weight": 1.0},
        ]

    env = RoutingEnvAsync(env_cfg)
    assert len(env.network.satellites) == 1584
    assert env.action_dim == 4 and env.obs_dim == 94

    if args.solver == "spf":
        solver = SPF()
        checkpoint_sha256 = None
    else:
        saved_path, saved_name = MODEL_PATHS[args.solver]
        cfg = NamedDict.load(f"{saved_path}/solver_config.json")
        cfg.device = "cpu"  # only runtime device changes; weights and network dimensions unchanged
        solver = create_solver(env.obs_dim, env.action_dim, cfg)
        path = Path(f"{saved_path}/models/best_model/{saved_name}.pth")
        checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        solver.load_models(str(path.parent))
    solver.set_eval()

    output_dir = Path(args.output_root) / args.mode / args.solver
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "environment.json").write_text(json.dumps(env_cfg.to_dict(), indent=2))
    records = []
    for k in range(args.seeds):
        seed = args.base_seed + 1000*k
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        env.reset(seed=seed, start_time=0)
        env.run(solver)
        record = compute_metrics(env, seed, solver.name, args.mode, args.seconds)
        record["checkpoint_sha256"] = checkpoint_sha256
        records.append(record)
        print("PAPER_EVAL_RESULT " + json.dumps(record, sort_keys=True), flush=True)

    (output_dir / "runs.json").write_text(json.dumps(records, indent=2))
    with (output_dir / "summary.csv").open("w") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print("PAPER_EVAL_DONE " + json.dumps({"solver": solver.name, "mode": args.mode, "seeds": args.seeds, "seconds": args.seconds, "dir": str(output_dir)}), flush=True)


if __name__ == "__main__":
    main()
